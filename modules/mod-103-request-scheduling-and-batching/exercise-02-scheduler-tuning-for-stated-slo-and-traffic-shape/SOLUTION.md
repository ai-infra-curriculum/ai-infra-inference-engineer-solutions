# Solution — Exercise 02: Scheduler Tuning for a Stated SLO and Traffic Shape

> Paired learning file: [`lessons/mod-103-request-scheduling-and-batching/exercises/exercise-02-scheduler-tuning-for-stated-slo-and-traffic-shape.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-103-request-scheduling-and-batching/exercises/exercise-02-scheduler-tuning-for-stated-slo-and-traffic-shape.md)

This exercise asks the learner to turn a *stated* workload / SLO spec into a *defended* scheduler configuration, pinned in version control with a regression test. The six artefacts are interdependent: Artefact A gates the sweep, Artefact B gates the sweep points, Artefact C makes the sweep reproducible, Artefact D is the sweep, Artefact E is the output, Artefact F is what outlasts the sweep.

The numbers in the worked example below are **illustrative** — they are not benchmark claims. The learner's sweep produces their numbers against their hardware; the reviewer's job is to check the *method*.

---

## 1. Solution overview

Six artefacts, in the order they must be produced:

| Artefact | Form | Pass bar |
|---|---|---|
| A — Workload / SLO spec | 1 page markdown | Specific numbers; no "approximate" / "typical" |
| B — First-pass config | 1 page prose, derivable from A | Reviewer can approve or push back on specific choices |
| C — Load generator | Script (Python recommended) | Reproducible given seed; writes machine-parseable trace |
| D — Pareto sweep | Table + two plots | ≥ 4 configs plotted, SLO lines overlaid, chosen point identified |
| E — Pinned config | YAML/JSON with reasoning comments | Reviewer unfamiliar with sweep can read and understand *why* |
| F — Regression test | CI-callable script | Fails cleanly on baseline drift |

The sweep is over the Cartesian product of `max-num-seqs ∈ {64, 128, 192, 256}` × `max-num-batched-tokens ∈ {4096, 8192, 16384, 32768}`, ideally 16 points, acceptable as a 4-point diagonal when GPU-hours are tight. The method assumes chunked prefill is **on** throughout this sweep (chunk-size tuning is deferred to [exercise 03](../exercise-03-chunked-prefill-ttft-tpot-tradeoff-sweep/SOLUTION.md)).

The pick rule from chapter 02: the Pareto-frontier point with **≥ 20% SLO headroom** on both TTFT P95 and TPOT P95, and the *lower* preemption rate among tied options.

---

## 2. Implementation

### 2.1 Artefact A — worked workload / SLO spec

The chapter's `chat-prod-us` pool, verbatim as the worked example:

```markdown
## Workload: chat-prod-us

- Model: Llama-3.1-70B-Instruct, bf16, TP=8 on 8×H200
- Mean RPS: 20
- Peak RPS (P95 over 1 min): 45
- Prompt length distribution: P50 = 800 tok, P95 = 4000 tok, P99 = 8000 tok
- Output length distribution: P50 = 200 tok, P95 = 600 tok, P99 = 1500 tok
- Workload mix: 70% chat, 30% RAG (longer prompts)
- Hit-max fraction: 4%

## SLO

- TTFT P95: 1500 ms
- TPOT P95: 50 ms
- E2E P95 (optional): 5000 ms
- Admission policy: 429 after queue depth > 100
- Cost ceiling: $1.20 per 1M input tokens, $3.00 per 1M output tokens
```

The hardware targets and the specific numbers above are **illustrative** from the chapter; a real `chat-prod-us` pool's targets should come from product / traffic telemetry.
### 2.2 Artefact B — worked first-pass config

```markdown
- max_num_seqs = 128
  Reason: On a dense 70B over TP=8 on H200, the decode knee is in the 64–256 range
  (chapter 02 rule-of-thumb). Starting mid-range keeps headroom both ways.
- max_num_batched_tokens = 8192
  Reason: P95 prompt is 4000 tokens; chunk size 1024 makes each iteration up to
  1024 prefill + up to 128 decode tokens = well under 8192 budget.
- enable_chunked_prefill = true, chunk-size ~ 1024
  Reason: P95 prompt is long enough that chunking is likely to protect decode TPOT.
  Final chunk size is tuned in exercise 03.
- scheduler_delay = off
  Reason: Peak-to-mean RPS ratio is 2.25× — not bursty enough to justify batching
  wait. Reconsider if burst mode in Artefact C's load generator shows queueing.
- swap_space = 16 GB
  Reason: Rough KV-per-sequence for 70B at 4k tokens is in the 1–2 GB range per
  sequence per TP slice; 16 GB CPU swap leaves room for ~10 evictions in flight.
  Validate via block-manager metrics; exercise 04 formalises this.
- gpu_memory_utilization = 0.92
  Reason: Leaves 8% for activations / CUDA workspace headroom per vLLM docs.
```

The reviewer reads this before approving the sweep. The reasoning must be derivable from Artefact A; if the learner's first-pass pick is unsupported by the spec, send them back.

### 2.3 Artefact C — reference load generator (sketch)

A reference scaffold; the learner adapts to their engine's API. The script below is OpenAI-SDK-shape (vLLM's `/v1/chat/completions` endpoint accepts it; SGLang does too; TRT-LLM via Triton wraps differently and needs its own client).

```python
# load_gen.py — reproducible SSE-streaming load generator
# Deps: httpx, numpy, openai (optional for SDK path)
import argparse
import asyncio
import csv
import os
import random
import time
from dataclasses import dataclass, asdict

import numpy as np
from openai import AsyncOpenAI


@dataclass
class TraceRow:
    arrival_ts: float
    prompt_tokens_requested: int
    output_tokens_requested: int
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    status: str  # "ok" | "error"


def sample_prompt_len(rng: random.Random) -> int:
    # Three buckets matching Artefact A's P50/P95/P99 markers.
    bucket = rng.choices(["p50", "p95", "p99"], weights=[0.70, 0.25, 0.05])[0]
    if bucket == "p50":
        return int(rng.gauss(800, 200))
    if bucket == "p95":
        return int(rng.gauss(4000, 500))
    return int(rng.gauss(8000, 1000))


def sample_output_len(rng: random.Random) -> int:
    bucket = rng.choices(["p50", "p95", "p99"], weights=[0.70, 0.25, 0.05])[0]
    if bucket == "p50":
        return max(32, int(rng.gauss(200, 50)))
    if bucket == "p95":
        return max(32, int(rng.gauss(600, 100)))
    return max(32, int(rng.gauss(1500, 200)))


async def one_request(client: AsyncOpenAI, rng: random.Random, rows: list, model: str):
    prompt_tokens = sample_prompt_len(rng)
    output_tokens = sample_output_len(rng)
    # Randomize each token so prompts do NOT share a long repeated-"token" prefix.
    # The pinned exercise-02 config enables prefix caching; a shared prefix would
    # let every request after warmup reuse cached blocks instead of prefilling,
    # biasing the sweep (and exercise 03's chunked-prefill measurement) toward
    # artificially low TTFT and prefill cost.
    prompt = " ".join(f"w{rng.randrange(1_000_000)}" for _ in range(prompt_tokens))
    arrival = time.perf_counter()
    first_token_at = None
    last_token_at = None
    token_count = 0
    status = "ok"
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=output_tokens,
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                last_token_at = now
                token_count += 1
    except Exception:
        status = "error"

    end = time.perf_counter()
    ttft_ms = (first_token_at - arrival) * 1000 if first_token_at else float("nan")
    tpot_ms = (
        ((last_token_at - first_token_at) * 1000 / max(token_count - 1, 1))
        if (first_token_at and last_token_at and token_count > 1)
        else float("nan")
    )
    e2e_ms = (end - arrival) * 1000
    rows.append(
        TraceRow(arrival, prompt_tokens, output_tokens, ttft_ms, tpot_ms, e2e_ms, status)
    )


async def driver(rps: float, duration_s: float, burst: bool, seed: int, base_url: str, model: str, out: str):
    rng = random.Random(seed)
    np.random.seed(seed)
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    rows: list = []
    tasks: list = []
    end = time.perf_counter() + duration_s
    while time.perf_counter() < end:
        # Poisson arrival: inter-arrival ~ Exp(1 / rps).
        current_rps = rps * (3.0 if burst and (time.perf_counter() - (end - duration_s)) < 60 else 1.0)
        dt = rng.expovariate(current_rps)
        await asyncio.sleep(dt)
        tasks.append(asyncio.create_task(one_request(client, rng, rows, model)))
    await asyncio.gather(*tasks, return_exceptions=True)

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TraceRow.__dataclass_fields__.keys())
        for r in rows:
            w.writerow(asdict(r).values())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rps", type=float, default=20.0)
    p.add_argument("--duration-s", type=float, default=600.0)
    p.add_argument("--burst", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base-url", default="http://localhost:8001/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    asyncio.run(driver(args.rps, args.duration_s, args.burst, args.seed, args.base_url, args.model, args.out))
```

Notes for the learner:

- The three prompt-length buckets *must* be heterogeneous. A fixed-length benchmark lies about tail latency; the chapter is explicit.
- The seed must be recorded alongside the sweep run; `git rev-parse HEAD` of the generator must be captured.
- SSE streaming is non-negotiable. TTFT without streaming is a placebo.
- The per-token randomization breaks prefix-cache reuse so the sweep measures real prefill cost against prefix caching enabled; for prefix-cache *hit-rate* experiments (later modules), substitute a corpus-backed generator that replays realistic shared-prefix patterns.
- vLLM's in-tree `benchmarks/benchmark_serving.py` is an acceptable base to extend if preferred; name the script's SHA in the writeup.
### 2.4 Artefact D — worked Pareto sweep

**Sweep protocol:**

For each `(max_num_seqs, max_num_batched_tokens)` cell:

1. Launch engine with the pinned Artefact B config overriding only the two swept knobs.
2. 5-minute warmup (discarded), 10-minute measured window.
3. Load generator at mean RPS (Artefact C, seed 0, baseline mode).
4. Record per `/metrics` scrape at 10 s: `vllm:gpu_cache_usage_perc`, `vllm:num_preemptions_total`, `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:time_to_first_token_seconds` histogram, `vllm:time_per_output_token_seconds` histogram.5. Compute from the trace CSV: TTFT P50/P95/P99, TPOT P50/P95/P99, E2E P95, throughput (tok/s and req/s), preemption rate, queue-time P95.

**Example output table** (shape; values are illustrative placeholders):

| `max_num_seqs` | `max_num_batched_tokens` | TTFT P95 (ms) | TPOT P95 (ms) | Tput (tok/s) | Preempt rate | KV usage max |
|---|---|---|---|---|---|---|
| 64 | 4096 | 950 | 32 | 1800 | 0.0% | 72% |
| 64 | 8192 | 820 | 34 | 1900 | 0.0% | 74% |
| 128 | 8192 | 1050 | 36 | 2900 | 0.2% | 84% |
| 128 | 16384 | 980 | 41 | 3100 | 0.3% | 86% |
| 192 | 16384 | 1300 | 48 | 3400 | 1.1% | 92% |
| 256 | 32768 | 1700 | 60 | 3600 | 3.8% | 97% |

(Placeholder values — the learner's sweep produces their numbers.)

**Plot 1 (TTFT P95 vs. throughput):** scatter the configs, overlay the horizontal line at `TTFT SLO = 1500 ms`. The Pareto frontier is the lower-right envelope.

**Plot 2 (TPOT P95 vs. throughput):** scatter the configs, overlay the horizontal line at `TPOT SLO = 50 ms`.

**Worked pick:** `(128, 8192)` sits at TTFT P95 1050 ms (30% headroom under 1500) and TPOT P95 36 ms (28% headroom under 50), with 0.2% preempt rate and 84% KV usage max. `(128, 16384)` and `(192, 16384)` push throughput higher (3100 and 3400 tok/s respectively) but fail the 20% TPOT headroom floor (18% and 4%). `(64, 8192)` has more headroom on both axes but gives up ≥ 30% throughput. The Pareto pick is `(128, 8192)` — the highest-throughput cell that still clears the 20% floor on both SLO axes.

The 20%-headroom rule is a floor, not a ceiling; a cell with higher headroom and lower preemption at a few percent throughput cost (e.g. `(64, 8192)` here) is also a valid pick with a different risk posture (favouring stability over throughput). The reviewer's job is to check the reasoning against the floor, not to insist on one cell — but a candidate below the floor on either axis must be rejected regardless of its throughput.

### 2.5 Artefact E — pinned config (worked)

```yaml
# serving-config/chat-prod-us.yaml
# Generated by exercise-02 sweep on 2026-XX-XX; sweep trace at s3://.../sweep-2026-XX-XX/.
# Engine: vllm==<pinned-version>, SHA <pinned-sha>.
# Hardware: 8× H200, TP=8.
model: meta-llama/Llama-3.1-70B-Instruct
dtype: bfloat16
tensor_parallel_size: 8
max_model_len: 16384
max_num_seqs: 128
max_num_batched_tokens: 8192
enable_chunked_prefill: true
enable_prefix_caching: true
swap_space: 16            # GB per worker
gpu_memory_utilization: 0.92

# Reasoning
# ---------
# - max_num_seqs = 128: Pareto knee for Llama-3.1-70B on 8×H200 under chat-prod-us
#   (TTFT P95 1050 ms vs. 1500 ms SLO, 30% headroom; TPOT P95 36 ms vs. 50 ms
#   SLO, 28% headroom — clears the 20% floor on both axes).
# - max_num_batched_tokens = 8192: highest-throughput sweep cell that still clears
#   the 20% TPOT headroom floor; preemption rate 0.2% sustained. Larger budgets
#   (16384, 32768) gain a few percent throughput but fall below the floor.
# - chunked prefill ON; chunk size to be refined in exercise 03.
# - swap_space 16GB: sized for 70B KV footprint (see block-manager notes);
#   re-verified in exercise 04 preemption drill.
# - gpu_memory_utilization = 0.92: vLLM default-ish headroom for activations
#   and CUDA workspace.
# Trace: sweep points, plots, CSVs under <artifact-store>.
```

### 2.6 Artefact F — regression test (sketch)

```bash
#!/usr/bin/env bash
# regression/chat-prod-us-sweep.sh
# Nightly regression against the pinned Artefact E config.
set -euo pipefail

CONFIG="serving-config/chat-prod-us.yaml"
BASELINE="regression/baseline-chat-prod-us.json"
TRACE_DIR="regression/traces/$(date -u +%Y-%m-%dT%H%M%SZ)"
mkdir -p "$TRACE_DIR"

# 1. Launch engine (background) from pinned config.
./scripts/launch-vllm-from-yaml.sh "$CONFIG" > "$TRACE_DIR/engine.log" 2>&1 &
ENGINE_PID=$!
trap 'kill $ENGINE_PID 2>/dev/null || true' EXIT

# 2. Wait for /metrics to come up (health probe).
./scripts/wait-for-metrics.sh http://localhost:8001/metrics 120

# 3. Run load generator at mean RPS for 10 min (5 min warmup is inside the script).
python load_gen.py \
  --rps 20 \
  --duration-s 900 \
  --seed 0 \
  --base-url http://localhost:8001/v1 \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --out "$TRACE_DIR/trace.csv"

# 4. Scrape final /metrics snapshot.
curl -s http://localhost:8001/metrics > "$TRACE_DIR/metrics.txt"

# 5. Compute today's TTFT/TPOT/Tput/preempt-rate and compare to baseline.
python regression/compare.py \
  --trace "$TRACE_DIR/trace.csv" \
  --metrics "$TRACE_DIR/metrics.txt" \
  --baseline "$BASELINE" \
  --tolerance-latency 0.10 \
  --tolerance-throughput 0.20 \
  --out "$TRACE_DIR/report.json"

# 6. compare.py exits non-zero on drift; CI fails here.
```

The `compare.py` helper reads baseline (stored at sweep-pick time) and today's numbers and exits non-zero if TTFT P95 or TPOT P95 drift > ±10% or throughput drifts > ±20%. The regression test is what outlasts the one-time sweep: when the engine upgrades, when the workload drifts, this is the alarm.

---

## 3. Validation steps

A reviewer checks:

1. **Artefact A completeness.** Every field filled with specific numbers; no "approximate." If any `<N>` remains, reject.
2. **Artefact B derivation.** Each chosen value traces to a line in Artefact A. If the reasoning references the chapter without referencing the learner's spec, the reasoning is generic — reject.
3. **Artefact C reproducibility.** Run the load generator twice with the same seed against a fixture engine; the two trace CSVs should be byte-identical (modulo wall-clock arrival timestamps, which differ by a few ms). The prompt-length and output-length samples per row must match.
4. **Artefact D method.** Count distinct sweep points on the plot (≥ 4). Confirm SLO lines are visibly overlaid. Confirm the chosen point is identified. Confirm it clears both SLO axes with ≥ 20% headroom. Confirm the preemption rate at the chosen point is the lower among tied points.
5. **Artefact E reasoning density.** A reviewer who did not run the sweep reads the YAML comments and understands *why* each value was chosen. If any value is pinned without a comment, reject that line.
6. **Artefact F end-to-end.** Run the regression script against the fixture engine — it exits 0 against the baseline. Perturb one knob (`max_num_seqs=1`) in a copy of the YAML and re-run; the script must exit non-zero.

---

## 4. Rubric or review checklist

Pass threshold: ≥ 16 / 22.

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| Artefact A specificity | Fields empty or "typical" | Fields filled but missing distribution tail (P99) or workload mix | All fields with concrete numbers; distributions have P50/P95/P99 |
| Artefact B derivation | Config values unsupported by A | Partial derivation; some values arbitrary | Every value derives from A with a cited rule |
| Artefact C reproducibility | Not deterministic across seeds | Deterministic but single-length prompts | Deterministic; ≥ 3 distinct prompt-length buckets |
| Artefact C streaming | TTFT measured after full response | TTFT via SSE but TPOT missing | SSE streaming; TTFT and TPOT both per-request |
| Artefact D coverage | < 4 points, no plots | 4+ points, one plot only | ≥ 4 points, both plots, SLO lines overlaid |
| Artefact D continuous metrics | Only final snapshot | Snapshots every minute | Snapshots every 5–10 s during each run |
| Artefact D pick justification | "The one that worked" | Chosen point named, but no headroom check | Pick defended against both SLO axes with named headroom and preemption rate |
| Artefact E reasoning comments | No comments | Partial reasoning | Every non-obvious value carries a reasoning comment |
| Artefact F clean fail path | Fails only on crash | Fails on latency drift only | Fails on latency *and* throughput drift; writes dated trace |
| Spec-first discipline | Sweep run before spec | Spec written during sweep | Spec committed before any sweep commits |
| Warmup honesty | No warmup | < 3 min warmup | ≥ 5 min warmup discarded |

---

## 5. Common mistakes

- **No-spec tuning.** Starting the sweep without a written spec means the learner can't tell "pass" from "fail." If Artefact A is missing, send them back before looking at D.
- **Fixed-length prompts.** A benchmark with a single prompt length lies about tail latency; chapter 02 is explicit that distributions must be heterogeneous.
- **No warmup.** The first few minutes of any new config are dominated by CUDA graph capture and prefix-cache fill. Discard at least 3 minutes; 5 is the chapter's rule.
- **Snapshot-only metrics.** A clean final P95 can hide a 2-minute mid-run spike. Scrape `/metrics` every 5–10 s during the sweep.
- **No preemption observation.** If every sweep run has zero preemptions, the engine is not under pressure — the Pareto pick is likely over-provisioned. Add the P95 burst mode or lower `max-num-seqs` in a stress run to confirm the pick holds.
- **Over-fitting the sweep.** 20% SLO headroom is a floor. Picking the point with 2% headroom leaves no room for traffic growth, engine upgrades, or workload drift.
- **Pinning without comments.** A config that is "the one that worked" without reasoning is one engine upgrade away from being re-derived from scratch.
- **Regression test that never fails.** A regression job that always exits 0 is noise. Verify it actually fails under a provoked perturbation.
- **Treating the 4-point diagonal as the full sweep.** It is a cost-constrained compromise; name the limitation explicitly (which cells were skipped, what risk comes with that).

---

## 6. References

Primary research the Pareto method draws on:

- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022 — <https://www.usenix.org/conference/osdi22/presentation/yu>.
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023 — <https://arxiv.org/abs/2309.06180>.
- Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, OSDI 2024 — <https://www.usenix.org/conference/osdi24/presentation/agrawal> — the TTFT/TPOT trade-off framework at the core of the Pareto.

Engine documentation:

- vLLM docs — <https://docs.vllm.ai/>; engine args reference via `vllm serve --help`; metrics reference — <https://docs.vllm.ai/en/latest/serving/metrics.html>.
- SGLang docs — <https://docs.sglang.ai/>.
- TensorRT-LLM docs — <https://nvidia.github.io/TensorRT-LLM/>.

Benchmarking harnesses to extend rather than replace:

- vLLM `benchmarks/benchmark_serving.py` (in-tree; verify path per release).
- SGLang `benchmark/` tree (in-tree).
- Locust — <https://locust.io/>; `wrk` — <https://github.com/wg/wrk>.

SLO / SRE methodology:

- SRE book — Service Level Objectives — <https://sre.google/sre-book/service-level-objectives/>.
- SRE workbook — Alerting on SLOs — <https://sre.google/workbook/alerting-on-slos/>.

Observability for sweep metrics scraping:

- Prometheus histogram practices — <https://prometheus.io/docs/practices/histograms/>.
- Full observability wiring lands in [exercise 05](../exercise-05-scheduler-metrics-into-prometheus/SOLUTION.md).

Paired learning content:

- Module 103 chapter 02 (Scheduler Configuration for a Stated SLO) in [`ai-infra-inference-engineer-learning`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning) — the arithmetic and worked `chat-prod-us` example this solution scaffolds against.
- Exercise 01 inner-loop cheatsheet — reused as the mental model when reading sweep results.
