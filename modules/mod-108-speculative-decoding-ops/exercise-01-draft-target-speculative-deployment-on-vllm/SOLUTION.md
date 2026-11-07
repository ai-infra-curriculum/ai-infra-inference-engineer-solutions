# Solution — Exercise 01: Draft-Target Speculative Deployment on vLLM

> Paired learning file: [`lessons/mod-108-speculative-decoding-ops/exercises/exercise-01-draft-target-speculative-deployment-on-vllm.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-108-speculative-decoding-ops/exercises/exercise-01-draft-target-speculative-deployment-on-vllm.md)
> Module: `mod-108-speculative-decoding-ops`
> Type: implementation / deployment walkthrough

## 1. Solution overview

The learner stands up a **draft-target speculative decoding** deployment on vLLM
and compares it against a non-speculative baseline for the same target model, same
prompts, and same concurrency. This is a *deployment* exercise, not a research
benchmark — the output is a working configuration, evidence that speculation is
actually firing (acceptance-rate counters are non-zero and non-trivial), and a
small table that shows where speculation helps and where it does not.

The three artefacts:

| Artefact | Form | Pass bar |
|---|---|---|
| A — Baseline + spec-decode launch commands | Shell script or Makefile | Both engines serve the same target model with pinned vLLM and identical sampling settings; only speculation differs |
| B — Correctness check | Script + output | Greedy outputs from baseline and spec-decode match for the same prompt and seed (within fp tolerance) |
| C — Acceptance + latency table | Table + method note | Reports `acceptance_rate`, `num_speculative_tokens`, p50/p95 latency, and throughput for at least two spec lengths; method note states prompt/output lengths, concurrency, warm-up |

Pin the vLLM version in the launch commands and the draft checkpoint. Speculative
decoding flag names and defaults shift between vLLM minor releases; verify every
flag against `vllm serve --help` in the pinned release before copying.

### Why speculative decoding exists at all

Autoregressive decoding is memory-bandwidth-bound: each decode step reads the full
weight set (or its KV working set) to produce one token. **Speculative decoding**,
introduced by [Leviathan et al. (ICML 2023)](https://arxiv.org/abs/2211.17192) and
independently by [Chen et al. (2023)](https://arxiv.org/abs/2302.01318), lets a
small, fast **draft model** propose `k` tokens that the target model **verifies in
a single forward pass**. Accepted tokens are emitted; rejected tokens trigger a
corrected sample from the target's distribution. The math guarantees that the
distribution of emitted tokens is identical to greedy or sampling from the target
model alone — speculation is **lossless** with respect to the target.

The speed-up depends on two things:

1. **Acceptance rate `α`**: how often the target accepts the draft's token.
2. **Draft/verify cost ratio**: cost of one draft forward pass + cost of one
   target verification pass, vs. cost of `E[accepted]` plain decode steps.

With acceptance rate `α` and speculation length `k`, the expected accepted-tokens
count per round is `(1 − α^(k+1)) / (1 − α)` (Leviathan §3). If the draft is much
cheaper than the target and `α` is high, the target's expensive forward pass
amortizes over more than one emitted token and total decode latency drops.

### The three flags that do the work

vLLM wires speculative decoding behind a small set of integer/string flags. Names
have shifted (the current form is a structured `--speculative-config` JSON string
in recent releases; older releases exposed separate flags). The learner must check
`vllm serve --help` against the pinned release.

| Concept | Current flag form (pin and verify) |
|---|---|
| Draft checkpoint | `--speculative-config '{"model": "<draft>"}'` or older `--speculative-model <draft>` |
| Speculation length `k` | `"num_speculative_tokens": <k>` or older `--num-speculative-tokens <k>` |
| Draft tensor-parallel size | `"draft_tensor_parallel_size": <n>` (optional; usually 1) |
| Spec method (draft / ngram / medusa / eagle) | `"method": "draft_model"` (default when `model` is set) |

The draft model must share the target's **tokenizer** and its **vocabulary**.
Rejection sampling compares per-token probabilities between draft and target, and
a mismatched vocabulary breaks that comparison. In practice this means picking a
draft from the same model family (e.g., Llama-3.2-1B draft for a Llama-3.1-70B
target), not a cross-family pairing.

## 2. Implementation — worked walkthrough

The commands below assume:

- A chat-sized dense target model, e.g. `meta-llama/Meta-Llama-3.1-70B-Instruct`.
- A same-family small draft model, e.g. `meta-llama/Llama-3.2-1B-Instruct`.
- One node of 8× 80 GiB-class accelerators (TP = 8 for the target; the draft is
  cheap enough to co-locate on the same ranks).
- A vLLM version pinned in `requirements.txt` or a container image tag.

Substitute model, draft, and hardware for whatever the learner's lab provides.

### 2.1 Baseline — target-only, no speculation

The control. Same target, same TP, same sampling settings — only speculation is
off. Latency and throughput numbers from this engine are what speculation has to
beat.

```bash
# config_baseline.sh
export HF_HOME=/scratch/hf
export VLLM_LOGGING_LEVEL=INFO

vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --disable-log-requests \
  --host 0.0.0.0 --port 8001
```

### 2.2 Spec-decode engine — draft + target

Same target, same TP, same max length, but with the draft model wired in and `k`
pinned. The exact flag form must be verified against the pinned vLLM release.
Below is the structured-config form; the older flat-flag form is shown in a
comment for readers on an older release.

```bash
# config_spec.sh
vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --disable-log-requests \
  --speculative-config '{
      "model": "meta-llama/Llama-3.2-1B-Instruct",
      "num_speculative_tokens": 5,
      "draft_tensor_parallel_size": 1
    }' \
  --host 0.0.0.0 --port 8002

# Older flat-flag equivalent (verify against your pinned vLLM release):
# --speculative-model meta-llama/Llama-3.2-1B-Instruct \
# --num-speculative-tokens 5 \
# --speculative-draft-tensor-parallel-size 1
```

Note: `--gpu-memory-utilization` is dropped a notch vs. baseline to leave headroom
for the draft model's weights and its KV cache. The draft is small (1 B class) but
its KV is not free; sizing it ahead of time avoids an OOM on the first long prompt.

Expected engine-log evidence that speculation is actually enabled:

```
INFO ... Initializing speculative decoding...
INFO ... Draft model: meta-llama/Llama-3.2-1B-Instruct
INFO ... num_speculative_tokens=5
INFO ... Scoring method: rejection-sampling
INFO ... Engine ready at http://0.0.0.0:8002
```

If these lines do not appear, speculation is not on and the measurements below are
meaningless.

### 2.3 Correctness check — greedy outputs must match

Speculative decoding is **lossless** for the target's distribution. Under greedy
sampling, the tokens emitted by the spec-decode engine must match the baseline
engine's tokens for the same prompt and seed, modulo floating-point tolerance.
This is the first thing to verify.

```python
# correctness_check.py
import httpx, json, sys

PROMPT = "List five properties of speculative decoding, each in one sentence."
BODY = {
    "prompt": PROMPT,
    "max_tokens": 128,
    "temperature": 0.0,
    "seed": 17,
    "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
}

def fetch(url):
    r = httpx.post(url, json=BODY, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["text"]

base = fetch("http://localhost:8001/v1/completions")
spec = fetch("http://localhost:8002/v1/completions")

if base == spec:
    print("MATCH — spec-decode output identical to baseline.")
    sys.exit(0)

print("DIVERGED — first 200 chars of each:\n--- baseline ---")
print(base[:200])
print("--- spec ---")
print(spec[:200])
sys.exit(1)
```

Divergence on greedy sampling is almost always one of:

- Draft and target use different tokenizers (vocab mismatch).
- Baseline and spec engines were launched with different sampling overrides.
- The pinned vLLM release has a known bug; check its GitHub issues against the
  speculative-decoding label.
- Numerical drift from a kernel difference — rare, but worth checking by matching
  only the first N tokens before declaring failure.

### 2.4 Load generator with acceptance-rate read-out

Use the same minimal generator pattern as module 107 (prompt-length bucket, output-
length bucket, concurrency, three runs), and additionally scrape vLLM's spec-decode
Prometheus metrics after the run.

```python
# load_gen_spec.py — minimal spec-aware generator
import argparse, asyncio, re, time
import httpx

METRICS_LINES = [
    "vllm:spec_decode_num_accepted_tokens",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_emitted_tokens",
    "vllm:spec_decode_draft_acceptance_rate",
]

async def one_request(client, url, body):
    t0 = time.perf_counter()
    r = await client.post(url, json=body, timeout=180)
    r.raise_for_status()
    return (time.perf_counter() - t0) * 1000.0

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--metrics-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--requests", type=int, default=128)
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--max-tokens", type=int, default=128)
    args = p.parse_args()

    prompt = "token " * args.prompt_tokens
    body = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
    }
    sem = asyncio.Semaphore(args.concurrency)

    async def guarded(client):
        async with sem:
            return await one_request(client, args.url, body)

    async with httpx.AsyncClient() as client:
        # Warm-up — discard
        await asyncio.gather(*[guarded(client) for _ in range(8)])

        t0 = time.perf_counter()
        results = await asyncio.gather(*[guarded(client) for _ in range(args.requests)])
        wall = time.perf_counter() - t0

        m = (await client.get(args.metrics_url)).text

    def grab(name):
        for line in m.splitlines():
            if line.startswith(name) and not line.startswith("#"):
                return float(line.split()[-1])
        return None

    lat = sorted(results)
    p50 = lat[len(lat)//2]
    p95 = lat[int(len(lat)*0.95)]
    thr = args.requests / wall
    print(f"wall={wall:.1f}s req/s={thr:.2f} p50={p50:.0f}ms p95={p95:.0f}ms")
    for n in METRICS_LINES:
        print(f"{n} = {grab(n)}")

asyncio.run(main())
```

Run against baseline and spec-decode:

```bash
python load_gen_spec.py \
  --url http://localhost:8001/v1/completions \
  --metrics-url http://localhost:8001/metrics \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --concurrency 16 --requests 128 \
  --prompt-tokens 512 --max-tokens 128

python load_gen_spec.py \
  --url http://localhost:8002/v1/completions \
  --metrics-url http://localhost:8002/metrics \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --concurrency 16 --requests 128 \
  --prompt-tokens 512 --max-tokens 128
```

### 2.5 Sweep `num_speculative_tokens`

The best `k` is a function of acceptance rate and draft/target cost ratio.
Sweep `k ∈ {3, 5, 7}` at the same concurrency and record `α` and p50/p95.

The qualitative pattern: as `k` grows, verification pays for more tokens per round
when `α` is high, but wasted draft work grows when `α` is low. A `k` that is "too
large" hurts tail latency at low acceptance.

### 2.6 Worked measurement table (illustrative)

The numbers below are **placeholders demonstrating expected relative ordering**;
the learner reports their own numbers. Reviewers judge the method, not these
figures. Expected sign of each column is called out below.

| Config | `k` | Draft | α (accept rate) | req/s | p50 lat (ms) | p95 lat (ms) |
|---|---|---|---|---|---|---|
| Baseline (no spec) | — | — | — | *measured* | *measured* | *measured* |
| Spec, draft=1B | 3 | Llama-3.2-1B | *measured* | *measured* | *measured* | *measured* |
| Spec, draft=1B | 5 | Llama-3.2-1B | *measured* | *measured* | *measured* | *measured* |
| Spec, draft=1B | 7 | Llama-3.2-1B | *measured* | *measured* | *measured* | *measured* |

Expected qualitative pattern on same-family, well-matched draft:

- `α` typically sits in the **0.6–0.8** band for chat workloads on open-weights
  Llama-class pairings in published practitioner reports; verify against measured
  numbers in the pinned vLLM release. Lower bands exist for mismatched domains
  (code vs. chat) and higher bands for narrow-domain fine-tunes.
- p50 latency drops by a factor roughly bounded by `E[accepted]` per round; the
  Leviathan paper's §3 formula is the right mental model.
- Throughput can **drop** at high concurrency: the engine spends GPU cycles on
  draft work that could have fed more parallel decode slots. If concurrency × decode
  work already saturates the target, speculation stops helping and starts hurting.

If measured `α` is below 0.3, speculation is fighting itself — the draft is wrong
too often, the extra verification pass dominates, and the engine is slower than
baseline. Investigate the draft/target mismatch (§5) before tuning `k`.

## 3. Validation steps

A reviewer validating a learner submission should perform the following in order.

1. **Launch commands pinned.** Both engines reference an explicit vLLM version
   (requirements or image tag), explicit target and draft model paths, explicit
   `num_speculative_tokens`, and identical sampling defaults.
2. **Spec is actually on.** Engine log from §2.2 shows the draft model name,
   `num_speculative_tokens`, and the scoring method. `/metrics` exposes
   `vllm:spec_decode_*` series.
3. **Correctness check passes.** §2.3 produces identical greedy outputs across
   the two engines, or the learner documents any token-position where they diverge
   and explains it (kernel fp drift, seed difference).
4. **Acceptance rate is realistic.** Measured `α` lies in the published range for
   same-family pairings (practitioner reports: ~0.5–0.8 for well-matched chat
   drafts). If below 0.3, the submission must investigate (§5) before reporting.
5. **Sweep over `k`.** At least two values of `num_speculative_tokens` are
   measured and compared, not just one.
6. **Method note.** Reports (a) prompt length, (b) output length, (c) concurrency,
   (d) warm-up policy, (e) explicit acknowledgement that this is a smoke
   measurement, not a benchmark. (Same discipline as module 107 exercise 02.)
7. **Teardown.** Clean shutdown of each engine; no zombie workers before the next
   config launches.

## 4. Rubric / review checklist

Score each item 0 (missing), 1 (partial), 2 (fully correct). Passing: ≥ 10 / 14.

| # | Item | Points |
|---|---|---|
| 1 | Baseline and spec-decode launches run clean on the same target model | 2 |
| 2 | Launch commands pin vLLM version, target, draft, and `num_speculative_tokens` | 2 |
| 3 | Engine logs cited that show speculation is on and the draft model loaded | 2 |
| 4 | Correctness check verifies greedy outputs match (or divergence is explained) | 2 |
| 5 | Acceptance-rate counter recorded; sweep over at least 2 values of `k` | 2 |
| 6 | Latency / throughput table filled with method note | 2 |
| 7 | Memory-fit sanity: `--gpu-memory-utilization` reduced vs. baseline to fit draft + its KV | 2 |

## 5. Common mistakes

- **Draft/target vocabulary mismatch.** Cross-family pairings (e.g., a Mistral
  draft against a Llama target) break rejection sampling. Acceptance collapses or
  the engine refuses to start. Fix: pick a draft from the same family / tokenizer.
- **Calling the draft "speculative" when it's actually prompt-lookup / ngram.**
  vLLM supports ngram speculation (`"method": "ngram"`) — useful for repetitive
  outputs — but it is a different method with different acceptance dynamics. Name
  the method explicitly in the launch command; do not conflate.
- **Measuring with temperature > 0 and no seed, then surprised at non-matching
  outputs.** Speculation preserves the target's distribution, not its samples; two
  independent sampling runs can and will diverge. Use greedy or a pinned seed for
  correctness checks.
- **Running the measurement without warm-up.** The first request pays CUDA graph
  capture and model-weight touch costs on both baseline and spec-decode. Discard
  the first N requests.
- **Omitting `--gpu-memory-utilization` reduction.** The draft model and its KV
  cache consume HBM; carrying the baseline's utilization forward will OOM on the
  first long prompt. Drop it by 5–10 points as a starting heuristic.
- **Testing with trivial prompts.** Short prompts generate few decode steps,
  speculation has few opportunities to help, and the measurement becomes noise.
  Use output lengths in the 128–512 band.
- **Picking `k` from a blog post.** The optimal `k` is specific to the draft/target
  pair, the hardware, and the workload. Sweep; do not copy a number.
- **Assuming speculation helps at high concurrency.** When the target engine is
  already running continuous batches that saturate decode, speculation steals
  cycles from that batch to run drafts. At some concurrency ceiling speculation
  stops helping and starts hurting; the sweep must include a high-concurrency
  point.
- **Reporting a win from aggregate throughput alone.** Speculation is primarily a
  **latency** optimization. Aggregate req/s can fall when speculation is on even
  as per-request latency drops, because the engine spends GPU time on draft work
  that would otherwise have fed parallel slots. Report both.

## 6. References

- Leviathan, Y., Kalman, M., Matias, Y. *Fast Inference from Transformers via
  Speculative Decoding.* ICML 2023. https://arxiv.org/abs/2211.17192 — the
  canonical formulation, rejection-sampling verification, expected-token formula.
- Chen, C. et al. *Accelerating Large Language Model Decoding with Speculative
  Sampling.* arXiv:2302.01318 — independent contemporaneous derivation from
  DeepMind.
- vLLM documentation. *Speculative Decoding.* https://docs.vllm.ai/ —
  practitioner implementation reference; verify flag names against the pinned
  release.
- vLLM. *Metrics / observability.* https://docs.vllm.ai/ — `vllm:spec_decode_*`
  Prometheus series, their meanings, and how they are reset.
- NVIDIA. *TensorRT-LLM Speculative Decoding.*
  https://docs.nvidia.com/deeplearning/tensorrt-llm/ — alternative practitioner
  implementation, useful cross-reference for terminology.
- Hugging Face. *Assisted Generation.*
  https://huggingface.co/docs/transformers/generation_strategies — reference
  practitioner implementation of draft-target decoding in the Transformers
  library.
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/)
  — not invoked for this deployment exercise. Draft-model provenance is covered
  under [[draft-model-lifecycle-contract]] in exercise 04.
- [mitre-atlas](https://atlas.mitre.org/) — not invoked for this deployment
  exercise.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework) —
  governance baseline; applies to the draft-model lifecycle in exercise 04, not
  to this deployment walkthrough directly.
