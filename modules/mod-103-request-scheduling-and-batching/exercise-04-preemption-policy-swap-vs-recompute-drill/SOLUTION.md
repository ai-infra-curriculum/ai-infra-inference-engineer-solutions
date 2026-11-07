# Solution — Exercise 04: Preemption Policy Swap-vs-Recompute Drill

> Paired learning file: [`lessons/mod-103-request-scheduling-and-batching/exercises/exercise-04-preemption-policy-swap-vs-recompute-drill.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-103-request-scheduling-and-batching/exercises/exercise-04-preemption-policy-swap-vs-recompute-drill.md)

Preemption is the scheduler's pressure-relief valve. This drill deliberately overcommits the engine under each policy mode, measures the recovery cost, computes a break-even prompt length between recompute and swap, verifies (or disproves) the starvation guard, and pins a policy with evidence.

Numbers below are **illustrative**. The learner produces their numbers against their model + hardware; the reviewer checks the method and that the numbers reconcile.

---

## 1. Solution overview

Four artefacts:

| Artefact | Form | Pass bar |
|---|---|---|
| A — Overcommit workload | Config writeup + load-generator invocation | Workload *provokes* preemption; counters move |
| B — Two preemption traces | Per-request traces for swap and recompute | Restore-TTFT tracked by request-id in both modes |
| C — Break-even analysis | Comparison table + one-paragraph verdict | Names a specific break-even prompt length for the learner's model / hardware |
| D — Starvation-guard verification | Tagged request's trace + conclusion | Either observed guard behaviour, or documented absence with mitigation |

The drill reuses: the engine from [exercise 01](../exercise-01-continuous-batching-inner-loop-walkthrough/SOLUTION.md), the load generator and metric scraping from [exercise 02](../exercise-02-scheduler-tuning-for-stated-slo-and-traffic-shape/SOLUTION.md), and the chunk-size setpoint from [exercise 03](../exercise-03-chunked-prefill-ttft-tpot-tradeoff-sweep/SOLUTION.md).

**Policy pick rule** (chapter 04):

- *Swap* preserves KV blocks on CPU DRAM; cost per preemption is a PCIe round-trip on resume. Favours long prompts (prefill cost >> transfer cost).
- *Recompute* frees KV blocks and re-prefills the prompt on resume. Favours short prompts (prefill cost << transfer cost); robust under constrained swap space.
- The **break-even prompt length** is where PCIe round-trip for that sequence's KV equals the GPU-time to re-prefill the same prompt. Longer than break-even → swap wins; shorter → recompute wins.

### Engine-specific knob mapping

**Verify flag spellings against the pinned release's docs.**
| Engine | Enable swap | Force recompute | Metric to confirm mode |
|---|---|---|---|
| vLLM | `--swap-space <GB>` sets the CPU swap pool size | Set `--swap-space 0` (or very small) so the allocator falls through | `vllm:num_preemptions_total`, plus swap-in / swap-out counters where exposed |
| SGLang | server-arg for CPU swap space | server-arg for recompute-only | verify via scheduler-manager metrics |
| TRT-LLM via Triton | `kv_cache_host_memory_bytes` in `config.pbtxt` | Set host memory to 0 | backend metrics |

Verify by watching the metric counters during a provocation run: if swap-out count stays at zero in "swap mode," the engine actually ran recompute.

---

## 2. Implementation

### 2.1 Artefact A — overcommit workload (reference recipe)

Force preemption by lowering `max-num-seqs` well below the exercise-02 Pareto setpoint while raising concurrent client count. Do **not** overcommit by raising RPS alone — it conflates prefill cost with admission pressure.

Worked config (adapt model / size to learner's hardware):

```bash
# Engine launch (swap-mode run)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --swap-space 16 \
  --gpu-memory-utilization 0.92 \
  --port 8001

# Load generator (reusing exercise-02 load_gen.py, extended per Artefact A of exercise 03
# so prompt-length is bucketed short/long).
python load_gen.py \
  --rps 60 \              # 3-5x the running-set cap
  --duration-s 600 \      # 5 min warmup + 5 min measured
  --seed 42 \
  --base-url http://localhost:8001/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --out traces/run-swap.csv

# At t=180s start a 60s burst of long-prompt-only traffic (second load_gen.py
# process with weights=[0.00, 1.00] so every prompt is long).
```

Document:

- Engine launch command (per mode).
- Load-generator seed and git SHA.
- Burst timing (t=180s, duration 60s).
- The expected preemption signature: counter rises visibly within the first 2 minutes of the measured window.

### 2.2 Artefact B — two preemption traces

**Run 1: swap mode.**

- Launch with `--swap-space 16` (enough room for several evicted sequences' KV).
- Preemption mode: swap. vLLM's default is "swap when space, else recompute"; **confirm mode by watching the swap-out counter**. If the counter does not move, swap space is undersized — raise it or acknowledge hybrid behaviour.
- Record for the 10-min run (5-min warmup + 5-min measured):
  - `vllm:num_preemptions_total` — overall preemption rate.
  - Swap-in / swap-out counters (exact names vary by release; verify against `/metrics`).  - `vllm:cpu_cache_usage_perc` (CPU swap-pool usage).
  - Per-request restore-TTFT: time from re-admission to first token on the second admission (not the arrival-to-first-token). This is the only honest measurement of "what did preemption cost this user."
  - E2E distribution for preempted vs. not-preempted sequences (split in post-processing by whether `preemptions_total` incremented for that request-id — requires scheduler-log instrumentation; a local print inside `Scheduler._preempt()` suffices).
  - Throughput tokens/sec.

**Run 2: recompute mode.**

- Launch with `--swap-space 0` (or a very small value — verify that swap-out stays at zero).
- Everything else identical: same seed, same workload spec, same measurement window.
- Record the same metrics.

Per-request trace schema (extend `load_gen.py`'s `TraceRow`):

```python
@dataclass
class TraceRow:
    request_id: str
    arrival_ts: float
    prompt_tokens: int
    output_tokens: int
    bucket: str
    first_admission_ts: float
    first_token_ts: float          # TTFT = first_token_ts - arrival_ts
    preemption_count: int          # from scheduler log
    re_admission_ts: float | None  # None if never preempted
    restore_ttft_ms: float | None  # time to first NEW token after re-admission
    e2e_ms: float
    tpot_ms: float
    status: str
```

### 2.3 Artefact C — worked break-even analysis

**Comparison table** (illustrative; numbers depend on model and PCIe):

| Metric | Run 1 (swap) | Run 2 (recompute) |
|---|---|---|
| Preemption rate | ~5% of requests | ~5% of requests |
| Mean cost per preemption (PCIe swap round-trip or GPU re-prefill time) | ~180 ms per 2k-token sequence | ~90 ms per 500-token prompt; ~1400 ms per 4000-token prompt |
| TTFT P95 on preempted sequences (restore-TTFT) | ~400 ms | ~220 ms for short; ~2200 ms for long |
| E2E P95 on preempted sequences | ~3900 ms | ~3200 ms for short; ~5100 ms for long |
| Throughput (tok/s) | ~slightly lower under swap (PCIe contention on heavy preemption runs) | ~slightly higher under recompute for short-prompt-dominant workloads, lower for long |
| GPU utilisation during post-preemption restore | drops briefly (idle waiting for KV copy-in) | spikes (re-prefill eats compute) |

**Break-even arithmetic** (chapter 04 intuition made concrete):

- *Swap cost per sequence* ≈ `KV_bytes_per_token × prompt_tokens × 2 / PCIe_GB_per_s` (×2 for round-trip: out and in). For an 8B model at bf16 the KV per token is on the order of 128 KB — verify against the model's architecture and attention head layout for your deployment.- *Recompute cost per sequence* ≈ `prefill_time_per_token × prompt_tokens`; `prefill_time_per_token` is model- and hardware-specific — compute empirically from the engine's prefill-time metric or the load generator's TTFT for cold requests.
- *Break-even*: solve `swap_cost(L) = recompute_cost(L)` for prompt length `L`. If PCIe bandwidth is 50 GB/s (PCIe Gen4 x16 effective), KV ≈ 128 KB/token, and prefill ≈ 0.5 ms/token on this model+hw, the break-even is roughly `L* ≈ (2 × 128 KB) / (50 GB/s × 0.5 ms) ≈ 1024 tokens`. **This arithmetic is illustrative** — use it as a template, substitute the learner's measured values.

**Worked verdict** (one paragraph, template):

> For this workload (80/20 short/long, 8B model, single H100) on this hardware, the break-even is ~1200 tokens (measured, see above). Our long-prompt bucket sits at P50 ≈ 500 tokens and P95 ≈ 5000 tokens — the bimodal distribution. **Recommendation:** prefer *swap mode* when `--swap-space` can accommodate 3–5 evicted sequences' KV (≈ 16 GB per worker here), because long-prompt preemptions pay back the PCIe cost; fall back to recompute when swap space is constrained. Preemption rate at the exercise-02 setpoint is only ~0.3%, so the policy-pick blast radius is small, but the long-prompt restore-TTFT savings matter for the tail users. Pin **swap mode** with the verified `--swap-space 16`.

The learner's verdict must name a specific break-even number (or a defensible range) and defend it against *their* measured values, not against this solution's placeholders. "It depends" fails the rubric.

### 2.4 Artefact D — worked starvation-guard verification

**Method 1: trace a tagged request.**

- Submit a single long-prompt (P95-sized) request with a distinctive tag in the API payload (e.g. `user="trace-canary-01"`).
- Watch the request-id in the scheduler log and the output stream.
- Record how many times it is preempted before completion.
- Compute whether the engine completed it within a reasonable upper bound (e.g. 2× expected E2E given its prompt+output lengths).

Expected observations under the worked workload:

```
[iter=0120] admit trace-canary-01 (prompt 5100 tok)
[iter=0128] preempt trace-canary-01 mode=swap
[iter=0151] re-admit trace-canary-01 (swap-in 2 blocks)
[iter=0183] preempt trace-canary-01 mode=swap
[iter=0210] re-admit trace-canary-01
[iter=0410] complete trace-canary-01
```

Learner records the count of preemption events and compares against the expected-E2E ceiling.

**Method 2: code review of the selection function.**

Reuses [exercise 01](../exercise-01-continuous-batching-inner-loop-walkthrough/SOLUTION.md) Artefact A's source tour. Look inside `Scheduler._schedule_running()` / the eviction-selection helper. Three things to check:

1. Is there a per-request *age* counter? (An admission timestamp passed through the policy.)
2. Does the eviction policy prefer the *most-recently-admitted* sequence (LIFO — a de-facto FIFO guard for the oldest-admitted) or the *least-recently-used* (LRU, which could evict a long-running sequence repeatedly)?
3. Is there a preemption cap per sequence, or a promotion-on-preemption that moves a repeatedly-preempted sequence to the head of the queue?
**Worked conclusion templates:**

- *Guard present:* "The tagged request was preempted 2 times and completed within 1.7× its expected E2E. The eviction policy is LIFO (verified at `scheduler.py:<line>`), which implicitly protects older requests. Pass."
- *Guard absent / unclear:* "The tagged request was preempted 7 times across 3 bursts and completed at 3.2× expected E2E. The eviction policy evicts the lowest-priority running sequence regardless of preemption history (verified at `scheduler.py:<line>`). **Mitigation:** add a per-request timeout at the API gateway (`X-Request-Timeout: 10s` → return 408 or shed to backup pool); application-level retry policy caps the long-tail blast radius."

Either outcome is acceptable; the artefact's value is the evidence.

---

## 3. Validation steps

A reviewer checks:

1. **Preemption fires at all.** Open Artefact B's metric capture. If `num_preemptions_total` did not move during either run, the drill failed to engage its own mechanism — send the learner back to Artefact A to tighten overcommit (lower `max-num-seqs`, raise concurrent clients).
2. **Mode was actually the mode claimed.** For Run 1 (swap), confirm the swap-out counter is non-zero. For Run 2 (recompute), confirm swap-out is zero and preemption count rose anyway.
3. **Restore-TTFT is per-request.** Grep the trace CSV for a `restore_ttft_ms` column (or equivalent). Aggregate preemption counts alone are not enough — the chapter is explicit.
4. **Break-even is a number.** Artefact C must name a specific prompt length (or a tight range). "It depends on the workload" fails.
5. **Starvation-guard artefact has evidence.** Either a trace of the tagged request with preemption-count history, or a code review pointing at the eviction-selection function with line numbers. Pure assertion ("the engine surely has one") fails.
6. **Policy pick is committed.** The chosen mode + verified `--swap-space` value is appended to the exercise-02 pinned config with a reasoning comment that cites this drill.

---

## 4. Rubric or review checklist

Pass threshold: ≥ 10 / 14.

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| Preemption actually fires | Counter at zero in both runs | Counter fires in one run only | Counter fires in both runs with ≥ dozens of events |
| Mode verified by counters | Mode not verified | Verified for one run | Verified for both runs; swap-count non-zero for swap, zero for recompute |
| Per-request restore-TTFT | Aggregate only | Per-request but no re-admission timestamp | Per-request restore-TTFT with request-id correlation |
| Break-even math | None, or "depends" | Formula without substituted numbers | Formula + substituted numbers + named break-even length |
| Policy verdict | Assertion without evidence | Verdict named, not justified | Verdict named, defended against the measured break-even, compared to the alternative |
| Starvation-guard evidence | Pure assertion | Code review OR trace but incomplete | Either path with line-number references or preemption-count history |
| Config pinning | Not pinned | Pinned but no comment | Pinned with reasoning comment citing drill evidence |

---

## 5. Common mistakes

- **No preemption fires.** The exercise is uninteresting if the engine never evicts. Lower `--max-num-seqs` and increase concurrent clients until the counter moves.
- **Mode confusion.** Many engines default to a hybrid "swap when space, else recompute." If the learner labels their trace "swap mode" without checking counters, half of what they call "swap" is actually recompute.
- **Overcommitting by raising RPS.** Raises prefill cost at the same time as admission pressure; conflates variables. Overcommit by lowering `--max-num-seqs`.
- **Aggregate metrics only.** TTFT P95 across all requests may look fine; what matters is TTFT P95 of the preempted sequences, tracked by request-id.
- **Starvation guard assumed, not verified.** "The engine must have one" is not evidence. Either trace it or read the selection function.
- **Pinning `--swap-space 16` without verification.** Swap sizing depends on the workload's KV footprint. Compute the per-sequence KV bytes × expected eviction depth; don't guess.
- **Ignoring the chunked-prefill interaction.** A sequence preempted mid-chunk in recompute mode loses chunk progress. If preemption rate is non-trivial and chunked prefill is on, prefer swap. Call this out explicitly in the verdict.
- **Not disabling the alert stack.** Running this drill against a production alert pipeline will spam on-call. Silence the Alertmanager stack for the drill window (or run against a dev engine).

---

## 6. References

Primary research:

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023 — <https://arxiv.org/abs/2309.06180> — PagedAttention block-level eviction.
- Sheng et al., *FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU*, ICML 2023 — <https://arxiv.org/abs/2303.06865> — swap vs. recompute intuition at the extreme.
- Yu et al., *Orca*, OSDI 2022 — <https://www.usenix.org/conference/osdi22/presentation/yu> — the baseline continuous-batching scheduler the preemption decision runs inside.
- Agrawal et al., *Sarathi-Serve*, OSDI 2024 — <https://www.usenix.org/conference/osdi24/presentation/agrawal> — interaction of chunked prefill with preemption.

Engine documentation (verify against pinned release):

- vLLM docs — <https://docs.vllm.ai/>; the preemption / swap docs and the `vllm serve --help` output for `--swap-space` and related flags.
- SGLang docs — <https://docs.sglang.ai/>.
- TensorRT-LLM docs — <https://nvidia.github.io/TensorRT-LLM/>; `kv_cache_host_memory_bytes` configuration in `config.pbtxt`.
- Triton model configuration — <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html>.

PCIe bandwidth references (for break-even arithmetic):

- NVIDIA technical briefs for the GPU generation under test (H100, H200, Blackwell) describe PCIe lane configuration and effective bandwidth; verify against the device's `nvidia-smi topo -m` and vendor documentation.

Paired learning content:

- Module 103 chapter 04 (*Preemption Policies*) in [`ai-infra-inference-engineer-learning`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning) — the break-even framework this exercise operationalises.
- Exercise 01 solution — the source tour the starvation-guard code review builds on.
- Exercise 02 solution — the pinned config the policy choice appends to.
- Exercise 03 solution — the chunked-prefill setpoint and its interaction with preemption.
- Exercise 05 solution — the preemption-rate and swap-count alerts this drill verifies.
