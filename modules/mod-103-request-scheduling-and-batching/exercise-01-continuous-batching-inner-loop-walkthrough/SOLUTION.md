# Solution — Exercise 01: Continuous Batching Inner-Loop Walkthrough

> Paired learning file: [`lessons/mod-103-request-scheduling-and-batching/exercises/exercise-01-continuous-batching-inner-loop-walkthrough.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-103-request-scheduling-and-batching/exercises/exercise-01-continuous-batching-inner-loop-walkthrough.md)

This is a **diagnostic-fluency** exercise. There is no single correct source map because the engines re-structure the scheduler module between releases; what this solution provides is (a) the *shape* of each of the four artefacts, (b) a worked vLLM source tour at a representative release used as a reference, (c) a defensible rubric a mentor can grade against, and (d) the common ways learners fail.

---

## 1. Solution overview

The deliverable is four artefacts per learner:

| Artefact | Form | Pass bar |
|---|---|---|
| A — Scheduler source-tour | ≤ 2 pages of markdown with ≥ 7 `file:function` references | Every reference `grep`-confirmable at the pinned engine SHA |
| B — Running engine with per-iteration log | Launch command + startup log + ≥ 10 lines of iteration log | ≥ 100 iteration lines emitted across a ≥ 5 min run |
| C — Traced iteration | ≤ 1 page covering admission, batch composition, preemption (or evidence it wasn't needed), forward dispatch, post-state | ≥ 4 cross-references to Artefact A with line numbers |
| D — Inner-loop cheatsheet | 1 page, reusable across module 103 | A sibling operator can launch the engine and name every iteration phase without opening the source tour |

The exercise is *not* a benchmark. Do not chase throughput numbers; chase the ability to point at `schedule()` in under 60 seconds and narrate what the next iteration will decide.

The recommended engine is **vLLM**: its scheduler is the most readable of the three (single Python class per iteration, vs. SGLang's manager-split design and TRT-LLM's C++ in-flight batcher wrapped by `tensorrtllm_backend`). All three engines implement continuous / iteration-level batching per the Orca paper; the inner-loop vocabulary (admission queue, running set, preemption, prefill vs. decode selection) is identical across engines, only the file layout differs.

Primary mental model: the inner loop is a loop over *iterations*, not over *requests*. Each iteration (a) pulls admissions from the waiting queue subject to `max_num_seqs` and KV budget, (b) selects prefill vs. decode work subject to `max_num_batched_tokens`, (c) may preempt to free KV space, and (d) dispatches a single forward pass over a mixed batch. Chapter 01 of the paired learning module defines this loop; Orca (OSDI 2022) is the primary paper.

---

## 2. Implementation

### 2.1 Artefact A — worked scheduler source-tour (vLLM, V0-shape scheduler)

The exact file layout has shifted across vLLM releases (the engine has been refactored toward a "V1" engine core); the worked tour below uses the pre-refactor layout as the canonical reference shape. **Learners must re-grep against their own pinned version before submitting.**
| Entry point | Suggested location (verify) | What it does |
|---|---|---|
| Engine step loop | `vllm/engine/async_llm_engine.py` → `AsyncLLMEngine.engine_step()` (or `LLMEngine.step()` in sync path) | Drives one iteration end-to-end: scheduler → model executor → output processing. |
| Scheduler class | `vllm/core/scheduler.py` → `class Scheduler` | Holds `waiting`, `running`, `swapped` queues and the policy. |
| One-iteration function | `vllm/core/scheduler.py` → `Scheduler.schedule()` | Returns `SchedulerOutputs` describing this iteration: who runs, who is prefill vs. decode, who got preempted. |
| Admission helper | `vllm/core/scheduler.py` → `Scheduler._schedule_prefills()` (or `_schedule_running()` for the already-admitted path) | Pulls from `waiting` into `running` subject to `max_num_seqs` and block budget. |
| Prefill-vs-decode selection | `vllm/core/scheduler.py` → `Scheduler._schedule_running()` + `_schedule_prefills()` + the chunked-prefill helpers | Decides which running sequences generate decode tokens and how many prefill tokens the batch carries this iteration. |
| Preemption helper | `vllm/core/scheduler.py` → `Scheduler._preempt()` / `_preempt_by_recompute()` / `_preempt_by_swap()` | Chooses the victim (default: last-in, i.e. the most recently admitted) and the mode (swap vs. recompute). |
| Mixed prefill+decode dispatch | `vllm/worker/model_runner.py` → `ModelRunner.execute_model()` and the attention backend wiring under `vllm/attention/` | The point at which chunked-prefill tokens and decode tokens share one forward pass. |

The policy knobs mapped to this tour:

| Flag | What it bounds | Code path |
|---|---|---|
| `--max-num-seqs` | Running-set cap | Loop guard inside `_schedule_prefills()` |
| `--max-num-batched-tokens` | Per-iteration token budget (prefill + decode) | Budget check inside `_schedule_prefills()` / `_schedule_running()` |
| `--enable-chunked-prefill` | Whether prefill is broken into chunks per iteration | Branch in `_schedule_prefills()` |
| `--swap-space` | Size of the CPU swap pool | Allocator (`block_manager`); referenced from `_preempt_by_swap()` |
| `--gpu-memory-utilization` | Fraction of GPU memory used for the KV pool (upstream of the scheduler) | Engine init; sets the total block count the scheduler sees |

For SGLang the equivalent tour lives under `python/sglang/srt/managers/scheduler.py` (verify path for your release); for TRT-LLM, the in-flight batching logic lives in the `tensorrt_llm/batch_manager` tree in C++ and is driven from `tensorrtllm_backend` in Triton.
### 2.2 Artefact B — running engine with per-iteration logging (vLLM reference recipe)

Launch command (reference values; adjust to the learner's hardware):

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --enable-chunked-prefill \
  --disable-log-requests=false \
  --port 8001 \
  --swap-space 4
```

The settings are chosen to make admission pressure and preemption *reachable*: a small running-set cap of 8 and a modest token budget of 2048 against a workload of 15–20 concurrent streaming clients will provoke admission and preemption events within the first minute.

**Instrumentation strategy.** vLLM does not emit a per-iteration structured log by default. Two acceptable approaches:

1. **Scrape `/metrics` at 1 Hz** during the run and derive iteration dynamics from counters (`vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:num_preemptions_total`, `vllm:prompt_tokens_total`, `vllm:generation_tokens_total` — verify exact names for the pinned release against `curl http://localhost:8001/metrics | head -40`). This is non-invasive.2. **Add a single `logger.info(...)` line inside `Scheduler.schedule()` in a local clone** that emits `(iteration_id, running_count, waiting_count, admitted, prefill_tokens, decode_tokens, preempted)` once per call. Warn the learner: this patch stays local — do not upstream and do not confuse any audit trail.

Both are acceptable; (2) is the more pedagogically valuable path because the log line is authored *inside* the function the source tour names.

A head-sample of 10 iteration lines from the instrumented engine should look like:

```
iter=0001 running=0 waiting=3 admitted=3 prefill_tok=1800 decode_tok=0 preempted=0
iter=0002 running=3 waiting=0 admitted=0 prefill_tok=0 decode_tok=3 preempted=0
iter=0003 running=3 waiting=5 admitted=5 prefill_tok=1920 decode_tok=3 preempted=0
iter=0004 running=8 waiting=2 admitted=0 prefill_tok=0 decode_tok=8 preempted=0
iter=0005 running=8 waiting=2 admitted=0 prefill_tok=1024 decode_tok=7 preempted=1
...
```

The learner's capture will differ in values but should show the same four shapes: an admit-heavy iteration, a decode-heavy iteration, a mixed prefill+decode iteration (chunked prefill engaged), and an iteration where preemption fires.

### 2.3 Artefact C — one traced iteration (template)

Pick an iteration where admission, prefill, decode, and preemption all occur (iteration 5 in the sample above). The write-up should follow the template:

> **Pre-iteration state.** `running=8`, `waiting=2`, KV pool ≈ 92% used.
>
> **Admission decision.** 0 admissions this iteration. `_schedule_prefills()` at `scheduler.py:<line>` checked the waiting head, found no KV block budget after preemption reservation, and declined.
>
> **Batch composition.** 1024 prefill tokens from 1 sequence (chunked prefill mid-stream) + 7 decode tokens from 7 of the 8 running sequences. Total 1031 batched tokens, well under `max-num-batched-tokens=2048`. The 8th sequence is the victim selected in the preemption step below.
>
> **Preemption decision.** 1 preemption. `_preempt()` at `scheduler.py:<line>` selected the most-recently-admitted sequence (LIFO policy) and called `_preempt_by_recompute()` (with `--swap-space 4`, swap was unavailable for this sequence's KV size); the sequence moves back to the waiting queue with its KV blocks freed and its prompt retained for re-prefill.
>
> **Forward dispatch.** `ModelRunner.execute_model()` at `model_runner.py:<line>` called into the attention backend (Flash Attention 2 in this build) with a mixed prefill+decode batch descriptor; the attention backend handled the chunked prefill path per [SARATHI / SARATHI-Serve](https://arxiv.org/abs/2308.16369).
>
> **Post-iteration state.** `running=7`, `waiting=3`, 7 new decode tokens emitted, 1024 prefill tokens of progress on 1 sequence.

The pass-bar is not that the learner's numbers match these — it is that each line **cross-references a specific line in Artefact A's source map**. "Line 237 of `scheduler.py`" is the discipline.

### 2.4 Artefact D — inner-loop cheatsheet (template)

One page, reusable for the rest of module 103:

```
=== Inner-Loop Cheatsheet — <engine>@<git-sha> ===

Entry point:        <file>:<function>           (how to find in 60s)
Scheduler class:    <file>:<class>              (lives next to engine step loop)

Order within one iteration:
  1. Admission     -> _schedule_prefills()      (max_num_seqs, KV blocks)
  2. Prefill/decode -> _schedule_running() + chunked-prefill branch
                                                (max_num_batched_tokens, chunk size)
  3. Preemption    -> _preempt() + mode helper  (swap vs. recompute)
  4. Forward       -> model_runner.execute_model() + attention backend

Five load-bearing knobs:
  --max-num-seqs            -> running-set cap       (step 1)
  --max-num-batched-tokens  -> per-iter token budget (steps 1–2)
  --enable-chunked-prefill  -> prefill chunking on   (step 2)
  --swap-space              -> CPU swap pool size    (step 3)
  --gpu-memory-utilization  -> total KV block count  (upstream of step 1)

First place to look when:
  TTFT spikes      -> step 1 (admission blocked) or step 2 (long prefill mid-batch)
  TPOT spikes      -> step 2 (prefill eating decode budget)
  Preemption ≠ 0   -> step 3 (KV pressure, see cache usage metric)
  Iteration stalls -> step 4 (kernel path; peer-track escalation)
```

Pinned alongside the module-102 five-layer map; refreshed on every engine upgrade.

---

## 3. Validation steps

A reviewer checks:

1. `grep` every `file:function` reference in Artefact A inside the learner's pinned engine clone at the recorded SHA. Zero broken references.
2. Launch the recorded command from Artefact B; confirm the engine starts and the listed port serves `/metrics` after warmup.
3. Confirm Artefact B's iteration log has ≥ 100 lines spanning ≥ 5 minutes and contains at least one iteration each of: admission > 0, decode-only, mixed prefill+decode, and preemption > 0 (or the learner has explicitly argued why preemption did not fire — e.g. workload too light, which is acceptable only if they engineered the load to try).
4. Open Artefact C; for each of the six subsections (pre-state, admission, batch composition, preemption, forward, post-state), confirm an Artefact A cross-reference with a line number is present.
5. Read Artefact D cold (reviewer skips Artefact A deliberately). Can a different operator launch the engine and name the four iteration phases in order using only this one page? If no, Artefact D fails.
6. Confirm all four artefacts are committed to version control and the engine clone SHA is pinned.

---

## 4. Rubric or review checklist

Use the rubric below for grading. Each row scores 0 (missing / wrong), 1 (partial), or 2 (meets bar). Pass threshold: ≥ 10 / 14.

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| Source tour completeness | < 5 entry points, or references don't grep | 5–6 entry points, most grep | ≥ 7 entry points, every reference grep-confirmable |
| Source tour pinning | No SHA / version | Version named but not SHA-pinned | Engine SHA-pinned, clone matches runtime |
| Engine launch and log capture | Engine didn't run, or no log | Engine ran, log has < 50 lines or missing phases | ≥ 100 lines, all four iteration shapes present |
| Traced iteration completeness | Missing ≥ 2 of the 6 sub-phases | Missing 1 sub-phase, or sub-phases present but without line references | All 6 sub-phases, ≥ 4 with explicit line references |
| Traced iteration honesty | Numbers don't reconcile with Artefact B | Numbers roughly consistent | Numbers reconcile cleanly with the iteration-log line referenced |
| Cheatsheet reusability | Over 1 page, or unreadable | Fits but a sibling operator still needs the source tour | Sibling operator can launch and narrate from the cheatsheet alone |
| Instrumentation hygiene | Logging patch upstreamed / merged to main | Patch local but no comment explaining "do not upstream" | Patch local, commented, and reversible |

---

## 5. Common mistakes

- **Chapter-copying.** Writing Artefact A from the chapter's prose instead of the installed engine. The chapter is deliberately vague about class names because they rotate; the writeup must be concrete.
- **No preemption in the log.** With `--max-num-seqs 8` and 20 concurrent clients it will fire; if it doesn't, drop to `--max-num-seqs 4` or send a single long prompt mid-batch. If the learner records a log without any preemption and doesn't try to provoke one, send them back.
- **Logging too noisy to read.** Multi-line log entries per iteration make the 10-line head sample unreadable. One line per iteration, with sampling parameters dumped to a side file if needed.
- **Benchmarking confusion.** Learners bring throughput numbers and claim they tuned something. This is a *diagnostic fluency* exercise; throughput belongs in exercise 02. Point out which artefact is missing.
- **Unpinned engine.** The source tour's addresses rot the moment the engine upgrades. If the clone is not SHA-pinned to match the running engine, every address in Artefact A is now a liability.
- **Instrumentation upstreamed.** A print statement merged to the engine's `main` is a supply-chain contamination risk in a team setting; keep the patch in a local branch and note "do not upstream" in the commit message.
- **Hybrid preemption mode confusion.** vLLM's preemption default is "swap if space, else recompute." If the learner labels their trace "swap mode" without checking the metric counters, the trace is wrong half the time. Exercise 04 addresses this directly; flag it here.

---

## 6. References

Primary research on continuous batching and the mixed-batch inner loop:

- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022 — <https://www.usenix.org/conference/osdi22/presentation/yu> — the paper that introduced iteration-level scheduling.
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023 — <https://arxiv.org/abs/2309.06180> — the KV allocator the scheduler calls into.
- Agrawal et al., *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills* — <https://arxiv.org/abs/2308.16369>.
- Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, OSDI 2024 — <https://www.usenix.org/conference/osdi24/presentation/agrawal>.

Engine documentation and source (verify versions):

- vLLM project home — <https://github.com/vllm-project/vllm>; docs — <https://docs.vllm.ai/>; metrics reference — <https://docs.vllm.ai/en/latest/serving/metrics.html>.
- SGLang project home — <https://github.com/sgl-project/sglang>; docs — <https://docs.sglang.ai/>.
- TensorRT-LLM — <https://github.com/NVIDIA/TensorRT-LLM>; `tensorrtllm_backend` — <https://github.com/triton-inference-server/tensorrtllm_backend>.

Paired learning content:

- Module 103 README and chapters 01 and 02 in [`ai-infra-inference-engineer-learning`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning).
- Module 102 exercise 01 (five-layer cheatsheet) — the scaffold Artefact D builds on top of.
