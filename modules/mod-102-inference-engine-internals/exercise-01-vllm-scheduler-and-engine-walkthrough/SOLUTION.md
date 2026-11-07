# Exercise 01 — vLLM Scheduler and Engine Walkthrough

> Paired learning exercise: `lessons/mod-102-inference-engine-internals/exercises/exercise-01-vllm-scheduler-and-engine-walkthrough.md`.

## 1. Solution overview

The learner is asked to trace a single prompt through the vLLM engine — from
the API layer down to GPU execution — and to produce an annotated call graph
plus a short written walkthrough covering scheduling, KV-cache allocation, and
output return. This solution gives the reference call path, the key data
structures touched at each hop, and a prose narrative the learner's submission
should match.

The core loop vLLM implements, at the engine layer, is: **(a)** admit new
requests into a `Scheduler` queue, **(b)** on every step build a *batch* of
running sequences subject to the available KV blocks and token budget,
**(c)** hand that batch to a `ModelRunner` that executes one forward pass on
GPU, **(d)** sample tokens, update sequences, free finished ones, **(e)**
stream outputs back to the API layer. The scheduler is cooperative: prefill and
decode share the same step function, continuous batching happens at the
granularity of a single `engine.step()`, and KV memory is pooled into
fixed-size blocks (PagedAttention) so admission becomes a block-availability
test rather than a contiguous-allocation problem.

<!-- verify-source: vLLM renames and refactors modules frequently (e.g. the v1
engine moved several of these responsibilities). Learner submissions should
cite the exact commit/tag they traced. The call path below is accurate for the
stable v0 engine series; v1 learners should match the symbols to their own
`vllm/v1/` tree. -->

## 2. Implementation — Worked walkthrough

### 2.1 Reference call graph (prompt → tokens → response)

```
AsyncLLMEngine.generate(prompt, sampling_params, request_id)
  └─ AsyncLLMEngine.add_request(...)
       └─ LLMEngine.add_request(...)
            ├─ Tokenizer.encode(prompt)                      # prompt → input_ids
            ├─ SequenceGroup(...)                            # wraps seq(s) + sampling params
            └─ Scheduler.add_seq_group(seq_group)            # pushed to waiting queue

AsyncLLMEngine._engine_step()  (loop, one step per forward pass)
  └─ LLMEngine.step()
       ├─ Scheduler.schedule()
       │    ├─ _schedule_running()       # continuing decodes; may preempt
       │    ├─ _schedule_waiting()       # admit new prefills if KV blocks free
       │    └─ returns (SchedulerOutputs, SequenceGroupMetadata[])
       ├─ ExecutorBase.execute_model(execute_model_req)
       │    └─ Worker.execute_model(...)
       │         └─ ModelRunner.execute_model(...)
       │              ├─ InputBuilder → token ids, block tables, attn metadata
       │              ├─ model.forward(input_ids, positions, kv_caches, attn_metadata)
       │              │     └─ PagedAttention kernels read/write the block-based KV cache
       │              └─ Sampler(logits, sampling_metadata) → next token ids
       └─ _process_model_outputs(...)
            ├─ Sequence.append_token_id(...)
            ├─ Scheduler.free_finished_seq_groups()
            └─ yield RequestOutput to the async queue
```

Key structures the learner must name in their write-up:

| Structure | Where it lives | Role |
|---|---|---|
| `SequenceGroup` / `Sequence` | `vllm/sequence.py` | per-request + per-beam token state, status, logprobs |
| `Scheduler` | `vllm/core/scheduler.py` (v0) | admission, batching, preemption |
| `BlockManager` | `vllm/core/block_manager*.py` | KV-block allocation, copy-on-write for shared prefixes |
| `ModelRunner` | `vllm/worker/model_runner.py` | tensor layout, CUDA graphs, calls into the model |
| PagedAttention kernels | `vllm/attention/ops/*` + model layer | block-indexed attention over non-contiguous KV |
| `Sampler` | `vllm/model_executor/layers/sampler.py` | logit processors, temperature/top-k/top-p, token sampling |

### 2.2 Prose walkthrough (reference answer)

1. **Admission.** The HTTP layer (`vllm.entrypoints.openai.api_server`, or a
   direct `AsyncLLMEngine` user) hands a prompt and `SamplingParams` to
   `LLMEngine.add_request`. The tokenizer encodes the prompt, the engine wraps
   the resulting token IDs in a `Sequence` (one per beam/n-sample) wrapped in a
   `SequenceGroup`, and the group goes on the scheduler's **waiting** queue.
   Nothing is on GPU yet.
2. **Scheduling.** On the next `engine.step()` the scheduler walks the running
   queue first — decodes keep their KV blocks, each takes one token slot and
   one position in the batch. It then tries to admit from waiting until either
   (a) the token budget for this step is exhausted, or (b) the block manager
   cannot allocate the prefill's required KV blocks. If a running sequence
   would need a new block but none is free, the scheduler **preempts** — either
   by swapping that sequence's blocks to CPU memory, or by recomputing it
   (configurable). The scheduler emits a `SchedulerOutputs` describing the
   batch plus a list of `SequenceGroupMetadata` the model worker will turn into
   tensors.
3. **Forward pass.** The executor (single-GPU `GPUExecutor`, multi-GPU
   `RayGPUExecutor` / `MultiprocessingGPUExecutor`, etc.) calls the worker,
   which calls the `ModelRunner`. The runner builds the input tensors: token
   IDs, position IDs, a **block table** per sequence (the list of KV block
   indices the attention kernel will read), and attention metadata separating
   the prefill tokens from decode tokens. The model forwards through its
   layers; attention layers call the PagedAttention kernels which gather K/V
   from the paged blocks. CUDA graphs are used when the batch shape matches a
   captured profile. The final logits go through the configured logit
   processors and the `Sampler`, which produces one token per sequence.
4. **Post-step.** The engine appends the sampled tokens to each `Sequence`,
   checks stopping criteria (max tokens, stop strings, EOS), and emits a
   `RequestOutput` on the async stream. Finished groups release their KV
   blocks back to the block manager; those blocks become available to pending
   prefills on the next step.
5. **Return path.** The API server reads from the async generator and streams
   SSE chunks (or returns a single response for non-streaming requests).

### 2.3 Scheduling policy — the three invariants the learner must name

- **Token budget per step** (`max_num_batched_tokens`): caps the total tokens —
  prefill tokens + decode tokens — admitted into a single forward pass.
- **Concurrency cap** (`max_num_seqs`): caps the number of sequences
  simultaneously running.
- **KV-block availability**: admission is gated by whether `BlockManager` can
  reserve the blocks a new prefill, or an incoming decode token, requires.

A correct walkthrough explains how these three *together* implement continuous
batching: decodes advance one token per step without waiting for peers; new
prefills are opportunistically folded into the same step when the budget and
blocks allow (chunked prefill splits long prefills across multiple steps).

## 3. Validation steps

1. **Static trace.** Run `python -m pip show vllm` to confirm the version
   studied. Then `grep -nR "def step" vllm/engine/` and
   `grep -nR "def schedule" vllm/core/` in the installed package or a clone;
   the learner's call graph should reference exactly the symbols they find —
   not the symbols in this solution if they diverge for their version.
2. **Smoke run.** Start a server, send a single prompt, and inspect logs with
   `VLLM_LOGGING_LEVEL=DEBUG`:
   ```bash
   VLLM_LOGGING_LEVEL=DEBUG python -m vllm.entrypoints.openai.api_server \
       --model facebook/opt-125m --port 8000
   curl -s http://localhost:8000/v1/completions \
       -H 'Content-Type: application/json' \
       -d '{"model":"facebook/opt-125m","prompt":"Hello","max_tokens":16}'
   ```
   The debug log should show: `add_request`, at least one `schedule` call with
   a non-zero running count, model execution, and request completion.
3. **Scheduler assertions.** Fire two prompts concurrently and confirm the
   second enters the running set during the first's decode (continuous
   batching) rather than after it finishes.
4. **Block accounting.** With `--enable-prefix-caching` off, record the number
   of free GPU blocks before and after a request (the engine prints this at
   init); it must return to the initial value after completion.

## 4. Rubric / review checklist

| # | Criterion | Weight | Pass bar |
|---|---|---|---|
| 1 | Call graph covers the API → scheduler → runner → sampler → stream-out path | 25% | Every hop in §2.1 named with real symbols from the learner's version |
| 2 | Scheduling invariants explained | 20% | All three of token budget, max_num_seqs, KV-block availability named and tied to continuous batching |
| 3 | KV cache model (paged blocks + block table) explained | 15% | Learner distinguishes paged, block-indexed KV from a contiguous `[B, T, H, D]` buffer |
| 4 | Prefill vs. decode separation shown | 10% | Attention metadata (or equivalent) described as splitting the batch |
| 5 | Preemption behaviour described | 10% | Both swap and recompute paths mentioned; trigger condition named |
| 6 | Validation evidence attached | 10% | At least one of debug-log snippet or two-request concurrency trace included |
| 7 | Version pinned | 10% | vLLM commit hash or release tag stated at top of submission |

Passing threshold: 70%.

## 5. Common mistakes

- **Treating the forward pass as "one request at a time."** A step is a batch;
  the whole point of continuous batching is that prefills and decodes share
  the same forward.
- **Confusing the waiting and running queues with a FIFO.** The scheduler
  admits from waiting *after* servicing running; priority and budget can skip
  requests that don't fit.
- **Describing the KV cache as `[B, max_seq_len, H, D]`.** That's the
  non-paged baseline. In vLLM each sequence's KV lives in a list of
  fixed-size blocks indexed by the block table — that's what lets
  heterogeneous-length requests share memory and what prefix caching reuses.
- **Skipping preemption.** Preemption is what makes vLLM robust under
  KV-exhaustion; a walkthrough that never mentions it misses a core design.
- **Mixing v0 and v1 symbols.** v1 (`vllm/v1/`) restructured the engine and
  scheduler; pinning the version prevents the walkthrough from drifting into
  nonsense.

## 6. References

- vLLM source repository — https://github.com/vllm-project/vllm (authoritative
  for all symbol names and scheduler behavior). <!-- verify-source: pin the
  learner's commit; main changes weekly. -->
- vLLM documentation — https://docs.vllm.ai/ (architecture, engine args).
- PagedAttention paper — Kwon et al., "Efficient Memory Management for Large
  Language Model Serving with PagedAttention," SOSP 2023 —
  https://arxiv.org/abs/2309.06180 (origin of the block-table KV layout).
- OWASP Machine Learning Security Top 10 —
  https://owasp.org/www-project-machine-learning-security-top-10/ (ML05: Model
  Theft and ML02: Data Poisoning are the surfaces most affected by what an
  inference engine exposes; reference when the walkthrough touches multi-tenant
  isolation).
- NIST AI Risk Management Framework —
  https://www.nist.gov/itl/ai-risk-management-framework ("Measure" function:
  observability of the serving stack is an AI RMF measurement concern;
  reference when discussing debug-log instrumentation).
