# Exercise 02 — SGLang Radix and Scheduler Walkthrough

> Paired learning exercise: `lessons/mod-102-inference-engine-internals/exercises/exercise-02-sglang-radix-and-scheduler-walkthrough.md`.

## 1. Solution overview

This exercise asks the learner to walk a prompt through SGLang the way
Exercise 01 walked it through vLLM, with one addition: explain how the
**RadixAttention** prefix cache influences admission, batching, and KV
allocation. SGLang's distinguishing design is that its KV cache is indexed by a
radix tree over token prefixes, so that any two requests sharing a prefix
(system prompt, few-shot block, chat history) share the underlying KV blocks
automatically — no application-level cache keys required.

This solution gives the reference call path, a diagram of how the radix tree
and the scheduler interact, and the policy knobs the learner should name.

<!-- verify-source: SGLang ships frequent refactors; symbol names below match
the maintained `python/sglang/srt/` tree. Learners must pin the commit they
traced and reconcile differences in their submission. -->

## 2. Implementation — Worked walkthrough

### 2.1 Reference call graph

```
HTTP endpoint (sglang.srt.entrypoints.http_server)
  └─ TokenizerManager.generate_request(obj)
       ├─ tokenizer.encode(prompt)
       └─ send to Scheduler via ZMQ

Scheduler.event_loop_normal()              # one iteration = one forward step
  ├─ recv_requests()                        # pull new reqs from tokenizer side
  ├─ process_input_requests()
  │    └─ Req objects added to waiting_queue
  ├─ get_next_batch_to_run()
  │    ├─ get_new_batch_prefill()
  │    │    ├─ tree_cache.match_prefix(req.input_ids)   ← RadixCache
  │    │    │    └─ returns cached prefix + node to attach new tokens under
  │    │    ├─ policy (LPM / FCFS / priority) selects admitted reqs
  │    │    └─ allocator.alloc(num_new_tokens)           ← token-pool / KV
  │    └─ update_running_batch()                         # decodes
  ├─ run_batch()
  │    └─ TpModelWorker.forward_batch(batch)
  │         └─ ModelRunner.forward(batch)
  │              ├─ attention: radix-aware kernel reads shared KV via indices
  │              ├─ sampling
  │              └─ returns next tokens
  └─ process_batch_result()
       ├─ append tokens to Req
       ├─ tree_cache.cache_finished_req(req) or cache_unfinished_req(req)
       └─ send stream chunks back to TokenizerManager → HTTP response
```

Key structures:

| Structure | Where it lives | Role |
|---|---|---|
| `Req` | `sglang/srt/managers/schedule_batch.py` | one in-flight request: tokens, state, sampling info |
| `Scheduler` | `sglang/srt/managers/scheduler.py` | admission, batching, step loop |
| `RadixCache` / tree nodes | `sglang/srt/mem_cache/radix_cache.py` | radix tree over token prefixes; nodes point into the KV pool |
| `TokenToKVPoolAllocator` + `ReqToTokenPool` | `sglang/srt/mem_cache/*` | the actual KV pool and the per-request → KV-slot mapping |
| `ModelRunner` / worker | `sglang/srt/model_executor/*` | tensor assembly, model forward, sampling |

### 2.2 RadixAttention — how the cache changes the hot path

1. **Match.** When a new request arrives, the scheduler walks the radix tree
   from the root, consuming the longest matching prefix of the request's
   `input_ids`. The result is a (matched-length, attach-node) pair. The
   matched tokens already have KV; only the suffix needs a prefill.
2. **Admit.** Admission tests whether the KV allocator can hand out slots
   for the unmatched suffix *plus* room for `max_new_tokens` decoded tokens
   (respecting any chunked-prefill limits). If yes, the request joins the
   batch; the attach-node is pinned so it cannot be evicted until the
   request finishes.
3. **Prefill.** The forward pass only recomputes the suffix tokens. The
   attention kernel is fed per-request index tables that reference the shared
   prefix KV directly — no copy.
4. **Grow the tree.** As tokens are decoded, new nodes are grafted onto the
   tree beneath the attach-node. When the request finishes
   (`cache_finished_req`) those nodes remain in the tree, available for
   future matches, until the eviction policy (LRU with a protected / pinned
   set) reclaims them under memory pressure.
5. **Eviction.** Under KV pressure the scheduler asks the radix cache for the
   least-recently-used *leaf* subtree that is not currently pinned by a
   running request, and the allocator frees those KV slots.

### 2.3 Scheduling policy — knobs the learner must name

- **Admission policy.** SGLang supports multiple policies (e.g. **LPM** —
  longest prefix match first — which maximises cache hit rate, and plain
  **FCFS**). LPM biases admission toward requests that would recompute the
  fewest tokens, which is the whole point of having the cache. <!-- verify-source:
  the exact set of policies has changed over releases; the learner should
  enumerate what their pinned version exposes. -->
- **Token budget per batch.** Caps the prefill tokens admitted per step so a
  very long new request does not starve decodes.
- **Chunked prefill.** Long prefills are split across several steps so
  decodes continue to advance — the decode tokens of other requests ride in
  the same forward.
- **Overlap scheduling / CUDA graphs.** The scheduler overlaps CPU-side batch
  prep with the GPU forward; CUDA graphs are captured per batch shape when
  enabled.

### 2.4 Prose walkthrough (reference answer)

1. **Admission with prefix match.** The HTTP server encodes the prompt and
   hands it to the tokenizer manager, which forwards it (over IPC) to the
   scheduler. The scheduler places the `Req` on the waiting queue. On the
   next iteration `match_prefix` walks the radix tree; any prefix that
   already has KV (system prompt, shared few-shot) will match. The matched
   length is subtracted from the prefill cost.
2. **Batch build.** The scheduler mixes decodes (one token per running
   request) with newly admitted prefills, subject to the token budget, max
   running requests, and KV allocator availability. Chunked prefill is used
   to cap the per-step prefill load.
3. **Forward pass.** The worker runs one model forward. Attention reads the
   shared prefix KV directly via index tables; only the suffix tokens are
   materialised anew. The sampler returns one token per running request.
4. **Post-step.** Tokens are appended; the radix tree is updated so the new
   tokens become new nodes under the request's attach-node. Finished
   requests call `cache_finished_req`, which leaves their tokens in the tree
   for reuse. Streaming chunks flow back out.
5. **Memory pressure.** If an incoming prefill cannot be admitted, the
   scheduler evicts cold subtrees (LRU, non-pinned) until there is room, or
   defers admission.

## 3. Validation steps

1. **Version pin.** Record the `sglang` version and the commit of the
   `sglang/srt/` tree the walkthrough matches.
2. **Launch a server.**
   ```bash
   python -m sglang.launch_server \
       --model-path meta-llama/Llama-3.2-1B-Instruct \
       --port 30000
   ```
3. **Cache-hit smoke test.** Send the same system prompt + two different user
   turns back-to-back; the second request's logged prefill length should be
   the suffix only, not the whole prompt. The `/get_server_info` endpoint
   (<!-- verify-source: field names change across releases -->) exposes
   cache hit counters.
4. **Scheduler assertions.** Fire N concurrent requests sharing a prefix and
   confirm throughput scales super-linearly compared to N requests with no
   shared prefix.
5. **Eviction.** Load the cache beyond its budget and confirm the oldest
   non-pinned subtrees are reclaimed (check the pool's free-slot count).

## 4. Rubric / review checklist

| # | Criterion | Weight | Pass bar |
|---|---|---|---|
| 1 | Call graph is grounded in real SGLang symbols | 20% | At least `Scheduler`, `RadixCache`, token/req pools, model worker named from the pinned commit |
| 2 | Radix tree mechanics explained | 25% | Match / attach / grow / evict cycle covered explicitly |
| 3 | Scheduling knobs listed | 15% | Admission policy (LPM vs. FCFS), token budget, chunked prefill all named |
| 4 | Impact on prefill cost shown | 10% | Learner states that matched tokens skip prefill |
| 5 | Validation evidence | 15% | A two-turn shared-prefix experiment attached |
| 6 | Memory-pressure behaviour described | 10% | Eviction ties back to the pinned-vs-leaf rule |
| 7 | Version pinned | 5% | Commit hash or release tag at top |

Passing threshold: 70%.

## 5. Common mistakes

- **Treating the radix cache as an application-level cache.** It is engine-
  internal and indexed by tokens; no cache keys from the user side.
- **Assuming cached prefixes live forever.** Only pinned (currently
  referenced by a running request) nodes are protected; everything else is
  LRU-evictable.
- **Confusing token-pool allocation with the radix tree.** The tree is the
  index; the token pool is the physical KV storage. A node in the tree
  points at tokens in the pool.
- **Ignoring chunked prefill.** Without chunking, a 32k-token new request
  would freeze every other request's decode for the duration of that
  prefill.
- **Benchmarking without priming.** First call of a shared prefix always
  misses; numbers must come from steady state.

## 6. References

- SGLang source repository — https://github.com/sgl-project/sglang (schedule,
  radix cache, token pool). <!-- verify-source: pin commit. -->
- SGLang documentation — https://docs.sglang.ai/ (architecture and launch
  arguments).
- RadixAttention — Zheng et al., "SGLang: Efficient Execution of Structured
  Language Model Programs," NeurIPS 2024 — https://arxiv.org/abs/2312.07104
  (original description of the prefix-sharing cache).
- OWASP Machine Learning Security Top 10 —
  https://owasp.org/www-project-machine-learning-security-top-10/ (shared
  prefix caches raise ML06: AI Supply Chain Attacks and multi-tenant
  isolation concerns; reference when the walkthrough discusses cross-user
  reuse).
- NIST AI Risk Management Framework —
  https://www.nist.gov/itl/ai-risk-management-framework ("Govern" and
  "Measure" functions: tie cache-hit observability to AI RMF measurement
  practice).
