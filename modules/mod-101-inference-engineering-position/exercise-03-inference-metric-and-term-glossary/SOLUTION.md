# Exercise 03 — Inference Metric and Term Glossary (Reference Solution)

Paired exercise: [`exercise-03-inference-metric-and-term-glossary.md`](../../../../ai-infra-inference-engineer-learning/lessons/mod-101-inference-engineering-position/exercises/exercise-03-inference-metric-and-term-glossary.md). Anchor chapter: [Chapter 03 — Inference Vocabulary](../../../../ai-infra-inference-engineer-learning/lessons/mod-101-inference-engineering-position/03-inference-vocabulary.md).

## 1. Solution overview

The exercise asks learners to turn Chapter 03 into a team-wiki-ready glossary that:

- defines every required term with a 7-field entry structure,
- includes a term-to-engine-primitive reference table,
- and names at least three dashboard-label anti-patterns with their rename.

This solution ships the glossary end-to-end against a synthetic team fleet (two regions, mixed Hopper + Blackwell, vLLM primary, SGLang on the MoE pool). Prometheus metric names below are the ones a learner should propose for their fleet if they do not already exist; where the metric does not exist, the entry carries `TODO: expose as <proposed-name>`.

## 2. Implementation — Worked answer

### 2.1 Entry format recap

Each entry uses this schema (from the exercise spec):

1. **Term** (and common synonyms)
2. **Definition** — operator-level, one or two sentences
3. **Unit** — SI unit / percentage / window / `N/A`
4. **Dominated by** — the single most load-bearing contributor
5. **Engine primitive** — the concrete scheduler / allocator / API code path
6. **Prometheus / exporter name in our fleet** — the metric name; `TODO: expose as ...` if not yet exposed
7. **Who else uses this term** — peers likely to reference it

### 2.2 Latency metrics

**TTFT (Time To First Token)**

1. Term: TTFT (synonyms: "first-token latency", "time to first token")
2. Definition: Wall-clock time from request admission to the first output token reaching the client, including queue wait, prefill, and the first-token network hop.
3. Unit: milliseconds (ms); typical reporting window P50 / P90 / P99 over 5-minute rolling buckets.
4. Dominated by: prefill cost and queue depth.
5. Engine primitive: serving front-end timer; influenced by the scheduler admission step, chunked-prefill chunk size, and prefix-cache hit path (PagedAttention block match in vLLM; RadixAttention node hit in SGLang).
6. Prometheus: `inference_ttft_ms_bucket` (histogram), reported as `inference_ttft_ms_p50`, `..._p90`, `..._p99` via a recording rule.
7. Who else: `ai-infra-sre` for SLO burn; AI Engineering consumers for user-perceived latency; `ai-infra-performance` for kernel-level attribution.

**TPOT (Time Per Output Token)**

1. Term: TPOT (synonyms: "per-token latency", "average decode-step latency")
2. Definition: Average wall-clock time to generate one output token after the first, measured as the time between the first and last output tokens divided by `N − 1` for a request producing `N` output tokens (the first token is excluded so the denominator counts only decode intervals, not the prefill-dominated first token).
3. Unit: milliseconds per token (ms/token); reported P50 / P99.
4. Dominated by: decode-phase compute, batch size, memory bandwidth, and speculative-decoding acceptance rate.
5. Engine primitive: per-token iteration in the continuous-batching loop (vLLM's `Scheduler`, SGLang's batch scheduler, TRT-LLM's in-flight batching).
6. Prometheus: `inference_tpot_ms_per_token_bucket` with `..._p50`, `..._p99`.
7. Who else: inference engineers (tuning target); `ai-infra-sre` for SLO.

**ITL (Inter-Token Latency)**

1. Term: ITL (synonyms: "inter-chunk latency", "stream glitch")
2. Definition: Per-token wall-clock gap between consecutive streamed tokens as the user perceives them; under preemption or stalls, spikes above TPOT.
3. Unit: milliseconds (ms/token); reported as a *distribution*, not a mean.
4. Dominated by: preemption events, batching glitches, scheduler stalls, and SSE flush boundaries at the HTTP layer.
5. Engine primitive: stream pipe between engine and HTTP server; backpressure handling in the OpenAI-compatible server for vLLM / SGLang.
6. Prometheus: `inference_itl_ms_per_token_bucket` with `..._p50`, `..._p99`.
7. Who else: `ai-infra-sre` for user-perceived streaming quality; AI Engineering consumers.

**E2E latency (End-to-End)**

1. Term: E2E latency (synonyms: "end-to-end", "request latency")
2. Definition: Wall-clock time from client submission to final output token, including queue, prefill, decode, and network; for streaming, approximately TTFT + output-token-count × TPOT.
3. Unit: milliseconds (ms); reported P50 / P99 per endpoint and model class.
4. Dominated by: total request duration; dominated by decode for long-output workloads, by TTFT for short-output workloads.
5. Engine primitive: request-level timer at the API gateway / server; aggregates TTFT + decode-loop duration.
6. Prometheus: `inference_e2e_latency_ms_bucket` with `..._p50`, `..._p99`.
7. Who else: `ai-infra-sre`; batch pipeline consumers; finance for cost-per-request reconciliation.

### 2.3 Throughput metrics

**Throughput**

1. Term: throughput (synonyms: "tok/s")
2. Definition: Total output tokens produced per second across the fleet (or per endpoint / per node).
3. Unit: tokens per second; aggregated over a 1-minute window.
4. Dominated by: batch size, model architecture (dense vs. MoE), parallelism topology, and kernel efficiency.
5. Engine primitive: fleet-wide token counter (sum of per-request output tokens over time).
6. Prometheus: `inference_output_tokens_per_second`.
7. Who else: capacity planner; finance (cost-per-million-token derivation); `ai-infra-performance` for kernel-regression diffs.

**Goodput**

1. Term: goodput (synonyms: "SLO-conformant throughput")
2. Definition: Throughput that satisfies the SLO; only the revenue-preserving fraction.
3. Unit: tokens per second (SLO-filtered); reported as a ratio against raw throughput.
4. Dominated by: SLO-compliance filtering; collapses with preemption spikes or TTFT drift.
5. Engine primitive: throughput filtered by per-request SLO conformance at the gateway.
6. Prometheus: `inference_goodput_tokens_per_second` with `inference_goodput_ratio` as a derived gauge.
7. Who else: finance; `ai-infra-sre`.

**Cache-hit rate (prefix cache)**

1. Term: cache-hit rate (synonyms: "prefix-cache hit rate")
2. Definition: Fraction of prompt tokens served by prefix-cache hits rather than re-prefilled.
3. Unit: percentage (0–100); reported as a ratio over a 5-minute window.
4. Dominated by: prefix-repetition pattern in the workload; shared system prompts, RAG preambles, agent-tool context.
5. Engine primitive: PagedAttention block hashing + block-table lookup (vLLM); RadixAttention tree node sharing (SGLang).
6. Prometheus: `inference_prefix_cache_hit_rate` (gauge), `inference_prefix_cache_hits_total` / `inference_prefix_cache_lookups_total` (counters) for rate derivation.
7. Who else: inference engineer (leading indicator of TTFT regression); capacity planner.

**Preemption rate**

1. Term: preemption rate
2. Definition: Fraction of in-flight requests preempted (swap-to-CPU or recompute) when the KV allocator runs out of blocks.
3. Unit: percentage of scheduled iterations, or preemptions per second.
4. Dominated by: KV memory pressure; indicates the fleet is bottlenecked on KV capacity.
5. Engine primitive: scheduler eviction policy on top of PagedAttention (vLLM); RadixAttention LRU eviction (SGLang).
6. Prometheus: `inference_preemption_rate` (gauge), `inference_preemption_events_total` (counter).
7. Who else: `ai-infra-sre` (non-zero rate predicts TTFT regression and goodput drop); capacity planner.

### 2.4 Scheduling and KV primitives

**Continuous batching**

1. Term: continuous batching (synonyms: "iteration-level batching", TRT-LLM's "in-flight batching")
2. Definition: Scheduling model where new requests join the running batch between decode iterations; replaces static / dynamic batching.
3. Unit: N/A (scheduling model); batch-size distribution reported separately.
4. Dominated by: scheduler admission policy.
5. Engine primitive: vLLM's `Scheduler`, SGLang's batch scheduler, TRT-LLM's in-flight batching.
6. Prometheus: `inference_batch_size_bucket` (distribution over iterations); `inference_queue_depth` for the inverse signal.
7. Who else: `ai-infra-sre` (queue-depth alerts); capacity planner.

**Chunked prefill**

1. Term: chunked prefill
2. Definition: Scheduler splits long prefill computations into chunks and interleaves them with decode steps, keeping TPOT stable while prefill progresses.
3. Unit: N/A (configuration); chunk size reported as a gauge.
4. Dominated by: chunk-size knob trade-off (smaller → lower TPOT variance, longer TTFT; larger → faster TTFT, more decode stalls).
5. Engine primitive: vLLM's `--enable-chunked-prefill` + `--max-num-batched-tokens`.
6. Prometheus: `inference_chunked_prefill_chunk_size` (config gauge); `inference_prefill_chunks_processed_total` (counter).
7. Who else: inference engineer; `ai-infra-performance` when tuning kernel path.

**Prefix caching**

1. Term: prefix caching
2. Definition: Reuse of KV-cache blocks for a prompt prefix that has already been computed (same system prompt, RAG preamble, etc.); converts worst-case prefill into a cache lookup.
3. Unit: N/A (mechanism); hit rate is `inference_prefix_cache_hit_rate` above.
4. Dominated by: workload prefix-repetition pattern.
5. Engine primitive: PagedAttention block hashing + block-table lookup (vLLM); RadixAttention tree-node sharing (SGLang).
6. Prometheus: covered by cache-hit rate; add `inference_prefix_cache_bytes` for capacity sizing.
7. Who else: inference engineer; cross-worker prefix sharing is tiered-KV territory (module 104).

**PagedAttention**

1. Term: PagedAttention
2. Definition: KV-cache allocator that chunks the cache into fixed-size blocks addressed via per-request block tables; enables fine-grained allocation, copy-on-write prefix sharing, and swap-to-CPU preemption.
3. Unit: block size (tokens per block); utilisation reported as percent.
4. Dominated by: block-size tuning.
5. Engine primitive: vLLM's block allocator; adopted in paged form by TRT-LLM's KV cache.
6. Prometheus: `inference_kv_cache_blocks_total`, `inference_kv_cache_utilization_percent`.
7. Who else: inference engineer; capacity planner.

**RadixAttention**

1. Term: RadixAttention
2. Definition: KV-cache organisation that indexes shared prefixes with a radix tree; requests sharing a token prefix share the same KV allocation node, with reference counting and LRU eviction.
3. Unit: node count; reported as a gauge.
4. Dominated by: workload prefix-reuse pattern; favoured for heavy, structured reuse (chat sessions, tool-call preambles).
5. Engine primitive: SGLang's radix tree + cache-aware worker routing.
6. Prometheus: `inference_radix_tree_nodes`, `inference_radix_cache_hit_rate`.
7. Who else: inference engineer (chat / agent workloads); capacity planner.

**Disaggregated prefill–decode**

1. Term: disaggregated prefill–decode (synonyms: "P/D-disaggregation")
2. Definition: Topology splitting request handling into prefill workers (compute-heavy, batch-parallel) and decode workers (memory-bandwidth-heavy, latency-sensitive), connected by KV transport.
3. Unit: N/A (topology); worker-utilisation reported per pool.
4. Dominated by: hardware specialisation; a single node cannot hit a tight TTFT *and* a tight TPOT simultaneously without a bottleneck.
5. Engine primitive: DistServe / Splitwise reference systems; NVIDIA Dynamo; vLLM V1 P/D-disaggregation mode.
6. Prometheus: `inference_prefill_worker_utilization_percent`, `inference_decode_worker_utilization_percent`.
7. Who else: capacity planner; hardware topology decision owner.

**Speculative decoding** (plus flavours)

1. Term: speculative decoding
2. Definition: Decoding scheme where a cheap "draft" (another model or an auxiliary head) proposes a token continuation and the main "target" model verifies in a single step; accepted tokens are kept, rejected ones are discarded.
3. Unit: acceptance rate (percentage); speed-up factor (dimensionless).
4. Dominated by: acceptance rate and draft-model cost; low acceptance × non-trivial draft produces a latency regression.
5. Engine primitive: verification path in the decode loop. Flavours: **draft-target** (two distinct models), **Medusa** (extra prediction heads on the target), **EAGLE** (feature-level drafting), **Lookahead** (self-speculative, no draft), **n-gram** (Markov-chain proposal).
6. Prometheus: `inference_spec_decode_acceptance_rate`, `inference_spec_decode_speedup_factor`, `inference_spec_decode_flavour_info` (info metric).
7. Who else: inference engineer (per-workload tuning in module 108); `ai-infra-performance` for draft-verify kernel path.

### 2.5 Parallelism primitives

**Tensor parallelism (TP)**

1. Term: tensor parallelism
2. Definition: Partitions the weights of individual tensors across GPUs in a tightly-coupled group (usually a single node over NVLink); all ranks participate in every forward step via collective communication.
3. Unit: TP degree (integer).
4. Dominated by: model size per GPU; collective-bandwidth ceiling.
5. Engine primitive: `--tensor-parallel-size` (vLLM / SGLang / TRT-LLM); NVLink / NVSwitch collectives (AllReduce / AllGather / ReduceScatter).
6. Prometheus: `inference_tensor_parallel_degree` (info metric); `inference_collective_latency_ms_bucket` for health.
7. Who else: capacity planner.

**Pipeline parallelism (PP)**

1. Term: pipeline parallelism
2. Definition: Partitions whole layers across stages; each stage holds a contiguous slice of layers; requests flow through like an assembly line; tolerant of inter-node interconnect.
3. Unit: PP degree (integer).
4. Dominated by: model depth; pipeline-bubble overhead with small micro-batches.
5. Engine primitive: `--pipeline-parallel-size`; inter-stage activation transport.
6. Prometheus: `inference_pipeline_parallel_degree` (info metric); `inference_pipeline_bubble_ratio` (gauge).
7. Who else: capacity planner.

**Expert parallelism (EP)**

1. Term: expert parallelism
2. Definition: For MoE models, partitions experts across GPUs; routing dispatches tokens to chosen experts (which may live on different GPUs); introduces AllToAll dispatch sensitive to interconnect.
3. Unit: EP degree (integer).
4. Dominated by: MoE sparsity + routing load; AllToAll bandwidth.
5. Engine primitive: `--expert-parallel-size` (vLLM / SGLang); engine-specific routing policy.
6. Prometheus: `inference_expert_parallel_degree` (info metric); `inference_expert_dispatch_imbalance` (gauge).
7. Who else: capacity planner (MoE: DeepSeek-V3, Mixtral, Qwen-MoE).

**DP attention**

1. Term: DP attention (synonyms: "data-parallel attention")
2. Definition: MoE-specific pattern where the attention block runs data-parallel while the MoE block runs expert-parallel; two parallelism axes compose.
3. Unit: N/A (mode flag).
4. Dominated by: MoE model shape; relieves attention from being EP-bound.
5. Engine primitive: SGLang's DP-attention reference implementation; also available in vLLM for MoE models (`verify` against release notes).
6. Prometheus: `inference_dp_attention_enabled` (info metric).
7. Who else: inference engineer (MoE tuning).

**LoRA swap / multi-LoRA**

1. Term: LoRA swap / multi-LoRA
2. Definition: Serving a shared base model with many LoRA adapters simultaneously; each adapter is small; engine time-slices or batches across adapters (S-LoRA, Punica reference systems, module 106).
3. Unit: active-adapter count (integer).
4. Dominated by: adapter count, swap overhead, and per-request routing policy.
5. Engine primitive: `--enable-lora` + per-request `lora_name` field; engine fuses adapter weights into the forward pass per request.
6. Prometheus: `inference_active_lora_adapters`, `inference_lora_swap_events_total`.
7. Who else: inference engineer (module 106); AI Engineering consumers for per-tenant adapter routing.

**Model sharding**

1. Term: model sharding
2. Definition: Any splitting of model state across devices; on the inference side, near-synonymous with "TP + PP (+ EP if MoE)."
3. Unit: N/A (topology description).
4. Dominated by: model size + topology choice.
5. Engine primitive: composition of TP, PP, EP.
6. Prometheus: no top-level metric; use `inference_tensor_parallel_degree`, `inference_pipeline_parallel_degree`, `inference_expert_parallel_degree` together.
7. Who else: capacity planner.

### 2.6 API primitives

**Streaming**

1. Term: streaming (synonyms: "SSE streaming", "incremental generation")
2. Definition: Returning output tokens as they are produced rather than waiting for the full response; over HTTP, almost always Server-Sent Events, occasionally WebSocket.
3. Unit: N/A (mode).
4. Dominated by: per-token output and backpressure handling.
5. Engine primitive: HTTP SSE emission tied to per-token output of the decode loop; backpressure and client-disconnect handling at the HTTP layer (module 111).
6. Prometheus: `inference_active_streaming_requests`, `inference_streaming_disconnects_total`.
7. Who else: `ai-infra-sre` (ITL contract); AI Engineering consumers.

**Structured output / constrained decoding**

1. Term: structured output (synonyms: "constrained decoding", "JSON mode", "grammar-guided generation")
2. Definition: Constraint on generated tokens so that the output is a parseable instance of a schema (JSON Schema, regex, grammar, Pydantic); enforced at the token-sampling layer.
3. Unit: percentage of structured-output requests (gauge) and invalid-output rate.
4. Dominated by: schema complexity and token-vocabulary overlap.
5. Engine primitive: Outlines / XGrammar backends (vLLM / SGLang); engine-specific guided-decoding modes.
6. Prometheus: `inference_structured_output_requests`, `inference_structured_output_invalid_rate`.
7. Who else: `ai-eval-engineer` (makes agent tool calls / JSON-mode / function-calling reliable without retries).

### 2.7 Term-to-engine-primitive reference table

| Term | Engine primitive(s) that implement it |
|---|---|
| TTFT | Serving front-end timer + scheduler admission + prefill + prefix-cache (PagedAttention / RadixAttention) |
| TPOT | Per-iteration decode timer in the continuous-batching loop (vLLM Scheduler / SGLang batch scheduler / TRT-LLM in-flight batching) |
| ITL | Streaming pipe between engine and HTTP layer (OpenAI-compatible server; backpressure) |
| E2E latency | Request-level timer at the API gateway |
| Throughput | Fleet-wide output-token counter |
| Goodput | Throughput filtered by per-request SLO conformance |
| Cache-hit rate | PagedAttention block-match counter / RadixAttention tree-hit counter |
| Preemption rate | Scheduler eviction counter |
| Continuous batching | vLLM `Scheduler`, SGLang batch scheduler, TRT-LLM in-flight batching |
| Chunked prefill | vLLM `--enable-chunked-prefill` + `--max-num-batched-tokens` |
| Prefix caching | PagedAttention block hashing (vLLM) or RadixAttention tree sharing (SGLang) |
| PagedAttention | vLLM block allocator |
| RadixAttention | SGLang radix-tree KV organisation |
| Disaggregated prefill–decode | NVIDIA Dynamo; vLLM V1 P/D; DistServe / Splitwise reference systems |
| Speculative decoding | Draft-target / Medusa / EAGLE / Lookahead / n-gram verification path in the decode loop |
| TP | `--tensor-parallel-size`, intra-node collectives |
| PP | `--pipeline-parallel-size`, inter-stage activation transport |
| EP | `--expert-parallel-size`, AllToAll dispatch |
| DP attention | SGLang DP-attention (and equivalent in vLLM, `verify`) |
| LoRA swap / multi-LoRA | `--enable-lora` + per-request adapter id; adapter fusion in forward pass |
| Model sharding | Composition of TP, PP, EP |
| Streaming | HTTP SSE emission tied to per-token output; backpressure and client-disconnect handling |
| Structured output | Outlines / XGrammar backends; engine guided-decoding modes |

### 2.8 Alias / anti-pattern section

The following dashboard labels on our fleet are not in the glossary. Each row names the rename needed to bring it into line:

| Observed label | Why it fails | Rename |
|---|---|---|
| `latency_p99` | Ambiguous: TTFT, ITL, E2E, or queue delay. Hides which SLI is actually burning. | Split into `inference_ttft_ms_p99`, `inference_itl_ms_per_token_p99`, `inference_e2e_latency_ms_p99` depending on the panel. |
| `tok_per_sec` (unscoped) | Does not distinguish throughput from goodput; hides SLO-conformance. | Rename to `inference_output_tokens_per_second` and add `inference_goodput_tokens_per_second` next to it on every throughput panel. |
| `cache_hit` (unscoped) | Overloaded — could be prefix cache, HTTP cache, or model-weight cache. | Rename to `inference_prefix_cache_hit_rate` and reserve `cache_hit` for the HTTP caching layer. |
| `queue` (as a bare metric) | Could be gateway queue, scheduler queue, or client-side queue. | Rename to `inference_scheduler_queue_depth` for the engine-internal queue and `gateway_queue_depth` for ingress. |
| `gpu_util` | NVIDIA GPU utilisation is a sampling-based proxy, not an engine-level signal; dashboards often conflate it with throughput headroom. | Keep `dcgm_gpu_utilization` as the exporter name but *add* `inference_kv_cache_utilization_percent` and `inference_batch_size_p50` on the same panel so operators see engine-level signal, not just device-level. |
| `errors` (unscoped) | Hides whether the error is a 429, a 500, a tool-call schema fail, or a streaming disconnect. | Split into `http_requests_total{status="..."}`, `inference_structured_output_invalid_rate`, `inference_streaming_disconnects_total`. |

## 3. Validation steps

1. **7-field completeness** — Scan every term. Reject any entry missing a field; placeholders ("TBD", "—") are not valid.
2. **Operator-level test** — Read each definition aloud. If it is longer than two sentences or dips into kernel math, cut.
3. **New-hire disambiguation test** — Hand the glossary to a new hire with three mock tickets: a TTFT regression, a TPOT regression, and an ITL regression. The hire must label them differently, each with the right fix-owner. If they merge two, the TTFT/TPOT/ITL definitions are not operator-level enough.
4. **SRE-plug-in test** — A peer from `ai-infra-sre` should be able to plug the latency SLI definitions straight into their SRE methodology without rewriting the terms. If they must rewrite, the definitions still smell kernel-level.
5. **Prometheus marker audit** — Every `TODO: expose as ...` entry has a tracked ticket. The glossary is not pin-worthy until each `TODO` either has a metric or a ticket.
6. **Anti-pattern application** — Open the team's current dashboards. For each observed-label row in section 2.8, verify the rename has either landed or has a tracked ticket.

## 4. Rubric / review checklist

| # | Criterion | Pass signal |
|---|---|---|
| 1 | All required terms present with the full 7-field entry | Term-count = required-count |
| 2 | Term-to-engine-primitive table covers every term | Row-count = term-count |
| 3 | At least 3 dashboard anti-patterns named with rename | 6 rows in section 2.8 |
| 4 | A new hire can label TTFT, TPOT, ITL regression tickets differently | Dry-run with new hire |
| 5 | An `ai-infra-sre` peer accepts SLI definitions as-is | Reviewed in SLO meeting |
| 6 | Every Prometheus metric name is either currently exposed or tracked under a `TODO: expose as ...` | Scan complete |
| 7 | No definition is longer than two sentences | Word-count scan |

## 5. Common mistakes

- **Collapsing TTFT, TPOT, ITL, and E2E into "latency."** Every entry here disambiguates them; the three-ticket dry run surfaces the collapse quickly.
- **Writing kernel-level definitions.** Operators need "what breaks and who fixes it," not a math derivation. Push the math to Chapter 03 citations.
- **Using `tok/s` as throughput without goodput.** Fleet-wide `tok/s` without SLO conformance is a marketing number; always pair with goodput.
- **Missing `TODO: expose as ...` markers.** A proposed Prometheus metric that does not exist yet must be tracked, or it will not be built.
- **Forgetting the ITL definition's HTTP-layer dependency.** ITL depends on the engine *and* the HTTP streaming layer; a glossary that attributes ITL only to the engine will misroute tickets.
- **Lumping speculative-decoding flavours.** Draft-target, Medusa, EAGLE, Lookahead, and n-gram have different acceptance-rate profiles; an `inference_spec_decode_flavour_info` metric is needed to attribute regressions to the right flavour.
- **Omitting the preemption-rate entry.** Preemption rate is the leading indicator that KV capacity is thin; teams routinely forget it until their first swap-to-CPU incident.

## 6. References

- Chapter 03 — Inference Vocabulary: `../../../../ai-infra-inference-engineer-learning/lessons/mod-101-inference-engineering-position/03-inference-vocabulary.md`
- PagedAttention: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," arXiv:2309.06180 — <https://arxiv.org/abs/2309.06180>
- RadixAttention / SGLang: Zheng et al., arXiv:2312.07104 — <https://arxiv.org/abs/2312.07104>
- Orca (continuous batching): Yu et al., OSDI 2022 — <https://www.usenix.org/conference/osdi22/presentation/yu>
- vLLM docs — <https://docs.vllm.ai/>
- SGLang docs — <https://docs.sglang.ai/>
- TensorRT-LLM docs — <https://nvidia.github.io/TensorRT-LLM/>
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> (relevant for linking inference SLIs to AI RMF measurement constructs).
