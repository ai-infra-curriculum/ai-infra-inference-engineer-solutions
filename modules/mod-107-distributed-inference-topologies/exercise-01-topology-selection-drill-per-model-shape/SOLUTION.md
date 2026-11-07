# Solution — Exercise 01: Topology Selection Drill per Model Shape

> Paired learning file: [`lessons/mod-107-distributed-inference-topologies/exercises/exercise-01-topology-selection-drill-per-model-shape.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-107-distributed-inference-topologies/exercises/exercise-01-topology-selection-drill-per-model-shape.md)
> Module: `mod-107-distributed-inference-topologies`
> Type: design / decision drill

## 1. Solution overview

The learner is handed several concrete **model shapes** (dense vs. MoE, parameter
count, context length, QPS target, available GPU fleet) and must pick a distributed
topology for each. A correct answer names one of the canonical shardings — **tensor
parallel (TP)**, **pipeline parallel (PP)**, **expert parallel (EP)**, **data
parallel (DP)**, a **hybrid** (e.g., TP×PP×DP), or **disaggregated prefill/decode** —
and defends the choice against two first-order constraints:

1. **Weights fit in device memory** (after KV cache reservation at the target
   concurrency and context length).
2. **The collective / communication pattern fits the interconnect** (NVLink/NVSwitch
   inside a node, PCIe or InfiniBand across nodes).

The drill is not about finding the "fastest" number on a benchmark; it is about
producing a defensible **first cut** that an SRE can prototype in half a day. The
rubric rewards learners who name the second-order lever they would pull if the first
cut misses the latency SLO.

### Selection priorities, in order

1. Does one replica of the weights (plus runtime KV budget) fit on a single GPU?
   → **DP only.** Launch N independent single-GPU engines behind a load balancer.
2. Does it fit within a single node (≤ 8 GPUs connected by NVLink/NVSwitch)?
   → **TP = node size.** Tensor parallel keeps all-reduces on NVLink where they
   belong.
3. Is it a dense model that straddles multiple nodes?
   → **TP = intra-node, PP = inter-node, DP = across replicas.** Keep TP's
   bandwidth-heavy all-reduces inside NVLink; pay inter-node cost only for PP
   send/recv, which is low-volume activations.
4. Is it a Mixture-of-Experts model with many experts?
   → **EP for the expert MLPs, TP for the attention + shared layers, DP as the
   outer replication.** Experts live on different ranks; `all_to_all` dispatches
   tokens.
5. Are prefill (compute-bound) and decode (memory-bound) competing for the same
   GPU cycles and blowing the TTFT SLO?
   → **Disaggregated prefill/decode:** separate GPU pools, KV cache transferred
   from the prefill pool to the decode pool over RDMA.

Papers and documentation underpinning each rule are enumerated in §6.

## 2. Implementation — worked answer

### 2.1 Drill workload table

The learner answers for each shape in a table. A worked example uses six shapes,
each with the same columns.

| ID | Model shape | Params | Context | Target QPS | Fleet |
|---|---|---|---|---|---|
| A | Dense 7B (e.g. Llama-3-8B class) | 8 B | 8 k | 100 | 1× node of 8× 80 GiB |
| B | Dense 70B | 70 B | 32 k | 20 | 1× node of 8× 80 GiB |
| C | Dense 405B | 405 B | 32 k | 20 | 4× nodes of 8× 80 GiB |
| D | MoE 8×22B active-39B (e.g. Mixtral-class) | 141 B total, ≈39 B active | 32 k | 50 | 2× nodes of 8× 80 GiB |
| E | MoE ≈671B total, ≈37 B active (DeepSeek-V3 class) | 671 B total, ≈37 B active | 128 k | 10 | 4× nodes of 8× 80 GiB |
| F | Dense 70B chat with P95 TTFT 1500 ms, bursty 128 k prompts | 70 B | 128 k | 40 | 8× nodes of 8× 80 GiB |

Parameter counts above are **public, published model specs** at the class level;
exact checkpoint values should be verified against the model card before any sizing
commitment.

### 2.2 Worked picks

The following picks are first-cut recommendations. Each defends against the two
first-order constraints (weights fit, collectives fit interconnect) and names one
second-order lever.

**Shape A — Dense 7B, 1 node of 8×80 GiB.**
- Weights at FP16: ≈ 16 GiB. Fits a single GPU with KV budget to spare.
- **Pick: DP = 8, TP = 1.** Eight independent single-GPU engine replicas behind the
  router. QPS scales linearly with replica count. No collectives on the hot path.
- Second-order lever: if P50 latency is fine but P99 degrades under burst, switch to
  **TP = 2, DP = 4** to shrink per-request latency at the cost of some throughput.
- Why not TP = 8? Needless intra-node all-reduce traffic; TP cost is paid per layer
  per decode step with no latency win when a single GPU already has headroom.

**Shape B — Dense 70B, 1 node of 8×80 GiB.**
- Weights at FP16: ≈ 140 GiB. Does **not** fit one GPU. Fits across 2+ GPUs at FP16
  or one GPU at INT4.
- **Pick: TP = 8, DP = 1.** Shard across the full NVLink domain. Keeps all-reduces on
  NVLink/NVSwitch where bandwidth is highest.
- Second-order lever: enable **chunked prefill** and tune `max-num-batched-tokens`
  (module 103) to protect TPOT under long prompts; if the fleet grows to multiple
  nodes, add DP across replicas, not PP, because PP's bubble is not worth it for a
  70B model that already fits one node.
- Why not PP within a node? PP's inter-stage traffic is small, but it adds the
  pipeline bubble at low batch sizes typical of decode and gains nothing when TP
  across NVLink already fits.

**Shape C — Dense 405B, 4 nodes of 8×80 GiB.**
- Weights at FP16: ≈ 810 GiB. Fits across 4 nodes at FP16 or 2 nodes at FP8.
- **Pick: TP = 8 (intra-node), PP = 4 (inter-node), DP = 1.** TP all-reduces stay on
  NVLink; PP send/recv crosses InfiniBand at low volume (activations only).
- Second-order lever: raise global batch via **chunked prefill + continuous batching**
  to amortize PP bubble. If TTFT suffers at long prompts, consider disaggregated
  prefill/decode (Shape F lever).
- Why not TP = 32 across all four nodes? Inter-node all-reduce per decode step would
  dominate the critical path on commodity IB; TP is bandwidth-bound and belongs on
  NVLink. (Megatron-LM §3; NVIDIA inference parallelism docs; vLLM distributed
  serving docs.)

**Shape D — MoE 8×22B (Mixtral-class), 2 nodes of 8×80 GiB.**
- Total weights at FP16: ≈ 280 GiB. Fits across 2 nodes, or 1 node at FP8.
- **Pick: TP = 8 (intra-node) on shared layers + attention; EP = 8 on the expert
  MLPs; DP = 2 across the two nodes.** With 8 experts per MoE layer and TP = 8, a
  common assignment places one expert per rank. Routed tokens travel via `all_to_all`
  within the node.
- Second-order lever: if routing is skewed (hot experts), increase **capacity
  factor** or enable **drop tokens** fallback; if `all_to_all` becomes the hot
  spot, consider grouping experts by top-k co-occurrence onto adjacent ranks.
- Why not TP = 16 across nodes? Inter-node TP all-reduce punishes every layer, and
  you still haven't addressed MoE routing — EP is the correct primitive for the
  MLP block.

**Shape E — MoE ≈671B total, ≈37B active (DeepSeek-V3 class), 4 nodes of 8×80 GiB.**
- Total weights at FP8: ≈ 671 GiB. Fits across 4 nodes at FP8 (comfortably), or 8
  nodes at FP16.
- **Pick: TP = 8 (intra-node) + EP across all 32 ranks for the expert MLPs + DP = 1
  as a baseline.** With many experts per MoE layer, each rank hosts multiple
  experts; `all_to_all` goes inter-node, so RDMA is mandatory and fabric-level
  tuning (GPUDirect RDMA, NCCL env) is in scope.
- Second-order lever: if `all_to_all` latency dominates, add **expert redundancy /
  replication** for the top-k hot experts so routing has local options, or increase
  concurrency to amortize the fixed `all_to_all` cost.
- Why not PP? With TP+EP already saturating 32 ranks and the model fitting at FP8,
  adding PP buys little and adds bubbles; keep PP as a lever if a future node-count
  cut forces it.

**Shape F — Dense 70B, 128 k context, strict P95 TTFT 1500 ms, 8 nodes.**
- Weights fit on any single node at FP16 (TP = 8 covers it). The constraint is
  **latency under bursty 128 k prompts**: prefill of a 128 k prompt dominates
  decode-engine cycles and blows TTFT for anyone whose request lands behind it.
- **Pick: disaggregated prefill/decode.** Split the fleet, e.g., 2 nodes of **prefill
  workers (TP = 8 each)** and 6 nodes of **decode workers (TP = 8 each, DP = 6)**.
  Route each request's prefill to a prefill pool worker, transfer the computed KV
  cache to a decode worker over RDMA, then run decode there. Scale prefill and
  decode pools independently to the two SLOs (TTFT vs. TPOT).
- Second-order lever: if KV transfer bandwidth becomes the bottleneck, co-locate
  prefill and decode GPUs under the same NVLink switch where feasible, or compress
  the KV in flight (module 105).
- Why not more TP for one big pool? A single pool still queues long prefills behind
  decode steps on the same engine; TP speeds up each prefill but does not break the
  head-of-line blocking. Disaggregation is a *topology* fix for a *scheduling*
  pathology.

### 2.3 Memory sanity check template

Every pick should be accompanied by this one-line sanity check. If it fails, the
pick fails, regardless of how clever it looks.

```
per_gpu_weight_bytes = total_param_bytes / (TP * PP)
per_gpu_kv_bytes     = bytes_per_token_kv(model) * target_concurrency * context_len / TP
per_gpu_overhead     = CUDA graphs + activations + framework (~2–6 GiB)
assert per_gpu_weight_bytes + per_gpu_kv_bytes + per_gpu_overhead <= gpu_hbm * target_util
```

`target_util` is typically 0.85–0.92 for vLLM `--gpu-memory-utilization`; the exact
number is an engine knob and should be pinned per deployment. `bytes_per_token_kv`
is the formula from module 104:
`2 * n_layers * n_kv_heads * head_dim * bytes_per_element`.

Before adopting any pick, perform this check with the actual model card's
`n_layers`, `n_kv_heads`, `head_dim`, dtype, and the fleet's accelerator HBM.
Approximate numbers are acceptable **only** if clearly labelled as illustrative.

## 3. Validation steps

A reviewer validating a learner submission should do the following, in order:

1. **Did they name a topology, not a product?** "Use vLLM" is not a topology. "TP = 8
   intra-node + DP = 2 across nodes" is.
2. **Does weight-fit check pass?** Run the §2.3 sanity check against each pick's
   per-GPU memory load. If it overflows, the pick is invalid.
3. **Does the collective fit the interconnect?** TP all-reduces must live on
   NVLink/NVSwitch; inter-node links should see PP's low-volume send/recv or EP's
   `all_to_all`, not TP all-reduces. (Megatron-LM; NCCL docs.)
4. **Was the "why not" column populated?** Reviewers cross-check the rejected
   alternatives: common misses are "why not TP across nodes" and "why not PP within
   a node."
5. **Was a second-order lever named?** A first cut without a second-order lever is a
   half answer. Chunked prefill, DP replication, disaggregation, FP8 weights, FP8/
   INT4 KV, admission control — all valid.
6. **For MoE shapes, is EP named for the experts, not TP?** A common failure is to
   treat an MoE like a dense model and scale with TP alone. EP routes tokens to the
   expert ranks; TP shards the shared layers. (Megatron-Core / DeepSpeed-MoE docs.)
7. **For latency-constrained shapes (F-style), is disaggregation named?** If the
   workload has strict TTFT with bursty long prompts and the one-pool pick does not
   protect TTFT, the learner should recognize the pathology. (DistServe; Splitwise;
   Mooncake.)

### Optional dynamic validation (small-scale)

If the learner has a 2-GPU box and a small model (e.g., a 7B-class open-weights
model), they can observe each topology's effect on per-token latency:

```bash
# TP = 1 baseline
python -m vllm.entrypoints.openai.api_server \
  --model <public-7b-model> \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85

# TP = 2 (requires NCCL-visible GPUs)
python -m vllm.entrypoints.openai.api_server \
  --model <public-7b-model> \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85
```

Expected qualitative result for a 7B model: TP = 2 on a single node *increases*
per-token latency slightly (collective overhead) with no throughput win, matching
the Shape A rejected pick. Observing this at small scale helps the learner internalize
that "more parallelism" is not automatically "better."

## 4. Rubric / review checklist

Score each item 0 (missing), 1 (partial), 2 (fully correct). Passing: ≥ 10 / 14.

| # | Item | Points |
|---|---|---|
| 1 | Correct topology named for each supplied shape (TP/PP/EP/DP/hybrid/disagg) | 2 |
| 2 | Memory-fit sanity check applied per pick (weights + KV + overhead ≤ HBM·util) | 2 |
| 3 | Intra-node vs. inter-node interconnect rule applied (TP ⟂ NVLink; PP/EP ⟂ IB) | 2 |
| 4 | MoE shapes use EP for experts (not only TP) and name `all_to_all` as the collective | 2 |
| 5 | Latency-constrained shape names disaggregated prefill/decode as the lever | 2 |
| 6 | Each pick includes a "why not <alternative>" sentence with a concrete reason | 2 |
| 7 | Each pick names a second-order lever to pull if first cut misses SLO | 2 |

## 5. Common mistakes

- **Scaling TP past the NVLink boundary.** TP's all-reduce per layer per step is
  bandwidth-dominated. Running TP = 16 across two nodes on commodity InfiniBand is
  a classic footgun; Megatron-LM §3 is explicit that TP belongs inside the
  high-bandwidth domain. Fix: cap TP at node size, scale further with PP or DP.
- **Using PP when TP within a node already fits.** PP adds a pipeline bubble that
  degrades decode latency at low batch size. PP is for the case where the model
  does not fit the NVLink domain, not a default.
- **Treating MoE as dense.** Running TP alone on an MoE wastes most of the fleet on
  cold experts and serializes routing. EP is the right primitive; tokens flow to
  experts via `all_to_all`. (Megatron-Core MoE; DeepSpeed-MoE.)
- **Ignoring KV cache in the sizing.** Weights fit, but the engine OOMs under the
  target concurrency × context length because KV was omitted from the budget.
  Module 104's bytes-per-token-KV formula is non-optional.
- **Picking disaggregation as the first lever.** Disaggregation adds operational
  complexity (two pools, KV transfer, independent autoscaling). It is the right
  fix when prefill queues block decode; it is overkill for shapes where the engine
  already meets SLO. (DistServe; Splitwise.)
- **Confusing TP = N with "needs N GPUs."** TP = N is a sharding choice, not a
  capacity floor. You can have TP = 2 with DP = 4 on an 8-GPU node, and that is
  often the right answer for 70B at FP8 or 13B at FP16.
- **Picking "whichever product supports it."** The paired learning file deliberately
  frames the question in terms of model shape, not engine; the answer is a topology,
  and multiple engines (vLLM, SGLang, TensorRT-LLM) support the same topologies.

## 6. References

- Shoeybi, M. et al. *Megatron-LM: Training Multi-Billion Parameter Language Models
  Using Model Parallelism.* arXiv:1909.08053 — the canonical TP + PP paper that
  establishes the intra-node / inter-node rule.
- Narayanan, D. et al. *Efficient Large-Scale Language Model Training on GPU
  Clusters Using Megatron-LM.* SC 2021 — TP × PP × DP composition details.
- Rajbhandari, S. et al. *DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and
  Training to Power Next-Generation AI Scale.* ICML 2022 — expert parallelism and
  `all_to_all`.
- Fedus, W., Zoph, B., Shazeer, N. *Switch Transformer: Scaling to Trillion
  Parameter Models with Simple and Efficient Sparsity.* JMLR 2022 — the baseline
  MoE routing model that EP is built around.
- Zhong, Y. et al. *DistServe: Disaggregating Prefill and Decoding for
  Goodput-Optimized Large Language Model Serving.* OSDI 2024 — disaggregated
  prefill/decode case for TTFT-constrained workloads.
- Patel, P. et al. *Splitwise: Efficient Generative LLM Inference Using Phase
  Splitting.* ISCA 2024 — independent-contribution on prefill/decode disaggregation.
- Qin, J. et al. *Mooncake: A KVCache-centric Disaggregated Architecture for LLM
  Serving.* arXiv:2406.17565 — production disaggregated serving architecture.
- NVIDIA. *NCCL User Guide.* https://docs.nvidia.com/deeplearning/nccl/user-guide/ —
  collective semantics, NVLink/NVSwitch topology considerations.
- vLLM documentation. *Distributed Serving.* https://docs.vllm.ai/ (practitioner
  implementation): `--tensor-parallel-size`, `--pipeline-parallel-size`,
  Ray-orchestrated multi-node.
- SGLang documentation. https://docs.sglang.ai/ (practitioner implementation):
  tensor and pipeline parallel runtime flags.
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/)
  — not invoked for this design exercise.
- [mitre-atlas](https://atlas.mitre.org/) — not invoked for this design exercise.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework) — governance
  baseline; applicable to topology choices that affect tenant isolation
  (cross-tenant KV sharing is covered in module 104 exercise 06).
