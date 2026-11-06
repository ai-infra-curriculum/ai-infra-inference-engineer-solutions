# Exercise 04 — GPU Fleet Landscape Cheatsheet (Reference Solution)

Paired exercise: [`exercise-04-gpu-fleet-landscape-cheatsheet.md`](../../../../ai-infra-inference-engineer-learning/lessons/mod-101-inference-engineering-position/exercises/exercise-04-gpu-fleet-landscape-cheatsheet.md). Anchor chapter: [Chapter 04 — GPU Fleet Landscape](../../../../ai-infra-inference-engineer-learning/lessons/mod-101-inference-engineering-position/04-gpu-fleet-landscape.md). Primary-source index: [`resources.md`](../../../../ai-infra-inference-engineer-learning/lessons/mod-101-inference-engineering-position/resources.md).

## 1. Solution overview

The exercise asks learners to produce two artefacts:

- **Artefact A** — a one-page cheatsheet with a GPU SKU table, the four operator-depth questions, rules-of-thumb sections, and a red-flag list.
- **Artefact B** — three worked fleet-pick scenarios (70 B dense chat, 400 B+ MoE, smaller-model long tail).

The chapter is explicit: *do not fill numeric cells from memory.* Every cell is either (a) cited to a vendor primary source, or (b) explicitly marked `verify`. The cheatsheet is an artefact to *maintain* — it will go stale on the next hardware launch — so the `verify` cells carry `<!-- verify-source: ... -->` markers for the merge gate.

## 2. Implementation — Worked answer: the cheatsheet (one page)

> **Formatting note.** This solution ships the cheatsheet as Markdown so it renders inline in the repository. In the real artefact it should be exported to a single landscape A4 / Letter page (dense formatting; ≤ 10-pt table). If it exceeds one page, cut rules-of-thumb before cutting SKU rows (per acceptance criterion 1 in the exercise spec).

### 2.1 GPU SKU table

| SKU | HBM (GB) | HBM bandwidth (TB/s) | Peak FP16/BF16 compute | FP8 | FP4 | Intra-node | Inter-node (typical) | Engines (first-class) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| **NVIDIA H100 SXM** | 80 | `verify` | `verify` | yes (hardware, E4M3 / E5M2) | no | NVLink 4 (NVLink Switch; HGX 8-GPU domain) | InfiniBand / RoCE | vLLM, SGLang, TensorRT-LLM, Dynamo | 2022-2023 workhorse; FP8 target `verify` <!-- verify-source: current NVIDIA H100 spec page --> |
| **NVIDIA H200** | 141 | `verify` | `verify` | yes (hardware) | no | NVLink 4 (HGX 8-GPU domain) | InfiniBand / RoCE | vLLM, SGLang, TensorRT-LLM, Dynamo | H100 refresh; more memory `verify` <!-- verify-source: NVIDIA H200 spec page --> |
| **NVIDIA L40S** | 48 (GDDR6) | `verify` | `verify` | yes (hardware) | no | PCIe only (no NVLink for TP across GPUs) | InfiniBand / RoCE | vLLM, SGLang, TensorRT-LLM | Enterprise serving SKU; not HBM; TP topology capped at single GPU unless the engine runs TP over PCIe `verify` |
| **NVIDIA L4** | 24 (GDDR6) | `verify` | `verify` | yes (hardware) | no | PCIe only | `verify` | vLLM, SGLang | Edge / low-cost serving; not a 70B-class target |
| **NVIDIA B200** | 192 | `verify` | `verify` | yes (hardware, E4M3 / E5M2) | yes (hardware, NVFP4 / MXFP4 microscaling) | NVLink 5 (HGX; `verify` current GPU-count-per-domain) | InfiniBand / RoCE | TensorRT-LLM, Dynamo day-one; vLLM / SGLang follow `verify` | Blackwell generation; FP4 signature capability `verify` <!-- verify-source: current engine support for FP4 on B200 --> |
| **NVIDIA GB200 NVL72** | 192 per GPU | `verify` | `verify` | yes (hardware) | yes (hardware) | NVLink 5 + NVLink Switch across 72 GPUs (single unified domain) | InfiniBand / RoCE | TensorRT-LLM, Dynamo | Rack-scale coherent pool; single NVLink Switch domain = 72 GPUs, extending TP ceiling `verify` |
| **NVIDIA GB300 NVL72** | `verify` | `verify` | `verify` | yes (hardware) | yes (hardware) | NVLink 5 + NVLink Switch | InfiniBand / RoCE | `verify` | Next generation; SKU matrix still shifting `verify` <!-- verify-source: GB300 NVL72 spec page and GA timing --> |
| **AMD MI300X** | 192 | `verify` | `verify` | yes (hardware) | no | AMD Infinity Fabric | Infinity Fabric + ROCm RCCL | vLLM (ROCm), SGLang (ROCm), TensorRT-LLM-adjacent via community paths `verify` | HBM capacity differentiator; feature parity with NVIDIA path varies per release `verify` <!-- verify-source: AMD MI300X spec page --> |
| **AMD MI325X** | 256 | `verify` | `verify` | yes (hardware, extended matrix) | `verify` | AMD Infinity Fabric | Infinity Fabric + RCCL | vLLM (ROCm), SGLang (ROCm) `verify` | Larger HBM than MI300X; FP4 status unconfirmed `verify` <!-- verify-source: AMD MI325X spec page and FP4 support --> |
| **AWS Inferentia2 (Inf2)** | `verify` | `verify` | `verify` | `verify` | `verify` | AWS proprietary (NeuronLink) | AWS EFA | vLLM via Neuron plug-in (narrower feature set) | Compile-ahead workflow with Neuron SDK; `verify` current vLLM/Neuron feature parity <!-- verify-source: AWS Inferentia2 spec page --> |
| **AWS Trainium2 (Trn2)** | `verify` | `verify` | `verify` | `verify` | `verify` | AWS proprietary (NeuronLink v2) | AWS EFA | vLLM via Neuron plug-in `verify` | Training-oriented current generation; inference use emerging `verify` <!-- verify-source: AWS Trainium2 spec page --> |
| **Google TPU v5e** | `verify` | `verify` | `verify` | `verify` | `verify` | Google ICI (pod) | Google ICI (pod) | JAX / XLA native; vLLM TPU backend | Serving / cost-optimised TPU class `verify` <!-- verify-source: Google TPU v5e product page --> |
| **Google TPU v5p** | `verify` | `verify` | `verify` | `verify` | `verify` | Google ICI (pod) | Google ICI (pod) | JAX / XLA native | Scale-up training TPU class; also used for large-model serving `verify` |
| **Google TPU v6 (Trillium)** | `verify` | `verify` | `verify` | `verify` | `verify` | Google ICI (pod) | Google ICI (pod) | JAX / XLA / Pallas | Current-generation branded Trillium `verify` <!-- verify-source: Google TPU v6 product page --> |
| **Google TPU v6 (Ironwood)** | `verify` | `verify` | `verify` | `verify` | `verify` | Google ICI (pod) | Google ICI (pod) | `verify` | Inference-focused successor; GA timing `verify` <!-- verify-source: GA status --> |

**Every cell marked `verify` must be filled from the vendor product page or engine release notes before this cheatsheet is pinned.** The `<!-- verify-source: ... -->` markers are intentional merge blockers.

### 2.2 Four operator-depth questions (decision loop)

Open these on the shared screen during every capacity meeting.

1. **Can the model fit?** — Memory sizing: weights (bytes-per-parameter × parameter-count), KV footprint per in-flight sequence, activation working set. Per numeric format: FP16/BF16 ≈ 2 bytes/param, FP8 ≈ 1 byte/param, FP4 ≈ 0.5 byte/param. KV per sequence ≈ `seq_len × 2 × num_layers × num_kv_heads × head_dim × bytes_per_element`.
2. **Will the interconnect carry it?** — Tensor Parallelism stays inside one NVLink Switch domain (8-GPU HGX for Hopper/Ada; up to 72-GPU for GB200 / GB300 NVL72). Pipeline Parallelism spans nodes over IB / RoCE / EFA. Expert Parallelism for MoE requires AllToAll-friendly interconnect and can regress sharply on undersized fabric.
3. **Does the numeric format match the hardware?** — FP8 requires hardware support on Hopper+ (or MI300-class on AMD). FP4 / NVFP4 / MXFP4 requires Blackwell+ hardware (NVIDIA) or `verify` on AMD. BF16 works everywhere. If the engine claims FP8/FP4 but the hardware is older, it is emulation — advertised throughput will not materialise.
4. **Does the engine support it as a first-class target?** — A SKU is only first-class when the engine's docs explicitly list it. Experimental or community-plug-in support is a risk flag, not a procurement input.

### 2.3 Rules of thumb

**Memory sizing**

- Weights (dense): FP16/BF16 ≈ 2 bytes/param; FP8 ≈ 1 byte/param; FP4 ≈ 0.5 byte/param; INT8 ≈ 1 byte/param; INT4 ≈ 0.5 byte/param.
- Weights (MoE): size the memory for the *total* expert set plus routing, not just the active experts; EP topology shards experts across GPUs.
- KV cache per in-flight sequence: `seq_len × 2 × num_layers × num_kv_heads × head_dim × bytes_per_element`.
- Reserve 10 – 30 % headroom for activations, workspace, and fragmentation; the exact fraction depends on batch size and chunk size — the engine profiler is the source of truth.
- Prefix caching trades memory for TTFT: budget KV capacity for the cache *on top of* in-flight sequences.

**Parallelism ceilings**

- TP inside one NVLink Switch domain only. HGX: 8 GPUs. GB200 / GB300 NVL72: up to 72 GPUs in the rack.
- PP spans nodes; sized to depth of the model and the inter-node fabric.
- EP for MoE sized to the AllToAll bandwidth of the interconnect; oversubscribing EP across a weak fabric is the classic MoE regression.
- DP attention pairs with EP on MoE; the composition is engine-specific (SGLang is the reference implementation; vLLM has equivalents, `verify`).
- When in doubt, keep TP within one node and let PP (dense) or EP (MoE) take the inter-node hops.

**Numeric-format fallbacks**

- If FP4 does not yet work in the engine today → fall back to FP8 on Hopper/Blackwell.
- If FP8 regresses accuracy on the specific model → fall back to BF16.
- Never fall back to INT8 without passing the model's own accuracy gate; INT8 PTQ is lossy on some architectures.
- If the hardware does not have the tensor-core path for the claimed format, the claim is emulation; drop it until the right SKU arrives.
- On MI300-class AMD, FP8 maturity trails NVIDIA; `verify` the engine supports FP8 on ROCm for the specific model before committing.

### 2.4 Red-flag list

Pause a fleet decision when any of these is true:

1. "FP8" is claimed but the engine release notes do not list hardware support on that SKU (= emulation; advertised throughput will not materialise).
2. TP size proposed larger than the NVLink Switch domain (collectives fall off-node and tail latency explodes).
3. Engine support status for a load-bearing feature is "experimental" or carries a `verify` marker on the cheatsheet.
4. Procurement timeline lands the hardware after the model-release deadline (the fleet cannot serve the model when it ships).
5. A single-vendor assumption is embedded in the plan without a fallback SKU (the primary vendor is the only path if things slip).
6. The MoE plan oversubscribes AllToAll across an inter-node fabric that has not been benchmarked for this model (classic MoE goodput cliff).

## 3. Worked answer — three fleet picks

### 3.1 Scenario 1 — 70 B dense chat model

**Workload:** 70 B dense; 4 k average input; 500 output tokens; 50 % prefix-cache hit on a shared system prompt; 10 k RPS peak; OpenAI-compatible API; P99 TTFT target ~100 ms.

**Pick:**

- GPU SKU: **NVIDIA H200** (fallback **H100 SXM** if H200 capacity is thin).
- Count: 8 GPUs per node (one HGX box) × enough nodes to carry 10 k RPS — size against goodput, not nameplate throughput; replicas sized by the canary measurement.
- Numeric format: **FP8 (E4M3)** on weights, BF16 KV.
- Engine: **vLLM** (OpenAI-compatible server first-class; PagedAttention + prefix caching + chunked prefill all first-class; FP8 supported on Hopper+ `verify`).
- Parallelism topology: **TP=8** inside one NVLink Switch domain. No PP (not required). No EP (dense model).
- Prefix-cache strategy: enable PagedAttention prefix caching globally; size KV capacity to carry the 50 % hit with the expected working set.

**Risk:** FP8 accuracy regression on this specific model. Mitigation: run the model's own accuracy gate on the FP8 build; if it fails, fall back to BF16 weights and re-size the fleet.

### 3.2 Scenario 2 — 400 B+ MoE model

**Workload:** DeepSeek-V3-scale MoE (`verify` current parameter / expert counts if real); 1 k input, 500 output; 1 k RPS peak; P99 TTFT acceptable at 500 ms.

**Pick:**

- GPU SKU: **NVIDIA GB200 NVL72** (fallback: **H200** 8-GPU nodes if GB200 capacity is not yet in-fleet; fallback carries a goodput haircut).
- Count per node: full rack — 72 GPUs in a single NVLink Switch domain.
- Nodes: sized against the measured goodput for the chosen EP × DP-attention topology; start with one NVL72 rack per region and scale on canary data.
- EP topology: experts sharded across the 72-GPU domain; DP attention paired with EP.
- Numeric format: **FP8** for a conservative first roll; **FP4 (NVFP4 / MXFP4)** when the engine ships FP4 for this model `verify`.
- Engine: **SGLang** (DP attention is first-class; MoE routing first-class) or **TensorRT-LLM via Dynamo** (first-class FP8 / FP4 on Blackwell; disaggregated prefill-decode via Dynamo). Pick between them on a benchmarked canary for this exact workload.
- Parallelism topology: EP across the NVL72 domain; DP attention; disaggregated prefill-decode on Dynamo if the chosen engine supports it.

**Risks:**

1. **MoE AllToAll bandwidth.** If the engine's EP routing oversubscribes the intra-rack fabric for the workload's token distribution, goodput cliffs. Mitigation: benchmark EP dispatch imbalance on canary; have a fallback EP size before promotion.
2. **FP4 / NVFP4 engine maturity.** Blackwell FP4 support is first-class on TensorRT-LLM; vLLM / SGLang timelines vary. Mitigation: start on FP8 and switch to FP4 once the engine's release notes list first-class support for the specific model.

### 3.3 Scenario 3 — Smaller-model long tail (1 B – 13 B, ~20 models, ≤ 100 RPS each)

**Workload:** Cost-sensitive; broad model catalogue; latency targets looser than Scenario 1.

**Pick:**

- GPU SKU: **NVIDIA L40S** (fallback **L4** for the smallest models in the catalogue; fallback **H100 SXM** for models that spike above 100 RPS or need FP8 headroom).
- Co-location strategy: **MIG partitions** or **Multi-Process Service (MPS)** on supported SKUs; **Dynamic Resource Allocation (DRA)** when the fleet is on a Kubernetes release that supports it (covered in module 109 — treat it as the forward path).
- Engine: **vLLM** (OpenAI-compatible first-class; broad hardware coverage; multi-adapter LoRA for models that share a base).
- Numeric format: **FP8** on L40S when the model has an FP8 recipe that clears the accuracy gate; **BF16** otherwise. For the smallest (1 B – 3 B) models, **AWQ / GPTQ** quantisation to shrink the per-replica memory footprint.

**Trade-off against going to a commercial provider.** A managed OpenAI-compatible provider (Together, Fireworks) is cheaper per token at low utilisation and faster to onboard. Self-hosting wins when (a) tenant isolation or data residency rules out managed providers, (b) multi-adapter LoRA cost beats per-token managed billing for the fleet's actual usage, or (c) steady-state utilisation is high enough to amortise the GPU spend. Decision rule: if the long-tail fleet's rolling 7-day average utilisation on L40S is below ~30 %, revisit the commercial-provider option before the next procurement cycle.

## 4. Validation steps

1. **Cell-citation scan** — Open the SKU table. Every cell that is not `verify` must be traceable to a vendor product page or engine release notes. Every `verify` cell must carry a `<!-- verify-source: ... -->` marker.
2. **One-page fit test** — Export the cheatsheet to a single landscape PDF. If it overflows, cut rules-of-thumb first (not SKU rows). If it still overflows, drop SKUs that are not in the fleet roadmap for the next 6 months.
3. **Scenario completeness** — Each scenario names (a) GPU SKU, (b) count, (c) numeric format, (d) engine, (e) parallelism topology. The MoE scenario names *two* risks; the other scenarios name at least one.
4. **Red-flag table-top** — Walk a procurement partner through the red-flag list in a dry run. If the partner cannot use the flags to pause a real proposal, rewrite the flag to be more specific.
5. **Rot calendar** — Set a quarterly refresh on this artefact and an event-triggered refresh on every vendor hardware launch (new NVIDIA generation, new ROCm MI SKU, new TPU generation).

## 5. Rubric / review checklist

| # | Criterion | Pass signal |
|---|---|---|
| 1 | Cheatsheet fits one page | Export passes |
| 2 | Every cell is cited to a primary source or marked `verify` with a `verify-source` marker | Scan complete |
| 3 | The four operator-depth questions are present and answerable from the SKU table alone | Dry-run with a peer |
| 4 | Each scenario names a specific GPU SKU, count, numeric format, engine, and parallelism topology | Explicit in each pick |
| 5 | Scenario 1 names ≥ 1 risk; Scenario 2 names ≥ 2 risks; Scenario 3 names the commercial-provider trade-off | Explicit in each pick |
| 6 | Red-flag list contains at least the four items in the exercise spec, specific enough to pause a real proposal | Checked row-by-row |
| 7 | Rules-of-thumb sections stay ≤ 5 bullets each | Scan complete |

## 6. Common mistakes

- **Filling numeric cells from memory.** HBM capacities and bandwidths rotate per generation; the only acceptable fill is a vendor-page citation or a `verify` marker.
- **Collapsing "FP8" into one row.** FP8 means different things on Hopper vs. Blackwell vs. MI300-class; distinguish hardware tensor-core support from library emulation.
- **Proposing TP > NVLink Switch domain.** TP=16 on HGX 8-GPU boxes is a classic regression. The ceiling is the NVLink Switch domain size, not the node count.
- **Picking GB200 for the dense 70 B scenario.** GB200 is overkill for a 70 B dense chat workload; H200 (or even H100 SXM) is the right first pick. Save NVL72 for 400 B+ MoE and larger.
- **Ignoring AllToAll bandwidth on MoE.** The MoE scenario's biggest risk is EP dispatch imbalance over an undersized fabric; a plan without an AllToAll benchmark is a plan with a goodput cliff.
- **Treating the cheatsheet as a one-shot.** It will go stale on the next hardware launch. The red-flag list is the single most load-bearing section; the SKU table is the one most likely to rot.
- **No fallback SKU for commercial / supply scenarios.** Scenario 2 depends on NVL72 availability. A plan without a fallback (H200 nodes) will stall when Blackwell delivery slips.

## 7. References

- Chapter 04 — GPU Fleet Landscape: `../../../../ai-infra-inference-engineer-learning/lessons/mod-101-inference-engineering-position/04-gpu-fleet-landscape.md`
- Module resources index: `../../../../ai-infra-inference-engineer-learning/lessons/mod-101-inference-engineering-position/resources.md`
- NVIDIA data-center products — H100: <https://www.nvidia.com/en-us/data-center/h100/>; H200: <https://www.nvidia.com/en-us/data-center/h200/>; Blackwell: <https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/>; GB200 NVL72: <https://www.nvidia.com/en-us/data-center/gb200-nvl72/>; NVLink / NVSwitch: <https://www.nvidia.com/en-us/data-center/nvlink/>
- AMD Instinct — MI300X: <https://www.amd.com/en/products/accelerators/instinct/mi300.html>; MI325X: <https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html>; ROCm docs: <https://rocm.docs.amd.com/>
- AWS — Trainium: <https://aws.amazon.com/machine-learning/trainium/>; Inferentia: <https://aws.amazon.com/machine-learning/inferentia/>; Neuron SDK docs: <https://awsdocs-neuron.readthedocs-hosted.com/>
- Google — Cloud TPU: <https://cloud.google.com/tpu>; JAX / Pallas: <https://docs.jax.dev/>
- Engine release notes (for cell verification): vLLM — <https://github.com/vllm-project/vllm/releases>; SGLang — <https://github.com/sgl-project/sglang/releases>; TensorRT-LLM — <https://github.com/NVIDIA/TensorRT-LLM/releases>; NVIDIA Dynamo — <https://docs.nvidia.com/dynamo/>
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> (relevant to the procurement / supply-chain red-flag rows).
