# Exercise 02 — Inference Engine Capability Map (Reference Solution)

Paired exercise: [`exercise-02-inference-engine-capability-map.md`](../../../../ai-infra-inference-engineer-learning/lessons/mod-101-inference-engineering-position/exercises/exercise-02-inference-engine-capability-map.md). Anchor chapter: [Chapter 02 — Engine Capability Map](../../../../ai-infra-inference-engineer-learning/lessons/mod-101-inference-engineering-position/02-engine-capability-map.md). Primary-source index: [`resources.md`](../../../../ai-infra-inference-engineer-learning/lessons/mod-101-inference-engineering-position/resources.md).

## 1. Solution overview

The exercise produces three artefacts:

- **Artefact A** — a capability spreadsheet with ten rows (Chapter 02's ten capability axes) and eight columns (vLLM, SGLang, TensorRT-LLM, NVIDIA Dynamo, Triton Inference Server, llama.cpp, MLC-LLM, TGI).
- **Artefact B** — a commercial compare-point summary across NVIDIA NIM, Fireworks, Together, Baseten, Modal, Groq, and Cerebras on the three axes *what's unique*, *build-vs-buy fit*, and *exit cost*.
- **Artefact C** — a written recommendation of ≤ 300 words that picks one engine for a stated workload.

The exercise explicitly forbids filling cells from Chapter 02 claims uncritically. Every capability claim is tied to a primary source from `resources.md`, or carries an `<!-- verify-source: ... -->` marker. The marker is a merge blocker in this repository.

## 2. Implementation — Worked answer

### 2.1 Artefact A — Capability spreadsheet

Legend: **first-class** (documented, GA, tested by the project) / **supported** (documented, GA, may have caveats) / **experimental** (labelled by the project as such, or behind a flag) / **unsupported** (project says no, or the hardware or kernel path is missing) / **N/A** (not applicable to the engine's design) / **verify** (primary source needed for the next refresh).

> This solution carries every cell that depends on current engine release notes as `verify` plus a `<!-- verify-source: ... -->` marker. The spreadsheet is designed to be edited on every release cycle, not asserted once. Chapter 02 was ambiguous or silent on many engine × axis combinations; the chapter is a point-in-time orientation, not a feature matrix.

| # | Axis | vLLM | SGLang | TensorRT-LLM (via Triton/Dynamo) | NVIDIA Dynamo | Triton Inference Server | llama.cpp | MLC-LLM | TGI (historical) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Scheduling model (continuous batching, chunked prefill, static) | first-class — continuous batching + chunked prefill + preemption with swap/recompute `verify` <!-- verify-source: current chunked-prefill defaults vs. release notes --> | first-class — continuous batching + cache-aware routing `verify` | first-class — in-flight batching + chunked context `verify` | N/A at engine layer — orchestrates request-level routing across prefill/decode workers | supported — generic dynamic/sequence batching; LLM scheduling via TRT-LLM backend | supported — server mode has batching; historically single-stream focus `verify` | supported — continuous batching in `mlc_llm serve` `verify` | `verify` <!-- verify-source: current maintenance status of TGI --> |
| 2 | KV-cache primitive (PagedAttention, RadixAttention, contiguous, tiered) | first-class — PagedAttention; prefix cache; copy-on-write; swap-to-CPU | first-class — RadixAttention (prefix sharing indexed by token-sequence prefix, LRU eviction) | first-class — paged KV cache + prefix caching `verify` | first-class — cross-worker KV transport (NIXL + plug-in backends) | `verify` — depends on backend | `verify` — contiguous historical default; prefix-cache support in flux <!-- verify-source: llama.cpp current prefix-cache status --> | `verify` <!-- verify-source: MLC-LLM KV-cache allocator details --> | `verify` <!-- verify-source --> |
| 3 | Attention / fused kernel path (Flash Attention version, vendor libs) | supported — FlashInfer + Flash Attention backends; custom CUDA / Triton kernels | supported — FlashInfer + Flash Attention `verify` | first-class — NVIDIA's own kernel suite + Flash Attention variants | N/A — orchestrator, not a kernel runtime | supported — passes through to backends | supported — custom kernels per backend (CUDA, Metal, Vulkan, CPU) `verify` | supported — compiler-generated kernels via TVM/Relax `verify` | `verify` <!-- verify-source --> |
| 4 | Quantization matrix (FP16/BF16, FP8, FP4/NVFP4/MXFP4, INT8/INT4, AWQ/GPTQ/SmoothQuant) | FP16/BF16 **first-class**; FP8 (E4M3) **supported on Hopper+**; FP4/NVFP4 **supported on Blackwell+** `verify`; INT8/INT4 supported; AWQ/GPTQ/SmoothQuant via upstream compressor `verify` | FP16/BF16 **first-class**; FP8 **supported** `verify`; GPTQ/AWQ supported `verify` <!-- verify-source: current FP4 roadmap in SGLang --> | FP8 (E4M3/E5M2) **first-class**; FP4/NVFP4 **first-class on Blackwell**; INT8/INT4 supported; AWQ/GPTQ/SmoothQuant supported `verify` | N/A at engine layer | passes through to backend | GGUF: Q2–Q8 variants, K-quants, imatrix calibration **first-class** | 4-bit / 8-bit supported `verify` | `verify` <!-- verify-source --> |
| 5 | Parallelism (TP, PP, EP, DP attention, sequence) | TP **first-class**; PP **supported**; EP **supported for MoE** `verify`; DP attention **supported** `verify` | TP **first-class**; EP **first-class** `verify`; DP attention **first-class** (reference implementation for MoE) | TP **first-class**; PP **first-class**; EP **supported** `verify` | request-level routing only — parallelism is at the engine layer | passes through to backend | TP/PP generally N/A — single-node focus | `verify` <!-- verify-source --> | `verify` <!-- verify-source --> |
| 6 | Disaggregated prefill–decode | **supported in V1** (cross-worker KV transport via NIXL and plug-in backends) `verify` | **supported** (P/D-disaggregated serving mode) `verify` | **supported via NVIDIA Dynamo** | **first-class** — this is the engine's signature capability | N/A | N/A | N/A | N/A |
| 7 | Multi-adapter serving (static LoRA, dynamic LoRA swap, multi-LoRA) | first-class — dynamic LoRA loading via `--enable-lora` `verify` | supported — LoRA loading `verify` <!-- verify-source: current multi-LoRA status in SGLang --> | supported — LoRA in build/runtime path `verify` | N/A at engine layer (routing only) | `verify` | `verify` <!-- verify-source: llama.cpp LoRA/adapter path --> | `verify` <!-- verify-source --> | `verify` <!-- verify-source --> |
| 8 | Speculative decoding (draft-target, Medusa, EAGLE, Lookahead, n-gram) | supported — draft-target, n-gram, EAGLE `verify` <!-- verify-source: current Medusa/Lookahead status --> | supported — EAGLE, draft-target `verify` | supported — draft-target, Medusa, EAGLE `verify` | N/A at engine layer | passes through to backend | `verify` | `verify` <!-- verify-source --> | `verify` <!-- verify-source --> |
| 9 | API surface (OpenAI-compatible, Triton gRPC, custom) | OpenAI-compatible server (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`) **first-class** | OpenAI-compatible **first-class**; SGLang front-end language for structured programs | served via Triton (HTTP/gRPC) or Dynamo | multi-engine endpoint; routes to vLLM/SGLang/TRT-LLM backends | Triton HTTP/gRPC **first-class**; OpenAI-compatible via add-ons | OpenAI-compatible (`llama-server`) **first-class**; C API | OpenAI-compatible server `verify` | OpenAI-compatible historically `verify` <!-- verify-source --> |
| 10 | Hardware backends (NVIDIA, AMD, AWS Neuron, Google TPU, Intel Gaudi, CPU, Apple) | NVIDIA **first-class**; AMD (ROCm) **supported**; AWS Neuron **supported** (narrower); Intel (CPU/XPU/Gaudi) **supported** `verify`; Google TPU **supported** `verify` — feature coverage varies per backend | NVIDIA **first-class**; AMD and others **supported** with maturity gaps `verify` <!-- verify-source: SGLang per-backend feature parity --> | NVIDIA **only** (by design) | NVIDIA primary `verify` | NVIDIA primary; CPU/other for non-LLM models | CPU, NVIDIA, AMD, Apple Metal, Vulkan **first-class** (broadest coverage) | NVIDIA, AMD, Apple, Qualcomm, WebGPU, mobile `verify` | `verify` <!-- verify-source --> |

**Cells that differ from a naive reading of Chapter 02** (acceptance criterion 2 requires at least three):

1. **vLLM × Quantization (FP4/NVFP4)** — Chapter 02 says FP4/NVFP4 is supported on Blackwell+; we mark `verify` because the exact release is a moving target and must be confirmed against vLLM release notes before a fleet commits to it. Reason: the chapter lands at a point in time; the release that first ships hardware FP4 may trail the Blackwell GA date in engine-land.
2. **vLLM × Speculative decoding** — Chapter 02 names draft-target, n-gram, and EAGLE. Medusa and Lookahead status is flagged as `verify` here rather than filled from the chapter. Reason: speculative-decoding offerings in vLLM have historically rotated.
3. **TGI × every axis** — Chapter 02 does not describe TGI in detail and `CURRICULUM.md` references a 2026 archival signal that itself requires verification. The TGI column is `verify` across the board with a single `<!-- verify-source: current maintenance status -->` marker. Reason: the chapter cannot vouch for a project's current state.
4. **Triton Inference Server × KV-cache / attention / parallelism** — Chapter 02 is silent on the LLM-specific internals because Triton routes to backends. We mark these `verify` rather than inferring.
5. **llama.cpp × Prefix caching** — Chapter 02 does not assert a specific allocator status for prefix caching in llama.cpp; we mark `verify` rather than extending the chapter.

Each of these is a legitimate cell edit: *the chapter is a point-in-time orientation; release notes are the source of truth for this spreadsheet.*

### 2.2 Artefact B — Commercial compare-point summary

| Vendor | What's unique (one sentence) | Build-vs-buy threshold | Exit cost |
|---|---|---|---|
| **NVIDIA NIM** | Containerised, enterprise-supported serving built on TensorRT-LLM + Triton + Dynamo, delivered as pullable images. | Use NIM when the team wants TRT-LLM-grade performance **without** owning the compile pipeline and a TRT-LLM licence boundary is acceptable; self-host when custom engine integration (new quantization recipe, non-standard scheduler) is load-bearing. | Container image portability is high across NVIDIA environments; portability off NVIDIA hardware is low (TRT-LLM is NVIDIA-only). |
| **Fireworks** | Proprietary serving stack with distinctive FP8 / FP4 speed claims and curated model catalogue (<!-- verify-source: FireAttention / speculative-decoding variants — vendor docs rotate -->). | Use Fireworks when peak-token-latency on an OpenAI-shape model is a product requirement **and** control over the engine is not; self-host when model mix is custom or regulatory constraints require dedicated hardware. | Medium — OpenAI-shape API portability is good, but any adopted vendor-specific feature (speculative recipes, caching semantics) is not portable. |
| **Together** | OpenAI-compatible endpoints for a broad catalogue of open-weight models, consumption pricing. | Use Together when consumption-shape overflow or breadth-of-models is the ask; self-host when fleet utilisation is high and sustained enough to amortise GPUs. | Medium — API portability is good; model-availability dependency is the lock-in. |
| **Baseten** | Model-deployment platform with auto-scaled endpoints and custom model code (Truss). | Use Baseten when a bespoke model or inference strategy needs rapid production hosting without an in-house platform; self-host when workload is large enough to justify a platform team. | Medium — Truss model format migration is the lift. |
| **Modal** | Serverless general-compute platform widely used for custom inference endpoints. | Use Modal when the inference workload is low-volume, spiky, or experimental and container-native; self-host when steady-state fleet utilisation is substantial. | Low-to-medium — container portability is high; environment-layer dependencies are the main migration cost. |
| **Groq** | Custom LPU hardware plus managed service with a distinctive latency profile (very low TPOT at small batches). | Use Groq when latency-critical, latency-tail-sensitive workloads benefit from LPU hardware and the model is in Groq's supported catalogue; self-host when custom models are required or batch workloads dominate. | High — hardware specificity means a workload migration is a re-engineering exercise. |
| **Cerebras** | WSE wafer-scale hardware plus managed service with a distinctive throughput profile (very high throughput at long context). | Use Cerebras when throughput at long-context, large-prompt workloads is the pricing lever; self-host when workload is short-context or batch cost at commodity GPUs beats Cerebras per-token. | High — hardware specificity, like Groq. |

### 2.3 Artefact C — Written recommendation (≤ 300 words)

**Workload:** 70B dense chat model (open-weight); 4k average input, 500 output tokens; 50% prefix-cache hit on a shared system prompt; 10k RPS peak; OpenAI-compatible API required; NVIDIA H200 fleet.

**Pick:** **vLLM** on H200, TP=8 within one NVLink Switch domain, FP8 weights (E4M3), continuous batching + chunked prefill + prefix caching enabled, OpenAI-compatible server fronting the fleet, autoscaling against queue depth and prefill-tokens-per-second.

**Rationale (against the spreadsheet).** The load-bearing axes for this workload are (1) scheduling and chunked prefill — vLLM is first-class on both; (2) PagedAttention plus prefix caching — the 50% prefix-cache hit translates directly into TTFT headroom; (3) FP8 quantization on Hopper-class hardware — vLLM lists it as supported and the engine target is first-class; (4) OpenAI-compatible server — first-class in vLLM. Parallelism (TP=8) stays inside one NVLink Switch domain, so the intra-node interconnect carries the collective traffic and we avoid EP / PP topology risk this quarter.

**Runner-up:** **TensorRT-LLM served via Triton or Dynamo**. TRT-LLM wins on kernel-level speed (first-class FP8 and FP4, NVIDIA's own kernel suite) and on disaggregated-serving maturity (via Dynamo). The axis our pick loses on is **FP8 kernel maturity and peak tok/s under identical hardware** — TRT-LLM is tuned harder. Mitigation: we stage a controlled TRT-LLM A/B against vLLM for the exact traffic shape on one canary pool next quarter, and if the measured goodput delta clears a stated threshold (post-rollout cost per 1M output tokens under target), we migrate.

## 3. Validation steps

1. **Primary-source scan** — Open every cell in column order. Any cell *not* marked `verify` must have an identifiable citation to one of the URLs in `resources.md`. Any cell marked `verify` must carry a `<!-- verify-source: ... -->` marker so the merge gate catches it.
2. **Point-in-time rot check** — For each engine, compare the capability claims to the engine's latest release notes *as of the day of the review*. If a cell matches Chapter 02 but not the release notes, flip to the release-notes version and add a short note explaining why.
3. **Hardware-conditional claim audit** — Any FP8 / FP4 / INT8 cell must distinguish "supported on hardware H" from "supported on all backends." Cells that collapse the distinction are wrong.
4. **Commercial threshold phrasing** — Each vendor row in Artefact B names a *threshold* for build-vs-buy ("use X when Y, self-host when Z"), not a description.
5. **Recommendation shape** — Artefact C names (a) the GPU SKU, (b) the engine, (c) the parallelism topology, (d) the numeric format, (e) the runner-up, and (f) the single axis on which the pick loses. Any missing element is a rewrite trigger.

## 4. Rubric / review checklist

| # | Criterion | Pass signal |
|---|---|---|
| 1 | Every non-trivial cell has a citation to a current primary source (not Chapter 02) or is marked `verify` with a `verify-source` marker | Spreadsheet scan |
| 2 | At least three cells differ from a naive reading of Chapter 02, with a note on *why* | Documented in-line |
| 3 | The commercial section names a build-vs-buy threshold for each vendor, not just a description | Row-by-row check |
| 4 | The written recommendation names the runner-up engine and the single axis the pick loses on | Explicit in Artefact C |
| 5 | A peer teammate who hasn't read Chapter 02 can pick a reasonable engine for a stated workload in under five minutes from the spreadsheet alone | Peer dry-run |
| 6 | Every `verify` cell carries a machine-detectable `<!-- verify-source: ... -->` marker for the merge gate | Marker scan |

## 5. Common mistakes

- **Copying Chapter 02 claims into cells as if they were current.** The chapter is a point-in-time orientation. Every cell must trace to an engine's release notes or docs page, or be marked `verify`.
- **Collapsing "supported on hardware H" into "supported."** Hardware FP8 Tensor Cores land on Ada (L4, L40S) and Hopper (H100, H200) and newer; Ampere (A100, A10) and older do not have FP8 Tensor Cores and fall back to emulation or a different precision. An A100 fleet does not get hardware FP8 even if the engine lists FP8; an L4 or L40S fleet does.
- **Missing the TGI maintenance-status gap.** Many learners carry TGI capability claims from memory. Mark the column `verify` end-to-end until an up-to-date announcement is cited.
- **Vendor row that describes the vendor instead of giving a threshold.** "Groq is a fast inference provider" is not a build-vs-buy threshold.
- **Picking the engine from vibes in Artefact C.** The pick must reference specific axes from the spreadsheet. If the rationale does not cite numbered axes, rewrite.
- **Skipping the mitigation on the runner-up axis.** The exercise asks for one axis on which the pick is weaker *and* how you plan to mitigate it.
- **Forgetting Dynamo is an orchestrator, not an engine.** Several axes are `N/A at engine layer` for Dynamo; writing kernel-level claims for Dynamo is a signal that the learner misread Chapter 02.

## 6. References

- Chapter 02 — Engine Capability Map: `../../../../ai-infra-inference-engineer-learning/lessons/mod-101-inference-engineering-position/02-engine-capability-map.md`
- Module resources index: `../../../../ai-infra-inference-engineer-learning/lessons/mod-101-inference-engineering-position/resources.md`
- vLLM: project — <https://github.com/vllm-project/vllm>; docs — <https://docs.vllm.ai/>; release notes — <https://github.com/vllm-project/vllm/releases>
- SGLang: project — <https://github.com/sgl-project/sglang>; docs — <https://docs.sglang.ai/>
- TensorRT-LLM: project — <https://github.com/NVIDIA/TensorRT-LLM>; docs — <https://nvidia.github.io/TensorRT-LLM/>
- NVIDIA Dynamo: project — <https://github.com/ai-dynamo/dynamo>; docs — <https://docs.nvidia.com/dynamo/>
- Triton Inference Server: project — <https://github.com/triton-inference-server/server>; docs — <https://docs.nvidia.com/deeplearning/triton-inference-server/>
- llama.cpp: project — <https://github.com/ggml-org/llama.cpp>; GGUF spec — <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>
- MLC-LLM: project — <https://github.com/mlc-ai/mlc-llm>; docs — <https://llm.mlc.ai/docs/>
- TGI (historical): project — <https://github.com/huggingface/text-generation-inference>; docs — <https://huggingface.co/docs/text-generation-inference/>
- Commercial compare-points: NVIDIA NIM — <https://docs.nvidia.com/nim/>, <https://build.nvidia.com/>; Fireworks AI — <https://docs.fireworks.ai/>; Together AI — <https://docs.together.ai/>; Baseten — <https://docs.baseten.co/>; Modal — <https://modal.com/docs>; Groq — <https://console.groq.com/docs>; Cerebras — <https://inference-docs.cerebras.ai/>
- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> (relevant to API-surface and quantization-related supply-chain risks called out in rows 4 and 9).
