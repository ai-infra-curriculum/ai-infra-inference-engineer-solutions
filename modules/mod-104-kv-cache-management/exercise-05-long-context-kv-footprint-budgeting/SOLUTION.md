# Exercise 05 — Long-Context KV Footprint Budget

> Module: `mod-104-kv-cache-management`
> Exercise: `exercise-05-long-context-kv-footprint-budgeting`
> Type: design / quantitative estimation

## 1. Solution overview

The exercise asks the learner to **budget KV-cache memory for a long-context serving
workload** and to decide which levers (batch size, context length, GQA, FP8/INT8/INT4
KV quantization, eviction/compression) to pull to make a target workload fit a given
GPU. The deliverable is a sizing worksheet, a prioritized list of levers with
expected memory savings and quality trade-offs, and a go/no-go recommendation.

The core formula:

```
bytes_per_token_kv = 2 * n_layers * n_kv_heads * head_dim * bytes_per_element
total_kv_bytes     = bytes_per_token_kv * seq_len * batch_size
```

The leading `2` is for **both** K and V. For grouped-query attention, `n_kv_heads <
n_query_heads` — this is the single largest "free" lever in the budget because it is
already baked into the model, not a runtime choice.

## 2. Implementation — worked budget

### 2.1 Workload under study

A hypothetical workload the learner can reuse:

| Parameter | Value |
|---|---|
| Model | Llama-3-70B class (public spec — see References) |
| `n_layers` | 80 |
| `n_query_heads` | 64 |
| `n_kv_heads` | 8 (GQA ratio 8) |
| `head_dim` | 128 |
| Weight precision | FP16 (2 B) |
| Target max context | 128 K tokens |
| Target concurrency (batch) | 16 active sequences |
| GPU | 1× 80 GiB-class accelerator (public spec class) |
| Weights footprint | ~140 GiB FP16, ~70 GiB FP8, ~35 GiB INT4 (approximate from `n_params * bytes_per_param`; verify for the exact checkpoint) |

Immediately: FP16 weights (~140 GiB) do not fit on one 80 GiB GPU. The exercise is
posed deliberately with weights that only fit at FP8 or INT4 — this is realistic for
70B-class long-context serving today and forces the KV budget onto a quantized-weight
baseline.

### 2.2 KV bytes per token

```
bytes_per_token_fp16 = 2 * 80 * 8 * 128 * 2 = 327,680 B  ≈ 320 KiB
bytes_per_token_fp8  = 2 * 80 * 8 * 128 * 1 = 163,840 B  ≈ 160 KiB
bytes_per_token_int8 = 2 * 80 * 8 * 128 * 1 = 163,840 B  ≈ 160 KiB
bytes_per_token_int4 = 2 * 80 * 8 * 128 * 0.5 = 81,920 B ≈  80 KiB
```

Note the factor-8 reduction already granted by GQA (8:1): without GQA, those numbers
would be eightfold larger. GQA is the first lever that was *already* pulled at
training time; recognize it rather than "take credit" for it in the sizing table.

### 2.3 Per-sequence and batch footprints

At 128 K context, per-sequence KV:

```
KV_fp16(128K) = 128*1024 * 320 KiB = 128*1024 * 327_680  = 40 GiB
KV_fp8 (128K) = 128*1024 * 160 KiB                       = 20 GiB
KV_int4(128K) = 128*1024 *  80 KiB                       = 10 GiB
```

At batch 16, FP16 KV alone is 640 GiB — impossible on a single 80 GiB GPU before we
even add weights. Practical configurations collapse to one of:

| Config | Weights | KV dtype | Max concurrent 128K seqs on 80 GiB |
|---|---|---|---|
| A | FP16 (does not fit) | — | 0 |
| B | FP8 (~70 GiB) | FP16 KV | 0 (KV even at batch 1 doesn't fit next to weights) |
| C | FP8 | FP8 KV | floor((80 - 70 - overhead) / 20) → **0 at 128K**; ~1 at 48K |
| D | INT4 (~35 GiB) | FP8 KV | floor((80 - 35 - overhead) / 20) ≈ 2 at 128K |
| E | INT4 | INT4 KV | floor((80 - 35 - overhead) / 10) ≈ 4 at 128K |
| F | INT4 | FP8 KV + sliding-window / H2O / SnapKV | depends on retention; ≥ 4 |

"Overhead" covers activations, cuda graphs, communication buffers, and the KV-pool
block allocator slack. A realistic reserve is 4–8 GiB on an 80 GiB card; learners
should pick a number, justify it, and include it in the worksheet.

### 2.4 Sizing worksheet

Fill the worksheet below for each configuration under consideration. This is the
artifact that goes into the report.

```
Model: ____________________________________
n_layers=__  n_kv_heads=__  head_dim=__

Weights:           n_params=____  dtype=____   -> ____ GiB
Activation slack:  ________________________   -> ____ GiB
CUDA graphs etc.:  ________________________   -> ____ GiB
---------------------------------------------------------
Budget remaining for KV pool  ...............  ____ GiB

Target: context=____ tokens, batch=____ seqs

KV dtype:   ____    bytes_per_elem=____
Per-token KV   = 2 * n_layers * n_kv_heads * head_dim * bytes_per_elem
               = ________ bytes
Per-seq KV     = per-token KV * context  = ________ GiB
Batched KV     = per-seq KV * batch      = ________ GiB

Fits?  _____________    Headroom?  _______________
```

### 2.5 Lever menu, ranked by headline impact

| Lever | Mechanism | Headline KV saving | Quality / correctness risk |
|---|---|---|---|
| GQA / MQA | Fewer KV heads per attention layer | Up to ~8× or more, model-dependent | Baked into the checkpoint; not a runtime choice |
| FP8 KV cache | Quantize K and V tensors to FP8 (E4M3 or E5M2) | 2× over FP16 KV | Small perplexity delta if calibrated; verify on the task |
| INT8 / INT4 KV | Integer-quantized KV with per-tensor/per-head scales | 2–4× over FP16 KV | Larger quality delta than FP8; always benchmark |
| Shorter effective context (sliding window) | Attend to most recent `W` tokens plus optional attention sinks | Linear in `W / seq_len` | Loses information outside window; StreamingLLM shows this can be kept coherent with sink tokens |
| H2O (Heavy-Hitter Oracle) eviction | Keep only tokens with historically high attention weight plus recent window | Typically keep ~20% of tokens | Degrades on long-range tasks that depend on pruned tokens |
| SnapKV | Observation-window-based KV compression for prompts | Large reduction on prompt KV only | Does not help decode-phase KV growth |
| Batch reduction | Fewer concurrent sequences | Linear in batch | Lowers throughput; raises queue time |
| Context cap | Reject or truncate prompts above some `L_max` | Linear | Product-visible; needs product sign-off |
| Tiered KV (LMCache/Mooncake) | Push evicted blocks to CPU/disk | Expands effective capacity, not per-request footprint | Covered by Exercise 04; adds reload latency |

**Priority order for a pass at the exercise:**

1. Confirm GQA ratio from the model config. If GQA is absent (MHA model), the budget
   almost never works for 128 K at batch 16 on 80 GiB and the exercise should escalate
   to a different model or a smaller context target.
2. Quantize KV to FP8. This is the lowest-quality-risk large lever; most recent
   inference engines (vLLM, TensorRT-LLM, SGLang) support FP8 KV with calibrated
   scales.
3. Decide weight dtype in lockstep with KV dtype. INT4 weights + FP8 KV is a common
   production point for 70B-class models.
4. Apply context truncation or sliding-window eviction when the product tolerates it.
   Document it in the API contract.
5. Add a tiered KV cache (Exercise 04) if the workload has prefix reuse; this expands
   effective capacity without further quantization.
6. H2O/SnapKV only after the simpler levers, and only with task-specific quality
   benchmarks.

### 2.6 Worked recommendation

For the workload in §2.1, a defensible single-GPU recommendation is **Config D with a
64 K context cap**:

- Weights INT4 → ~35 GiB.
- Reserve ~5 GiB for activations + CUDA graphs + runtime overhead.
- KV pool ≈ 40 GiB.
- Per 64K sequence at FP8 KV: `64 * 1024 * 160 KiB = 10 GiB`.
- Concurrent sequences supported: `floor(40 / 10) = 4`.

If the business requires 16 concurrent 128K sequences the recommendation changes to
*not* colocate on one 80 GiB GPU — options are: shard across GPUs (Exercise/Module on
distributed topologies), move to a larger-memory accelerator, or restructure the
workload to use shared prefixes + tiered KV (Exercise 04) so the long common context
is amortized.

## 3. Validation steps

1. **Checkpoint math.** Pull the model's `config.json`. Confirm `num_hidden_layers`,
   `num_key_value_heads` (or `num_kv_heads`), `head_dim` (or `hidden_size /
   num_attention_heads`). Plug into the formula and compare against `nvidia-smi` on a
   single-sequence run with no KV reuse.
2. **Live measurement.** Launch vLLM with `--max-model-len L` and
   `--kv-cache-dtype fp8_e4m3` (names vary by engine version). From Prometheus
   metrics, read KV blocks total × bytes-per-block and confirm within a few percent
   of the worksheet number. The engine's own logs will emit "KV cache size: X GB" at
   startup.
3. **Quality sanity check.** For whichever KV dtype is chosen, run the engine's or
   the task's quality harness with the **same** sampling seed before and after the
   dtype change. The difference should be below the project's quality bar; if not,
   fall back to the next-higher-precision KV dtype.
4. **Reject infeasible configurations early.** A learner who gets to the end with a
   budget that only works if "overhead" is set to zero has set overhead wrong.

## 4. Rubric / review checklist

Score 0–2; passing ≥ 12 / 16.

| # | Item | Pts |
|---|---|---|
| 1 | Correct per-token KV formula with factor of 2 and correct dtype bytes | 2 |
| 2 | Weights, activations, CUDA graphs, and KV pool all line-itemed | 2 |
| 3 | Lever table covers GQA, KV quantization, context cap, eviction, tiered cache | 2 |
| 4 | Recommendation explicitly matches budget; infeasible configs flagged | 2 |
| 5 | Quality-risk column filled in with task-appropriate caveats (not just "small") | 2 |
| 6 | Live `nvidia-smi` / engine-log cross-check performed | 2 |
| 7 | Clear escalation path if single-GPU budget cannot meet the workload target | 2 |
| 8 | GQA recognized as a model-level (not runtime) choice | 2 |

## 5. Common mistakes

- **Dropping the `2` for K and V.** Immediately halves the estimate; every
  downstream number is wrong.
- **Forgetting activations and CUDA-graph reservations.** Learners allocate the full
  "free" HBM to KV and then watch the engine OOM under load. 4–8 GiB on a modern
  80 GiB accelerator is a reasonable reserve.
- **Using `num_attention_heads` instead of `num_key_value_heads` for KV.** On a GQA
  model this overestimates KV footprint by the GQA ratio.
- **Assuming FP8 KV is free.** FP8 KV is low-risk, but still a quality lever. Running
  the task's own quality benchmark before/after is non-optional.
- **Averaging KV over sequence length.** Learners sometimes model the "average"
  decoded length and underestimate footprint for tail sequences that go to the full
  `max_model_len`. Budget against the worst-case admitted length.
- **Treating tiered KV (Exercise 04) as a per-sequence shrink.** It is not — it
  expands the *shared* cache across requests, not the per-sequence working set.
- **Ignoring communication buffers in a multi-GPU setup.** The budget sketch here is
  single-GPU. Distributed topologies (Module 107) add NCCL buffers and
  parallelism-specific reserves.

## 6. References

- Ainslie, J. et al. *GQA: Training Generalized Multi-Query Transformer Models from
  Multi-Head Checkpoints.* arXiv:2305.13245 — the GQA paper.
- Shazeer, N. *Fast Transformer Decoding: One Write-Head is All You Need.*
  arXiv:1911.02150 — original MQA; GQA's ancestor.
- Xiao, G. et al. *Efficient Streaming Language Models with Attention Sinks.*
  ICLR 2024, arXiv:2309.17453 — StreamingLLM / attention sinks for sliding-window
  retention.
- Zhang, Z. et al. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of
  Large Language Models.* NeurIPS 2023, arXiv:2306.14048.
- Li, Y. et al. *SnapKV: LLM Knows What You are Looking for Before Generation.*
  arXiv:2404.14469.
- vLLM documentation — "KV cache dtype", "`--max-model-len`", and KV-cache memory
  logs — https://docs.vllm.ai/ (practitioner implementation).
- TensorRT-LLM documentation — INT8/FP8/INT4 KV cache feature pages
  (practitioner implementation).
- Llama 3 model card / config — for the `n_layers`, `n_kv_heads`, `head_dim` values
  used in the worked example (verify against the exact checkpoint you plan to
  deploy).
