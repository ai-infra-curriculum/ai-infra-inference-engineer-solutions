# Exercise 03 — TensorRT-LLM Engine Build and Triton Serve

> Paired learning exercise: `lessons/mod-102-inference-engine-internals/exercises/exercise-03-trt-llm-engine-build-and-triton-serve.md`.

## 1. Solution overview

Unlike vLLM and SGLang, TensorRT-LLM is a **build-then-serve** stack: you
compile a model into a per-GPU, per-config *engine* (a `.engine` file), and
then serve it under NVIDIA Triton Inference Server via the TensorRT-LLM
backend (which wraps the C++ `Executor` runtime that owns the inflight
batcher and the KV cache manager). The learner is asked to produce the two
artefacts — the build step and the Triton deployment — and to explain why
the engine is coupled to the exact GPU, precision, and max-shape envelope it
was built for.

This solution provides a minimal reference build command, a Triton model
repository layout, the `config.pbtxt` fields that must be set correctly, and
a short deployment narrative.

<!-- verify-source: TensorRT-LLM and the tensorrtllm_backend rename APIs and
reorganise the model-repository layout across releases; the learner must pin
the container image / repo commit they used. -->

## 2. Implementation — Worked build and deploy

### 2.1 Build the engine

TensorRT-LLM exposes build flows via its `tensorrt_llm` Python package and
example scripts under the project's `examples/` tree. The high-level shape
is:

1. **Load / convert weights** from a Hugging Face checkpoint into the
   TensorRT-LLM checkpoint format (`convert_checkpoint.py` in the model's
   example directory).
2. **Build the engine** with `trtllm-build`, specifying max input / output /
   batch shapes, precision (FP16 / BF16 / FP8 / INT4-AWQ, etc.), tensor-
   parallel size, plugins (paged KV cache, GPT attention plugin), and any
   speculative-decoding or LoRA options.

Reference command (adapt paths and shapes to the pinned version):

```bash
# 1. Convert HF checkpoint → TRT-LLM checkpoint
python examples/llama/convert_checkpoint.py \
    --model_dir /models/hf/llama-3.1-8b-instruct \
    --output_dir /models/trt_ckpt/llama-3.1-8b-instruct-bf16 \
    --dtype bfloat16 \
    --tp_size 1

# 2. Build the engine
trtllm-build \
    --checkpoint_dir /models/trt_ckpt/llama-3.1-8b-instruct-bf16 \
    --output_dir     /models/trt_engines/llama-3.1-8b-instruct-bf16-tp1 \
    --gemm_plugin   bfloat16 \
    --gpt_attention_plugin bfloat16 \
    --paged_kv_cache enable \
    --remove_input_padding enable \
    --max_batch_size 32 \
    --max_input_len 4096 \
    --max_seq_len   8192 \
    --max_num_tokens 16384
```

The learner must explain each of these flags:

| Flag | Why it matters |
|---|---|
| `--gpt_attention_plugin` | Fused multi-head attention kernel; required for inflight batching |
| `--paged_kv_cache enable` | Block-based KV so inflight batching can pack heterogeneous lengths |
| `--remove_input_padding enable` | Packs batched inputs without padding; required for inflight batching |
| `--max_batch_size` | Compile-time cap on concurrent requests; the runtime cannot exceed it |
| `--max_input_len` / `--max_seq_len` | Compile-time context caps; longer requests are rejected |
| `--max_num_tokens` | Bound on tokens packed into one iteration (prefill + decode mixed) |

A correct submission notes explicitly that **the engine is locked to the GPU
SM arch, the precision, and these max shapes at build time** — move it to a
different GPU family or need a longer context and you must rebuild.

### 2.2 Serve with Triton (TensorRT-LLM backend)

Minimal model repository layout (per the `tensorrtllm_backend` conventions):

```
model_repo/
├── preprocessing/
│   ├── 1/model.py
│   └── config.pbtxt          # tokenizer: text in → token ids out
├── tensorrt_llm/
│   ├── 1/                    # engine files + config.json land here
│   │   └── (symlink or copy of /models/trt_engines/llama-...-tp1/)
│   └── config.pbtxt          # the TRT-LLM backend instance
├── postprocessing/
│   ├── 1/model.py
│   └── config.pbtxt          # token ids → text
└── ensemble/
    └── config.pbtxt          # chains preprocessing → tensorrt_llm → postprocessing
```

The key fields the learner must get right in
`tensorrt_llm/config.pbtxt` (names stable across recent releases, exact
values from the backend's provided template):

| Parameter | Role |
|---|---|
| `backend` | Must be `tensorrtllm` |
| `gpt_model_path` | Path to the directory containing the built engine + `config.json` |
| `gpt_model_type` | `inflight_fused_batching` (not `V1`) to enable continuous batching |
| `kv_cache_free_gpu_mem_fraction` | Fraction of free GPU memory reserved for the KV pool |
| `enable_kv_cache_reuse` | Enables prefix/KV reuse across requests |
| `max_beam_width`, `max_tokens_in_paged_kv_cache` | Runtime caps that must fit the engine's compile-time caps |

<!-- verify-source: exact parameter names and defaults are in
tensorrtllm_backend/all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt
on the matching release branch. Learners must copy from the release they use. -->

Launch:

```bash
docker run --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002 \
    -v $(pwd)/model_repo:/models \
    nvcr.io/nvidia/tritonserver:<pinned-tag>-trtllm-python-py3 \
    tritonserver --model-repository=/models
```

### 2.3 Deployment narrative

- **Build phase** produces a `.engine` file that is specific to (a) the GPU
  compute capability (e.g. SM90 for H100, SM80 for A100), (b) the TensorRT
  / TensorRT-LLM version in the container, (c) the precision and plugin
  choices, and (d) the max-shape envelope. Moving the file across a GPU
  family or TRT version will silently fail to load or will fail an input
  that exceeds `max_seq_len`.
- **Serve phase** is handled by Triton and the TensorRT-LLM backend. The
  backend owns the inflight batcher (the TRT-LLM analogue of a scheduler),
  the paged KV cache manager, and KV reuse. The ensemble routes requests:
  tokenizer → engine → detokenizer. Streaming is available via
  `decoupled_mode: true` in `tensorrt_llm/config.pbtxt` plus gRPC
  streaming.
- **Capacity** is set by two knobs that must agree:
  `kv_cache_free_gpu_mem_fraction` (how much of the free GPU memory the KV
  pool consumes) and the engine's `max_num_tokens` (how many tokens an
  iteration can execute). The learner should walk through how to size
  these given the model, precision, and expected concurrency.

## 3. Validation steps

1. **Engine sanity.** Run the TRT-LLM `run.py` reference script (under the
   model's `examples/` directory) against the built engine with a short
   prompt; a successful generation confirms the engine loads.
2. **Triton health.** `curl -v http://localhost:8000/v2/health/ready` must
   return 200 after launch; `curl http://localhost:8000/v2/models/ensemble`
   must show the ensemble as ready.
3. **End-to-end request.** POST a generate request to the ensemble:
   ```bash
   curl -s http://localhost:8000/v2/models/ensemble/generate \
       -H 'Content-Type: application/json' \
       -d '{"text_input": "Hello", "max_tokens": 16}'
   ```
   The response must contain generated text. For streaming, use the
   `generate_stream` endpoint and `decoupled_mode: true`.
4. **Inflight batching check.** Fire two concurrent requests; the backend's
   metrics (Prometheus endpoint on port 8002) should show `>1` active
   requests during the second request's prefill.
5. **KV reuse check.** With `enable_kv_cache_reuse: true`, repeat the same
   system prompt across two requests; the KV-reuse metric should increment.

## 4. Rubric / review checklist

| # | Criterion | Weight | Pass bar |
|---|---|---|---|
| 1 | Build command is reproducible | 20% | Includes convert + build, with precision, TP, and the paged-KV / remove-padding / attention plugins |
| 2 | Max-shape envelope documented | 10% | Learner explains that batch/input/seq caps are compile-time |
| 3 | Model repository laid out correctly | 15% | preprocessing, tensorrt_llm, postprocessing, ensemble all present |
| 4 | `config.pbtxt` fields set | 15% | `backend`, `gpt_model_path`, `gpt_model_type=inflight_fused_batching`, KV mem fraction all set |
| 5 | GPU / version coupling called out | 10% | Learner states the engine is not portable across SM archs or TRT versions |
| 6 | Validation evidence | 20% | Health probe + end-to-end request + concurrency metric all attached |
| 7 | Version pinned | 10% | Both the TRT-LLM commit / wheel and the Triton container tag recorded |

Passing threshold: 70%.

## 5. Common mistakes

- **Expecting portability.** Rebuilding is mandatory when any of GPU family,
  TRT-LLM version, precision, or max shapes change.
- **Setting `gpt_model_type: V1`.** That disables inflight batching — the
  engine will serve one request at a time.
- **Over-committing KV memory.** Setting `kv_cache_free_gpu_mem_fraction`
  too high leaves no headroom for activations and causes OOMs under load.
- **Skipping `--remove_input_padding` or the GPT attention plugin.** Both
  are preconditions for the inflight batcher; builds without them silently
  degrade to padded, lower-throughput execution.
- **Serving the engine directly.** TensorRT-LLM provides a Python runtime
  for experimentation, but production serving in this exercise goes
  through Triton + the TRT-LLM backend; learners sometimes skip Triton and
  claim completion.
- **Ignoring tokenizer parity.** The `preprocessing` / `postprocessing`
  Python models must use the same tokenizer version/config as the one used
  at build time.

## 6. References

- TensorRT-LLM repository — https://github.com/NVIDIA/TensorRT-LLM (build
  scripts, examples, `trtllm-build`). <!-- verify-source: pin commit. -->
- TensorRT-LLM backend for Triton —
  https://github.com/triton-inference-server/tensorrtllm_backend
  (`config.pbtxt` templates, model repository layout).
- NVIDIA Triton Inference Server documentation —
  https://docs.nvidia.com/deeplearning/triton-inference-server/ (ensembles,
  decoupled mode, metrics).
- NVIDIA NGC catalog —
  https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver (pinned
  Triton container images with the TRT-LLM backend baked in).
- OWASP Machine Learning Security Top 10 —
  https://owasp.org/www-project-machine-learning-security-top-10/ (ML06:
  AI Supply Chain Attacks — pinning TRT-LLM and the Triton image is
  directly relevant).
- NIST AI Risk Management Framework —
  https://www.nist.gov/itl/ai-risk-management-framework ("Manage" function:
  version pinning and reproducible engine builds are AI RMF manage-phase
  controls).
