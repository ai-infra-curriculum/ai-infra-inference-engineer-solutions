# Solution — Exercise 01: VLM Serving with Visual-Token Cache

> Paired learning file: [`lessons/mod-112-multimodal-and-emerging-serving/exercises/exercise-01-vlm-serving-with-visual-token-cache.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-112-multimodal-and-emerging-serving/exercises/exercise-01-vlm-serving-with-visual-token-cache.md)
> Module: `mod-112-multimodal-and-emerging-serving`
> Type: implementation / serving walkthrough

## 1. Solution overview

The learner stands up a **vision-language model (VLM)** serving deployment, measures
the baseline cost of re-encoding the same image across calls, then enables a
**visual-token cache** so that the vision encoder is amortized across requests
that share an image. The deliverables are a working server, a pair of micro-
benchmarks (cold image vs. repeat image), and a short note on where the cache
helps and where it does not.

Three artefacts:

| Artefact | Form | Pass bar |
|---|---|---|
| A — VLM serving launch | Shell script / Makefile | A single pinned model (e.g. Qwen2-VL-7B-Instruct or LLaVA-OneVision-7B) is served with the vision tower and LLM in the same engine; image input works through the OpenAI-compatible endpoint |
| B — Visual-token cache enabled + verified | Config diff + log excerpt | The engine's cache for pre-computed image embeddings is on, and a second call with the same image hits the cache (shown by a log counter or a latency drop on repeat calls) |
| C — Latency table | Table + method note | Reports time-to-first-token (TTFT) and total latency for (i) cold image, (ii) repeat of the same image, (iii) new image after cache eviction. Method note states image size, prompt length, output length, and warm-up |

Keep the deliverable tight. This is a deployment exercise about the vision
encoder's placement in the serving graph, not a VLM accuracy study.

### Why visual tokens are the interesting state

A VLM has two pipelines joined at the embedding layer. The image passes through a
**vision encoder** (typically a ViT) which produces a sequence of patch
embeddings; a lightweight projector maps those into the LLM's embedding space.
The projected sequence is then concatenated with the text token embeddings and
run through the LLM. From the LLM's perspective, an image is simply a span of
tokens at a specific position in the input.

That span is not small. For a 7-B class open-weights VLM served at native
resolution, one image typically contributes several hundred to a few thousand
"visual tokens" (exact count depends on the model's patching strategy, image
resolution, and whether the model uses dynamic tiling). Two consequences:

1. The vision encoder's forward pass is a non-trivial fraction of per-request
   cost — often it dominates TTFT for short text prompts.
2. Visual tokens occupy KV-cache space just like text tokens do. The KV footprint
   of a VLM request is `len(text) + len(visual)`, not just `len(text)`.

A **visual-token cache** addresses the first point: if two calls use the same
image, the vision encoder's output can be computed once and reused. Serving
frameworks that support this expose it under names like `mm_processor_cache` or
`image_feature_cache`; the exact flag depends on the engine and its pinned
release. Verify flag names against the engine's `--help` output for the pinned
version.

The second point — KV-cache contribution — interacts with [[kv-cache-management]]
and is the reason sizing a VLM engine is not a direct port from a text-only
engine. Visual tokens are long, lossless, and locally-dense.

## 2. Implementation — worked walkthrough

The commands below assume vLLM as the serving engine, Qwen2-VL-7B-Instruct as the
VLM, and a single-node 1× or 2× 80 GiB-class accelerator. Other VLMs supported by
vLLM's multimodal stack (LLaVA family, Pixtral, InternVL, MiniCPM-V) follow the
same pattern but with their own `--max-model-len` and tile-count tradeoffs. The
flag names shown must be verified against the pinned vLLM release.

### 2.1 Launch — VLM engine

```bash
# config_vlm.sh
export HF_HOME=/scratch/hf
export VLLM_LOGGING_LEVEL=INFO

vllm serve Qwen/Qwen2-VL-7B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --limit-mm-per-prompt '{"image": 2}' \
  --mm-processor-kwargs '{"min_pixels": 262144, "max_pixels": 1048576}' \
  --disable-log-requests \
  --host 0.0.0.0 --port 8010
```

Notes:

- `--limit-mm-per-prompt` is the engine-level cap on how many multimodal items
  one request can carry. Keep it small (1–2) for a first launch; raising it
  increases the worst-case KV footprint.
- `--mm-processor-kwargs` for Qwen2-VL controls the dynamic image resolution
  (min/max pixel count after resize). Pinning this bounds the per-image visual-
  token count and makes latency measurements comparable across runs.
- `--max-model-len` must comfortably exceed `(visual_tokens_per_image ×
  images_per_request) + text_prompt_tokens + max_output_tokens` for the worst-
  case request in your workload. For Qwen2-VL at 1 MP resolution this is roughly
  1.5k–2k visual tokens per image; verify by checking the engine log on the
  first request.

Expected engine-log evidence that multimodal is wired in:

```
INFO ... Model type detected: Qwen2VLForConditionalGeneration (vision + LLM)
INFO ... Vision tower initialized; patch size=14
INFO ... Multimodal processor cache: ENABLED, max entries=256
INFO ... Engine ready at http://0.0.0.0:8010
```

If the "Multimodal processor cache" line is missing or shows `ENABLED=false`,
cache measurements in §2.3 will show no repeat-call speedup.

### 2.2 Call the engine with an image

Use the OpenAI-compatible chat endpoint; images are passed as URLs or base64 in
the `image_url` content part.

```python
# call_vlm.py
import base64, httpx, sys, time, pathlib

def encode(p):
    return "data:image/jpeg;base64," + base64.b64encode(pathlib.Path(p).read_bytes()).decode()

img = encode(sys.argv[1])
body = {
    "model": "Qwen/Qwen2-VL-7B-Instruct",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": img}},
            {"type": "text",      "text": "What is in this image? Answer in one sentence."},
        ],
    }],
    "max_tokens": 64,
    "temperature": 0.0,
}

for i in range(3):
    t0 = time.perf_counter()
    r = httpx.post("http://localhost:8010/v1/chat/completions", json=body, timeout=120)
    r.raise_for_status()
    dt = (time.perf_counter() - t0) * 1000
    print(f"call {i}: {dt:.0f} ms  text={r.json()['choices'][0]['message']['content'][:60]!r}")
```

Run twice: once against a cold image (fresh base64 payload each call, so the
preprocessor hash differs) and once against a repeated image (the identical
base64 string reused, so the preprocessor hash matches).

### 2.3 Measure the cache effect

The multimodal processor cache is keyed on a hash of the pre-processed image
content. Two calls with the *same* post-preprocess image bytes should re-use the
vision encoder's output. Two calls with different images or different
preprocessor settings should each pay the encoder cost.

Run the following in sequence against the running engine:

```bash
# Cold: encoder cost on every call (each call receives a visually different image)
for p in img_a.jpg img_b.jpg img_c.jpg; do python call_vlm.py "$p"; done

# Repeat: encoder cost on first call, cache hit on next two
for i in 1 2 3; do python call_vlm.py img_a.jpg; done
```

Scrape the engine's Prometheus `/metrics` endpoint after each run. The relevant
counters are the multimodal / processor cache series exposed by the engine;
names vary by release (`vllm:mm_cache_hits_total`,
`vllm:mm_cache_misses_total`, or equivalent). A rising hit counter on the
"repeat" run is the evidence the cache is doing work.

### 2.4 Worked measurement table (illustrative)

| Scenario | TTFT (ms) | Total (ms) | MM-cache hit? |
|---|---|---|---|
| Cold image (first ever call) | *measured* | *measured* | miss |
| Repeat same image, call 2 | *measured* | *measured* | hit |
| Repeat same image, call 3 | *measured* | *measured* | hit |
| New image after cache eviction (fill cache beyond its size) | *measured* | *measured* | miss |

Expected qualitative pattern:

- TTFT on a cache hit drops materially; the vision encoder's forward pass is
  skipped and only the LLM prefill over cached visual-token embeddings runs.
  Total latency drops proportionally for short outputs and less so for long
  outputs (the decode phase is unchanged).
- Throughput on a cache hit is bounded by the LLM's prefill+decode cost alone.
- An image that is pre-processed differently (different resize, different crop)
  will miss even if "the same" semantically — the cache is keyed on post-
  processing bytes, not on human-perceived identity.

### 2.5 Sizing note

The engine must hold:

1. Model weights (ViT + projector + LLM).
2. KV cache for in-flight requests, sized against
   `(visual_tokens_per_image × images_per_request) + text_prompt + max_output`.
3. The multimodal-processor cache's stored embeddings (one entry per unique
   pre-processed image, up to its max-entries cap).

Dropping `--gpu-memory-utilization` by 5–10 points vs. a text-only deployment on
the same hardware is a safe starting point; verify against actual peak-memory
readings after a representative workload.

## 3. Validation steps

A reviewer validating a learner submission should perform, in order:

1. **Engine starts clean with the model's vision tower loaded.** The engine log
   confirms the vision-encoder initialization, not just LLM load.
2. **One-image call succeeds through the OpenAI-compatible endpoint.** The
   response is coherent and references the actual image content, not a generic
   placeholder answer.
3. **Multimodal cache is enabled.** Either a startup log line or a `/metrics`
   series confirms the cache is on.
4. **Repeat-call measurement shows cache hits.** The repeat run in §2.3
   increments the hit counter and TTFT drops on the second and third calls.
5. **Different-image measurement shows cache misses.** Two visually distinct
   images do not share a cache entry.
6. **KV-cache sizing is honest.** Submission states the visual-token count per
   image and computes `max_model_len` budget against it, rather than reusing a
   text-only cap.
7. **Method note.** States image resolution / preprocessor settings, prompt
   length, output length, concurrency, and warm-up policy.

## 4. Rubric / review checklist

Score each item 0 (missing), 1 (partial), 2 (fully correct). Passing: ≥ 10 / 14.

| # | Item | Points |
|---|---|---|
| 1 | VLM engine starts with vision tower and LLM joined; model + engine version pinned | 2 |
| 2 | OpenAI-compatible chat endpoint accepts `image_url` content parts and returns a coherent answer | 2 |
| 3 | Visual-token / multimodal-processor cache is explicitly enabled; evidence cited from log or metrics | 2 |
| 4 | Latency table reports cold vs. repeat-image TTFT and total latency with a clear method note | 2 |
| 5 | Submission computes KV budget from visual-token-per-image count, not from text-only assumptions | 2 |
| 6 | Cache-eviction case is measured (fill cache > capacity, then retry a past image) | 2 |
| 7 | Reviewer can reproduce from the submitted launch script and call script without edits | 2 |

## 5. Common mistakes

- **Running a VLM but calling it text-only.** Learners sometimes submit a working
  VLM engine with only text prompts, which does not exercise any of the visual-
  token path. The pass bar is a call that carries an image part.
- **Measuring with base64-identical but preprocessor-different images.** The
  cache is keyed post-preprocess. Changing `min_pixels`/`max_pixels` or sending
  the same source image at a different HTTP-level resolution misses the cache
  even though it "looks" the same. Pin preprocessor settings for the
  measurement.
- **Setting `--limit-mm-per-prompt` too high on the first launch.** Allowing 4–8
  images per request multiplies the worst-case KV footprint and can OOM the
  engine on the first long call. Start at 1–2, measure, then raise.
- **Carrying a text-only `--max-model-len` forward.** A 4k text-only context
  that was comfortable on a 7-B LLM is not comfortable on a 7-B VLM once each
  image contributes 1–2k visual tokens. Recompute against the per-image
  visual-token count.
- **Confusing the visual-token cache with KV-cache prefix reuse.** They are
  different layers of caching: the visual-token / multimodal-processor cache
  avoids re-running the vision encoder; the LLM's prefix / automatic-prefix-cache
  avoids re-running the LLM's prefill over an identical token prefix. Both can
  be on at once; both are keyed differently.
- **Reporting speedup from a single-call measurement.** TTFT varies run-to-run.
  Warm the engine, then average or median at least five repeats per cell of the
  table.
- **Assuming the vision encoder dominates for all workloads.** For long outputs
  (hundreds of decode steps), the LLM decode phase swamps vision-encoder cost
  even when the cache misses. The cache helps most for short-output,
  image-heavy workloads (classification, OCR-style, visual-QA with one-sentence
  answers).
- **Blindly enabling tensor-parallel splitting of the vision tower.** For 7-B
  class VLMs on one 80 GiB accelerator, TP=1 is almost always correct. Splitting
  the ViT adds collective-communication overhead that dwarfs its compute
  savings at this scale.

## 6. References

- vLLM documentation. *Multimodal models and multimodal inputs.*
  https://docs.vllm.ai/ — practitioner implementation reference; pin flag
  names (`--limit-mm-per-prompt`, `--mm-processor-kwargs`, multimodal cache
  flags) to the deployed release.
- Qwen team. *Qwen2-VL model card.*
  https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct — resolution and dynamic-
  tiling behaviour that drive visual-token counts.
- LLaVA project. *LLaVA-OneVision.*
  https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov — reference
  practitioner implementation of a VLM at the same scale; useful comparison
  point for patching strategy.
- Hugging Face Transformers. *Vision-language models guide.*
  https://huggingface.co/docs/transformers/model_doc/auto — the AutoProcessor
  pattern the serving engines wrap.
- SGLang project. *Multimodal serving.*
  https://docs.sglang.ai/ — alternative practitioner implementation, useful
  cross-reference for terminology on visual-token caching.
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/)
  — image inputs expand the model's input surface. Cover prompt-injection via
  image (text rendered in pixels) when designing abuse paths; see
  [[server-vs-edge-decision-drill]] for data-residency when images contain PII.
- [mitre-atlas](https://atlas.mitre.org/) — visual prompt-injection and
  adversarial image patches sit under the input-manipulation tactic; not
  invoked for this latency walkthrough but relevant when the same deployment
  carries production images.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework) —
  governance baseline for data handling when image inputs may carry PII.
