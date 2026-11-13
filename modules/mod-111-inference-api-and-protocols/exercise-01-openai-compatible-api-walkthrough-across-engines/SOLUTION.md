# Solution — Exercise 01: OpenAI-Compatible API Walkthrough Across Engines

> Paired learning file: [`lessons/mod-111-inference-api-and-protocols/exercises/exercise-01-openai-compatible-api-walkthrough-across-engines.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-111-inference-api-and-protocols/exercises/exercise-01-openai-compatible-api-walkthrough-across-engines.md)
> Module: `mod-111-inference-api-and-protocols`
> Type: implementation / cross-engine conformance walkthrough

## 1. Solution overview

The learner stands up the same open-weights model behind **three different
inference engines** — vLLM, SGLang, and TGI (or Triton with the TRT-LLM backend)
— and demonstrates that an identical OpenAI-formatted client request produces
semantically equivalent responses across all three. The point is **protocol
conformance**, not benchmarking: identify which fields each engine honours,
which it silently drops, and where behaviour diverges from the OpenAI reference
in ways that will bite the gateway or the application layer.

The three artefacts:

| Artefact | Form | Pass bar |
|---|---|---|
| A — Three engine launches | Scripts or Compose file | Same model, same tokenizer, same context length, exposed on three ports with OpenAI-compatible routes |
| B — Conformance matrix | Markdown table | For each OpenAI field (model, messages, max_tokens, temperature, top_p, n, stop, stream, tools, response_format, logprobs, seed, user), status is **honoured / silently ignored / rejected** per engine |
| C — Semantic-equivalence script | Python script + output | Same prompt + greedy sampling + seed produces responses that match to a documented fp tolerance, or divergence is explained |

Pin every engine to an explicit version (image tag or `pip` constraint). The
OpenAI route surface and field support shifts materially between minor versions
of vLLM, SGLang, and TGI; a matrix compiled against unpinned images rots in
weeks. Verify every field against the live `/v1/chat/completions` endpoint,
not only the engine's docs.

### Why an OpenAI-compatible surface is the default

OpenAI's REST shape — `POST /v1/chat/completions` with a `messages` array, a
mirrored streaming response, and tool-call blocks — is now the de-facto wire
protocol for LLM serving. Every production open-source engine (vLLM, SGLang,
TGI, Triton TRT-LLM, llama.cpp server, LMDeploy) exposes it. Client SDKs
(`openai-python`, `openai-node`, LangChain's `ChatOpenAI`, LiteLLM, tool-use
frameworks) target it. If a new engine wants adoption, the first thing it
ships is OpenAI compatibility.

The learner's job is not to prove the surface works — vendors already claim it
does — but to find out **where the surface lies**. Fields that appear accepted
but get silently ignored are the single most common integration bug class in
this space. The matrix is the deliverable because the matrix is what the
gateway team actually needs.

### Four classes of divergence

Across engines, behaviour divergence falls into four buckets. Name them in the
matrix so the reader can filter.

| Class | What it means | Typical example |
|---|---|---|
| **Honoured** | Field is parsed and influences the response as the OpenAI reference says | `temperature`, `max_tokens` on all three engines |
| **Silently ignored** | Field is accepted (no 400), but the engine does not apply it | `user`, `logit_bias`, `n > 1` on some engines |
| **Rejected** | Engine returns an error if the field is set | `tools` on an engine without tool-call support; `response_format: json_schema` on older versions |
| **Honoured with divergent semantics** | Field is applied, but the output shape or validation rule differs from OpenAI | `stop` sequences counted at the token vs. the string level; `stream_options.include_usage` emitted at a different point in the stream |

The last class is the dangerous one. The call succeeds, the output looks
right, and the application only notices during an incident at the 99th
percentile prompt.

## 2. Implementation — worked walkthrough

The commands below assume:

- A chat-sized dense target model, e.g. `meta-llama/Meta-Llama-3.1-8B-Instruct`.
- One node of 1–2× 80 GiB-class accelerators (TP = 1 is fine for an 8 B model).
- Three engines pinned to explicit versions.
- A gateway or `curl` running on the same host.

Substitute the model and hardware for whatever the learner's lab provides.

### 2.1 Three engine launches

Each engine binds to a different port. The model, tokenizer, max length, and
default sampling are held constant so conformance is the only moving part.

```bash
# vLLM
docker run --rm --gpus all --ipc=host \
  -p 8001:8000 \
  -e HF_TOKEN \
  vllm/vllm-openai:v0.6.3 \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice --tool-call-parser hermes

# SGLang
docker run --rm --gpus all --ipc=host \
  -p 8002:30000 \
  -e HF_TOKEN \
  lmsysorg/sglang:v0.3.6 \
    python3 -m sglang.launch_server \
      --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
      --context-length 8192 \
      --mem-fraction-static 0.85 \
      --tool-call-parser llama3

# TGI
docker run --rm --gpus all --shm-size 1g \
  -p 8003:80 \
  -e HF_TOKEN \
  ghcr.io/huggingface/text-generation-inference:2.4.0 \
    --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --max-input-length 7168 \
    --max-total-tokens 8192
```

Verify each engine is serving the OpenAI route, not only its native one:

```bash
curl -s http://localhost:8001/v1/models | jq
curl -s http://localhost:8002/v1/models | jq
curl -s http://localhost:8003/v1/models | jq
```

All three must return a `data` array containing an object with an `id` field
matching the loaded model. TGI historically exposed the OpenAI shape under
`/v1/chat/completions` while keeping its native `/generate` route live; both
still work in 2.x releases.

### 2.2 Conformance probe

Send identical requests to each engine, varying one OpenAI field at a time,
and record the response or error. A minimal driver:

```python
# probe.py
import httpx, json, sys

ENGINES = {
    "vllm":   "http://localhost:8001/v1/chat/completions",
    "sglang": "http://localhost:8002/v1/chat/completions",
    "tgi":    "http://localhost:8003/v1/chat/completions",
}
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

PROBES = [
    ("baseline",       {"temperature": 0.0, "max_tokens": 32}),
    ("stop_string",    {"temperature": 0.0, "max_tokens": 64, "stop": ["\n\n"]}),
    ("logit_bias",     {"temperature": 0.0, "max_tokens": 32, "logit_bias": {"0": -100}}),
    ("n_gt_1",         {"temperature": 0.7, "max_tokens": 32, "n": 3, "seed": 17}),
    ("user_tag",       {"temperature": 0.0, "max_tokens": 32, "user": "probe-user"}),
    ("response_json",  {"temperature": 0.0, "max_tokens": 128,
                        "response_format": {"type": "json_object"}}),
    ("logprobs",       {"temperature": 0.0, "max_tokens": 16,
                        "logprobs": True, "top_logprobs": 3}),
    ("stream_usage",   {"temperature": 0.0, "max_tokens": 16, "stream": True,
                        "stream_options": {"include_usage": True}}),
]

def send(url, extra):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with the word OK and stop."}],
        **extra,
    }
    r = httpx.post(url, json=body, timeout=60)
    return r.status_code, r.text[:400]

for name, extra in PROBES:
    print(f"\n=== {name} ===")
    for eng, url in ENGINES.items():
        try:
            code, body = send(url, extra)
            print(f"{eng:7s} HTTP {code}  {body[:200]}")
        except Exception as e:
            print(f"{eng:7s} EXCEPTION  {e}")
```

Run once, read the responses, and record the result per cell of the matrix.
Important: a `200 OK` with the requested field missing from the response is
**silently ignored**, not **honoured** — inspect the response body for
confirmation that the field took effect (`n_gt_1` must return three choices;
`logprobs` must include a `logprobs` block per choice; `stream_usage` must
emit a final `data:` chunk containing a `usage` object).

### 2.3 Conformance matrix (illustrative skeleton — fill with measured values)

The learner's output. Each cell is **H** (honoured), **I** (silently ignored),
**R** (rejected with error), or **D** (honoured with divergent semantics —
footnote the divergence).

| OpenAI field | vLLM | SGLang | TGI | Notes |
|---|---|---|---|---|
| `model` | H | H | H | Rejected (400) if ID does not match the loaded model on vLLM and SGLang; TGI historically ignores the field and serves the single loaded model |
| `messages` | H | H | H | Chat template is applied inside the engine; verify same template is used across engines (§2.4) |
| `max_tokens` | H | H | H | — |
| `max_completion_tokens` (newer OpenAI name) | H in recent vLLM | *measure* | *measure* | Verify against pinned release |
| `temperature`, `top_p`, `top_k` | H | H | H | `top_k` is a non-OpenAI extension but all three accept it |
| `n` | H | *measure* | I on older TGI | TGI historically ignored `n > 1` on the OpenAI route |
| `stop` | H | H | D | TGI's stop is applied at the token level; strings that span tokens may not fire |
| `seed` | H | H | H | — |
| `logit_bias` | *measure* | I | I | Patchy support; test, do not assume |
| `logprobs`, `top_logprobs` | H | H | *measure* | Verify shape matches OpenAI (object with `content` array of per-token entries) |
| `response_format: json_object` | H | H | H | Enforcement strength varies — covered in exercise 03 |
| `response_format: json_schema` | H in recent vLLM | H in recent SGLang | *measure* | New field; pin a recent version |
| `tools`, `tool_choice` | H with `--enable-auto-tool-choice` | H with `--tool-call-parser` | *measure* | Covered in exercise 04 |
| `stream` | H | H | H | Covered in exercise 02 |
| `stream_options.include_usage` | H | H | *measure* | TGI emits `usage` only on the final non-chunked frame in some versions |
| `user` | I | I | I | Field is accepted but engines do not forward it anywhere; the gateway must log it |

Call out every **D** and every **I** explicitly. Those are the rows that change
the gateway's behaviour, not the **H** rows that work everywhere.

### 2.4 Semantic equivalence — do identical requests produce identical answers?

Conformance of the field surface is not conformance of the model output. Two
engines can both honour `temperature: 0.0` yet produce different greedy
decodes if their chat-template, tokenizer edge-case handling, or kernel
implementation differs. The learner verifies **greedy equivalence** on a small
battery of prompts.

```python
# equivalence.py
import httpx

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
URLS = {
    "vllm":   "http://localhost:8001/v1/chat/completions",
    "sglang": "http://localhost:8002/v1/chat/completions",
    "tgi":    "http://localhost:8003/v1/chat/completions",
}
PROMPTS = [
    "What is 17 * 23? Answer with just the number.",
    "List five prime numbers between 50 and 100, separated by commas.",
    "Write a one-sentence definition of inference latency.",
]

for prompt in PROMPTS:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 64,
        "seed": 17,
    }
    out = {}
    for eng, url in URLS.items():
        r = httpx.post(url, json=body, timeout=60)
        out[eng] = r.json()["choices"][0]["message"]["content"].strip()
    agree = len(set(out.values())) == 1
    print(f"\nPROMPT: {prompt}\n  agree={agree}")
    for eng, text in out.items():
        print(f"  {eng:7s}: {text}")
```

When greedy outputs diverge — and they often do across engines — the most
common causes are:

- **Chat-template drift.** Each engine applies the tokenizer's chat template
  to the `messages` array. If one engine's template version differs (e.g., TGI
  bundles the template inside the image and SGLang reads it from the HF repo),
  the prompt the model actually sees differs, and the outputs diverge.
- **BOS/EOS handling.** Some engines prepend `<|begin_of_text|>`; others
  assume the template already includes it. One extra BOS token at the start
  shifts the entire generation.
- **Sampling-numerics tie-breaking.** When two logits are numerically
  equal and `argmax` picks one or the other, the first divergence cascades.
- **Kernel differences.** Flash-attention vs. xformers vs. a vendor custom
  kernel can produce different last-digit fp values. Divergence that first
  appears after ~N tokens is usually this.

Document, do not paper over, divergence. "Tokens 0–37 match, divergence
starts at token 38 on a numerically tied logit" is a strong answer; "the
outputs look similar" is not.

## 3. Validation steps

A reviewer validating a learner submission should perform the following in
order.

1. **Three engines pinned.** The launch commands reference explicit image
   tags (e.g., `vllm/vllm-openai:v0.6.3`), the same open-weights model, and
   the same context length.
2. **All three expose `/v1/models` and `/v1/chat/completions`.** Simple `curl`
   confirms the OpenAI route is live on each engine.
3. **Conformance matrix is filled in from measurement, not from vendor
   documentation.** Every cell is evidenced by a status code and response
   snippet, not an assertion.
4. **The four classes (H/I/R/D) are used correctly.** In particular, `200 OK`
   with a missing effect in the response body is **I** (silently ignored),
   not **H**.
5. **Semantic-equivalence run.** §2.4 is executed; divergence is either zero
   or explained with a named root cause from the §2.4 list.
6. **Gateway implications are called out.** For every row that is **I** or
   **D**, the matrix names what the gateway or the application layer must do
   to compensate (reject the field, translate it, log it for audit).
7. **Teardown.** Clean shutdown of each engine; no zombie containers before
   the next probe runs.

## 4. Rubric / review checklist

Score each item 0 (missing), 1 (partial), 2 (fully correct). Passing:
≥ 10 / 14.

| # | Item | Points |
|---|---|---|
| 1 | Three engines launched against the same model with pinned versions | 2 |
| 2 | OpenAI route verified live on each engine via `/v1/models` and a round-trip call | 2 |
| 3 | Conformance matrix covers at least 12 OpenAI fields across all three engines | 2 |
| 4 | Every cell is labelled H / I / R / D, with evidence quoted for I, R, D rows | 2 |
| 5 | Semantic-equivalence run executed; greedy divergence is quantified | 2 |
| 6 | Gateway / client-layer implications named for each divergence | 2 |
| 7 | Common-mistake list from §5 reviewed; matrix does not repeat those mistakes | 2 |

## 5. Common mistakes

- **Trusting vendor docs without probing.** Every engine's README claims
  "OpenAI-compatible". The probe tells you where that is a lie of omission.
- **Treating `200 OK` as honour.** Silent ignores return `200 OK`. Inspect
  the response body for the field's effect, not the status code.
- **Comparing engines on different model revisions.** HF repos get re-uploaded
  with tokenizer changes; one engine cached the old version and another
  downloaded the new. Pin the model revision (`--revision <sha>` or an
  immutable mirror).
- **Different chat templates behind the same model name.** The chat template
  lives in `tokenizer_config.json` in the HF repo; some engines bundle a stale
  copy. If greedy outputs diverge on a trivial prompt, dump the
  post-templating string from each engine and compare.
- **Testing with temperature > 0 and no seed.** You cannot distinguish an
  honoured field from sampling noise. Use greedy or a pinned seed.
- **Omitting the streaming surface from the matrix.** Streaming is where
  divergence hides — chunk shape, keep-alive behaviour, where `usage` is
  emitted, how tool-call deltas are chunked. Exercise 02 is the deep dive;
  exercise 01's matrix must at minimum say whether each engine streams and
  whether `stream_options.include_usage` is honoured.
- **Forgetting the gateway/client side.** The matrix is only half the
  deliverable. A row that says "SGLang silently ignores `logit_bias`" is not
  actionable until the matrix also says "gateway must strip the field and
  surface a warning to the caller".
- **Picking an engine from the matrix.** This exercise does not rank engines.
  Ranking is a separate conversation with latency, throughput, and operator
  cost in play. The matrix informs that conversation; it does not conclude it.

## 6. References

- OpenAI. *Chat completions API reference.*
  https://platform.openai.com/docs/api-reference/chat — authoritative field
  list. Treat the version that ships in the reference client SDK as the
  conformance target.
- vLLM. *OpenAI-compatible server.*
  https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html —
  supported fields, flags, tool-call parsers.
- SGLang. *Backend runtime.*
  https://sgl-project.github.io/backend/runtime.html — OpenAI-compatible
  endpoint, structured output, tool-call parsers.
- Hugging Face. *Text Generation Inference — Messages API.*
  https://huggingface.co/docs/text-generation-inference/en/messages_api —
  TGI's OpenAI compatibility surface, divergence notes.
- LiteLLM. *Provider routing.* https://docs.litellm.ai/ — a working
  reference for how application-layer gateways handle cross-engine
  divergence in practice.
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/)
  — not directly invoked for this conformance exercise. Prompt-injection and
  output-handling concerns are covered when the API surface lands in
  production (exercise 05 and project 103).
- [mitre-atlas](https://atlas.mitre.org/) — not invoked for this walkthrough.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework) —
  governance baseline; the `user` field's handling (§2.3) is the audit hook
  the RMF expects. If the gateway does not log it, the audit trail is broken.
