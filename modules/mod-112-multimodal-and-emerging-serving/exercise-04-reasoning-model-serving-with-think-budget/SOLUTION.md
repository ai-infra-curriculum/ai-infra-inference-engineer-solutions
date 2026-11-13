# Solution — Exercise 04: Reasoning-Model Serving with Think Budget

> Paired learning file: [`lessons/mod-112-multimodal-and-emerging-serving/exercises/exercise-04-reasoning-model-serving-with-think-budget.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-112-multimodal-and-emerging-serving/exercises/exercise-04-reasoning-model-serving-with-think-budget.md)
> Module: `mod-112-multimodal-and-emerging-serving`
> Type: implementation / policy design

## 1. Solution overview

The learner serves an **open-weights reasoning model** (DeepSeek-R1 distilled,
QwQ, Qwen3-reasoning, or an equivalent) and makes the engine honour an explicit
**think budget**: a per-request cap on the number of tokens the model is
allowed to spend inside its reasoning / thinking span before it must produce a
user-visible answer. The deliverables show the engine configuration, the
wire-level API contract for setting the budget, how the budget is enforced
server-side, and a short measurement showing the latency vs. quality tradeoff.

Three artefacts:

| Artefact | Form | Pass bar |
|---|---|---|
| A — Reasoning-model serving + API shape | Launch script + API doc | A pinned reasoning-capable model is served; the API accepts a `max_thinking_tokens` (or equivalent) parameter and returns the thinking span separately from the answer span |
| B — Server-side budget enforcement | Server code + test | When `max_thinking_tokens` is reached, the engine deterministically closes the thinking span and forces the answer span — it does not simply truncate mid-thought |
| C — Budget sweep | Table + policy note | Reports latency and answer quality at ≥3 budget tiers (e.g. 0, 1k, 4k tokens); policy note names the default and when to allow the client to override |

Reasoning-model APIs are still evolving in the open-weights community. OpenAI's
Responses API, Anthropic's extended-thinking, and the open-weights
implementations (DeepSeek-R1's `<think>...</think>` tags, Qwen3's
`enable_thinking` flag) all solve the same problem differently. Pin the
model, the serving engine, and the parameter names.

### Why "think budget" is a serving concern, not an application concern

Reasoning models are trained to emit a block of internal reasoning tokens
before their final answer. For DeepSeek-R1-distill and QwQ this is wrapped in
literal `<think>...</think>` tags; for Anthropic's models it is exposed as a
distinct `thinking` content block; for OpenAI reasoning models the thinking is
an opaque counted span (`reasoning_tokens`) not returned to the user by
default. In all cases, the thinking span can be **long** — easily several
thousand tokens on hard problems, occasionally tens of thousands.

That has three serving consequences:

1. **Latency.** Thinking tokens cost full decode time. A 10,000-token thinking
   span is a multi-second (to tens-of-seconds) wait before the user sees any
   answer-span token.
2. **Cost.** Metered deployments pay for every reasoning token. A reasoning
   model can quietly cost 10× a non-reasoning model for the same user-visible
   answer.
3. **KV footprint.** Thinking tokens occupy KV cache. A fleet that serves
   mixed reasoning + non-reasoning traffic must size for reasoning requests'
   worst case, not the average.

A **think budget** is the serving-side knob that bounds all three. It is **not
equivalent to `max_tokens`** — `max_tokens` is a cap on total output tokens
including the answer, and simply truncating at `max_tokens` can cut off
mid-thought and leave the client with an unanswered thinking span. The budget
must gracefully terminate the thinking span and force the answer span.

This is the practitioner analogue of OpenAI's Responses API `reasoning.effort`
levels and Anthropic's `thinking.budget_tokens` parameter — both are
serving-layer caps, not sampling-layer caps.

## 2. Implementation — worked walkthrough

The walkthrough uses `DeepSeek-R1-Distill-Qwen-14B` as a concrete open-weights
example: it produces `<think>...</think>` spans and is small enough for a
single-node deployment. Any model with a textual thinking-span convention
follows the same pattern through its tokenizer's special tokens.

### 2.1 Launch the engine

```bash
# config_reasoning.sh
export HF_HOME=/scratch/hf
export VLLM_LOGGING_LEVEL=INFO

vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 \
  --max-num-seqs 16 \
  --disable-log-requests \
  --host 0.0.0.0 --port 8040
```

The context is larger than a typical non-reasoning deployment because the
thinking span alone can consume several thousand tokens. Size `max_model_len`
against `(prompt + max_thinking_tokens + answer_tokens)`.

### 2.2 API shape

The engine exposes the standard OpenAI-compatible chat endpoint. The exercise
layer adds a thin proxy that accepts `max_thinking_tokens` and enforces it
server-side. The wire shape:

```http
POST /v1/chat/completions
{
  "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
  "messages": [{"role": "user", "content": "Prove that sqrt(2) is irrational."}],
  "max_tokens": 2048,
  "max_thinking_tokens": 1024,
  "temperature": 0.6
}
```

Response:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "thinking": "<redacted or echoed thinking span>",
      "content": "<final answer>"
    },
    "finish_reason": "stop",
    "think_finish_reason": "budget"
  }],
  "usage": {
    "prompt_tokens": 32,
    "thinking_tokens": 1024,
    "completion_tokens": 196
  }
}
```

Three contract points are worth calling out:

- `thinking` and `content` are separate fields. Clients should never need to
  parse `<think>` tags.
- `usage.thinking_tokens` is reported so the client (and billing) can see how
  much of the budget was spent.
- `think_finish_reason` names *why* thinking stopped (`natural`, `budget`,
  `cutoff`). This is critical for debugging: if quality drops at a given
  budget, the answer is in the finish reason distribution.

### 2.3 Server-side budget enforcement

Enforcement is the hard part. Three approaches, in increasing graciousness:

**Approach 1 — hard stop with injected answer-opening.** When the thinking-token
counter reaches `max_thinking_tokens`, the engine stops decoding, appends the
model-specific close-thinking marker (`</think>` for DeepSeek-R1-style), and
continues decoding into the answer span. This is the *minimum* correct
behaviour. Quality is noticeably worse than a naturally-closed thinking span.

**Approach 2 — soft nudge near budget.** Starting at `max_thinking_tokens −
nudge_window` (e.g. −128 tokens), inject a soft steering suffix at the next
natural boundary: `... So, to summarize: </think>`. The model gets a chance to
gracefully wrap up. Quality is noticeably better than approach 1.

**Approach 3 — tiered budget.** Expose `reasoning_effort ∈ {low, medium, high}`
with discrete budgets (e.g. `{256, 2048, 8192}`). Internally this maps to
approach 2 per tier but gives the API a stable, reviewable surface. This is
the pattern vendor APIs converge on for a reason: individual numeric budgets
leak implementation detail.

A sketch of approach 2 as a vLLM-side `stop`-string plus a client-side
re-dispatch (the practical open-weights pattern until engines expose a native
`max_thinking_tokens`):

```python
# proxy.py — think-budget enforcement proxy over an OpenAI-compatible engine
import httpx
from fastapi import FastAPI
from pydantic import BaseModel

UPSTREAM = "http://localhost:8040/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

CLOSE_THINK = "</think>"
THINK_OPEN = "<think>"

app = FastAPI()

class ChatReq(BaseModel):
    messages: list
    max_tokens: int = 2048
    max_thinking_tokens: int = 1024
    temperature: float = 0.6

@app.post("/v1/chat/completions")
def chat(req: ChatReq):
    # Phase 1: decode up to max_thinking_tokens with stop=["</think>"].
    phase1 = httpx.post(UPSTREAM, json={
        "model": MODEL,
        "messages": req.messages,
        "max_tokens": req.max_thinking_tokens,
        "temperature": req.temperature,
        "stop": [CLOSE_THINK],
    }, timeout=180).json()
    phase1_text = phase1["choices"][0]["message"]["content"]
    phase1_finish = phase1["choices"][0]["finish_reason"]

    # Normalize: the model may or may not have emitted <think> itself.
    if not phase1_text.startswith(THINK_OPEN):
        thinking_body = phase1_text
    else:
        thinking_body = phase1_text[len(THINK_OPEN):]

    think_finish = "natural" if phase1_finish == "stop" else "budget"

    # Phase 2: force the answer span. Feed back the conversation plus the
    # already-decoded thinking, close it, and decode the answer span.
    forced = (
        THINK_OPEN + thinking_body + CLOSE_THINK +
        ("\n\n" if think_finish == "budget" else "")
    )
    phase2_messages = req.messages + [
        {"role": "assistant", "content": forced}
    ]
    phase2 = httpx.post(UPSTREAM, json={
        "model": MODEL,
        "messages": phase2_messages,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
    }, timeout=180).json()
    answer = phase2["choices"][0]["message"]["content"]

    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "thinking": thinking_body,
                "content": answer,
            },
            "finish_reason": phase2["choices"][0]["finish_reason"],
            "think_finish_reason": think_finish,
        }],
        "usage": {
            "prompt_tokens": phase1["usage"]["prompt_tokens"],
            "thinking_tokens": phase1["usage"]["completion_tokens"],
            "completion_tokens": phase2["usage"]["completion_tokens"],
        },
    }
```

This is a transparent proxy pattern. Engines that expose `max_thinking_tokens`
natively (as newer releases are starting to) collapse the two phases into one
request, but the wire shape and the finish-reason contract remain the same.

### 2.4 Budget sweep

Hold the prompt set and model fixed; vary only `max_thinking_tokens`. Example
prompts span easy / medium / hard:

- Easy: "What's the capital of Portugal?" (thinking is near-zero value)
- Medium: "If `f(x) = x^3 − 3x`, where are its local extrema?"
- Hard: GSM8K-style multi-step word problem; a short proof; a logic puzzle.

Cells to measure:

| Budget | Easy | Medium | Hard |
|---|---|---|---|
| 0 (thinking disabled) | | | |
| 256 | | | |
| 1024 | | | |
| 4096 | | | |
| unbounded (natural stop) | | | |

Per cell: p50 total latency, mean `think_finish_reason` distribution, mean
answer correctness (judge model or human panel), mean `usage.thinking_tokens`.

Expected qualitative pattern:

- Easy prompts: quality is flat across budgets. Any budget > 0 is wasted on
  these; a `reasoning_effort=low` default is the right policy.
- Medium prompts: quality rises then plateaus. There is a budget below which
  quality drops sharply — the thinking span is cut off before the model has
  completed a chain-of-reasoning step.
- Hard prompts: quality continues to rise with budget well past the medium-
  prompt plateau. Hard problems benefit from more deliberation, up to a limit
  specific to the model and the problem class.
- Unbounded: latency and cost worst-case become very large. P99 thinking-span
  length for reasoning models is heavy-tailed; a fleet-wide SLO without a
  budget cap is very hard to hold.

### 2.5 Policy note

A one-page policy note is part of the deliverable. Example structure:

> **Default.** `max_thinking_tokens = 1024` (equivalent to
> `reasoning_effort=medium`). Chosen because it (a) captures ≥ 90% of the
> quality of unbounded reasoning on the medium-prompt set, (b) caps p95 total
> latency at `<budget>` on a single `<accelerator>`, (c) bounds the worst-case
> KV footprint to `<N>` tokens per request.
>
> **Client-override policy.** Clients may request higher budgets up to
> `max_thinking_tokens = 8192`, rate-limited at a lower RPS and billed at a
> higher rate. Lower budgets (down to 0, thinking disabled) are always
> allowed.
>
> **Thinking visibility policy.** Thinking is returned to the client under
> `message.thinking` by default; an admin flag can redact it for models whose
> thinking span may leak training-data verbatims. Thinking is always logged
> internally for debugging.
>
> **Finish-reason SLO.** At the chosen default, `think_finish_reason=budget`
> should fire on < 20% of medium prompts and < 60% of hard prompts. A rising
> `budget` rate signals the default is misaligned with real traffic.

## 3. Validation steps

1. **Model serves.** A pinned reasoning model with a textual thinking-span
   convention starts cleanly and answers one easy prompt end-to-end.
2. **`max_thinking_tokens` is accepted and enforced.** A request with a
   deliberately low budget (e.g. 128) returns within the budget's wall-clock
   bound and the thinking span's length does not exceed the budget.
3. **Thinking and answer are separated.** The response has distinct
   `thinking` and `content` fields; clients do not have to parse
   `<think>` tags.
4. **`think_finish_reason` is reported correctly.** Low-budget requests
   report `budget`; high-budget requests on easy prompts report `natural`.
5. **No mid-thought truncation reaches the user.** Even at `budget` finishes,
   the user sees a coherent answer span, not a cut-off thought.
6. **Budget sweep complete.** At least 3 budget tiers × 3 prompt-difficulty
   tiers reported with latency, quality judgement, and finish-reason
   distribution.
7. **Policy note.** Names default, client-override policy, thinking-visibility
   policy, and the finish-reason SLO.

## 4. Rubric / review checklist

Score each item 0 (missing), 1 (partial), 2 (fully correct). Passing: ≥ 10 / 14.

| # | Item | Points |
|---|---|---|
| 1 | Reasoning model served with pinned engine and model version | 2 |
| 2 | API exposes `max_thinking_tokens` separately from `max_tokens` | 2 |
| 3 | Thinking span and answer span returned as distinct fields | 2 |
| 4 | Budget enforced gracefully (close-marker injected, answer forced; no mid-thought cutoff seen by client) | 2 |
| 5 | `think_finish_reason` reported with at least `natural` and `budget` values | 2 |
| 6 | Budget sweep covers ≥3 budgets × ≥3 difficulty tiers; reports latency + quality + finish-reason distribution | 2 |
| 7 | Policy note names default, client-override policy, thinking-visibility policy, and a finish-reason SLO | 2 |

## 5. Common mistakes

- **Treating `max_tokens` as the budget.** `max_tokens` caps total output and
  will cut off mid-thought. The budget must specifically cap thinking and
  trigger answer-span synthesis afterwards.
- **Streaming the thinking span as if it were the answer.** Clients that
  expect SSE-streamed `content` tokens will mis-render `<think>` tags as part
  of the answer. Either strip thinking from the stream or deliver it on a
  separate SSE `event` channel.
- **Enforcing by regex on `<think>...</think>`.** Model drift between releases
  shifts the convention (DeepSeek-R1 uses `<think>`, Qwen3 uses a different
  pair, OpenAI's thinking is opaque). The engine should enforce via the
  tokenizer's special tokens, not substring search.
- **Not reporting `think_finish_reason`.** Without it, the policy owner cannot
  see whether the default budget is cutting real thinking off or just catching
  natural stops. It is the primary signal for tuning the default.
- **Confounding budget with sampling.** Two independent runs at the same
  budget and `temperature > 0` can produce different thinking lengths. For
  A/B testing budget policies, pin the seed or average over many prompts.
- **Billing `thinking_tokens` as `prompt_tokens`.** Thinking tokens are
  **completion** tokens generated by the model, not prompt tokens. Mixing
  them up produces incorrect usage reporting and incorrect billing.
- **Returning the thinking span to the user in production when the model's
  training data is unknown.** Reasoning models have been shown to emit
  near-verbatim training content inside thinking spans in some cases. The
  policy note must cover whether to surface, redact, or always log thinking.
- **Designing per-prompt budgets by hand.** Clients will not reliably pick the
  right budget per prompt. Expose tiers (`low/medium/high`), not raw integers,
  as the primary surface.
- **Assuming reasoning models always help.** On easy prompts, enabling
  thinking burns tokens for zero quality gain. A fleet-level routing policy
  that disables thinking for prompts matching an "easy" classifier can cut
  cost materially without hurting quality.

## 6. References

- DeepSeek-AI. *DeepSeek-R1 model card.*
  https://huggingface.co/deepseek-ai/DeepSeek-R1 — convention of
  `<think>...</think>` reasoning span in open-weights reasoning models.
- Qwen team. *QwQ-32B model card.*
  https://huggingface.co/Qwen/QwQ-32B — alternative open-weights reasoning
  model with its own thinking-span convention.
- OpenAI. *Reasoning models and the Responses API.*
  https://platform.openai.com/docs/guides/reasoning — vendor reference for
  `reasoning.effort` tiers and `reasoning_tokens` usage reporting.
- Anthropic. *Extended thinking.*
  https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking —
  vendor reference for a `thinking.budget_tokens` parameter and the
  `thinking` content block pattern.
- vLLM documentation. *Reasoning outputs and chat templates.*
  https://docs.vllm.ai/ — practitioner implementation reference; pin
  parameter names to the deployed release.
- Hugging Face Transformers. *Chat templates.*
  https://huggingface.co/docs/transformers/chat_templating — how
  thinking-span tokens enter the prompt via the model's chat template.
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/)
  — thinking-span exposure is a disclosure-risk surface; see also
  [[server-vs-edge-decision-drill]] for data-handling implications.
- [mitre-atlas](https://atlas.mitre.org/) — thinking-span inspection can
  reveal prompt-contents to an attacker who can observe the completion
  stream; not directly invoked for this serving walkthrough.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework) —
  governance baseline for metered-reasoning deployments and user-visible
  thinking policy.
