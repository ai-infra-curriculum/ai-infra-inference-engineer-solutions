# Solution — Exercise 04: Tool-Calling Protocol Plumbing

> Paired learning file: [`lessons/mod-111-inference-api-and-protocols/exercises/exercise-04-tool-calling-protocol-plumbing.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-111-inference-api-and-protocols/exercises/exercise-04-tool-calling-protocol-plumbing.md)
> Module: `mod-111-inference-api-and-protocols`
> Type: implementation / protocol plumbing walkthrough

## 1. Solution overview

The learner wires an **OpenAI-shaped tool-calling** loop end-to-end against
two open-source engines (e.g. vLLM and SGLang), demonstrates that the
chat-template's tool-call tokens are parsed correctly into the OpenAI
`tool_calls` field, exercises both the non-streaming and streaming shapes,
and handles the three protocol edge cases that the OpenAI reference papers
over: parallel tool calls, malformed arguments, and tool errors.

The three artefacts:

| Artefact | Form | Pass bar |
|---|---|---|
| A — Two-engine tool-calling loop | Python script + engine launch commands | A fixed prompt + two tools (`get_weather`, `search`) round-trips through both engines, with assistant-side tool calls decoded correctly and tool results fed back to produce a final answer |
| B — Streaming tool-call parser | Python async client | Tool-call deltas from the streamed response are reassembled into complete `tool_calls` objects; parser handles arguments that stream in across frames |
| C — Three edge-case tests | Scripts + outputs | Parallel tool calls (`tool_choice: "required"` producing >1 call), malformed-arguments recovery, and tool-execution-error feedback each produce documented outcomes |

Pin the engine versions and the **tool-call parser**. Each engine ships
multiple parsers (hermes, llama3, mistral, granite, phi-4). The parser
matches the model's chat template; a mismatch produces responses where the
tool-call appears as free text in `content` instead of as `tool_calls`.

### Why tool calling is a protocol problem, not a model problem

Open-weights models emit tool calls as **tokens in the output stream**.
Llama-3.1 emits `<|python_tag|>{"name": "get_weather", ...}<|eot_id|>`.
Hermes emits `<tool_call>{"name": ...}</tool_call>`. The model's job ends
there; the engine's job is to **parse those tokens out of the raw text** and
repackage them into the OpenAI `tool_calls` array. That translation layer —
the "tool-call parser" — is what the learner must understand and verify.

A well-wired loop has four stages:

1. **Caller sends tools.** `POST /v1/chat/completions` with `tools: [...]`
   and `tool_choice`. Engine inlines the tools into the chat template
   according to the model's conventions.
2. **Model emits a tool-call.** Decoding produces the tool-call tokens.
   Engine's parser finds them and returns them in the response as
   `message.tool_calls[*]`, with `message.content: null` and
   `finish_reason: "tool_calls"`.
3. **Caller executes the tool.** Caller runs the function, gets a result.
4. **Caller sends the result back.** A new request with the full message
   history, plus a `role: "tool"` message carrying the result and
   `tool_call_id` matching the one from step 2. Engine re-renders, model
   produces a natural-language reply.

Each stage has an engine-side failure mode. Stage 1: the tools get inlined
with the wrong template. Stage 2: the parser does not match the model and
the call shows up in `content`. Stage 3: no engine involvement. Stage 4:
the `role: "tool"` message is not rendered into the template correctly and
the model does not see the result.

### The parser-model matrix

Picking the right parser is the single most common source of silent
breakage. The table below is the practitioner-facing version.

| Model family | vLLM `--tool-call-parser` | SGLang `--tool-call-parser` |
|---|---|---|
| Llama-3.1 / 3.2 / 3.3 Instruct | `llama3_json` or `pythonic` | `llama3` |
| Mistral-7B/8x7B/8x22B Instruct | `mistral` | `mistral` |
| Hermes-2/3 fine-tunes, Qwen2.5 chat (recent) | `hermes` | `hermes` |
| Qwen2.5 (default chat template) | `hermes` (via chat template) | `qwen25` |
| Granite 3 | `granite-20b-fc` | `granite` |
| Phi-4 | `phi-4` | *check release* |

If the parser name does not appear in `vllm serve --help` or
`python -m sglang.launch_server --help`, your pinned release does not
support the model's tool-call format and the call will land in `content`
instead of `tool_calls`.

## 2. Implementation — worked walkthrough

The commands below assume:

- vLLM and SGLang pinned to recent releases.
- A Llama-3.1-8B-Instruct target on both engines.
- A tiny tool registry (`get_weather`, `search`) implemented locally; no
  outbound calls needed to exercise the loop.

### 2.1 Engine launches with tool parsers

```bash
# vLLM
docker run --rm --gpus all -p 8001:8000 \
  vllm/vllm-openai:v0.6.3 \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --max-model-len 8192

# SGLang
docker run --rm --gpus all -p 8002:30000 \
  lmsysorg/sglang:v0.3.6 \
    python3 -m sglang.launch_server \
      --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
      --tool-call-parser llama3 \
      --context-length 8192
```

`--enable-auto-tool-choice` on vLLM is load-bearing: without it, `tool_choice`
is honoured only when the caller sets it explicitly to a named function.

### 2.2 Non-streaming tool-call round-trip

```python
# roundtrip.py
import httpx, json, uuid

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "required": ["city"],
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["c", "f"], "default": "c"}
                },
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web for a query, return top N results.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10}
                },
                "additionalProperties": False
            }
        }
    }
]

def execute(name: str, args: dict) -> str:
    if name == "get_weather":
        return json.dumps({"city": args["city"], "temp_c": 14, "condition": "cloudy"})
    if name == "search":
        return json.dumps({"results": [{"title": "Example result", "snippet": "..."}]})
    return json.dumps({"error": f"unknown tool: {name}"})

def run(url: str, user_prompt: str):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when they help."},
        {"role": "user",   "content": user_prompt},
    ]
    body = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0.0,
        "max_tokens": 512,
    }

    r = httpx.post(url, json=body, timeout=120).json()
    msg = r["choices"][0]["message"]
    finish = r["choices"][0]["finish_reason"]
    assert finish == "tool_calls", f"expected tool_calls, got {finish}: {msg}"

    messages.append(msg)  # assistant turn with tool_calls

    for tc in msg["tool_calls"]:
        fn   = tc["function"]["name"]
        args = json.loads(tc["function"]["arguments"])  # may fail — see §2.4
        result = execute(fn, args)
        messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": result,
        })

    body["messages"] = messages
    r2 = httpx.post(url, json=body, timeout=120).json()
    final = r2["choices"][0]["message"]["content"]
    return final

if __name__ == "__main__":
    for eng, url in [("vllm", "http://localhost:8001/v1/chat/completions"),
                      ("sglang", "http://localhost:8002/v1/chat/completions")]:
        print(f"\n=== {eng} ===")
        print(run(url, "What's the weather in Paris in Fahrenheit?"))
```

What a healthy response looks like on the first round:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"city\": \"Paris\", \"unit\": \"f\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

Two invariants worth asserting in your test:

- `finish_reason == "tool_calls"` when there is a tool call; **not**
  `"stop"`. If `finish_reason` is `"stop"` but `message.content` contains
  JSON that looks like a tool call, the parser is not matching.
- `message.content` is `null` (or `""` on some engines) when there are
  tool calls. Text in `content` plus tool calls in `tool_calls` means the
  model "prefixed" the call with prose — rare on Llama-3.1 with
  `llama3_json`, more common on `pythonic`.

### 2.3 Streaming tool-call parser

Streaming tool calls is where most of the parser bugs live. The engine emits
tool-call fields in `choices[0].delta.tool_calls[*]` — with `function.arguments`
streaming in chunks as a partial string. The caller must accumulate by
tool-call index.

```python
# stream_tools.py
import httpx, json

async def stream(url: str, body: dict):
    tool_calls = {}  # index -> accumulator
    content = ""
    async with httpx.AsyncClient(timeout=120) as c:
        async with c.stream("POST", url, json=body) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                ev = json.loads(payload)
                delta = ev["choices"][0]["delta"]
                if delta.get("content"):
                    content += delta["content"]
                for tc in delta.get("tool_calls", []) or []:
                    idx = tc.get("index", 0)
                    acc = tool_calls.setdefault(idx, {
                        "id": None,
                        "function": {"name": None, "arguments": ""},
                    })
                    if tc.get("id"):
                        acc["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        acc["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        acc["function"]["arguments"] += fn["arguments"]
    return content, [tool_calls[i] for i in sorted(tool_calls)]
```

Three things the parser must get right, in order of how often they are
broken:

- **Accumulate by `index`, not by `id`.** The `id` field often only arrives
  on the first delta of a given call; subsequent deltas identify the call
  by its `index`. A naïve parser that keys by `id` creates one bucket per
  frame.
- **`arguments` streams as text, not as JSON.** You cannot parse a partial
  `arguments` string mid-stream. Accumulate the raw string and parse once
  the stream ends. (Some callers do speculative partial parsing for UI
  purposes; keep that separate from the authoritative parse.)
- **Expect `content` deltas even when a tool call happens.** Some templates
  interleave a short prose preface before the tool call. The parser must
  not drop them, and the UI must not render them as "the answer".

### 2.4 Edge case 1 — parallel tool calls

`tool_choice: "required"` and prompts that genuinely need two tools (e.g.
"What's the weather in Paris and in Tokyo?") can produce a response with
**more than one entry** in `message.tool_calls`. The loop must execute each,
append one `role: "tool"` message per `tool_call_id`, and send the whole
thing back.

Mistake pattern: executing only `tool_calls[0]` and feeding back one result.
The follow-up call now has a dangling `tool_call_id` in the history for
which no response exists. Engines that strictly render the template will
reject this with a 400; engines that silently drop it will produce a
confused reply. Always execute all tool calls, in order, before the
follow-up request.

### 2.5 Edge case 2 — malformed `arguments`

`arguments` is a free-form string the model emits; nothing in the loop
guarantees it is valid JSON that matches the tool's parameter schema. The
model can invent keys, omit required keys, type-mismatch, or truncate at
`max_tokens`.

```python
try:
    args = json.loads(tc["function"]["arguments"])
    jsonschema.validate(args, TOOLS_BY_NAME[tc["function"]["name"]]
                              ["function"]["parameters"])
except (json.JSONDecodeError, jsonschema.ValidationError) as e:
    result = json.dumps({"error": f"invalid arguments: {e}"})
else:
    result = execute(tc["function"]["name"], args)

messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
```

Feeding the error back as the tool's result — instead of throwing an
exception out of the loop — gives the model a chance to retry with
corrected arguments. In practice Llama-3.1 and Qwen2.5 recover on the next
turn; weaker models loop. Cap the number of retry turns (2–3 is standard).

### 2.6 Edge case 3 — tool execution errors

When the actual tool errors (API timeout, invalid credentials, backend
500), the loop must pass a structured error back, not raise. The tool
message is where that happens:

```python
try:
    result = my_real_tool(args)
except ToolError as e:
    result = json.dumps({"error": {"code": e.code, "message": str(e)}})

messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
```

The schema the model sees for the tool result is up to you — there is no
protocol-level standard. A common convention is
`{"ok": true, "data": ...}` on success and
`{"ok": false, "error": {"code": ..., "message": ...}}` on failure. Keep
it stable across tools or the model will not learn to handle errors well.

## 3. Validation steps

A reviewer validating a learner submission should perform the following in
order.

1. **Engines launched with the correct parser.** Launch commands name the
   `--tool-call-parser` explicitly; parser matches the model family per
   §1's matrix.
2. **Non-streaming round-trip works on both engines.** `roundtrip.py` returns
   a natural-language final answer on both, using the tool result.
3. **Response invariants hold.** `finish_reason == "tool_calls"`;
   `message.content` is null/empty when there are tool calls;
   `tool_calls[*].id` is a non-empty string; `function.arguments` is a
   string (not an object).
4. **Streaming parser round-trips.** `stream_tools.py` reconstructs the
   same `tool_calls` the non-streaming response produced, byte-for-byte.
5. **Parallel tool calls handled.** A prompt that triggers two tool calls
   produces two entries in `tool_calls`; the loop appends two `role: "tool"`
   messages before the follow-up; the follow-up succeeds.
6. **Malformed arguments handled.** Injecting a prompt that provokes
   missing-required-key arguments produces an `invalid arguments` error
   message back to the model, and the model recovers on the next turn.
7. **Tool-execution errors handled.** A simulated tool error produces a
   structured error payload in the `role: "tool"` message; the final
   assistant reply acknowledges it.

## 4. Rubric / review checklist

Score each item 0 (missing), 1 (partial), 2 (fully correct). Passing:
≥ 10 / 14.

| # | Item | Points |
|---|---|---|
| 1 | Both engines launched with an explicit `--tool-call-parser` matching the model | 2 |
| 2 | Non-streaming round-trip produces a final natural-language reply that uses the tool result | 2 |
| 3 | Streaming parser accumulates `arguments` across deltas, keyed by `index`, and matches the non-streaming output | 2 |
| 4 | Parallel-tool-call case appends one `role: "tool"` message per `tool_call_id` and the follow-up succeeds | 2 |
| 5 | Malformed-arguments case feeds a structured error back; the model recovers | 2 |
| 6 | Tool-execution-error case produces an `ok: false` payload that the final assistant reply acknowledges | 2 |
| 7 | Common-mistake list from §5 reviewed; submission avoids each | 2 |

## 5. Common mistakes

- **Wrong parser for the model.** The single most common failure. The tool
  call shows up in `message.content` as free text; `finish_reason` is
  `"stop"`. Fix: match parser to model per §1 matrix, verified against
  `--help` of the pinned release.
- **Not setting `--enable-auto-tool-choice` on vLLM.** Without it,
  `tool_choice: "auto"` is silently coerced to "no tools"; the model never
  emits a call.
- **Keying streaming-tool-call accumulators by `id` instead of `index`.** `id`
  arrives only on the first delta; subsequent deltas identify by `index`.
  Keying by `id` creates one bucket per frame and the final reconstruction
  is empty.
- **Trying to `json.loads` `arguments` mid-stream.** Partial JSON is never
  parseable. Accumulate the raw string; parse once at stream end.
- **Executing only the first tool call in a parallel-call response.** Leaves
  a dangling `tool_call_id` in the message history; the engine rejects or
  silently drops the follow-up. Execute all; append one `role: "tool"`
  message per call; preserve order.
- **Raising on malformed arguments.** Blocks the loop and loses the chance
  to let the model self-correct. Feed the error back as a tool result.
- **Passing tool result as an object, not a string.** OpenAI's wire shape
  for `role: "tool"` is `content: <string>`. Serialise to JSON first. Some
  engines will accept an object and others will 400; be strict.
- **Omitting `tool_call_id` on the `role: "tool"` message.** The field is
  required when there is a corresponding tool call; some engines are
  lenient, some return 400.
- **Not reading engine logs after a parser mismatch.** `vllm serve` logs a
  warning when the configured parser does not find any tool-call tokens in
  the generation. If every response "silently" lacks a tool call, grep for
  that warning before debugging the client.
- **Assuming tool calling is model-independent.** Smaller or less-trained
  models do not reliably emit well-formed tool calls at all. If
  `tool_call_rate` on a battery of tool-eligible prompts is below 50% at
  `tool_choice: "auto"`, the problem is model capability, not plumbing.

## 6. References

- OpenAI. *Function calling.*
  https://platform.openai.com/docs/guides/function-calling — authoritative
  wire shape for `tools`, `tool_choice`, `message.tool_calls`,
  `role: "tool"`.
- vLLM. *Tool calling.*
  https://docs.vllm.ai/en/latest/features/tool_calling.html — parser list,
  per-model notes, `--enable-auto-tool-choice`.
- SGLang. *Function calling.*
  https://sgl-project.github.io/backend/function_calling.html — SGLang
  equivalent; parser list and streaming shape.
- Meta. *Llama 3.1 prompt format.*
  https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/ —
  reference for the `<|python_tag|>` tool-call tokens that the `llama3_json`
  parser matches.
- Hermes 3 model card.
  https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B — reference
  for the `<tool_call>` format that the `hermes` parser matches.
- Hugging Face. *Chat templating — tool use.*
  https://huggingface.co/docs/transformers/chat_templating — how
  tools and `role: "tool"` messages render into the template that the
  model sees.
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/)
  — M2 (Prompt Injection) is the core risk here. A tool whose results are
  untrusted strings (search, email) can inject instructions into the next
  turn. Validate tool-argument schemas at the client (§2.5); do not
  concatenate tool output directly into the prompt without sanitisation.
- [mitre-atlas](https://atlas.mitre.org/) — AML.T0051 (LLM Prompt
  Injection) and AML.T0054 (LLM Jailbreak) both apply to tool-call loops,
  especially where the tool returns attacker-controlled text.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework) —
  governance baseline; tool-call loops widen the agent's capability
  surface, and the RMF's measure/manage functions should include a
  tool-level allowlist and per-tool authorisation scope.
