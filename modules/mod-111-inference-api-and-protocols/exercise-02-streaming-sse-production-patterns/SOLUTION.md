# Solution — Exercise 02: Streaming SSE Production Patterns

> Paired learning file: [`lessons/mod-111-inference-api-and-protocols/exercises/exercise-02-streaming-sse-production-patterns.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-111-inference-api-and-protocols/exercises/exercise-02-streaming-sse-production-patterns.md)
> Module: `mod-111-inference-api-and-protocols`
> Type: implementation / production-patterns walkthrough

## 1. Solution overview

The learner wires an OpenAI-shaped **Server-Sent Events (SSE)** stream from an
inference engine, through a reverse proxy, out to a browser and a server-side
Python client. The deliverable is a stream that survives the five things that
break SSE in production: idle-timeout kills, proxy buffering, mid-stream
backend disconnects, client-side cancellation that must propagate to the
engine, and token-accounting that must stay correct when the stream terminates
early.

The three artefacts:

| Artefact | Form | Pass bar |
|---|---|---|
| A — End-to-end streaming stack | Compose file or scripts | Engine → reverse proxy (nginx/Envoy) → two clients (browser JS and Python). Chunks arrive at the client within a bounded time of being produced by the engine |
| B — Resilience probe suite | Shell + Python scripts | Six named probes (slow client, proxy idle, mid-stream kill, client cancel, usage-chunk, error mid-stream) each with an expected outcome |
| C — Operational runbook | Markdown section | For each probe, states (a) what a healthy stack does, (b) what a mis-configured stack does, (c) the single config knob that fixes it |

### Why SSE, and not WebSockets or HTTP/2 server push

SSE — a single long-lived HTTP response with `Content-Type: text/event-stream`
and `data:` lines terminated by `\n\n` — won the LLM-streaming protocol fight
for three reasons:

1. **It's HTTP.** It traverses every reverse proxy, CDN, and L7 load balancer
   that already understands long-lived responses. No upgrade handshake, no
   separate TLS session, no WebSocket-specific proxy config.
2. **It's unidirectional, and that's enough.** LLM streaming is
   server-to-client. The client already finished its half when the request
   went out. WebSockets' bidirectional channel is wasted bandwidth and
   complexity.
3. **The OpenAI reference uses it.** Every client SDK (`openai-python`,
   `openai-node`, LangChain) expects an SSE stream back.

The OpenAI SSE shape:

```
data: {"id":"...","choices":[{"delta":{"content":"Hel"}}]}

data: {"id":"...","choices":[{"delta":{"content":"lo"}}]}

data: {"id":"...","choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]

```

Each `data:` line is a JSON chunk mirroring a non-streaming response, with
one field changed — `delta` replaces `message`, carrying the incremental
tokens. The stream ends with `data: [DONE]` and the server closes the
connection. The `[DONE]` sentinel is an OpenAI convention, not RFC-6202; if
the client doesn't see it, treat the stream as a mid-stream disconnect.

### The five failure modes you must handle

| # | Failure | Where it shows up | The fix |
|---|---|---|---|
| 1 | **Proxy buffering** | Client sees chunks in large bursts or nothing until the very end | Disable proxy buffering per response; emit flush hints |
| 2 | **Idle timeout** | Long first-token latency (big prefill) kills the connection before the first token emits | Raise read-timeout on every hop; emit an opening comment frame |
| 3 | **Backend disconnect mid-stream** | Client stream ends without `[DONE]` and without `finish_reason` | Client detects missing sentinel → retry policy decides what to do |
| 4 | **Client cancel not propagated** | User navigates away, engine keeps decoding, GPU time wasted | Propagate TCP close through the proxy; engine aborts on disconnect |
| 5 | **Usage accounting lost** | `[DONE]` sentinel received but `usage` block missing → billing breaks | Request `stream_options.include_usage: true`; validate the final chunk |

A production submission addresses all five. A demo submission addresses one.

## 2. Implementation — worked walkthrough

The commands below assume:

- vLLM as the engine (SGLang and TGI differ only in a few flag names).
- nginx as the reverse proxy; Envoy is a drop-in analogue with different
  directive names.
- A chat-sized dense model, e.g. `meta-llama/Meta-Llama-3.1-8B-Instruct`.

### 2.1 Engine launch — streaming enabled, usage emitted

```bash
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --max-model-len 8192 \
  --disable-log-requests \
  --host 0.0.0.0 --port 8001
```

No special flag is needed to enable SSE — it is enabled by any request with
`"stream": true`. The usage block on the final chunk requires
`stream_options.include_usage: true` on the request body:

```bash
curl -N http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Count to 10."}],
    "stream": true,
    "stream_options": {"include_usage": true},
    "max_tokens": 128
  }'
```

`curl -N` disables curl's own line-buffering so you can watch chunks land
live. The last two non-DONE chunks should look like:

```
data: {"id":"...","choices":[{"delta":{},"finish_reason":"stop"}], "usage": null}

data: {"id":"...","choices":[], "usage": {"prompt_tokens":22,"completion_tokens":18,"total_tokens":40}}
```

Some engines put `usage` on the same frame as the `finish_reason`; some emit
a trailing frame with `choices: []`. Both are legal — the invariant is
`[DONE]` is strictly the last line and `usage` is strictly before it.

### 2.2 Reverse proxy — disable buffering, extend timeouts

nginx buffers responses by default. For SSE that is fatal: chunks accumulate
in the proxy's buffer and arrive in bursts.

```nginx
# /etc/nginx/conf.d/llm.conf
upstream llm_engine {
    server 127.0.0.1:8001;
    keepalive 32;
}

server {
    listen 443 ssl http2;
    server_name llm.example.com;

    location /v1/ {
        proxy_pass http://llm_engine;

        # --- SSE essentials ---
        proxy_buffering off;
        proxy_cache off;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        chunked_transfer_encoding on;
        gzip off;

        # --- Timeouts — must exceed worst-case first-token latency ---
        proxy_connect_timeout 10s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;
        send_timeout 600s;

        # --- Propagate client disconnect to upstream ---
        proxy_ignore_client_abort off;

        # --- Pass through the auth and request id headers ---
        proxy_set_header X-Request-ID $request_id;
        proxy_set_header Authorization $http_authorization;
    }
}
```

Three directives earn their own callout:

- `proxy_buffering off` — the headline fix. Without it the client sees
  nothing until nginx's `proxy_buffer_size` fills or the response completes.
- `gzip off` — SSE and gzip interact badly. gzip's flush strategy is
  compatible with streaming only in theory; in practice nginx accumulates
  until the gzip window fills and the client stalls. Turn it off for the
  streaming route.
- `proxy_ignore_client_abort off` — when the client TCP-closes, nginx must
  tear down the upstream connection. Without this the engine keeps decoding
  into `/dev/null`.

For Envoy the equivalents are
`per_connection_buffer_limit_bytes: 0`,
`stream_idle_timeout: 600s`, and
`http2_protocol_options.stream_error_on_invalid_http_messaging: false`.

### 2.3 Node-less fallback pattern: keep-alive comments

If the first prefill is slow (long prompt, tool-filled context), the engine
produces no `data:` lines for seconds. Proxies and clients may interpret
that silence as death and close the connection. The OpenAI convention — and
what vLLM/SGLang/TGI emit — is a **comment frame** (a line starting with `:`)
every 15–30 seconds to keep intermediaries awake:

```
: keep-alive

```

SSE spec (WHATWG HTML §9.2) requires clients to ignore comment lines. If your
gateway inserts its own, make sure they appear on their own frame, terminated
by `\n\n`, or they will be mis-parsed as part of the previous event.

### 2.4 Browser client

The browser's built-in `EventSource` object does **not** support POST, so
you cannot use it for OpenAI-style requests. Use `fetch` with a streaming
response and a ReadableStream reader:

```html
<!-- client.html -->
<script>
async function stream(prompt) {
  const res = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: {"Content-Type": "application/json",
              "Authorization": "Bearer " + TOKEN},
    body: JSON.stringify({
      model: "meta-llama/Meta-Llama-3.1-8B-Instruct",
      messages: [{role: "user", content: prompt}],
      stream: true,
      stream_options: {include_usage: true},
      max_tokens: 1024,
    }),
  });
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  const out = document.getElementById("out");

  for (;;) {
    const {done, value} = await reader.read();
    if (done) break;
    buf += dec.decode(value, {stream: true});

    // SSE frame boundary is \n\n
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, i); buf = buf.slice(i + 2);
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") return;
        const ev = JSON.parse(payload);
        const delta = ev.choices?.[0]?.delta?.content;
        if (delta) out.textContent += delta;
        if (ev.usage) console.log("usage", ev.usage);
      }
    }
  }
}
</script>
```

Two bugs this snippet avoids:

- **Partial-frame parsing.** Chunks arrive at arbitrary byte boundaries. Scan
  for `\n\n` and buffer the tail, don't split on newline.
- **Invalid JSON on empty keep-alive.** Comment frames (`: keep-alive`)
  must be skipped — the `data:` prefix check does that for free.

### 2.5 Python server-side client — with cancellation

The server-side case is where cancellation propagation matters most. If the
caller cancels mid-stream, the engine should hear about it immediately so
the GPU stops working on a dead request.

```python
# stream_client.py
import httpx, json, asyncio

async def stream(prompt: str, cancel_after_s: float | None = None):
    body = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 1024,
    }
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0)
    emitted = 0
    usage = None
    finished_cleanly = False

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST",
                "https://llm.example.com/v1/chat/completions",
                json=body) as resp:
            resp.raise_for_status()
            start = asyncio.get_event_loop().time()
            async for line in resp.aiter_lines():
                if cancel_after_s and (asyncio.get_event_loop().time() - start
                                        > cancel_after_s):
                    raise asyncio.CancelledError("caller cancelled")
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    finished_cleanly = True
                    break
                ev = json.loads(payload)
                if ev.get("choices"):
                    delta = ev["choices"][0]["delta"].get("content", "")
                    if delta:
                        emitted += len(delta)
                        print(delta, end="", flush=True)
                if ev.get("usage"):
                    usage = ev["usage"]

    return {
        "finished_cleanly": finished_cleanly,
        "chars_emitted": emitted,
        "usage": usage,
    }
```

When the async context manager exits — whether cleanly, via exception, or
via `CancelledError` — httpx closes the TCP connection. With the nginx
config from §2.2 (`proxy_ignore_client_abort off`), that close propagates
to vLLM, which notices the dead connection and aborts the running request
on its next iteration of the scheduler loop.

### 2.6 Six resilience probes

Each probe is a one-line script that stresses one failure mode. The
runbook (§3) describes the expected pass / fail output per probe.

```bash
# probe-1 — slow client, 100ms between reads
python slow_reader.py --delay 0.1

# probe-2 — proxy idle, 90-second prefill-heavy prompt
python slow_prompt.py --prompt-tokens 7500

# probe-3 — mid-stream backend kill (restart vllm while streaming)
python stream_client.py &
sleep 2 && docker restart vllm

# probe-4 — client cancel, 2s into a 60s stream
python stream_client.py --cancel-after 2

# probe-5 — usage chunk check
python stream_client.py --require-usage

# probe-6 — engine error mid-stream (OOM from an oversize generation)
python stream_client.py --max-tokens 1000000
```

Expected outcomes on a healthy stack:

| # | Expected |
|---|---|
| 1 | Stream completes; chunks continue to arrive; no proxy timeout |
| 2 | First keep-alive comment arrives within 30s; first `data:` frame arrives when prefill completes |
| 3 | Client sees stream end without `[DONE]`; client retries or surfaces an error — must not silently claim success |
| 4 | vLLM metrics show the request aborted; GPU utilisation drops within ~1 s |
| 5 | Final frame contains a `usage` object with `prompt_tokens`, `completion_tokens`, `total_tokens` |
| 6 | Engine returns a well-formed error event on the stream, not a TCP RST; client surfaces the error |

Probe 6 is engine-dependent. vLLM emits an SSE error frame
(`data: {"error": {"message": "...", "type": "..."}}`); TGI historically closed
the connection. Document what your pinned engine does.

## 3. Validation steps

A reviewer validating a learner submission should perform the following in
order.

1. **Three-tier stack is up.** Engine, proxy, and client are running from
   the pinned configs.
2. **A baseline stream round-trips.** `curl -N` against the public hostname
   produces visible, incremental `data:` frames (not batched, not blank).
3. **nginx config is reviewed against §2.2.** The three load-bearing
   directives (`proxy_buffering off`, `gzip off`,
   `proxy_ignore_client_abort off`) are present. Read-timeouts exceed the
   worst-case first-token latency.
4. **All six probes run.** Each has a recorded outcome vs. §2.6's expected
   outcome. Any deviation is explained with a config-side root cause.
5. **Usage accounting is intact after cancel.** Probe 4 (client cancel) is
   followed by a check that the engine's token counters advanced only by
   the tokens that were actually emitted, not by the request's `max_tokens`
   ceiling.
6. **Runbook maps probes to config knobs.** For each of the five failure
   modes, the runbook names the single directive that fixes it.
7. **Cross-engine note.** The submission names which directives change for
   SGLang / TGI / Envoy, even if only one is tested end-to-end.

## 4. Rubric / review checklist

Score each item 0 (missing), 1 (partial), 2 (fully correct). Passing:
≥ 10 / 14.

| # | Item | Points |
|---|---|---|
| 1 | End-to-end stack launches from pinned configs; baseline stream works | 2 |
| 2 | nginx (or Envoy) config disables buffering, disables gzip, sets long read-timeouts, propagates client abort | 2 |
| 3 | Browser client parses SSE with partial-frame buffering; skips comment frames | 2 |
| 4 | Python server-side client propagates cancellation to the engine (verified against engine metrics) | 2 |
| 5 | All six probes run with recorded pass/fail outputs | 2 |
| 6 | Usage block arrives on the final chunk; billing-side accounting stays correct after cancel | 2 |
| 7 | Runbook maps the five failure modes to the single config knob that fixes each | 2 |

## 5. Common mistakes

- **Leaving `proxy_buffering on`.** Chunks arrive in bursts at the end of
  the response or after `proxy_buffer_size` worth of output. This is the
  single most common "it works on localhost, breaks behind nginx" bug.
- **Leaving gzip on the streaming route.** gzip flushing is nominally
  compatible with streaming but behaves unpredictably on mid-stream flushes.
  Scope gzip to non-stream routes only.
- **Using `EventSource` in the browser.** It cannot POST, cannot set
  headers, and locks you into GET-shaped requests. Use `fetch` + a
  ReadableStream reader.
- **Treating the absence of `[DONE]` as success.** Any stream that ended
  without the `data: [DONE]` sentinel is a mid-stream failure, even if the
  text looks complete. Clients must raise, not swallow.
- **Not requesting `stream_options.include_usage`.** The default is off on
  most engines. Without it the usage block never arrives, billing silently
  loses tokens, and the gap only shows up at the end-of-month invoice.
- **Not propagating client cancellation.** Without
  `proxy_ignore_client_abort off`, a user who closes the tab leaves vLLM
  decoding for the full `max_tokens` budget. At scale this is tens of
  percent of wasted GPU time.
- **Parsing SSE line-by-line and splitting on `\n`.** SSE frames are
  terminated by `\n\n`, not `\n`. A line split will produce partial JSON
  under load and throw intermittently.
- **Logging full stream bodies at INFO.** SSE streams can be megabytes.
  Log the frame count, the usage block, and the error-frame flag — not the
  full payload.
- **Short `proxy_read_timeout`.** A 60-second default kills requests whose
  prefill is longer than that. Streaming requests must have
  per-route-overridden long read timeouts.
- **No keep-alive on the engine → proxy hop.** vLLM's SSE emits data for a
  while and then stops to prefill a new batch; without keep-alive the
  connection pool thrashes.

## 6. References

- WHATWG. *HTML Living Standard §9.2 Server-sent events.*
  https://html.spec.whatwg.org/multipage/server-sent-events.html — the
  wire-format source of truth: `data:`, `event:`, `id:`, `retry:`, comment
  lines, frame boundary.
- OpenAI. *Streaming responses.*
  https://platform.openai.com/docs/api-reference/streaming — the
  `data: [DONE]` sentinel convention and `stream_options.include_usage`.
- nginx documentation. *HTTP proxy module.*
  https://nginx.org/en/docs/http/ngx_http_proxy_module.html — directive
  reference for the §2.2 config.
- Envoy. *HTTP routing.*
  https://www.envoyproxy.io/docs/envoy/latest/configuration/http/ —
  equivalent directives for operators on Envoy / Istio.
- vLLM. *OpenAI-compatible server.*
  https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html —
  streaming behaviour and error-frame conventions.
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/)
  — M1 (Output Handling) applies here. Streaming does not change the risk
  that model output ends up in an HTML context unescaped; the client in
  §2.4 renders to `textContent`, not `innerHTML`, for exactly this reason.
- [mitre-atlas](https://atlas.mitre.org/) — not directly invoked. A
  denial-of-wallet via client-cancel-abuse scenario is a reasonable
  threat-model entry if the stack bills on stream start rather than stream
  complete.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework) —
  governance baseline. The usage-accounting invariant (§2.5) is the audit
  hook that keeps per-tenant cost attribution honest.
