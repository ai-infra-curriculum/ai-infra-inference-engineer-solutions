# Solution — Exercise 01: TTFT / TPOT / ITL Decomposition and Trace Wiring

> Paired learning file: [`lessons/mod-110-observability-slo-and-cost/exercises/exercise-01-ttft-tpot-itl-decomposition-and-trace-wiring.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-110-observability-slo-and-cost/exercises/exercise-01-ttft-tpot-itl-decomposition-and-trace-wiring.md)
> Module: `mod-110-observability-slo-and-cost`
> Type: implementation / observability

## 1. Solution overview

Token-streaming inference has three first-class latency signals, and a stack
that only records one of them is operationally blind. This exercise lands all
three as **named, measured, traced** quantities with unambiguous definitions:

| Signal | Definition (per request) | What it answers |
|---|---|---|
| **TTFT** — Time To First Token | Wall-clock from the moment the client's bytes finish uploading to the moment the first output byte is flushed back. Includes router, queue, prefill, and sampling of token 0. | "Did the model start talking in time?" |
| **TPOT** — Time Per Output Token | Mean decode-step duration across generated tokens 2..N. Usually computed as `(E2E − TTFT) / (output_tokens − 1)`. | "Is the stream fast enough once it's going?" |
| **ITL** — Inter-Token Latency | Distribution of wall-clock gaps between successive output tokens as the client observes them. Equivalent instantaneous view of TPOT. | "Does the stream stall, or is it smooth?" |

TPOT is a scalar per request; ITL is a distribution per request. Reporting
only TPOT hides stalls — a stream with one 2 s pause and 99 fast tokens can
have the same TPOT as one with 100 steady tokens.

The four deliverables:

| Artefact | Form | Pass bar |
|---|---|---|
| A — Latency decomposition diagram + budget table | Markdown + ASCII diagram | Every hop between client and last byte is named and has a budget share; TTFT / TPOT / ITL boundaries drawn precisely |
| B — OTel span tree specification | Table + sample JSON span | Required spans exist per request with named attributes; TTFT, TPOT, and ITL derivable from span timestamps alone |
| C — Engine + router instrumentation patch | Python snippet (engine wrapper) + router middleware | Spans produced on every request; counters and histograms exposed for the Prom scraper in [[exercise-03-prom-grafana-otel-inference-stack-base]] |
| D — Verification trace | 1 captured trace (JSON + screenshot) | TTFT, TPOT, ITL computed from the trace match engine-emitted metrics within ± one scrape interval |

### Why decomposition matters before SLOs

SLOs authored on E2E alone leave the on-call guessing: a breach could mean
queue, prefill, or decode. The decomposition gives each span a budget and lets
the alert surface *which* stage is bleeding. The SLOs in
[[exercise-02-slo-authoring-per-workload-class]] assume this decomposition is
already in place.

## 2. Implementation — worked answer

### 2.1 Latency decomposition (artefact A)

The canonical request path, from client send to last byte:

```
 client.send_last_byte                                                    client.recv_last_byte
         │                                                                        ▲
         ▼                                                                        │
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐   │
  │  ingress /   │  │   router /   │  │   engine     │  │   engine decode     │   │
  │  TLS / auth  ├─▶│  admission   ├─▶│  prefill     ├─▶│   loop (N steps)    ├───┘
  └──────────────┘  └──────┬───────┘  └──────┬───────┘  └──────┬──────────────┘
                           │                 │                 │
                      queue_wait        prefill_duration   per-step ITL
                      ───────────       ────────────────   ─────────────
                      scheduler         compute-bound,     memory-bound,
                      picks req         grows with         grows with
                                        prompt tokens      batch pressure

 ◄── TTFT ────────────────────────────────────────────────────────────────┐
     client.send_last_byte → first output byte flushed to client (token 0) │
                                                                           ▼
                                                                  first_byte_flush
 ◄── ITL(i) ──────────────────────────────────────────────────────────────┐
     (flush of token i−1) → (flush of token i)   for i = 1 .. N−1          │
                                                                           ▼
 ◄── E2E ─────────────────────────────────────────────────────────────────▶
     client.send_last_byte → client.recv_last_byte

 TPOT = (E2E − TTFT) / (output_tokens − 1)
 ITL_p95 = p95 of {ITL(1), ..., ITL(N−1)} per request;
           or p95 of ITL events across a time window at the fleet level
```

Named hop budgets for a reference **interactive chat** request
(70 B class target, streaming, prompt ≈ 1 k tokens, output ≈ 200 tokens).
The budget line is a worked example a learner should tune to their own
hardware and workload — it is not a fleet default.

| Hop | Typical share of TTFT | Typical share of E2E |
|---|---|---|
| Ingress / TLS / auth | ≤ 5 % of TTFT | small |
| Router / routing / admission | ≤ 5 % of TTFT | small |
| Engine queue wait | ≤ 20 % of TTFT (target); blows past this under overload | small |
| Prefill (prompt-length-proportional) | 50–70 % of TTFT | small (one-shot) |
| First token sample + first flush | ≤ 5 % of TTFT | small |
| Decode loop | 0 | dominates — `(output_tokens − 1) × TPOT` |

A learner submitting artefact A must produce a similar table for their own
workload class with measured numbers from a baseline run.

### 2.2 OTel span tree (artefact B)

OpenTelemetry's trace model carries this decomposition cleanly. The required
span tree per request:

```
Span: inference.request                     (root — client-facing)
  attrs: peer.service, http.route, tenant, model, request_id
  events: — (none)
├── Span: router.route
│     attrs: router.decision, pool, engine_endpoint
│     duration: router latency incl. load-balancer selection
├── Span: engine.request
│     attrs: engine, engine.version, model, model.revision,
│            prompt_tokens, max_tokens, sampling.temperature, sampling.top_p,
│            spec_decode.enabled
│     ├── Span: engine.queue
│     │     attrs: queue.depth_at_admission
│     │     duration: time in waiting queue
│     ├── Span: engine.prefill
│     │     attrs: prefill.chunks, prefill.tokens, prefill.prefix_cache_hit
│     │     duration: prompt prefill, may span multiple engine iterations
│     │     events:
│     │       - "first_output_token_ready"   ← timestamp fixes TTFT
│     └── Span: engine.decode
│           attrs: decode.output_tokens, decode.preempted, decode.spec_accepted
│           duration: last decode step − first_output_token_ready
│           events:
│             - "token_emitted"   per token (or sampled)  ← timestamps give ITL
```

TTFT, TPOT, and ITL are *derivable from timestamps alone* — no separate
measurement path. The events are what matter:

```
TTFT      = event("first_output_token_ready").timestamp − root.start_time
E2E       = root.end_time − root.start_time
TPOT      = (E2E − TTFT) / (output_tokens − 1)
ITL[i]    = event("token_emitted")[i].timestamp − event("token_emitted")[i−1].timestamp
ITL_p95   = percentile(ITL[], 0.95)
```

Attributes follow OpenTelemetry semantic conventions for GenAI
(`gen_ai.request.model`, `gen_ai.usage.input_tokens`,
`gen_ai.usage.output_tokens`); the trace wiring is interoperable with any
OTel-aware backend (Tempo, Jaeger, Honeycomb, Datadog, etc.). Verify exact
attribute names against the OpenTelemetry semantic conventions release pinned
in [[exercise-03-prom-grafana-otel-inference-stack-base]].

**Per-token events: cost control.** A 4 k-token response produces 4 k events.
At fleet scale this swamps the trace backend. Two pragmatic options:

- **Sample events.** Emit `token_emitted` every k-th token (k = 16 or 32 is a
  common choice); keep `first_output_token_ready` unconditionally.
- **Summarise.** Emit per-request histogram attributes on `engine.decode`
  (`decode.itl_p50_ms`, `decode.itl_p95_ms`, `decode.itl_max_ms`) and drop
  per-token events. Loses stall-location signal but keeps TTFT/TPOT/ITL.

Either path is acceptable; the choice must be documented so dashboards and
alerts in [[exercise-03-prom-grafana-otel-inference-stack-base]] know what
they can rely on.

### 2.3 Instrumentation patch (artefact C)

Two touch points: the router (for `inference.request` + `router.route`) and
the engine wrapper (for `engine.request`, `engine.queue`, `engine.prefill`,
`engine.decode`). Sketch in Python using the OTel SDK; the shape ports
directly to Go, Rust, Java, etc.

```python
# router/middleware.py (FastAPI example)
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
import time

tracer = trace.get_tracer("inference.router")

async def inference_middleware(request, call_next):
    with tracer.start_as_current_span("inference.request") as root:
        root.set_attribute("http.route", request.url.path)
        root.set_attribute("tenant", request.headers.get("x-tenant", "unknown"))
        with tracer.start_as_current_span("router.route") as route_span:
            pool = pick_pool(request)             # routing decision
            route_span.set_attribute("router.decision", pool.name)
            route_span.set_attribute("pool", pool.name)
            route_span.set_attribute("engine_endpoint", pool.endpoint)
        response = await call_next(request)       # engine span is created below
        root.set_attribute("http.status_code", response.status_code)
        if response.status_code >= 500:
            root.set_status(Status(StatusCode.ERROR))
        return response
```

```python
# engine/wrapper.py (vLLM AsyncLLMEngine shown; adapt for SGLang / TRT-LLM)
from opentelemetry import trace
from prometheus_client import Histogram, Counter
import time

tracer = trace.get_tracer("inference.engine")

# Metrics are the Prom-scrapable twin of the trace spans.
# Buckets chosen for typical streaming chat; tune per workload class.
TTFT_HIST = Histogram(
    "inference_ttft_seconds",
    "Time to first token",
    labelnames=("pool", "engine", "model", "workload_class"),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0),
)
TPOT_HIST = Histogram(
    "inference_tpot_seconds",
    "Time per output token (mean over decode)",
    labelnames=("pool", "engine", "model", "workload_class"),
    buckets=(0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5),
)
ITL_HIST = Histogram(
    "inference_itl_seconds",
    "Inter-token latency (per-token, sampled)",
    labelnames=("pool", "engine", "model", "workload_class"),
    buckets=(0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0),
)
E2E_HIST = Histogram(
    "inference_e2e_seconds",
    "End-to-end request latency",
    labelnames=("pool", "engine", "model", "workload_class"),
    buckets=(0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0),
)

async def generate_with_tracing(req):
    labels = dict(
        pool=req.pool, engine=req.engine, model=req.model,
        workload_class=req.workload_class,
    )
    t_root = time.perf_counter()
    with tracer.start_as_current_span("engine.request") as eng:
        eng.set_attribute("gen_ai.request.model", req.model)
        eng.set_attribute("gen_ai.usage.input_tokens", req.prompt_tokens)
        eng.set_attribute("sampling.temperature", req.temperature)

        # Queue wait — admission span opened on enqueue, closed on scheduler pick
        with tracer.start_as_current_span("engine.queue") as q:
            await req.admitted()                   # future resolved by scheduler
            q.set_attribute("queue.depth_at_admission", req.queue_depth_at_pick)

        # Prefill — may span multiple engine iterations under chunked prefill
        t_prefill_start = time.perf_counter()
        with tracer.start_as_current_span("engine.prefill") as pf:
            await req.first_token_ready()
            pf.set_attribute("prefill.tokens", req.prompt_tokens)
            pf.set_attribute("prefill.chunks", req.prefill_chunks)
            pf.set_attribute(
                "prefill.prefix_cache_hit", req.prefix_cache_hit,
            )
            pf.add_event("first_output_token_ready")   # fixes TTFT
        ttft = time.perf_counter() - t_root
        TTFT_HIST.labels(**labels).observe(ttft)

        # Decode — per-token events (optionally sampled) + ITL histogram
        with tracer.start_as_current_span("engine.decode") as dec:
            last = time.perf_counter()
            k = 16                                   # event-sampling stride
            itl_samples = []
            async for i, token in enumerate_stream(req):
                now = time.perf_counter()
                itl = now - last
                last = now
                ITL_HIST.labels(**labels).observe(itl)
                itl_samples.append(itl)
                if i % k == 0:
                    dec.add_event("token_emitted", {"token.index": i})
            dec.set_attribute("decode.output_tokens", len(itl_samples) + 1)
            dec.set_attribute("decode.itl_p95_ms",
                              _p95(itl_samples) * 1000.0)

        e2e = time.perf_counter() - t_root
        E2E_HIST.labels(**labels).observe(e2e)
        if len(itl_samples) >= 1:
            tpot = (e2e - ttft) / len(itl_samples)
            TPOT_HIST.labels(**labels).observe(tpot)
```

Three deliberate design decisions:

- **Metrics and traces are twins.** Every histogram has a corresponding span
  event or attribute. A reviewer can cross-check a dashboard panel against a
  captured trace and expect agreement to within one scrape interval. If they
  disagree, one of them is wrong.
- **ITL is a histogram, not a gauge.** Gauging ITL loses the distribution. The
  stall-detection story needs p95 / p99, not mean.
- **Labels stay low-cardinality.** `pool`, `engine`, `model`, `workload_class`
  — no `request_id` or `tenant` on histograms. Per-tenant breakdowns belong on
  traces and exemplars, not on label sets.

**Exemplars.** When Prometheus exemplars are enabled, attach the trace ID to
each histogram observation. On an SLO breach the on-call clicks the exemplar
and lands on the exact slow trace — the "which hop was slow?" question is
answered before the runbook is opened.

### 2.4 Verification trace (artefact D)

Fire a single streaming request with a known prompt and sampled output length,
capture the resulting trace, and show that trace-derived numbers match
metrics-derived numbers.

```
# capture one trace
curl -N -H 'content-type: application/json' \
  -d '{"model":"llama-3.1-70b-instruct","prompt":"<N-token prompt>",
       "max_tokens":128,"stream":true,"temperature":0.0,
       "trace_sampling":"always_on"}' \
  https://inference.internal/v1/chat/completions

# Inspect the trace in Tempo / Jaeger. Record:
#  - root.start_time, root.end_time
#  - event timestamps: first_output_token_ready, token_emitted[i]

# Compute offline from span timestamps:
#   TTFT  = event("first_output_token_ready").ts − root.start
#   E2E   = root.end − root.start
#   TPOT  = (E2E − TTFT) / (output_tokens − 1)
#   ITLs  = diffs of token_emitted timestamps
#
# Cross-check against histograms via Prometheus:
#   histogram_quantile(0.5, sum by (le) (
#     rate(inference_ttft_seconds_bucket{request_id="..."}[1m])))
# (use exemplar lookup if request_id is not a histogram label)
```

Agreement within one scrape interval (typically ≤ 15 s) is the pass bar.
Larger drift is one of:

- Clock skew between router and engine (fix: single source for
  `root.start_time`, or NTP-tighten both hosts).
- TTFT recorded before the first byte actually hits the socket (fix: emit
  `first_output_token_ready` after the flush, not before the sample).
- ITL sampled only every k-th token but TPOT computed over all tokens
  (consistent by design, but makes the ITL p95 from the histogram look higher
  than the trace's per-token diffs — document the sampling stride).

## 3. Validation steps

A reviewer validating a learner submission runs the following, in order.

1. **Diagram covers every hop.** Artefact A names ingress, router, queue,
   prefill, decode, flush; each has a budget share; TTFT/E2E/ITL boundaries
   are drawn precisely (not just "around here").
2. **Span tree matches §2.2.** `inference.request` → `router.route` and
   `engine.request` → `engine.queue`, `engine.prefill` (with
   `first_output_token_ready` event), `engine.decode` (with `token_emitted`
   events or sampled-event + summary attributes).
3. **Attributes present on every span.** `model`, `pool`, `engine`,
   `prompt_tokens`, `output_tokens`, `workload_class`; OpenTelemetry GenAI
   conventions respected.
4. **Metrics exported.** `inference_ttft_seconds`, `inference_tpot_seconds`,
   `inference_itl_seconds`, `inference_e2e_seconds` scrape cleanly;
   `promtool query instant` returns non-empty on each.
5. **Trace / metric cross-check.** Artefact D shows a captured trace plus
   PromQL output; TTFT, TPOT, ITL agree within one scrape interval.
6. **Event sampling documented.** If `token_emitted` is sampled, the stride
   is in artefact B and reflected in dashboards and SLO windows downstream.
7. **No per-request labels on histograms.** `request_id`, `tenant`, and other
   high-cardinality attributes live on spans, not on label sets.

## 4. Rubric / review checklist

Score each item 0 (missing), 1 (partial), 2 (fully correct). Pass: ≥ 10 / 14.

| # | Item | Points |
|---|---|---|
| 1 | TTFT / TPOT / ITL definitions precise, with the TPOT formula and the ITL-vs-TPOT distinction stated | 2 |
| 2 | Decomposition diagram names every hop; TTFT / ITL / E2E boundaries drawn precisely | 2 |
| 3 | Span tree matches §2.2 with correct parent/child relations and semantic-conv attributes | 2 |
| 4 | Instrumentation emits spans and Prom histograms as twins; low-cardinality label hygiene | 2 |
| 5 | Event sampling strategy chosen and documented (per-token vs. stride-k vs. summary) | 2 |
| 6 | Verification trace captured; TTFT / TPOT / ITL from trace agree with metrics within one scrape interval | 2 |
| 7 | Exemplars wired (or an explicit trace-ID label on a dedicated exemplar-friendly histogram) | 2 |

## 5. Common mistakes

- **Reporting TPOT without ITL.** A mean hides stalls. A 200-token stream with
  a 2 s pause and 199 fast tokens has the same TPOT as a smooth stream;
  users notice the stall. Always carry both.
- **Measuring TTFT at sample time, not at flush time.** The first-byte event
  must fire after the socket write, or TTFT undercounts the actual user
  experience by the serialization + network cost.
- **High-cardinality labels on histograms.** Putting `request_id` or `tenant`
  on a histogram explodes Prometheus series count. Use exemplars for
  per-request drill-down; keep labels to the pool/engine/model/class grid.
- **Per-token span events at fleet scale.** A 4 k-token response per request
  times fleet QPS swamps the trace backend. Sample or summarize; document
  which.
- **Confusing per-request TPOT with fleet TPOT.** `(E2E − TTFT) / (N − 1)` is
  a per-request scalar. The fleet "TPOT P95" is the p95 of those scalars, not
  a p95 over ITL events. Both are useful; label them unambiguously on
  dashboards.
- **Clock skew between router and engine.** If `root.start_time` comes from
  the router and `first_output_token_ready` comes from the engine without NTP
  tightening, TTFT drifts by the skew. One source of truth, or tight NTP.
- **Dropping the queue span.** A low TTFT with a high queue wait is a
  scheduler problem, not a prefill problem; without the `engine.queue` span
  the two are indistinguishable.
- **Hardcoded histogram buckets.** The default Prometheus buckets are tuned
  for RPC latency, not token streaming. Choose buckets that put at least one
  bucket edge within 10 % of each SLO threshold; otherwise percentile
  estimation is noisy near the SLO.
- **Forgetting prefix-cache hits.** Prefill time varies by an order of
  magnitude on prefix-cache hit vs. miss. Attribute the hit/miss onto
  `engine.prefill` so TTFT dashboards can be split by cache status.

## 6. References

- OpenTelemetry. *Specification.* <https://opentelemetry.io/docs/specs/otel/>
  — span and event model, attribute conventions.
- OpenTelemetry. *Semantic conventions for Generative AI.*
  <https://opentelemetry.io/docs/specs/semconv/gen-ai/> — `gen_ai.*` attribute
  names used above; verify against the pinned semantic-conv release.
- Prometheus. *Histogram and summary practices.*
  <https://prometheus.io/docs/practices/histograms/> — bucket choice,
  `histogram_quantile` idiom.
- Prometheus. *Exemplars.*
  <https://prometheus.io/docs/prometheus/latest/feature_flags/#exemplars-storage>
  — attaching trace IDs to histogram observations.
- Google SRE book. *Monitoring distributed systems.*
  <https://sre.google/sre-book/monitoring-distributed-systems/> — the "four
  golden signals"; TTFT, TPOT, ITL, E2E are their token-streaming instance.
- vLLM. *Metrics / observability.* <https://docs.vllm.ai/> — engine-side
  counters this exercise wraps with a trace-shaped twin; verify names against
  the pinned release.
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/)
  — ML06-adjacent: a silent throughput regression (TTFT fine but TPOT drifted)
  is an output-integrity concern at the service level.
- [mitre-atlas](https://atlas.mitre.org/) — detection patterns for
  model-serving systems; latency anomalies are a monitored signal.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework) —
  "Measure" function; decomposed latency instrumentation is the concrete
  measurement surface for a deployed inference system.
