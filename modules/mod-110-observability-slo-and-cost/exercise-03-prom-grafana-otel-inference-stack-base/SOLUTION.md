# Solution — Exercise 03: Prometheus / Grafana / OTel Inference Stack — Base

> Paired learning file: [`lessons/mod-110-observability-slo-and-cost/exercises/exercise-03-prom-grafana-otel-inference-stack-base.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-110-observability-slo-and-cost/exercises/exercise-03-prom-grafana-otel-inference-stack-base.md)
> Module: `mod-110-observability-slo-and-cost`
> Type: implementation / infrastructure

## 1. Solution overview

The **base** stack is the single reproducible bundle every downstream
observability exercise in this module reuses. It stands up five services
(engine → OTel Collector → Prometheus → Tempo → Grafana), carries the three
signal types (metrics, traces, logs) through to Grafana, and lets a reviewer
replay the whole thing with one `docker compose up`.

The six artefacts:

| Artefact | Form | Pass bar |
|---|---|---|
| A — Topology diagram | ASCII diagram + Markdown | Every service and port named; data-plane and control-plane flows separated |
| B — `docker-compose.yml` | YAML | Stack comes up clean; versions pinned; volumes persist across restart |
| C — OTel Collector config | YAML | Receives OTLP from engine, routes metrics to Prometheus, traces to Tempo, logs to Loki (optional) |
| D — Prometheus config + rules | YAML | Scrapes engine `/metrics` and collector's self-metrics; recording rules for TTFT / TPOT / ITL percentiles; rule files validate with `promtool` |
| E — Grafana provisioning | YAML + JSON | Datasources auto-provisioned (Prometheus, Tempo, optional Loki); three seed dashboards imported on startup |
| F — End-to-end smoke test | Shell script + log | One scripted request produces exactly one trace in Tempo and non-zero histogram buckets in Prometheus within 30 s |

The stack is the substrate for
[[exercise-01-ttft-tpot-itl-decomposition-and-trace-wiring]]'s traces and
[[exercise-02-slo-authoring-per-workload-class]]'s burn-rate alerts. It is
not the *final* production bundle — downstream exercises add Alertmanager,
cost-per-token accounting, GPU telemetry (DCGM), and multi-tenant dashboards.
This exercise lands the floor, not the ceiling.

### Why OTel Collector in the middle

The engine can push OTLP directly to each backend, but a Collector in the
middle earns its keep:

- **One wire protocol**, OTLP, between the engine and the observability
  plane. Engines (vLLM, SGLang, TRT-LLM) and backends (Prometheus, Tempo,
  Loki, Datadog, Honeycomb) change independently; the Collector absorbs the
  churn.
- **Per-signal routing.** Metrics → Prometheus remote-write or
  `/metrics`-style pull, traces → Tempo, logs → Loki — one receiver, three
  exporters.
- **Sampling, redaction, and label hygiene.** The Collector is where
  per-tenant / per-request attributes get stripped or hashed before they
  hit storage. The engine should not be in the compliance business.
- **Resilience.** The Collector buffers if a backend blips; the engine does
  not block waiting for observability.

If the fleet is small and a single backend is involved, direct push is
fine — but the exercise lands the Collector form because the fleet grows.

## 2. Implementation — worked answer

### 2.1 Topology (artefact A)

```
                        ┌──────────────────────────────────────────┐
                        │ Docker-compose network: "obs-net"        │
                        └──────────────────────────────────────────┘

  ┌──────────────┐   OTLP/gRPC (4317)   ┌─────────────────────┐
  │  inference   │ ──────────────────▶  │                     │
  │  engine      │                      │   OTel Collector    │
  │  (vLLM)      │                      │   (contrib)         │
  │  :8001       │ ◄── (optional)       │                     │
  │              │     scrape /metrics  │                     │
  └──────┬───────┘                      └──────┬──────────────┘
         │                                     │
         │ Prom scrape                         │ remote_write   OTLP      OTLP
         │ (:8001/metrics)                     │ to /api/v1/   (traces)  (logs)
         │                                     │ write          │        │
         ▼                                     ▼                ▼        ▼
  ┌──────────────┐                     ┌───────────────┐  ┌─────────┐  ┌──────┐
  │  Prometheus  │ ◄───────────────────│   collector   │  │  Tempo  │  │ Loki │
  │   :9090      │  (pull scrape of    │ self-metrics  │  │  :3200  │  │:3100 │
  │              │  engine + collector)└───────────────┘  └────┬────┘  └──┬───┘
  └──────┬───────┘                                             │          │
         │                                                     │          │
         └─────────────┬─────────────────────────┬─────────────┘          │
                       ▼                         ▼                        │
                 ┌───────────┐              ┌──────────┐                  │
                 │  Grafana  │ ◄────────────┤datasources│ ◄───────────────┘
                 │   :3000   │              └──────────┘
                 └───────────┘
```

Three flow notes:

- **Metrics travel both ways.** The engine exposes `/metrics` for Prometheus
  pull, *and* pushes histograms (incl. TTFT / TPOT / ITL) via OTLP to the
  Collector. The two paths overlap on purpose: pull wins on completeness
  and alert reliability, push wins on OTel-native histograms with
  exemplars. Dashboards favour the pull series; exemplar lookups use the
  push series.
- **Traces go one way, through the Collector.** No Prometheus hop for
  traces.
- **Logs are optional in the base bundle.** Include Loki when log-based
  correlation is a must; omit otherwise. Grafana datasources degrade
  gracefully if Loki is absent.

### 2.2 `docker-compose.yml` (artefact B)

Pinned versions below are illustrative — bump to the current LTS at build
time and record the versions used in a `VERSIONS.md` sibling.

```yaml
# docker-compose.yml
version: "3.9"

networks:
  obs-net:
    driver: bridge

volumes:
  prom-data:
  tempo-data:
  grafana-data:

services:
  otel-collector:
    image: otel/opentelemetry-collector-contrib:0.111.0
    command: ["--config=/etc/otel/collector.yaml"]
    volumes:
      - ./collector/collector.yaml:/etc/otel/collector.yaml:ro
    ports:
      - "4317:4317"    # OTLP gRPC — engine → collector
      - "4318:4318"    # OTLP HTTP — curl-friendly path for smoke tests
      - "8888:8888"    # collector self-metrics (Prometheus scrape)
    networks: [obs-net]

  prometheus:
    image: prom/prometheus:v2.55.0
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./prometheus/rules:/etc/prometheus/rules:ro
      - prom-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=15d'
      - '--web.enable-remote-write-receiver'
      - '--enable-feature=exemplar-storage'
    ports:
      - "9090:9090"
    networks: [obs-net]

  tempo:
    image: grafana/tempo:2.6.0
    command: ["-config.file=/etc/tempo/tempo.yaml"]
    volumes:
      - ./tempo/tempo.yaml:/etc/tempo/tempo.yaml:ro
      - tempo-data:/var/tempo
    ports:
      - "3200:3200"    # tempo http
      - "4319:4319"    # tempo OTLP receiver (collector → tempo)
    networks: [obs-net]

  grafana:
    image: grafana/grafana:11.2.0
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin        # dev only
      - GF_FEATURE_TOGGLES_ENABLE=traceToMetrics,traceqlEditor
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
      - grafana-data:/var/lib/grafana
    ports:
      - "3000:3000"
    depends_on: [prometheus, tempo]
    networks: [obs-net]

  # The engine itself is run OUT of compose when GPUs are involved; wire it
  # into obs-net with `--network obs-net` so `otel-collector:4317` resolves,
  # OR run a CPU dummy-server for smoke tests via the sibling service below.
  mock-engine:
    image: ghcr.io/aicg/mock-inference-engine:latest   # or build from ./mock-engine
    environment:
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
      - OTEL_SERVICE_NAME=inference-engine
    ports:
      - "8001:8001"
    networks: [obs-net]
```

**On the real engine.** vLLM, SGLang, and TRT-LLM each support an
`OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_EXPORTER_OTLP_PROTOCOL` environment
variable (verify the exact flag against the pinned release); set those to
point at `otel-collector:4317` and the trace wiring from
[[exercise-01-ttft-tpot-itl-decomposition-and-trace-wiring]] starts
reporting. A mock engine ships with this exercise to make the stack bring-up
step independent of GPU availability — a reviewer can exercise the whole
pipeline on a laptop.

### 2.3 OTel Collector config (artefact C)

```yaml
# collector/collector.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    send_batch_size: 8192
    timeout: 5s
  memory_limiter:
    check_interval: 1s
    limit_mib: 512
  resource/common:
    attributes:
      - key: deployment.environment
        value: dev
        action: upsert
  attributes/redact:
    actions:
      # Strip request bodies if the engine mis-attributes them to a span.
      - key: gen_ai.prompt
        action: delete
      - key: gen_ai.completion
        action: delete
      # Hash tenant id instead of exposing raw strings.
      - key: tenant
        action: hash

exporters:
  prometheusremotewrite:
    endpoint: "http://prometheus:9090/api/v1/write"
    resource_to_telemetry_conversion:
      enabled: true
  otlp/tempo:
    endpoint: "tempo:4319"
    tls:
      insecure: true
  # loki (optional)
  # loki:
  #   endpoint: "http://loki:3100/loki/api/v1/push"

extensions:
  health_check:
    endpoint: 0.0.0.0:13133

service:
  extensions: [health_check]
  pipelines:
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, resource/common, batch]
      exporters: [prometheusremotewrite]
    traces:
      receivers: [otlp]
      processors: [memory_limiter, resource/common, attributes/redact, batch]
      exporters: [otlp/tempo]
    # logs:
    #   receivers: [otlp]
    #   processors: [memory_limiter, resource/common, attributes/redact, batch]
    #   exporters: [loki]
  telemetry:
    metrics:
      address: 0.0.0.0:8888
```

Two deliberate choices in the Collector config:

- **`attributes/redact` runs on traces, not metrics.** Metrics should
  already be low-cardinality (the engine wrapper in
  [[exercise-01-ttft-tpot-itl-decomposition-and-trace-wiring]] enforces
  this); traces are where accidental PII / prompt content leaks, so the
  redactor sits on the trace pipeline.
- **`memory_limiter` is the first processor.** Backpressure under a trace
  storm hits here first, not the exporters. Without this the Collector OOMs
  during an incident, which is exactly when traces are most needed.

### 2.4 Prometheus config and recording rules (artefact D)

```yaml
# prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: dev
    stack: inference-base

scrape_configs:
  - job_name: 'inference-engine'
    scrape_interval: 10s
    metrics_path: /metrics
    static_configs:
      - targets: ['mock-engine:8001']
        labels:
          engine: vllm
          pool: dev-pool
          model: mock-model
          workload_class: chat

  - job_name: 'otel-collector'
    scrape_interval: 15s
    static_configs:
      - targets: ['otel-collector:8888']

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

rule_files:
  - /etc/prometheus/rules/*.yml
```

Recording rules — the TTFT / TPOT / ITL percentiles that dashboards and
burn-rate rules in
[[exercise-02-slo-authoring-per-workload-class]] consume. Expensive
`histogram_quantile` expressions get pre-computed once rather than at every
dashboard load.

```yaml
# prometheus/rules/inference-latency.yml
groups:
  - name: inference-latency-percentiles
    interval: 30s
    rules:
      - record: inference:ttft:p50:5m
        expr: |
          histogram_quantile(
            0.5,
            sum by (le, pool, engine, model, workload_class) (
              rate(inference_ttft_seconds_bucket[5m])
            )
          )
      - record: inference:ttft:p95:5m
        expr: |
          histogram_quantile(
            0.95,
            sum by (le, pool, engine, model, workload_class) (
              rate(inference_ttft_seconds_bucket[5m])
            )
          )
      - record: inference:ttft:p99:5m
        expr: |
          histogram_quantile(
            0.99,
            sum by (le, pool, engine, model, workload_class) (
              rate(inference_ttft_seconds_bucket[5m])
            )
          )
      - record: inference:tpot:p95:5m
        expr: |
          histogram_quantile(
            0.95,
            sum by (le, pool, engine, model, workload_class) (
              rate(inference_tpot_seconds_bucket[5m])
            )
          )
      - record: inference:itl:p95:5m
        expr: |
          histogram_quantile(
            0.95,
            sum by (le, pool, engine, model, workload_class) (
              rate(inference_itl_seconds_bucket[5m])
            )
          )
      - record: inference:e2e:p95:5m
        expr: |
          histogram_quantile(
            0.95,
            sum by (le, pool, engine, model, workload_class) (
              rate(inference_e2e_seconds_bucket[5m])
            )
          )
      - record: inference:requests:qps:1m
        expr: |
          sum by (pool, engine, model, workload_class) (
            rate(inference_e2e_seconds_count[1m])
          )
```

Validate with `promtool check rules /etc/prometheus/rules/*.yml`; the stack's
CI job should refuse to start Prometheus on a rule failure.

### 2.5 Grafana provisioning (artefact E)

Three files make Grafana self-provision on startup.

```yaml
# grafana/provisioning/datasources/datasources.yml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    jsonData:
      exemplarTraceIdDestinations:
        - name: trace_id
          datasourceUid: tempo
          urlDisplayLabel: "View trace"
  - name: Tempo
    type: tempo
    uid: tempo
    access: proxy
    url: http://tempo:3200
    jsonData:
      tracesToMetrics:
        datasourceUid: prometheus
        tags:
          - { key: 'pool', value: 'pool' }
          - { key: 'workload_class', value: 'workload_class' }
```

```yaml
# grafana/provisioning/dashboards/dashboards.yml
apiVersion: 1
providers:
  - name: inference-base
    folder: Inference
    type: file
    options:
      path: /var/lib/grafana/dashboards
```

Three seed dashboards live in `grafana/dashboards/`:

1. **`00-overview.json`** — top-of-fleet summary.
   - Panels: QPS by pool; TTFT p95 by pool; E2E p95 by pool; error rate.
   - Queries use the recording rules in §2.4; cheap to render.
2. **`01-latency-decomposition.json`** — per-pool decomposition.
   - Panels: TTFT p50/p95/p99 time series; TPOT p95; ITL p95 and p99; a
     heatmap of `inference_ttft_seconds_bucket`.
   - Every panel supports exemplar click-through into Tempo via the
     `exemplarTraceIdDestinations` wiring above.
3. **`02-request-explorer.json`** — Tempo-first panel plus metrics side-by-side.
   - Tempo TraceQL search by `workload_class`, `pool`, `tenant_hash`.
   - Linked panels show the TTFT / TPOT / ITL time series for the same
     window; traces-to-metrics narrows to the same `pool` label.

Dashboards are reviewed by shape, not by pixel. A reviewer imports them,
fires the smoke test, and confirms every panel renders with non-empty data.

### 2.6 Tempo config (minimal)

```yaml
# tempo/tempo.yaml
server:
  http_listen_port: 3200
distributor:
  receivers:
    otlp:
      protocols:
        grpc:
          endpoint: 0.0.0.0:4319
ingester:
  trace_idle_period: 10s
  max_block_duration: 5m
storage:
  trace:
    backend: local
    local:
      path: /var/tempo/traces
```

### 2.7 End-to-end smoke test (artefact F)

```bash
#!/usr/bin/env bash
# smoke.sh — brings the stack up and asserts metrics + traces flow.
set -euo pipefail

docker compose up -d
echo "waiting for stack to come up"
for i in {1..30}; do
  if curl -sf http://localhost:9090/-/ready > /dev/null \
     && curl -sf http://localhost:3200/ready > /dev/null \
     && curl -sf http://localhost:3000/api/health > /dev/null; then
    break
  fi
  sleep 2
done

# One streaming request against the mock engine.
TRACE_ID=$(python3 -c "import secrets; print(secrets.token_hex(16))")
curl -s -H "x-trace-id: ${TRACE_ID}" \
     -H "content-type: application/json" \
     -d '{"model":"mock-model","prompt":"hello","max_tokens":32,"stream":true,
          "workload_class":"chat"}' \
     http://localhost:8001/v1/chat/completions \
     > /tmp/smoke.out

sleep 20   # recording-rule evaluation + remote-write flush

# Assertion 1: Prometheus has a TTFT observation.
TTFT_SAMPLES=$(curl -sG 'http://localhost:9090/api/v1/query' \
  --data-urlencode 'query=sum(rate(inference_ttft_seconds_count[1m]))' \
  | python3 -c "import sys,json; print(float(json.load(sys.stdin)['data']['result'][0]['value'][1]))")
if awk "BEGIN {exit !(${TTFT_SAMPLES} > 0)}"; then
  echo "OK: Prometheus received TTFT observations (${TTFT_SAMPLES}/s)"
else
  echo "FAIL: no TTFT observations in Prometheus"
  exit 1
fi

# Assertion 2: Tempo has the trace.
TRACE_RESP=$(curl -s "http://localhost:3200/api/traces/${TRACE_ID}")
if echo "${TRACE_RESP}" | grep -q 'batches'; then
  echo "OK: Tempo returned the trace for ${TRACE_ID}"
else
  echo "FAIL: trace ${TRACE_ID} not found in Tempo"
  exit 1
fi

# Assertion 3: Grafana datasources reachable.
curl -sf http://admin:admin@localhost:3000/api/datasources/name/Prometheus > /dev/null
curl -sf http://admin:admin@localhost:3000/api/datasources/name/Tempo > /dev/null
echo "OK: Grafana datasources provisioned and reachable"

echo "smoke test PASSED"
```

A pass is the complete three-way roundtrip: engine → OTLP → Collector →
(Prom + Tempo) → Grafana datasources. Any failure localises to the step
that returned non-zero.

## 3. Validation steps

A reviewer validates a learner submission by running the following, in order.

1. **Stack comes up.** `docker compose up -d` reaches a clean state within
   60 s; `/ready` endpoints on Prometheus, Tempo, and Grafana all return 200.
2. **Prometheus targets are UP.** `/targets` lists the engine job, the
   collector self-metrics job, and Prometheus itself, all `up=1`.
3. **Recording rules evaluate cleanly.** `promtool check rules` exits 0;
   `promtool query instant` on each recorded series returns non-empty.
4. **Collector receives OTLP.** `otelcol_receiver_accepted_spans_total`
   and `otelcol_receiver_accepted_metric_points_total` increase after a
   request is fired.
5. **Dashboards render.** The three seed dashboards load; each panel has
   non-empty data after the smoke test.
6. **Traces are searchable in Tempo.** The known trace ID from the smoke
   test returns a trace with the expected `engine.queue`, `engine.prefill`,
   `engine.decode` spans.
7. **Exemplar click-through.** On the TTFT heatmap, clicking an exemplar
   lands in Tempo on the corresponding trace. Any missing exemplar wire is
   a rubric-partial.
8. **Smoke script exits 0.** `./smoke.sh` runs the three assertions and
   prints `smoke test PASSED`.

## 4. Rubric / review checklist

Score each item 0 (missing), 1 (partial), 2 (fully correct). Pass: ≥ 14 / 20.

| # | Item | Points |
|---|---|---|
| 1 | Topology diagram names every service and port; metrics / traces / logs flows separate | 2 |
| 2 | `docker-compose.yml` brings the stack up in < 60 s; versions pinned; volumes persist | 2 |
| 3 | Collector pipelines split metrics / traces (/ logs); redaction on trace pipeline | 2 |
| 4 | Prometheus scrapes engine + collector self-metrics; remote-write receiver enabled | 2 |
| 5 | Recording rules compute TTFT / TPOT / ITL / E2E percentiles; `promtool` passes | 2 |
| 6 | Grafana datasources auto-provisioned; three seed dashboards imported | 2 |
| 7 | Exemplar wiring from Prometheus histograms to Tempo traces | 2 |
| 8 | Tempo receives OTLP from the Collector; traces searchable | 2 |
| 9 | Smoke script exercises engine → Prom + Tempo → Grafana and asserts data flow | 2 |
| 10 | README documents bring-up, teardown, and how to point a real engine at the Collector | 2 |

## 5. Common mistakes

- **Pushing metrics to Prometheus remote-write without enabling the
  receiver.** Prometheus refuses writes unless started with
  `--web.enable-remote-write-receiver`; the Collector exporter silently
  retries, no data lands, and the dashboards are empty.
- **Collector buffering OOMs.** Omitting `memory_limiter` makes the Collector
  the first thing to die under a trace storm. Always first in the processor
  chain.
- **Running the engine on the host but the Collector in the compose network.**
  The engine cannot resolve `otel-collector`. Either run the engine on the
  compose network (`docker run --network obs-net …`) or point its OTLP
  endpoint at `host.docker.internal:4317` with the matching compose
  `extra_hosts` line.
- **Dashboards query raw histograms.** Every panel calls
  `histogram_quantile(...)` inline instead of referencing the recording
  rules. Panels crawl; Prometheus load climbs. Use the recorded series.
- **Labels missing on scrape.** `pool`, `engine`, `model`, `workload_class`
  must be added at scrape time (or by the Collector's resource processor).
  Adding them later by relabel_config is possible but makes recording-rule
  joins brittle.
- **Grafana datasource UIDs not stable.** If a dashboard references
  `datasource: "${DS_PROMETHEUS}"` instead of the provisioned UID, imports
  break. Pin the UID in the provisioning file and reference it from
  dashboards.
- **Exemplars enabled in Prometheus but not emitted by the engine.**
  Exemplars require the engine wrapper to attach a `trace_id` to each
  observation. Without that, the exemplar wiring in Grafana is dead.
  Exercise 01's instrumentation does this; verify it is on before relying on
  click-through.
- **Local storage with no retention.** The default Tempo config will fill
  the dev disk; set `max_block_duration` and local retention appropriate
  to the review scenario.
- **Secrets in compose.** `GF_SECURITY_ADMIN_PASSWORD=admin` is a dev
  convenience and must not leak to any shared environment. Document this in
  the README; move to a proper secrets store before any non-local use.
- **No README.** The stack is useless to a reviewer who cannot find the
  smoke script and the bring-up command. A short README that lists the
  three commands (`up`, `smoke`, `down`) is mandatory.

## 6. References

- OpenTelemetry Collector docs — <https://opentelemetry.io/docs/collector/>
  — receiver / processor / exporter model; exporter list for
  `prometheusremotewrite`, `otlp`, `loki`.
- OpenTelemetry Semantic Conventions. *Generative AI.*
  <https://opentelemetry.io/docs/specs/semconv/gen-ai/> — the attribute
  names the engine instrumentation emits.
- Prometheus. *Configuration.*
  <https://prometheus.io/docs/prometheus/latest/configuration/configuration/>
  — scrape config; remote-write receiver flag.
- Prometheus. *Exemplars.*
  <https://prometheus.io/docs/prometheus/latest/feature_flags/#exemplars-storage>
  — feature flag and storage model.
- Prometheus. *Histogram practices.*
  <https://prometheus.io/docs/practices/histograms/> — bucket design and
  `histogram_quantile` idiom used in §2.4.
- Grafana Tempo. *Overview.*
  <https://grafana.com/docs/tempo/latest/> — ingester / distributor; the
  minimal config in §2.6.
- Grafana. *Provisioning.*
  <https://grafana.com/docs/grafana/latest/administration/provisioning/>
  — datasources, dashboards, UID pinning.
- Grafana. *Trace to metrics / exemplars.*
  <https://grafana.com/docs/grafana/latest/datasources/tempo/configure-tempo-data-source/>
  — wiring exemplars from Prometheus to Tempo.
- Google SRE book. *Monitoring distributed systems.*
  <https://sre.google/sre-book/monitoring-distributed-systems/> — four
  golden signals that the dashboards in §2.5 instantiate for token streaming.
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/)
  — ML06 Output Integrity and ML04 Denial-of-Service: the redaction and
  memory-limiter choices in §2.3 are concrete mitigations for the
  observability plane's exposure to these risks.
- [mitre-atlas](https://atlas.mitre.org/) — detection patterns applicable
  to a model-serving telemetry pipeline; the Collector is the enforcement
  surface for most of them.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework) —
  the "Measure" function; this base stack is the measurement substrate for
  every other SLO, cost, and incident exercise in the module.
