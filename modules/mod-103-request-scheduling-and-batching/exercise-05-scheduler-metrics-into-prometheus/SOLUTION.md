# Solution — Exercise 05: Scheduler Metrics into Prometheus

> Paired learning file: [`lessons/mod-103-request-scheduling-and-batching/exercises/exercise-05-scheduler-metrics-into-prometheus.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-103-request-scheduling-and-batching/exercises/exercise-05-scheduler-metrics-into-prometheus.md)

This exercise lands the observability bundle for the scheduler: a running Prometheus scraping the engine, five dashboards covering the five SLO axes, five alert rules, and — critically — a verification log that *each alert has actually been provoked and seen to fire*. An un-provoked alert is an un-trusted alert.

Metric names below are the vLLM-shaped names used in chapter 05. **Verify the exact spellings against your pinned engine release's `/metrics` output** — vLLM has renamed and reshaped metrics multiple times across releases, and the precise set depends on whether the engine was launched with the V0 or V1 engine core.
---

## 1. Solution overview

Five artefacts:

| Artefact | Form | Pass bar |
|---|---|---|
| A — Prometheus scrape config | `prometheus.yml` fragment + engine launch command | Running stack; `/metrics` returns ≥ 15 families; labels applied at scrape time |
| B — Five dashboards | Grafana JSON (or screenshots) | Each dashboard ≥ 4 panels; every panel references an existing metric |
| C — Five alert rules | Prometheus rules file | Each has `for:`, `severity`, `annotations.runbook`, standard labels |
| D — Alert verification log | One log entry per rule | Each alert *fired* under a named provocation and *resolved* cleanly |
| E — Observability cheatsheet | 1 page | Reusable by another operator; names the three biggest metric-drift risks |

The whole bundle is reproducible via a docker-compose (Prometheus + Grafana, optionally Alertmanager) + rules files + dashboard JSON. Another engineer should be able to bring it up in under 15 minutes.

---

## 2. Implementation

### 2.1 Artefact A — Prometheus scrape config (worked)

`prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'inference-engine'
    scrape_interval: 10s
    metrics_path: /metrics
    static_configs:
      - targets:
          - 'localhost:8001'
        labels:
          engine: vllm
          pool: chat-prod-us
          model: llama-3.1-70b-instruct
          region: local-dev
          tenant: internal

rule_files:
  - /etc/prometheus/rules/scheduler-alerts.yml

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - 'localhost:9093'    # optional; omit if Alertmanager is not deployed
```

`docker-compose.yml` fragment (reference; drop Alertmanager if not needed):

```yaml
version: "3"
services:
  prometheus:
    image: prom/prometheus:latest
    ports: ["9090:9090"]
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./rules:/etc/prometheus/rules:ro
    extra_hosts:
      - "host.docker.internal:host-gateway"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=7d'
  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./grafana-dashboards:/var/lib/grafana/dashboards
  alertmanager:
    image: prom/alertmanager:latest
    ports: ["9093:9093"]
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
```

Engine launch command alongside (reference — vLLM):

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 8 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 16384 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --swap-space 16 \
  --gpu-memory-utilization 0.92 \
  --port 8001
```

Verify the metrics endpoint comes up and reports a healthy family count:

```bash
curl -s http://localhost:8001/metrics | grep -E "^# HELP" | wc -l
# Expect ≥ 15 for a modern vLLM; actual count varies by release and config.
```

Capture a `curl http://localhost:8001/metrics | head -40` snapshot into Artefact A. The snapshot is the ground truth for metric spellings used in Artefacts B and C.

### 2.2 Artefact B — five dashboards (panel specifications)

Grafana JSON is long; the review artefact is the *panel specification* list below plus the exported JSON file. Verify that each PromQL expression evaluates against the learner's `/metrics` snapshot before publishing.

Metric names below use the vLLM-shaped spelling. For SGLang / TRT-LLM substitute the equivalents; confirm against engine docs.
#### Dashboard 1 — Scheduler overview

| Panel | PromQL (template; verify against engine release) |
|---|---|
| Queue depth | `vllm:num_requests_waiting{pool="$pool"}` |
| Running set | `vllm:num_requests_running{pool="$pool"}` |
| Admission rate | `rate(vllm:request_success_total{pool="$pool"}[1m])` |
| Preemption rate | `rate(vllm:num_preemption_tokens_total{pool="$pool"}[1m]) / clamp_min(rate(vllm:request_success_total{pool="$pool"}[1m]), 1)` |
| Iteration time distribution | `histogram_quantile(0.95, sum(rate(vllm:time_per_output_token_seconds_bucket{pool="$pool"}[5m])) by (le))` (as a proxy when a direct iteration-time histogram isn't exposed) |

#### Dashboard 2 — Admission / queue dynamics

| Panel | PromQL |
|---|---|
| Queue depth over time | `vllm:num_requests_waiting` |
| Arrival vs. admission rate | `rate(vllm:request_prompt_tokens_sum[1m])` vs. `rate(vllm:request_generation_tokens_sum[1m])` (or arrival counter if exposed) |
| Time-in-queue histogram P50/P95/P99 | `histogram_quantile(0.5|0.95|0.99, sum(rate(vllm:time_in_queue_seconds_bucket[5m])) by (le))` |
| Request-aborted rate | `rate(vllm:request_error_total[5m])` (or the engine's equivalent counter) |

#### Dashboard 3 — Prefill vs. decode mix

| Panel | PromQL |
|---|---|
| Prefill fraction | `rate(vllm:prompt_tokens_total[1m]) / (rate(vllm:prompt_tokens_total[1m]) + rate(vllm:generation_tokens_total[1m]))` |
| Chunked-prefill chunk count | counter as exposed by engine; verify |
| Iteration-tokens-per-step distribution | `histogram_quantile(0.95, sum(rate(vllm:iteration_tokens_bucket[5m])) by (le))` (verify presence) |
| TPOT P50/P95 | `histogram_quantile(0.5|0.95, sum(rate(vllm:time_per_output_token_seconds_bucket[5m])) by (le))` |

#### Dashboard 4 — Preemption and KV pressure

| Panel | PromQL |
|---|---|
| Preemption rate | `rate(vllm:num_preemption_tokens_total[5m])` |
| GPU cache usage | `vllm:gpu_cache_usage_perc` |
| CPU cache (swap-pool) usage | `vllm:cpu_cache_usage_perc` |
| Swap-in / swap-out counts | `rate(vllm:swap_out_tokens_total[1m])`, `rate(vllm:swap_in_tokens_total[1m])` (verify names) |

#### Dashboard 5 — SLO roll-up

| Panel | PromQL |
|---|---|
| TTFT P50/P95/P99 | `histogram_quantile(0.5|0.95|0.99, sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le))` |
| TPOT P50/P95/P99 | `histogram_quantile(0.5|0.95|0.99, sum(rate(vllm:time_per_output_token_seconds_bucket[5m])) by (le))` |
| E2E P95 | `histogram_quantile(0.95, sum(rate(vllm:e2e_request_latency_seconds_bucket[5m])) by (le))` |
| Request-aborted rate | `rate(vllm:request_error_total[5m])` |
| SLO compliance | TTFT P95 compared to SLO constant; TPOT P95 compared to SLO constant (two single-stat panels with thresholds) |

Export each dashboard's JSON into the solutions repo (or capture screenshots for the artefact).

### 2.3 Artefact C — five alert rules (worked `scheduler-alerts.yml`)

```yaml
# scheduler-alerts.yml
# Companion to serving-config/chat-prod-us.yaml (exercise 02 pinned config).
groups:
  - name: scheduler-slo
    interval: 30s
    rules:
      - alert: TTFT_P95_Breach
        expr: |
          histogram_quantile(
            0.95,
            sum by (le, pool, engine) (
              rate(vllm:time_to_first_token_seconds_bucket[5m])
            )
          ) > 1.5
        for: 5m
        labels:
          severity: page
          slo_axis: ttft
          pool: chat-prod-us
          engine: vllm
          model: llama-3.1-70b-instruct
        annotations:
          summary: "TTFT P95 above SLO for pool {{ $labels.pool }}"
          description: "TTFT P95 is {{ $value | humanizeDuration }} (SLO 1.5s)."
          runbook: "https://runbooks.internal/inference/ttft-breach.md"

      - alert: TPOT_P95_Breach
        expr: |
          histogram_quantile(
            0.95,
            sum by (le, pool, engine) (
              rate(vllm:time_per_output_token_seconds_bucket[5m])
            )
          ) > 0.050
        for: 5m
        labels:
          severity: page
          slo_axis: tpot
          pool: chat-prod-us
          engine: vllm
          model: llama-3.1-70b-instruct
        annotations:
          summary: "TPOT P95 above SLO for pool {{ $labels.pool }}"
          description: "TPOT P95 is {{ $value | humanizeDuration }} (SLO 50ms)."
          runbook: "https://runbooks.internal/inference/tpot-breach.md"

      - alert: Preemption_Thrash
        expr: |
          (
            rate(vllm:num_preemption_tokens_total[5m])
            /
            clamp_min(rate(vllm:prompt_tokens_total[5m]) + rate(vllm:generation_tokens_total[5m]), 1)
          ) > 0.02
        for: 5m
        labels:
          severity: page
          slo_axis: preemption
          pool: chat-prod-us
          engine: vllm
        annotations:
          summary: "Preemption rate sustained above 2% on {{ $labels.pool }}"
          description: "Preempted-token fraction {{ $value | humanizePercentage }} over 5m."
          runbook: "https://runbooks.internal/inference/preemption-thrash.md"

      - alert: Queue_Time_P95_Breach
        expr: |
          histogram_quantile(
            0.95,
            sum by (le, pool) (rate(vllm:time_in_queue_seconds_bucket[5m]))
          ) > 1.0
        for: 5m
        labels:
          severity: page
          slo_axis: queue_time
          pool: chat-prod-us
          engine: vllm
        annotations:
          summary: "Queue time P95 above 1s for pool {{ $labels.pool }}"
          description: "Queue-time P95 is {{ $value | humanizeDuration }}."
          runbook: "https://runbooks.internal/inference/queue-backup.md"

      - alert: KV_Cache_Saturation
        expr: vllm:gpu_cache_usage_perc{pool="chat-prod-us"} > 0.9
        for: 10m
        labels:
          severity: ticket
          slo_axis: kv_saturation
          pool: chat-prod-us
          engine: vllm
        annotations:
          summary: "KV cache usage > 90% sustained on {{ $labels.pool }}"
          description: "GPU cache usage {{ $value | humanizePercentage }} for 10m."
          runbook: "https://runbooks.internal/inference/kv-saturation.md"
```

Notes for the learner:

- Metric names are vLLM-shaped and version-sensitive. Before committing, run `promtool check rules scheduler-alerts.yml` and then `promtool query instant http://localhost:9090 '<expr>'` for each `expr:` field; any expression that returns empty is wrong.
- `for: 5m` is the practical minimum for SLO alerts. `for: 0s` flaps on every scrape jitter.
- Each rule carries a runbook URL. The runbook can be a stub page at first — a stub is still infinitely more useful than nothing when the page fires at 3am.
- Severity convention: `page` wakes someone; `ticket` goes to a backlog.
- Standard labels: `pool`, `engine`, `model`, `severity`, `slo_axis` — chapter 05's hygiene list. Add `region`, `tenant` at the first production rollout.

### 2.4 Artefact D — worked verification log

For each alert, the log entry is: timestamp, provocation command, metric value at firing, confirmation of firing (and of resolution). Templates below; the learner's log has real timestamps and real values.

```markdown
## TTFT_P95_Breach — verified 2026-XX-XX HH:MM

Provocation: relaunch engine with `--max-num-batched-tokens=1024` (below the long-prompt
P95), run `load_gen.py --rps 20 --duration-s 600 --seed 0` with 20% long prompts.

Observation window: 10 min.
- Dashboard 5 TTFT P95 panel climbed from ~980 ms to ~2300 ms by t=4 min.
- `promtool query instant` on the alert expression at t=5m: 2.31 (seconds).
- Prometheus `/alerts` showed TTFT_P95_Breach in `firing` state at t=5m02s with labels
  {pool=chat-prod-us, engine=vllm, severity=page, slo_axis=ttft}.
- Alertmanager delivered a webhook with the runbook URL
  https://runbooks.internal/inference/ttft-breach.md — opened, served, HTTP 200.

Reversion: returned `--max-num-batched-tokens=16384` (exercise-02 pinned value).
- TTFT P95 fell back to ~980 ms by t=13m.
- Alert cleared (`firing` → `inactive`) at t=14m30s — within `for: 5m` + a few scrape
  intervals of reversion. Clean resolve.
```

Repeat the pattern for the other four alerts; the provocations are listed in the paired learning file:

- **TPOT_P95_Breach**: launch with `--enable-chunked-prefill=false` + long-prompt workload → TPOT P95 rises → alert fires.
- **Preemption_Thrash**: run the exercise-04 overcommit workload with `--max-num-seqs 16` → preemption rate crosses 2% → alert fires.
- **Queue_Time_P95_Breach**: run `load_gen.py --rps 40` (2× pinned admission rate) → arrival queue backs up → alert fires.
- **KV_Cache_Saturation**: long-prompt + long-output workload sustains > 90% GPU cache → alert fires at t=10m. Note the longer `for:` window.

Each run must *resolve* cleanly on reversion. An alert that fires but doesn't resolve is as bad as one that doesn't fire; the bug is usually in the PromQL (missing / mismatched labels in the `rate()` window) and must be fixed before shipping.

### 2.5 Artefact E — worked observability cheatsheet (one page)

```
=== Scheduler Observability Cheatsheet — <engine>@<version> on <pool> ===

Launch with metrics on:
  vllm serve <model> ... --port 8001          (or engine equivalent)

Point Prometheus at it:
  static_configs: targets=['<host>:8001'], labels={pool,engine,model,region,tenant}
  Verify: curl :8001/metrics | wc -l          (expect dozens of metric families)

Five dashboards (Grafana folder: inference/scheduler-<pool>):
  1. Scheduler overview        (queue, running, admit, preempt, iter-time)
  2. Admission / queue dynamics (depth, arrivals, time-in-queue, aborts)
  3. Prefill vs. decode mix     (prefill fraction, iter-tokens dist, TPOT)
  4. Preemption and KV pressure (preempt rate, GPU/CPU cache, swap counts)
  5. SLO roll-up                (TTFT, TPOT, E2E, aborts, compliance)

Five alerts (file: scheduler-alerts.yml):
  TTFT_P95_Breach      -> dashboard 5 TTFT P95
  TPOT_P95_Breach      -> dashboard 5 TPOT P95
  Preemption_Thrash    -> dashboard 4 preempt-rate
  Queue_Time_P95_Breach -> dashboard 2 time-in-queue
  KV_Cache_Saturation  -> dashboard 4 GPU cache usage

Three biggest metric-drift risks across engine upgrades:
  1. vLLM renames and reshapes metrics (e.g. `vllm:` prefix / no prefix variance
     across releases; preemption counter moved from an event count to a token
     count in some versions). Verify names with every upgrade.
  2. SGLang / TRT-LLM use different spellings (e.g. no `vllm:` prefix,
     different histogram bucket boundaries). Dashboards and rules must be
     per-engine.
  3. Older engine releases may be missing metrics the alerts depend on
     (e.g. no time-in-queue histogram). Audit the upgrade's /metrics
     before the upgrade lands.
```

Pinned alongside the exercise-01 inner-loop cheatsheet; refreshed on every engine upgrade.

---

## 3. Validation steps

A reviewer checks:

1. **Stack comes up.** Bring up docker-compose, confirm Prometheus `/targets` shows the engine job UP, Grafana loads all five dashboards, every panel renders with data.
2. **Metric names resolve.** For every PromQL expression in the alert rules and dashboards, run it against Prometheus `/api/v1/query`; no empty results on a running engine. A panel that renders empty is a metric-name typo or a missing label.
3. **`promtool` passes.** `promtool check rules scheduler-alerts.yml` exits 0. Mandatory before merge.
4. **Alerts fire under provocation.** Re-run each of the five provocations in a scratch environment; confirm the Prometheus `/alerts` endpoint shows the expected rule in `firing` state within the `for:` window. If any alert does *not* fire, the rule is wrong.
5. **Alerts resolve.** After reversion, each alert returns to `inactive` within `for:` + two scrape intervals. If one stays stuck, the expression's label or `rate()` window is wrong.
6. **Runbook URLs resolve.** Every alert's `annotations.runbook` URL returns HTTP 200 (even to a stub page).
7. **Cheatsheet one page.** Artefact E prints to one page in a reasonable rendering. Longer than that, it stops being a cheatsheet.

---

## 4. Rubric or review checklist

Pass threshold: ≥ 12 / 18.

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| Scrape config and labels | Missing labels | Has pool/engine but not region/tenant | pool/engine/model/region/tenant applied at scrape |
| Metric families count | < 10 | 10–14 | ≥ 15 metric families scraped |
| Dashboards count and depth | < 5 dashboards | 5 dashboards, some < 4 panels | 5 dashboards, each ≥ 4 panels |
| Metric-name verification | Dashboards reference metrics that don't exist | Verified for some panels | Every panel's metrics confirmed in `/metrics` snapshot |
| Alert rules count and completeness | < 5 rules | 5 rules, one missing `for:` / `runbook` / labels | 5 rules each with `for:`, `severity`, `runbook`, standard labels |
| `promtool` passes | Fails | Passes with warnings | Passes clean |
| Alert verification coverage | 0–2 alerts provoked | 3–4 alerts provoked | All 5 provoked with metric-value evidence |
| Alert resolution check | None | Some resolve | All resolve cleanly in-window |
| Reproducibility | Stack is undocumented | docker-compose present but not standalone | docker-compose + rules + dashboards bring up in < 15 min |

---

## 5. Common mistakes

- **Alert names look right; expressions silently never fire.** PromQL typos (wrong metric prefix, wrong histogram bucket name, missing `sum by (le,...)`) are easy; the only honest check is provocation.
- **Metric drift across engine versions.** vLLM has renamed metrics (prefix changes, moved from event-count to token-count for preemption, histogram bucket boundary changes). Verify every name against the installed version's `/metrics` output before writing the alert.- **Dashboards with empty panels.** A metric-name typo renders an empty panel. Fix at authoring time; a dashboard that is 30% empty will be ignored.
- **`for: 0s` alerts.** Pages flap on every 10s scrape jitter. 5 minutes is the practical minimum for SLO alerts; `KV_Cache_Saturation` uses 10 minutes to avoid transient-burst pages.
- **Missing runbook links.** On-call clicks "runbook" when the page fires. If the URL is absent or 404s, the page is just noise.
- **Alerts tested only in staging, never provoked.** The verification step is non-negotiable; an un-provoked alert is an un-trusted alert.
- **Not silencing during drill runs.** Exercise 04 provokes real preemption, which trips `Preemption_Thrash`. Mute Alertmanager for the drill window, or route the drill to a dev engine.
- **Label hygiene neglected.** Missing `pool`, `engine`, `model` labels at scrape time is the lazy choice that costs the most at year 2 of fleet scale. Apply from day 1.
- **Alert expression without `sum by (le, ...)`.** `histogram_quantile` over a multi-pod engine without grouping by `le` returns nonsense. Standard idiom: `histogram_quantile(0.95, sum by (le, pool) (rate(..._bucket[5m])))`.

---

## 6. References

Prometheus / Grafana / Alertmanager documentation:

- Prometheus docs — <https://prometheus.io/docs/>.
- Prometheus histogram practices (`histogram_quantile`) — <https://prometheus.io/docs/practices/histograms/>.
- Alertmanager docs — <https://prometheus.io/docs/alerting/latest/alertmanager/>.
- Grafana docs — <https://grafana.com/docs/>.
- `promtool` reference (bundled with Prometheus) — see Prometheus release notes.

SLO / alerting methodology:

- SRE book — Service Level Objectives — <https://sre.google/sre-book/service-level-objectives/>.
- SRE workbook — Alerting on SLOs — <https://sre.google/workbook/alerting-on-slos/>.
- OpenTelemetry docs (for future transport unification) — <https://opentelemetry.io/docs/>.

GPU observability (companion to the scheduler metrics):

- NVIDIA DCGM — <https://docs.nvidia.com/datacenter/dcgm/>.
- DCGM Exporter for Prometheus — <https://github.com/NVIDIA/dcgm-exporter>.
- `nvidia-smi` reference — <https://docs.nvidia.com/deploy/nvidia-smi/>.
- AMD equivalents — <https://rocm.docs.amd.com/>.

Engine metrics references (verify against pinned release):

- vLLM metrics reference — <https://docs.vllm.ai/en/latest/serving/metrics.html>.
- SGLang docs — <https://docs.sglang.ai/> (search for "metrics").
- TensorRT-LLM metrics — <https://nvidia.github.io/TensorRT-LLM/> (search for "metrics" / "stats").
- Triton inference server metrics — <https://docs.nvidia.com/deeplearning/triton-inference-server/>.

Paired learning content:

- Module 103 chapter 05 (*Instrumenting the Scheduler*) in [`ai-infra-inference-engineer-learning`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning) — the five-axis instrumentation framework this exercise lands.
- Exercise 01 inner-loop cheatsheet — the structure Artefact E reuses.
- Exercise 02 solution — the pinned config and workload spec the alerts' SLO lines reference.
- Exercise 03 solution — the chunk-size setpoint whose tuning this dashboard makes visible.
- Exercise 04 solution — the preemption drill whose alert verification lands here.

Downstream: module 110 (*Observability, SLO, and Cost*) extends this bundle with burn-rate alerts, cost-per-token accounting, and cross-fleet dashboards.
