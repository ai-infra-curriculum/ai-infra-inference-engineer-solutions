# Solution — Exercise 03: Acceptance-Rate Monitoring and Alerts

> Paired learning file: [`lessons/mod-108-speculative-decoding-ops/exercises/exercise-03-acceptance-rate-monitoring-and-alerts.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-108-speculative-decoding-ops/exercises/exercise-03-acceptance-rate-monitoring-and-alerts.md)
> Module: `mod-108-speculative-decoding-ops`
> Type: implementation / observability

## 1. Solution overview

Speculative decoding only pays for itself when the **acceptance rate** `α` stays
above a workload-specific break-even point. This exercise wires that fact into
the production stack: scrape vLLM's speculative-decoding counters, derive a
rolling acceptance rate, chart it alongside tokens-per-second and p95 latency,
and alert when speculation stops earning its cost.

The four artefacts:

| Artefact | Form | Pass bar |
|---|---|---|
| A — Prometheus scrape config + a derived-metric recording rule | YAML | Scrapes `vllm:spec_decode_*` from each engine; recording rule computes `rate()`-based acceptance |
| B — Grafana (or equivalent) dashboard | JSON / screenshot | Four panels: α, emitted tokens/s, p95 latency, acceptance by request slice if available |
| C — Alert rules | YAML (Prometheus Alertmanager or equivalent) | At least three rules: low α, draft-starved, speculation-off-silently |
| D — Runbook stub | Markdown | One-page, named steps, links to the paired dashboard and to [[draft-model-lifecycle-contract]] |

### What "acceptance rate" means — and the trap

The vLLM counters you scrape are **monotonic counters**, not instantaneous
gauges:

| Counter | Meaning |
|---|---|
| `vllm:spec_decode_num_draft_tokens` | Total tokens the draft proposed since engine start. |
| `vllm:spec_decode_num_accepted_tokens` | Total draft tokens the target accepted. |
| `vllm:spec_decode_num_emitted_tokens` | Total tokens actually emitted to clients (accepted drafts + bonus tokens). |
| `vllm:spec_decode_draft_acceptance_rate` | Engine-reported rolling acceptance rate (gauge). Verify presence and window against the pinned vLLM release. |

The **operationally correct** acceptance rate is a derived metric:

```
α_window = increase(vllm:spec_decode_num_accepted_tokens[5m])
         / increase(vllm:spec_decode_num_draft_tokens[5m])
```

Computed from cumulative counters over a rolling window. Charting the engine's
built-in gauge alone is tempting but hides two failure modes:

- The gauge may be a lifetime average (resets only at engine restart), which
  masks recent regressions behind a stable history.
- If the draft silently stops firing (e.g., head failed to load, vLLM fell back
  to no-spec on a specific request path), `num_draft_tokens` goes flat and the
  gauge is undefined or stale. A derived-metric rate-of-rate explicitly surfaces
  this.

Scrape the counters; derive `α_window`; alert on the derived metric. Keep the
engine's gauge as a secondary signal only.

### The break-even inequality (so alert thresholds are defensible)

From Leviathan et al. (ICML 2023), speculative decoding pays for itself when the
expected tokens accepted per round exceed the overhead of the draft and
verification steps combined. For spec length `k` and draft/target cost ratio
`c = T_draft / T_target`, the break-even condition is approximately:

```
E[accepted | α, k] = (1 − α^(k+1)) / (1 − α)   must exceed   1 + k*c
```

The right-hand side captures the "wasted work" of drafting `k` tokens even when
some are rejected. For a well-matched same-family draft where `c ≈ 0.05–0.10` and
`k = 5`, break-even sits roughly near `α ≈ 0.35–0.45`. Measure `c` on your
hardware before pinning a numeric threshold; the alert should fire somewhat
above break-even to leave headroom for workload drift.

Pin the alert threshold per deployment, not as a global constant. A 70 B target
with a 1 B draft on H100s has a different break-even than a 7 B target with a
500 M draft on L4s.

## 2. Implementation — worked walkthrough

The commands below assume:

- A vLLM engine with speculative decoding enabled per exercise 01 (or exercise
  02 for a self-speculative method).
- A Prometheus scraper reachable from the engine.
- A Grafana (or equivalent) instance pointed at that Prometheus.
- A Loki, Alertmanager, or OpsGenie / PagerDuty sink for alerts. The YAML below
  uses Prometheus Alertmanager syntax; translate to the local sink as needed.

### 2.1 Prometheus scrape config (artefact A)

The engine's `/metrics` endpoint is already live. Scrape it with a short interval
and give the job a stable label so the derived metric and alerts stay
resilient to engine restarts.

```yaml
# prometheus.yml (fragment)
scrape_configs:
  - job_name: 'vllm-spec-decode'
    scrape_interval: 15s
    scrape_timeout: 10s
    static_configs:
      - targets:
          - 'inference-0.internal:8001'
          - 'inference-1.internal:8001'
        labels:
          cluster: 'prod-us-east'
          engine: 'vllm'
          spec_method: 'draft-target'   # or 'medusa' / 'eagle'
          target_model: 'llama-3.1-70b-instruct'
          draft_model: 'llama-3.2-1b-instruct'
```

Recording rule that produces the derived acceptance rate:

```yaml
# recording_rules.yml
groups:
  - name: vllm-spec-decode-derived
    interval: 30s
    rules:
      - record: vllm:spec_decode_accepted_per_draft:ratio5m
        expr: |
          sum by (cluster, target_model, draft_model) (
            rate(vllm:spec_decode_num_accepted_tokens[5m])
          )
          /
          clamp_min(
            sum by (cluster, target_model, draft_model) (
              rate(vllm:spec_decode_num_draft_tokens[5m])
            ),
            1e-9
          )

      - record: vllm:spec_decode_emitted_per_draft:ratio5m
        expr: |
          sum by (cluster, target_model, draft_model) (
            rate(vllm:spec_decode_num_emitted_tokens[5m])
          )
          /
          clamp_min(
            sum by (cluster, target_model, draft_model) (
              rate(vllm:spec_decode_num_draft_tokens[5m])
            ),
            1e-9
          )

      - record: vllm:spec_decode_draft_tokens_per_sec:rate5m
        expr: |
          sum by (cluster, target_model, draft_model) (
            rate(vllm:spec_decode_num_draft_tokens[5m])
          )
```

`clamp_min` prevents a division-by-zero when the draft is silently off; the
derived α stays finite and the "draft starved" alert can fire on the rate instead.

### 2.2 Grafana dashboard (artefact B)

Four panels, laid out in a 2×2 grid. Keep panel names in the exact form below so
the alert rules in §2.3 can be traced back to the dashboard that led to them.

1. **Acceptance rate — `α_window`**. Series:
   `vllm:spec_decode_accepted_per_draft:ratio5m`. Y-axis 0–1. Threshold lines
   at the measured break-even and at the alert threshold. Legend keyed on
   `{cluster, target_model, draft_model}`.
2. **Emitted tokens per draft token**. Series:
   `vllm:spec_decode_emitted_per_draft:ratio5m`. This is the "effective
   speed-up" metric; at `α = 0` it stays at 1 (the target still emits its own
   fallback token); at high α it climbs toward `k+1`.
3. **Draft throughput — tokens/sec**. Series:
   `vllm:spec_decode_draft_tokens_per_sec:rate5m`. Zero here means the draft is
   not firing; alert on this independently from α.
4. **p95 request latency** vs. **α**. Joint panel: p95 from the normal vLLM
   request-latency histogram on the left Y-axis; α on the right Y-axis. The
   operational story of speculation lives in whether these two curves move
   together.

Export the dashboard JSON alongside the solution so a reviewer can replay it.

### 2.3 Alert rules (artefact C)

Three alerts, in rough order of operational urgency:

```yaml
# alert_rules.yml
groups:
  - name: vllm-spec-decode
    rules:
      # Alert 1 — speculation is silently OFF. Draft tokens are not being produced
      # despite the engine being configured for speculation. This is the single
      # most dangerous failure because the engine still serves traffic; it is
      # simply slower than it should be, and dashboards still show p95 "within
      # SLO" so nothing else fires.
      - alert: SpecDecodeDraftStarved
        expr: |
          vllm:spec_decode_draft_tokens_per_sec:rate5m < 1
          and on (cluster, target_model) (
            # correlate with the engine serving non-trivial request traffic
            sum by (cluster, target_model) (rate(vllm:request_success_total[5m])) > 0.5
          )
        for: 10m
        labels:
          severity: warning
          runbook: 'https://runbooks.internal/vllm/spec-decode-draft-starved'
        annotations:
          summary: "Speculative decoding configured but draft not firing on {{ $labels.target_model }}"
          description: |
            Draft tokens/sec has been under 1 for 10 minutes while the engine is
            serving traffic. Speculation is effectively off; either the draft
            model failed to load, a request path is bypassing spec, or a recent
            vLLM upgrade changed flag semantics. Check engine logs and
            {{ $labels.cluster }} dashboard.

      # Alert 2 — acceptance rate below break-even. The engine is drafting
      # tokens that the target mostly rejects; speculation is now a net cost.
      # Threshold is deployment-specific — set from the measured break-even
      # (see §1 and §2.4) with 10 pts headroom.
      - alert: SpecDecodeAcceptanceBelowBreakEven
        expr: |
          vllm:spec_decode_accepted_per_draft:ratio5m < 0.45
          and vllm:spec_decode_draft_tokens_per_sec:rate5m > 10
        for: 15m
        labels:
          severity: warning
          runbook: 'https://runbooks.internal/vllm/spec-decode-low-acceptance'
        annotations:
          summary: "α={{ $value | humanize }} on {{ $labels.target_model }} below break-even"
          description: |
            Acceptance rate has sustained below the configured break-even
            threshold for 15 minutes. Either the workload shifted (new
            domain, longer prompts, different sampling), the draft drifted from
            the target (unpinned target revision), or the draft model
            regressed. Investigate before raising severity.

      # Alert 3 — acceptance rate collapsed. Much tighter threshold, shorter
      # window; this is the "something is actively broken" signal.
      - alert: SpecDecodeAcceptanceCollapsed
        expr: |
          vllm:spec_decode_accepted_per_draft:ratio5m < 0.20
          and vllm:spec_decode_draft_tokens_per_sec:rate5m > 10
        for: 3m
        labels:
          severity: page
          runbook: 'https://runbooks.internal/vllm/spec-decode-collapsed'
        annotations:
          summary: "SPEC DECODE COLLAPSED — α={{ $value | humanize }} on {{ $labels.target_model }}"
          description: |
            Acceptance rate has fallen below 0.20 for 3 minutes. The draft is
            actively hurting latency; the engine would be faster with
            speculation disabled. Page the on-call and consider toggling
            speculation off until root-caused.
```

Three deliberate design decisions worth defending:

- **Alert 1 is a rate-of-rate check**, not a derived-α check. A derived-α can
  be ill-defined when the draft is off; the explicit "draft tokens/sec is
  near-zero while the engine is serving traffic" condition catches the silent
  fallback.
- **Alerts 2 and 3 gate on `draft_tokens_per_sec > 10`**. Without this gate, a
  low-traffic engine will trip the acceptance alert the moment a few drafts
  get rejected. The gate silences rules during low-traffic troughs.
- **Alert 3 is page-grade, alert 2 is warning-grade**. An α of 0.4 when
  break-even is 0.45 is inefficient but not broken; an α of 0.2 means
  speculation is actively making things worse.

### 2.4 Measuring break-even on your hardware

The threshold in alert 2 should be derived, not copied. The procedure:

1. Take the same measurement artefact from exercise 01, but sweep
   `num_speculative_tokens ∈ {1, 3, 5}` and record both p50 latency and
   α for each.
2. Fit the Leviathan §3 inequality: find the α at which measured p50 (spec on)
   equals p50 (baseline) for your chosen `k`. That is break-even.
3. Set the alert threshold 10 points above break-even. Below break-even the
   engine is slower than baseline; the headroom gives the team time to
   investigate before a page.

### 2.5 Runbook stub (artefact D)

A one-page Markdown file, linked from each alert's `runbook` annotation. Short,
direct, named steps.

```markdown
# Runbook — Speculative Decoding Acceptance Issues

Linked alerts: SpecDecodeDraftStarved, SpecDecodeAcceptanceBelowBreakEven,
SpecDecodeAcceptanceCollapsed.

## 1. Confirm the alert

- Open the "vLLM Spec Decode" dashboard.
- Which clusters / targets / drafts are firing?
- Is `draft_tokens_per_sec` zero, non-zero-but-low, or normal?

## 2. Branch

### 2a. `draft_tokens_per_sec ≈ 0` (alert 1)

- Check engine logs for the most recent pod: does the "Speculative decoding
  initialized" line still appear?
- Check for a recent vLLM upgrade: was a spec-decode flag renamed? See the
  pinned-release notes.
- Check that the draft checkpoint is reachable (model registry / shared FS).
- If silently off: redeploy with speculation forced on; do NOT treat "baseline
  latency returned" as resolution.

### 2b. `α` low but non-zero (alert 2 or 3)

- Has the workload shifted? Compare current prompt-length and sampling
  parameters to the last stable window.
- Has the target model been updated? Draft checkpoints are pinned to target
  revisions — see [[draft-model-lifecycle-contract]]. If target was rolled and
  draft was not, pin acceptance to the mismatch.
- Has sampling temperature changed? High temperature reduces α.
- As a mitigation: lower `num_speculative_tokens` by 2; α tends to recover at
  smaller k. If α does not recover, toggle speculation off for the affected
  target and open an incident.

## 3. Rollback / mitigation

- Disable speculation for the affected target by redeploying with
  `--speculative-config` omitted. Document the rollback in the incident ticket.
- Keep the draft-model lifecycle contract in mind: do not silently swap in a
  new draft checkpoint without governance review.

## 4. Postmortem

- Record the α trajectory, workload change (if any), and the resolution.
- Update the break-even threshold if the hardware/model mix has materially
  changed.
```

## 3. Validation steps

A reviewer validating a learner submission should perform the following in
order.

1. **Scrape config present and correct.** Prometheus config includes the
   `vllm-spec-decode` job with labels for `cluster`, `target_model`, and
   `draft_model` (or `spec_method` for Medusa/EAGLE).
2. **Recording rule correct.** The derived `α_window` uses
   `rate(accepted) / rate(draft)` over a stable window, with `clamp_min` on
   the denominator so a draft-off state does not produce NaN panels.
3. **Dashboard panels match §2.2.** All four panels exist and the α panel
   draws a threshold line at the break-even derived in §2.4.
4. **Alert rules match §2.3.** Three rules (starved, below-break-even,
   collapsed) are present. The two acceptance alerts gate on draft rate.
5. **Thresholds justified.** The break-even threshold in alert 2 is derived
   from a measurement, not copied; the submission references the sweep that
   produced it.
6. **Runbook linked.** Each alert's annotation points to a runbook URL, and
   the runbook has named branches for "draft starved" vs. "low acceptance".
7. **Smoke test.** The learner shows one of:
   - A deliberately-wrong draft (cross-family) triggers alert 3 within 3 min.
   - A deliberately-stopped draft process triggers alert 1 within 10 min.
   - A screenshot of the dashboard during a successful run with α visibly
     tracking expected band.

## 4. Rubric / review checklist

Score each item 0 (missing), 1 (partial), 2 (fully correct). Passing: ≥ 10 / 14.

| # | Item | Points |
|---|---|---|
| 1 | Scrape config includes stable labels (cluster, target_model, draft_model, spec_method) | 2 |
| 2 | Recording rule computes α as `rate(accepted)/rate(draft)` with `clamp_min` | 2 |
| 3 | Dashboard has α, emitted/draft ratio, draft tokens/sec, and p95-vs-α panels | 2 |
| 4 | Three alert rules present, severities distinct (warning, warning, page) | 2 |
| 5 | Break-even threshold derived from a measurement, not copied | 2 |
| 6 | Runbook linked from each alert and has "draft starved" vs. "low α" branches | 2 |
| 7 | Smoke test demonstrates at least one alert firing on an injected fault | 2 |

## 5. Common mistakes

- **Alerting on the engine's lifetime-average α gauge.** A lifetime average
  changes very slowly; by the time it crosses your threshold, the fault has
  already been happening for a long time. Always derive α from the counters
  over a rolling window.
- **No gate on draft rate.** Without gating, a quiet engine at low QPS will
  produce noisy acceptance alerts during idle periods. The gate (`draft_rate >
  10 tokens/sec` or similar) silences rules when speculation is simply not
  being exercised.
- **One global threshold.** Different target/draft pairings and different
  hardware have different break-even points. A threshold copied from one
  deployment to another is likely wrong in both directions.
- **Missing the "silently off" failure.** If a vLLM upgrade quietly changes a
  flag name or moves speculation behind a different path, the engine may run
  at baseline latency without anyone noticing — the only signal is that
  `num_draft_tokens` goes flat. Alert 1 is specifically for this.
- **Treating low α as a bug in speculation.** Low α is a signal, not a cause.
  Causes include workload drift, draft-target version mismatch, sampling
  temperature change, or domain shift (chat deployment started receiving code
  prompts). Diagnosis is in the runbook, not in the alert.
- **Pausing speculation as the first action.** Toggling speculation off is a
  mitigation, not a resolution. It hides the symptom and makes the postmortem
  harder. Diagnose first; toggle second.
- **Forgetting to update break-even after a hardware refresh.** Replacing H100s
  with H200s or moving from 70 B to a different target class changes `c` (the
  draft/target cost ratio) and therefore break-even. Re-measure after any
  hardware or model-size change.
- **Scraping `/metrics` through a cache or ingress that re-labels.** Prometheus
  relies on counter continuity; a cache that resets labels on engine restart
  will produce rate spikes that look like α collapses. Scrape the engine
  directly.

## 6. References

- Leviathan, Y., Kalman, M., Matias, Y. *Fast Inference from Transformers via
  Speculative Decoding.* ICML 2023. https://arxiv.org/abs/2211.17192 —
  break-even inequality in §3.
- vLLM documentation. *Metrics / observability.* https://docs.vllm.ai/ —
  `vllm:spec_decode_*` counter definitions; verify counter names against the
  pinned release.
- Prometheus. *Recording rules.*
  https://prometheus.io/docs/prometheus/latest/configuration/recording_rules/ —
  reference syntax.
- Prometheus. *Alerting rules.*
  https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/ —
  reference syntax; `for:` semantics.
- Grafana. *Panels.* https://grafana.com/docs/grafana/latest/panels-visualizations/
  — dashboard construction reference.
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/)
  — ML06 Output Integrity and related items; silent speculation fallback is an
  integrity-adjacent failure mode (engine produces correct tokens but not at
  advertised performance), covered as a lifecycle concern in
  [[draft-model-lifecycle-contract]] (exercise 04).
- [mitre-atlas](https://atlas.mitre.org/) — monitoring and detection patterns
  for model-serving systems.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework) —
  "Measure" function; acceptance-rate monitoring is a concrete instance of
  continuous measurement for a deployed inference system.
