# Solution — Exercise 02: SLO Authoring per Workload Class

> Paired learning file: [`lessons/mod-110-observability-slo-and-cost/exercises/exercise-02-slo-authoring-per-workload-class.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-110-observability-slo-and-cost/exercises/exercise-02-slo-authoring-per-workload-class.md)
> Module: `mod-110-observability-slo-and-cost`
> Type: design / decision drill

## 1. Solution overview

A single global "p95 latency < X seconds" SLO is the most common failure mode
of inference observability: it over-protects batch summarization (which does
not care about TTFT at all) and under-protects code completion (where a 500 ms
TTFT is unusable). The exercise teaches **SLO-per-workload-class authoring**:
name the workloads, pick the SLIs that matter *for that workload*, choose
targets grounded in user research or a comparable public baseline, and
attach a burn-rate alerting policy that pages the right people at the right
time.

The three deliverables:

| Artefact | Form | Pass bar |
|---|---|---|
| A — Workload-class catalogue | Table | Six (or more) classes defined, each with a user-visible contract, traffic profile, and streaming / non-streaming shape |
| B — SLO table per class | Table | Each class has SLIs on TTFT, TPOT or ITL, E2E, availability; targets justified by user research or a cited reference; 28-day windows; error-budget policy |
| C — Burn-rate alert set | Prometheus rules YAML | Multi-window multi-burn-rate alerts per SLO, mapping to severity and runbook |

All three artefacts assume the decomposition and instrumentation from
[[exercise-01-ttft-tpot-itl-decomposition-and-trace-wiring]] is in place —
without named, measured signals, SLO authoring is fiction.

### What an SLO is (and isn't)

Per the Google SRE workbook, an SLO is:

- **An SLI** (service-level indicator): a measurable signal, framed as
  "fraction of events that met a threshold within a window."
- **A target**: the acceptable fraction (e.g., 99.0 %).
- **A compliance window**: calendar or rolling, typically 28 days for
  user-facing services.
- **An error budget**: `1 − target` × total events over the window. The
  budget is explicitly spendable.
- **A policy**: what the team does when the budget is burning too fast
  (page, freeze releases, prioritize paydown, etc.).

An SLO is *not* a latency threshold; the threshold is the SLI boundary. An
SLO is *not* an alert; the alert is a burn-rate signal that the budget is
being consumed too fast to make the window.

### Why per-workload class, not per-tenant or per-endpoint

Workload class is the correct axis because:

- TTFT / TPOT / ITL targets depend on **user interaction shape**, not on
  tenant identity. A chat endpoint used by Tenant A and Tenant B needs the
  same TTFT.
- Per-endpoint SLOs balloon the policy space; most endpoints are one of
  ~6 classes.
- Capacity planning and scheduler tuning (modules 103, 104, 107) operate at
  the workload-class level.

Tenant-level agreements (business-level SLAs) layer on top of the
workload-class SLOs; they are not a substitute.

## 2. Implementation — worked answer

### 2.1 Workload-class catalogue (artefact A)

Six classes cover the common cases. Treat this as the starter set; a real
deployment may collapse or split classes. Reasoning follows each row.

| Class | User-visible contract | Streaming | Typical prompt | Typical output | Traffic shape |
|---|---|---|---|---|---|
| **chat** | Interactive conversational UI; user reads as the model types | Yes | 0.5–4 k tokens | 100–800 tokens | Steady with daily peaks; TTFT must feel immediate |
| **rag** | Chat-like UI with retrieved context; same reading pace | Yes | 4–32 k tokens (with retrieved docs) | 200–1 000 tokens | Steady; prefill-heavy; prefix-cache hits common |
| **agent-tool** | Agent orchestrator calling the model to pick a tool, then act on the result | Usually no (needs full response to parse) | 1–8 k tokens | 50–300 tokens | Bursty; many sequential short calls per task |
| **code-completion** | IDE inline completion; user is typing | Yes (optional) | 1–8 k tokens (context) | 10–100 tokens | Spiky; TTFT dominates perceived quality |
| **batch-summarize** | Offline queue; summaries ready "by morning" | No | 2–64 k tokens | 200–2 000 tokens | Bulk; latency per request matters only loosely |
| **embedding** | Vector produced for downstream search / RAG indexing | N/A | 0.1–8 k tokens | 1 vector per input | Mixed; often high QPS small inputs |

Rationale, briefly:

- **chat** and **rag** share a streaming contract but have very different
  prefill shapes; separate because TTFT budget is dominated by prefill.
- **agent-tool** is not streaming because the orchestrator parses the full
  response before acting; TTFT is irrelevant, E2E is what matters.
- **code-completion** is the extreme TTFT class; users abandon the suggestion
  if it does not appear within one or two keystrokes.
- **batch-summarize** trades latency for throughput; SLO is about
  queue drain, not per-request speed.
- **embedding** has no decode loop; TTFT / TPOT / ITL do not apply — latency
  SLO is E2E only.

### 2.2 SLO table per class (artefact B)

Targets below are **worked examples**; a learner must justify their numbers
with user research, a public baseline (e.g., the chat competitor's measured
TTFT), or a measured baseline from their own deployment. Marking an SLO
"copied from the workbook" without justification is a rubric-zero.

All SLOs use a **28-day rolling window** unless noted. Availability is a
success-fraction SLI (`successful_requests / total_requests`), where
"successful" excludes 5xx, request timeouts, and aborted streams; 4xx caused
by client input is **not** a service error and does not spend budget.

| Class | TTFT SLO | TPOT / ITL SLO | E2E SLO | Availability SLO |
|---|---|---|---|---|
| **chat** | 99.0 % of requests have TTFT ≤ 1.5 s | 95 % of requests have ITL p95 ≤ 80 ms within the request | 99.0 % E2E ≤ 15 s for `max_tokens ≤ 1024` | 99.9 % successful |
| **rag** | 99.0 % TTFT ≤ 3.0 s (prefill-heavy; scales with prompt length) | 95 % ITL p95 ≤ 100 ms | 99.0 % E2E ≤ 30 s for `max_tokens ≤ 1024` | 99.9 % |
| **agent-tool** | — (non-streaming) | — | 99.0 % E2E ≤ 4.0 s for `max_tokens ≤ 256` | 99.95 % (agents retry — tight availability matters more than tight latency) |
| **code-completion** | 99.5 % TTFT ≤ 500 ms | 99 % ITL p99 ≤ 50 ms | 99.5 % E2E ≤ 2.0 s | 99.9 % |
| **batch-summarize** | — | — | 95 % of jobs complete within their enqueue→done SLO (e.g., ≤ 30 min from submission) | 99.5 % |
| **embedding** | — | — | 99.0 % E2E ≤ 300 ms for input ≤ 2 k tokens | 99.95 % |

Notes on the choices:

- **TTFT percentile (99.0 vs. 99.5).** Code completion runs at 99.5 % because
  the user *immediately* notices a slow suggestion; chat and rag run at
  99.0 % because occasional slow starts are tolerated when the response is
  being read.
- **ITL percentile (p95 vs. p99).** p99 is used for code completion because
  stalls inside a 60-token completion are unacceptable; chat / rag use p95
  because the reader absorbs occasional hiccups.
- **E2E conditioning on `max_tokens`.** Without a token-count gate, the E2E
  SLO becomes "infinity if any user requests `max_tokens=16384`." Condition
  the SLO on a token-count bucket, or report E2E / token instead of raw E2E.
- **Availability vs. latency.** Agent-tool and embedding have *tighter
  availability* and *looser latency* than chat. Agents retry on failure, so
  a single slow response is forgiven; a 500 is not.

### 2.2.1 Error budget policies

The SLOs above imply error budgets:

| Class | Monthly latency budget (fraction out of SLO) | Monthly availability budget |
|---|---|---|
| chat | 1.0 % of requests may miss TTFT | 0.1 % of requests may fail |
| rag | 1.0 % | 0.1 % |
| agent-tool | 1.0 % | 0.05 % |
| code-completion | 0.5 % | 0.1 % |
| batch-summarize | 5.0 % | 0.5 % |
| embedding | 1.0 % | 0.05 % |

Policy (same for all user-facing classes, relaxed for batch):

1. **Budget ≥ 50 % remaining**: normal release cadence, no action.
2. **Budget 10–50 % remaining**: freeze non-critical config changes on the
   affected pool; prioritize paydown of known contributors.
3. **Budget ≤ 10 % remaining**: release freeze, incident declared, engineering
   effort re-aimed at closing the budget for the window.
4. **Budget exhausted**: post-incident required; the SLO and workload class
   are reviewed in the next week's SLO review.

Batch classes use the same ladder but the top-of-window action is "notify
batch owner" rather than "freeze release."

### 2.3 Burn-rate alert set (artefact C)

Multi-window multi-burn-rate alerts (SRE workbook, "Alerting on SLOs") are
the canonical method. The intuition: an alert that fires only when the
whole budget is gone is useless; an alert that fires at every scrape
flaps. The compromise is **two burn rates × two windows** per severity:

- **Fast burn (page)**: a 2 % budget burn sustained over a short window
  (typically 1 h and 5 m). Mathematically: `error_rate ≥ 14.4 × budget_rate`
  over 1 h **and** over 5 m. A 1 h burn at this rate exhausts a 30-day budget
  in ~2 days.
- **Slow burn (ticket)**: a 5 % budget burn sustained over a longer window
  (6 h and 30 m). Rate ≥ 6 × budget rate.

The 14.4 × and 6 × multipliers come from the SRE workbook's reference table
for 30-day windows; reconfirm the exact multipliers against the workbook if
the window length changes.

Worked rules for the **chat** class TTFT SLO (99.0 % at 1.5 s, 28-day window):

```yaml
# slo-chat-ttft.rules.yml
groups:
  - name: slo-chat-ttft-burn-rate
    interval: 30s
    rules:
      # Recording rules — compute the "fraction out of SLO" at several windows.
      - record: slo:chat_ttft_bad_ratio:5m
        expr: |
          (
            sum by (pool, engine, model) (
              rate(inference_ttft_seconds_bucket{
                workload_class="chat", le="+Inf"}[5m])
            )
            -
            sum by (pool, engine, model) (
              rate(inference_ttft_seconds_bucket{
                workload_class="chat", le="1.5"}[5m])
            )
          )
          /
          clamp_min(
            sum by (pool, engine, model) (
              rate(inference_ttft_seconds_bucket{
                workload_class="chat", le="+Inf"}[5m])
            ),
            1e-9
          )

      - record: slo:chat_ttft_bad_ratio:1h
        expr: |
          # same shape over [1h]
          (
            sum by (pool, engine, model) (
              rate(inference_ttft_seconds_bucket{
                workload_class="chat", le="+Inf"}[1h])
            )
            -
            sum by (pool, engine, model) (
              rate(inference_ttft_seconds_bucket{
                workload_class="chat", le="1.5"}[1h])
            )
          )
          /
          clamp_min(
            sum by (pool, engine, model) (
              rate(inference_ttft_seconds_bucket{
                workload_class="chat", le="+Inf"}[1h])
            ),
            1e-9
          )

      - record: slo:chat_ttft_bad_ratio:30m
        expr: |
          (
            sum by (pool, engine, model) (
              rate(inference_ttft_seconds_bucket{
                workload_class="chat", le="+Inf"}[30m])
            )
            -
            sum by (pool, engine, model) (
              rate(inference_ttft_seconds_bucket{
                workload_class="chat", le="1.5"}[30m])
            )
          )
          /
          clamp_min(
            sum by (pool, engine, model) (
              rate(inference_ttft_seconds_bucket{
                workload_class="chat", le="+Inf"}[30m])
            ),
            1e-9
          )

      - record: slo:chat_ttft_bad_ratio:6h
        expr: |
          (
            sum by (pool, engine, model) (
              rate(inference_ttft_seconds_bucket{
                workload_class="chat", le="+Inf"}[6h])
            )
            -
            sum by (pool, engine, model) (
              rate(inference_ttft_seconds_bucket{
                workload_class="chat", le="1.5"}[6h])
            )
          )
          /
          clamp_min(
            sum by (pool, engine, model) (
              rate(inference_ttft_seconds_bucket{
                workload_class="chat", le="+Inf"}[6h])
            ),
            1e-9
          )

      # Budget target: 1 − SLO = 1 − 0.99 = 0.01
      # Fast-burn: 14.4 × 0.01 = 0.144 — pages when sustained over 1h AND 5m
      - alert: SLOChatTTFTFastBurn
        expr: |
          slo:chat_ttft_bad_ratio:1h > (14.4 * 0.01)
          and
          slo:chat_ttft_bad_ratio:5m > (14.4 * 0.01)
        labels:
          severity: page
          slo: chat_ttft
          workload_class: chat
          slo_axis: ttft
        annotations:
          summary: "Chat TTFT burning budget fast on {{ $labels.pool }}"
          description: |
            TTFT bad-ratio over 1h = {{ $value | humanizePercentage }},
            budget rate 1 %, fast-burn threshold 14.4×. At this rate a 30-day
            budget is exhausted in ~2 days.
          runbook: "https://runbooks.internal/inference/chat-ttft-fast-burn.md"

      # Slow-burn: 6 × 0.01 = 0.06 — tickets when sustained over 6h AND 30m
      - alert: SLOChatTTFTSlowBurn
        expr: |
          slo:chat_ttft_bad_ratio:6h > (6 * 0.01)
          and
          slo:chat_ttft_bad_ratio:30m > (6 * 0.01)
        labels:
          severity: ticket
          slo: chat_ttft
          workload_class: chat
          slo_axis: ttft
        annotations:
          summary: "Chat TTFT budget burning over 6h on {{ $labels.pool }}"
          description: |
            Sustained slow burn — budget will exhaust the 30-day window
            within ~5 days if unaddressed.
          runbook: "https://runbooks.internal/inference/chat-ttft-slow-burn.md"
```

Replicate the pattern for each (class × SLI) pair. The repeated shape —
record `bad_ratio` at multiple windows, alert on fast + slow burn — is
exactly what the SRE workbook prescribes. Keep the burn-rate multipliers
and the percentage target as variables in the rules file so only two numbers
change per SLO.

#### Two design decisions worth defending

- **Burn-rate alerts gate on two windows, not one.** A single-window burn
  rate flaps on arrival-rate spikes (a one-minute burst of slow requests
  from a single client). Requiring the burn rate to be sustained over both a
  long and a short window prevents the flap while keeping the alert fast.
- **Severity is derived from burn rate, not from latency magnitude.** A
  12 × burn that is *just under* the 14.4 × page threshold does not page even
  if TTFT is 10 s — the budget is not in danger. A 14.4 × burn pages even if
  TTFT is only slightly above 1.5 s — the budget *is* in danger.

### 2.4 How to derive a target when there is no baseline

A rubric-safe target is one of:

1. **Measured user behaviour.** Instrument the client to report "user
   abandoned stream" events; set TTFT at the latency at which abandonment
   rises above baseline.
2. **A published competitor latency.** For consumer chat: "our TTFT must not
   be worse than $COMPETITOR's advertised value by more than X."
3. **Hardware-floor reasoning.** For code completion: the smallest
   round-trip (ingress + tokenization + 1 prefill step) on the pinned
   hardware is ~N ms; the SLO target is some margin above that.

Copy-pasting a number from a different workload or a different hardware
generation is not a justification.

## 3. Validation steps

A reviewer validates a learner submission by running the following, in
order.

1. **Catalogue coverage.** Artefact A has ≥ 6 classes; each row names the
   user-visible contract, streaming / non-streaming, and traffic shape.
2. **SLI alignment per class.** Non-streaming classes (agent-tool,
   embedding, batch-summarize) have no TTFT / ITL SLOs. Streaming classes
   have both TTFT and ITL (or TPOT) SLOs. Mis-assignment is a rubric-zero.
3. **Targets justified.** Every numeric target has a cited justification
   (user research, competitor baseline, measured floor, workbook default
   with rationale). No bare numbers.
4. **Windows and budgets stated.** Every SLO names its window and its
   explicit error budget as a percentage *and* as an absolute count
   (requests/month) at a documented traffic baseline.
5. **Error-budget policy present.** Artefact B includes a four-step policy
   tied to budget remaining, with a named owner for each step.
6. **Burn-rate alerts use two windows.** Artefact C has at least one fast-burn
   (page) and one slow-burn (ticket) rule per SLO, each gating on two
   window lengths.
7. **Alert labels cross-reference the SLO.** Every alert carries `slo`,
   `workload_class`, `slo_axis`, `severity`, and a `runbook` annotation.
8. **`promtool check rules` passes.** Artefact C validates clean.

## 4. Rubric / review checklist

Score each item 0 (missing), 1 (partial), 2 (fully correct). Pass: ≥ 12 / 16.

| # | Item | Points |
|---|---|---|
| 1 | Workload catalogue: ≥ 6 classes with contract, streaming shape, traffic profile | 2 |
| 2 | SLI selection per class: TTFT / TPOT / ITL only where they apply; E2E and availability always | 2 |
| 3 | Numeric targets each have a cited justification | 2 |
| 4 | Explicit compliance window (e.g., 28 days) and error budget per SLO | 2 |
| 5 | Error-budget policy in four steps with owners | 2 |
| 6 | Multi-window multi-burn-rate alerts per SLO (one fast-burn page, one slow-burn ticket) | 2 |
| 7 | Alert labels and annotations (`slo`, `workload_class`, `slo_axis`, `severity`, `runbook`) | 2 |
| 8 | `promtool check rules` passes on artefact C | 2 |

## 5. Common mistakes

- **A single global latency SLO.** The top-of-page failure mode. Different
  workloads deserve different numbers; one global number is wrong for all of
  them.
- **TTFT SLO on a non-streaming class.** Agent-tool and embedding have no
  user-visible first token; a TTFT SLO on them is a category error.
- **Targets copied from the SRE workbook without translation.** 99.9 %
  availability for a chat endpoint sounds safe; 99.9 % of a billion monthly
  requests is still a million failures. Always convert to absolute counts.
- **Unconditional E2E SLO.** A fixed E2E ceiling is unreachable for
  `max_tokens=16384`; condition on token count or report E2E / token.
- **Single-window burn-rate alerts.** They flap. The workbook's two-window
  construction is deliberate.
- **Severity assigned by latency magnitude, not burn rate.** A 10 s TTFT
  outlier that happens once an hour is noise; a 1.7 s outlier that happens
  on 15 % of traffic burns the budget in a day. Severity follows budget
  consumption.
- **Treating 4xx as budget-spending failures.** Client-input errors do not
  spend the service's error budget. Only 5xx, timeouts, and aborted streams
  do.
- **No error-budget policy.** An SLO without a policy is a number on a
  dashboard; the policy — what the team *does* when the budget burns — is
  what makes it operational.
- **No per-class dashboards.** If the SLOs are per class and the dashboards
  are not, the on-call cannot see which class is bleeding. Dashboards and
  SLOs must share the same partitioning.
- **Including degraded output in the "success" SLI.** A streaming response
  that returned 500 output tokens where 50 expected is a successful HTTP but
  a questionable product outcome. If this failure mode matters, add a
  *quality* SLO (e.g., response-length distribution) alongside the latency
  ones; do not quietly include them in availability.

## 6. References

- Google SRE book. *Service Level Objectives.*
  <https://sre.google/sre-book/service-level-objectives/> — the SLI / SLO /
  error-budget framework this exercise applies.
- Google SRE workbook. *Alerting on SLOs.*
  <https://sre.google/workbook/alerting-on-slos/> — multi-window multi-burn-rate
  alerting; source of the 14.4× and 6× multipliers for 30-day windows.
- Google SRE workbook. *Implementing SLOs.*
  <https://sre.google/workbook/implementing-slos/> — practical worked
  examples; error-budget policy ladder.
- Prometheus. *Recording rules.*
  <https://prometheus.io/docs/prometheus/latest/configuration/recording_rules/>
  — pre-computing bad-ratios at multiple windows.
- Prometheus. *Alerting rules.*
  <https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/>
  — rule syntax; `for:` semantics.
- OpenAI usage guidance for serving models — <https://platform.openai.com/docs>
  — public reference for interactive-latency contracts; cite workloads whose
  SLOs you match to this baseline.
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/)
  — availability-adjacent concerns: a tenant-level DoS (ML04) consumes error
  budget as quickly as an infrastructure outage.
- [mitre-atlas](https://atlas.mitre.org/) — detection patterns; sustained
  burn-rate alerts are the service-level instance of ATLAS "performance
  degradation" detections.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework) —
  "Measure" and "Manage" functions; SLOs operationalise measurement, and
  the error-budget policy operationalises management.
