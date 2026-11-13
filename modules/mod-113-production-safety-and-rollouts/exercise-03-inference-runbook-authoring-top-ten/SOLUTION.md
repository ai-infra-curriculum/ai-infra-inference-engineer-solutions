# Solution — Exercise 03: Inference Runbook Authoring — the Top-Ten

> Paired learning file: [`lessons/mod-113-production-safety-and-rollouts/exercises/exercise-03-inference-runbook-authoring-top-ten.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-113-production-safety-and-rollouts/exercises/exercise-03-inference-runbook-authoring-top-ten.md)
> Module: `mod-113-production-safety-and-rollouts`
> Type: design / documentation

## 1. Solution overview

A runbook is a pre-written response to a specific alert. The job it does is
to shrink the on-call engineer's decision space to a short, ordered set of
steps that have been reviewed in calm conditions. An inference platform
accumulates a handful of failure modes that are specific enough to deserve
their own runbooks — not generic "page the model team," but concrete first-
steps that reflect how vLLM / SGLang / downstream services actually fail.

This exercise authors the **top ten** inference runbooks, chosen because
each has a distinct detection signal and a distinct first action. The ten
cover the common surfaces: the engine, the cache, the queue, the model
artefact, the downstream fanout, and the rollout machinery. For each, we
write:

- **Alert / symptom** — what the on-call sees.
- **Initial check** — the first question to answer, and how (dashboards,
  logs, kubectl).
- **Diagnostic branch** — the two or three directions the initial check can
  send you.
- **Mitigation** — the short-term action that restores the SLO.
- **Rollback / kill switch** — the escape hatch if mitigation does not take.
- **Follow-up** — what gets filed (ticket, RCA trigger, backlog item).

The ten below are expressed as a table of contents plus a representative
full runbook. A submission must include all ten at the depth shown in §2.3.

### Scope and tone

Runbooks are **short**. A page a human reads at 3am should fit on one or
two screens. Theory lives in the module's reading, not in the runbook.
Where a step requires judgement, name the judgement explicitly ("if latency
is still regressed after mitigation, escalate to the Model Owner — do not
continue trying variants from this runbook").

Runbooks are **owned**. Each has an owning rotation (platform, model,
router) and a last-reviewed date. An unowned runbook is a stale runbook.

Runbooks are **versioned**. The runbook text lives next to the code / IaC
that caused the alert, so a change to the alert threshold touches both.

## 2. Implementation — worked answer (the top ten)

### 2.1 Table of contents

| # | Runbook | Primary alert | Owning rotation | Links |
|---|---|---|---|---|
| 1 | TTFT SLO burn | `ttft_p95_ms > SLO for 5m` | Platform | [[exercise-01-canary-with-response-diffing-and-auto-rollback]] |
| 2 | Decode latency / TPOT regression | `tpot_p95_ms > SLO for 5m` | Platform | [[exercise-01-canary-with-response-diffing-and-auto-rollback]] |
| 3 | Error rate: 5xx on engine | `engine_5xx_rate > 1% for 1m` | Platform | — |
| 4 | OOM / KV-cache exhaustion | `cuda_oom_total > 0` or `kv_block_pool_free == 0` | Platform | — |
| 5 | Prefix / prompt-cache collapse | `cache_hit_rate < floor for 10m` | Platform | — |
| 6 | Queue saturation / shed traffic | `queue_depth > ceiling` or admission drops > 0 | Platform | — |
| 7 | Model-artefact integrity failure at startup | engine startup fails with SHA mismatch | Model Owner | [[exercise-02-blue-green-model-swap-walkthrough]] |
| 8 | Canary regression (response-diff bad-fraction) | `diff_bad_fraction > threshold` | Rollout Lead | [[exercise-01-canary-with-response-diffing-and-auto-rollback]] |
| 9 | Downstream consumer errors post-swap | `downstream_consumer_error_rate > 1%` | Rollout Lead | [[exercise-02-blue-green-model-swap-walkthrough]] |
| 10 | Eval-gate CI failure on a candidate release | eval-gate CI job red | Model Owner | [[exercise-05-eval-gate-contract-with-peer-tracks]] |

### 2.2 Shared template

Every runbook uses this skeleton; keep it under one page.

```markdown
# Runbook: <short name>

- **Alert**: <exact alert name and query>
- **Owning rotation**: <platform | model | rollout>
- **Last reviewed**: YYYY-MM-DD
- **Related runbooks**: <list>
- **Related dashboards**: <list>

## Symptom

<one sentence: what the on-call sees>

## Initial check (≤ 2 minutes)

1. ...
2. ...

## Diagnostic branch

- If <signal A>: go to <section>
- If <signal B>: go to <section>
- Else: <escalate | fall through to mitigation>

## Mitigation

1. ...
2. ...
3. Verify: <what changes on the dashboard to call it resolved>

## Rollback / kill switch

- <single-action command or UI step>
- Authority: <who may trigger>

## Follow-up

- [ ] File: <ticket | RCA trigger | backlog item>
- [ ] Update runbook: <date, revision>
```

### 2.3 Representative full runbook — #4, OOM / KV-cache exhaustion

```markdown
# Runbook: OOM / KV-cache exhaustion

- **Alert**: `cuda_oom_total > 0` within 5m, or `vllm:num_requests_waiting` rising while `vllm:gpu_cache_usage_perc` ≈ 1.0
- **Owning rotation**: Platform
- **Last reviewed**: 2026-11-12
- **Related runbooks**: #2 (TPOT regression), #5 (prefix-cache collapse)
- **Related dashboards**: Inference / GPU memory, Inference / request queue

## Symptom

Requests start timing out or 5xx-ing; GPU cache utilisation near 100%;
sometimes CUDA OOM errors in the engine log.

## Initial check (≤ 2 minutes)

1. Open the GPU memory dashboard. Confirm `gpu_cache_usage_perc` ≈ 1.0 and
   `num_requests_waiting` ≠ 0. If neither, this is not a cache-exhaustion
   event; see runbook #3.
2. Check whether the workload mix has shifted: the long-context cohort may
   have spiked. The dashboard panel "requests by max_tokens bucket" is the
   place to look.
3. Confirm the engine version: a recent engine upgrade that changed the
   paged-attention block size will show up here before anywhere else.

## Diagnostic branch

- **Workload mix shifted toward long context**: a few users or a specific
  endpoint suddenly submitting 32k-token requests. Mitigation = admission
  shed at the long-context endpoint.
- **Cache-usage high at normal workload**: engine config regressed.
  `--max-model-len`, `--gpu-memory-utilization`, or `--block-size` changed
  in the last rollout. Mitigation = revert the engine config.
- **Both normal**: check `vllm:num_preemptions_total` — a spike in
  preemptions with normal workload is the model artefact size changing
  (someone shipped a bigger checkpoint than the manifest claimed).

## Mitigation

1. If workload-shift: enable per-endpoint request-size cap, documented in
   the router config under `endpoint_caps`. Keep the cap in place until the
   long-context cohort is isolated to its own fleet.
2. If engine-config regression: revert the engine config to the prior
   known-good manifest. This is a router flag flip if the prior fleet is
   still running (blue-green), else an engine redeploy.
3. If artefact size regression: roll the artefact back per
   [[exercise-02-blue-green-model-swap-walkthrough]] procedure; model owner
   paged.
4. Verify: `gpu_cache_usage_perc` drops below 0.95; `num_requests_waiting`
   trends to zero; p95 TTFT recovers within 5 minutes.

## Rollback / kill switch

- Manifest revert PR template: `ops/rollback-engine-config`.
- Authority: on-call engineer (unilateral).
- Full fleet restart is not the default. Restart as a last resort only
  after mitigations above have been attempted.

## Follow-up

- [ ] File: incident ticket, even if resolved cleanly. OOM events are
      always RCA-worthy per [[exercise-04-mock-incident-rca-and-postmortem]].
- [ ] If workload-shift: open a backlog item for a long-context fleet
      split.
- [ ] If engine-config regression: open the rollout retrospective.
```

### 2.4 Summary of the other nine (what the full submission contains)

Each follows the same template as §2.3. The abbreviated scope table below
is enough to show what a passing submission covers.

**#1 TTFT SLO burn** — initial check is p95 TTFT dashboard; branch is
cold-start (fleet recently autoscaled, prefix-cache cold) vs model-side
regression (new revision landed) vs upstream (router or admission queue).
Mitigation: for cold-start, warm the fleet with synthetic load; for model
regression, roll back per [[exercise-02-blue-green-model-swap-walkthrough]];
for upstream, follow runbook #6.

**#2 Decode latency / TPOT regression** — initial check is TPOT panel
alongside batch-size and continuous-batching occupancy. Branch is batch
collapse (one replica stuck on a long-decode request), speculative-decoding
acceptance-rate drop
([[acceptance-rate-monitoring-and-alerts]] from mod-108), or quantisation
format mismatch. Mitigation: evict the stuck replica; fall back to non-
speculative decoding by router flag.

**#3 Engine 5xx** — initial check is log tail by error class. Branch is
request-format regression (client upgrade sent new schema), engine-internal
exception, or OOM-adjacent (feeds runbook #4). Mitigation: client-side
rollback if request-format; engine restart with saved config if internal.

**#5 Prefix / prompt-cache collapse** — initial check is cache-hit-rate
panel, cross-referenced against prompt-template version. Branch is a system-
prompt rotation (new prompt invalidated the cache) vs a cache eviction
storm (new workload pushed out warm entries). Mitigation: for template
rotation, confirm expected; for eviction, raise cache budget or isolate the
noisy workload.

**#6 Queue saturation / traffic shed** — initial check is admission-queue
depth and token-budget admission panel. Branch is incoming RPS exceeding
provisioned capacity vs admission threshold set too tight. Mitigation:
engage autoscaler; relax admission until stable; if RPS is a sustained step
up, trigger the capacity-expansion workflow.

**#7 Model-artefact integrity failure at startup** — initial check: look
for SHA mismatch between the registry entry and the computed checkpoint
SHA. Branch is a swapped-upstream artefact (hub republished) vs a corrupt
local cache. Mitigation: refuse startup and page the Model Owner; never
"patch over" by re-pinning the SHA to the new value. This is the
[[exercise-02-blue-green-model-swap-walkthrough]] manifest-pin control
firing as designed.

**#8 Canary regression** — initial check: open the canary dashboard from
exercise 01 (diff mean, bad-fraction, floor breaches). Branch: if hard
rollback trigger, the controller has already rolled back; the on-call
confirms. If soft, Rollout Lead + Model Owner decide. Mitigation: ensure
rollback took effect at the router; open the forensic-replay ticket.

**#9 Downstream consumer errors post-swap** — initial check: downstream
error rate panel plus a sampled request/response trace. Branch: schema
divergence (tool-call or structured-output format changed), token-budget
divergence (new model is more verbose), semantic divergence (downstream
parser expects a specific phrasing). Mitigation: roll the inference swap
back per exercise 02; downstream fix follows on its own cadence.

**#10 Eval-gate CI failure on a candidate release** — initial check: open
the eval-gate run from exercise 05; identify which peer-track suite
failed. Branch: suite failure is a known-regression on acceptable axes
(e.g., cost went up, quality held) vs a hard regression on a blocking
axis. Mitigation: hold the release; Model Owner responds. This is not a
production page but a release-gate page; keep the escalation path shorter.

### 2.5 Operating principles the ten share

- **Two-minute initial check.** A runbook that opens with "read the engine
  logs" fails. Open a dashboard, confirm the symptom, branch.
- **Single-action mitigation where possible.** If mitigation requires a
  code change, name the PR template; if it requires a redeploy, name the
  prior-revision manifest; if it requires a config flag, name the flag.
- **Named authority.** Every unilateral action names who may take it.
  Implicit authority creates "I didn't know I could" during incidents.
- **Follow-up is non-optional.** The runbook that resolves cleanly without
  a ticket is the runbook whose root cause will fire again next week.
- **Each runbook links to related runbooks and external procedures.** The
  on-call never dead-ends.

## 3. Validation steps

A reviewer checking a learner submission should:

1. **Count ten.** Fewer than ten is incomplete; a submission that combines
   TTFT and TPOT into one runbook misses the fact that their first checks
   are different panels and their branches diverge.
2. **Each runbook fits on one screen.** Open the rendered Markdown; if it
   scrolls past two screens, the runbook is actually a design doc.
3. **Initial check is a dashboard, not a log tail.** Reject any whose first
   step is "SSH to the pod."
4. **Mitigation is distinct from rollback.** Mitigation is the short-term
   restoration of SLO; rollback is the escape hatch. Collapsing them hides
   the point where the on-call stops trying and escalates.
5. **Named authority for unilateral actions.** Especially for runbooks #4,
   #7, #8, and #9, where rollback or kill-switch is likely.
6. **Each runbook has an owning rotation and last-reviewed date.** Unowned
   runbooks are a review-finding, not a suggestion.
7. **Cross-links to other exercises.** Runbooks #1, #2, #7, #8, #9, #10
   should link to the appropriate paired solutions.
8. **No invented metrics.** Metric names used in the alerts should match
   those a learner could find in the vLLM / SGLang documentation or
   collector exposition.

## 4. Rubric / review checklist

Score each item 0 / 1 / 2. Passing: ≥ 14 / 20.

| # | Item | Points |
|---|---|---|
| 1 | All ten runbooks present with named alerts and owning rotations | 2 |
| 2 | Shared template used consistently; each runbook fits one screen | 2 |
| 3 | Initial check is dashboard-first, not log-first | 2 |
| 4 | Mitigation distinct from rollback / kill switch in every runbook | 2 |
| 5 | Named authority for every unilateral action | 2 |
| 6 | Runbooks linked to related exercises (ex-01, ex-02, ex-05) | 2 |
| 7 | Follow-up section present in every runbook (ticket, RCA, backlog) | 2 |
| 8 | Alert queries use real engine / collector metric names | 2 |
| 9 | Review cadence stated (last-reviewed date; owning rotation reviews quarterly) | 2 |
| 10 | External framework mapping (§6) covers at least OWASP ML, ATLAS, NIST AI RMF once each | 2 |

## 5. Common mistakes

- **Runbook as reading material.** Theory, context, and "how the system
  works" belong in the lesson text. The runbook is a step-list the on-call
  executes. If a reader needs to understand a paper before they can
  mitigate, the runbook has failed its purpose.
- **Branching that leaves the reader stranded.** Every branch ends in
  either a mitigation step or an explicit escalation. "Else investigate
  further" is not a terminal node.
- **One mega-runbook for latency.** TTFT and TPOT are separately monitored
  because they fail for different reasons. Combining them produces a
  runbook that is wrong for both.
- **Kill switch buried in prose.** The kill-switch command belongs in a
  dedicated subsection with the single-line command and the named
  authority. Hidden kill switches do not get used.
- **Runbooks without owners.** An unowned runbook is a stale runbook; a
  stale runbook is worse than no runbook because the on-call still trusts
  it. Add rotation + last-reviewed date and require quarterly refresh.
- **No link to incident / RCA tooling.** The Follow-up section is the
  bridge between on-call response and
  [[exercise-04-mock-incident-rca-and-postmortem]]. Omitting it is how
  RCAs don't happen.
- **Invented alert queries.** A runbook whose alert query doesn't exist in
  the collector is a runbook whose alert never fires, which is the same as
  no runbook.
- **No cross-reference to rollout and eval-gate procedures.** The ten
  inference runbooks live in an ecosystem with the canary (ex-01), the
  blue-green walkthrough (ex-02), the eval-gate (ex-05), and the
  deprecation runbook (ex-06). The on-call reading one of the ten must be
  able to get to the next step without guessing.

## 6. References

- vLLM documentation — *Metrics* and *Performance* —
  https://docs.vllm.ai/ — source for the engine-side metric names
  referenced in alerts (e.g., `vllm:num_requests_waiting`,
  `vllm:gpu_cache_usage_perc`, `vllm:num_preemptions_total`).
- SGLang documentation — https://docs.sglang.ai/ — equivalent metrics on
  the SGLang runtime; runbooks are engine-agnostic by design.
- Google SRE Book — *Being On-Call*, *Managing Incidents* —
  https://sre.google/sre-book/ — generic runbook authoring standards this
  exercise specialises for inference.
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/):
  - **ML05 Model Theft / tampered artefact** — Runbook #7 (artefact
    integrity failure at startup) is the direct response.
  - **ML06 Output Integrity Attack** — Runbook #8 (canary regression) is
    the primary runbook; runbook #9 is the downstream follow-through.
  - **ML10 Insufficient Supply Chain Security** — Runbook #7 again; the
    discipline of refusing to "patch over" a SHA mismatch is the point.
- [mitre-atlas](https://atlas.mitre.org/):
  - **AML.T0010 ML Supply Chain Compromise** — Runbook #7.
  - **AML.T0043 Craft Adversarial Data / regression detection** — Runbook
    #8.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework):
  - **Govern** — named authority, owning rotation, review cadence.
  - **Measure** — dashboard-first initial checks; every runbook ties to a
    metric.
  - **Manage** — mitigation, rollback, follow-up.

## Appendix — control-to-framework mapping

| Control | OWASP ML | MITRE ATLAS | NIST AI RMF |
|---|---|---|---|
| Runbook #7 refuses engine start on SHA mismatch | ML05, ML10 | AML.T0010 | Manage |
| Runbook #8 confirms canary auto-rollback and opens forensic replay | ML06 | AML.T0043 | Measure, Manage |
| Runbook ownership + review cadence | ML-governance adjacent | n/a | Govern |
| Dashboard-first initial check across all ten | ML06 | Detection discipline | Measure |
| Follow-up ticket / RCA linkage | ML-governance adjacent | n/a | Govern, Manage |
