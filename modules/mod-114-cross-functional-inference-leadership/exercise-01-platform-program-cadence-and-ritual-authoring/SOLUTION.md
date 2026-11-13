# Solution — Exercise 01: Platform Program Cadence and Ritual Authoring

> Paired learning file: [`lessons/mod-114-cross-functional-inference-leadership/exercises/exercise-01-platform-program-cadence-and-ritual-authoring.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-114-cross-functional-inference-leadership/exercises/exercise-01-platform-program-cadence-and-ritual-authoring.md)
> Module: `mod-114-cross-functional-inference-leadership`
> Type: leadership / governance artefact set

## 1. Solution overview

Running an inference platform *on cadence* means the program is governed by four
written artefacts (charter, SLO contract pointer, change calendar, runbook set +
on-call rotation) and five recurring rituals (weekly SLO review, monthly change
calendar, quarterly migration roadmap, quarterly contract re-signature, ad-hoc
peer escalation). This exercise produces the full written set at a quality a
successor engineer can pick up cold.

The worked answer is anchored to a plausible hypothetical fleet so every
artefact has concrete nouns and numbers. **Everything that would carry a real
signer's name is explicitly marked `SIMULATED` in brackets.** Any numeric
estimate flagged as plausible — not measured — is tagged
`<!-- pending-verification: ... -->` where the real version would want validation.

### 1.1 Simulation preamble (applies to every artefact)

> The following artefacts are authored against a **simulated inference
> platform**. The fleet shape, model inventory, workload classes, SLO numbers,
> and change schedule are plausible-hypothetical and derived from the open
> `vLLM` / `SGLang` ecosystem. All sign-off names are of the form
> `[SIMULATED: <role>]`. The discipline and the shape of the artefacts are
> the deliverable; real numbers and real signatures replace these when a team
> adopts the templates.

### 1.2 Reference fleet

| Property | Value |
|---|---|
| Hardware | 40× H100-SXM5 nodes × 8 GPU = 320 H100 GPUs, 2× PCIe T4 utility nodes |
| Primary model | `meta-llama/Meta-Llama-3.1-70B-Instruct` FP16 (serving tier A) |
| Secondary models | 20 long-tail open-weights models via a LiteLLM-shape in-house router (tier B/C) |
| Routing | In-house router → vLLM engines, SGLang fallback pool |
| Average GPU utilisation | 55 % |
| Peak GPU utilisation | 85 % |
| Workload classes | `tier-A chat` (interactive, 70B), `tier-B long-tail` (routed to smaller models), `tier-C batch embedding` (off-peak), `internal eval-runner` (scheduled) |
| Token volume | ≈ 3.5 B output tokens / month across all tiers <!-- pending-verification: replace with measured monthly token count --> |
| Regions | `us-east-1`, `eu-west-1` (two clusters each) |
| Tenancy | Single-org, per-API-key quota tiers |

### 1.3 Deliverable map

| § | Deliverable | Pairs with chapter |
|---|---|---|
| §2 | A. Charter | 01 (platform program cadence) |
| §3 | B. SLO contract pointer | 01 + module 110 |
| §4 | C. Change calendar (template + 5 entries) | 01 |
| §5 | D. Runbook set pointer + on-call rotation + paging matrix | 01 + module 113 ch. 03 |
| §6 | E. Weekly SLO review (dry-run) | 01 |
| §7 | F. Quarterly migration roadmap + one full migration plan | 01 |
| §8 | G. Anti-pattern self-review | 01 |

---

## 2. Deliverable A — the charter

### 2.1 Platform charter (one page)

> **`[SIMULATED]` Inference Platform Charter — v1.0 — effective 2026-Q4**
>
> **Owns.** The production inference serving stack from request ingress to
> response egress: vLLM / SGLang engines, routing, autoscaler, warm pool,
> KV-cache tiering, deploy ritual, runbook execution, on-call rotation, SLO
> contracts, cost accounting per workload class, model-artefact ingress once
> the ML-platform peer has handed off a signed artefact.
>
> **Does not own.** Kernel source files or fused-kernel authorship
> (performance peer, module 102), model training or training-to-serving
> handoff before ingress (ML-platform peer), capability/benchmark evals and
> application evals (eval peers, module 113 ch. 05), application product
> features (consumer tracks), inference-edge security policy authorship
> (security peer; see Deliverable D of exercise 05).
>
> **Consumers.**
>
> | Consumer track | Workload class | Tenant tier |
> |---|---|---|
> | `search-rag` | tier-A chat, tier-C embedding | T1 (contractual SLO) |
> | `assistant-product` | tier-A chat | T1 |
> | `research-sandbox` | tier-B long-tail | T2 |
> | `eval-runner` | internal eval | T3 (best-effort, off-peak) |
>
> **SLOs committed.** See Deliverable B (pointer to the SLO contract).
> Headline numbers reproduced here for one-page readability: tier-A TTFT
> p95 ≤ 1500 ms, tier-A E2E p95 ≤ 7500 ms, tier-A availability ≥ 99.5 %.
>
> **Current scope boundary.** Workload classes above; clusters
> `us-east-1-a`, `us-east-1-b`, `eu-west-1-a`, `eu-west-1-b`; vLLM primary,
> SGLang fallback; token-generating workloads only (no diffusion or ASR —
> those live on a sibling platform).
>
> **Review cadence.** Quarterly, aligned with the migration-roadmap review
> (§7). Any scope change requires written sign-off from the sponsor and each
> affected consumer-track lead.
>
> **Signed.** `[SIMULATED: Platform Lead]`, `[SIMULATED: Sponsor — VP
> Infrastructure]`, 2026-10-14.

### 2.2 Why the charter is one page

The charter's job is to be read in ninety seconds by a consumer-track lead
who is choosing whether to stand up a dependency on this platform. A
three-page charter dilutes the one sentence that matters — *what we own and
what we don't* — and lets the boundary drift at contract renewal. The
discipline is: every quarterly review, re-prove the charter fits on one
page; cuts are more valuable than additions.

---

## 3. Deliverable B — the SLO contract pointer

### 3.1 Pointer document (one page)

> **`[SIMULATED]` SLO Contract Pointer — Inference Platform — v1.0**
>
> **Location.** `platform-infra/slo-contracts/inference/v2026.10.yaml` on
> the primary ops repo. Rendered at `[SIMULATED: internal wiki URL]`. The
> contract itself is the module 110 chapter 06 deliverable; if that
> deliverable is not yet complete on this platform, a TODO stub lives at
> the location above and the review cadence below applies to filling it in.
>
> **SLOs covered by workload class.**
>
> | Workload class | TTFT p95 | ITL p95 | E2E p95 | Availability | Cost ceiling per 1M output tokens |
> |---|---|---|---|---|---|
> | tier-A chat | 1500 ms | 60 ms | 7500 ms | 99.5 % | `[SIMULATED: $]` <!-- pending-verification: replace with measured per-1M cost --> |
> | tier-B long-tail | 2500 ms | 100 ms | 15 000 ms | 99.0 % | `[SIMULATED: $]` |
> | tier-C batch embedding | n/a (batch) | n/a | 60 s per batch of 1 k | 99.0 % | `[SIMULATED: $]` |
> | internal eval | best-effort | best-effort | best-effort | best-effort | n/a |
>
> **Signers.**
>
> | Side | Role | Named signer |
> |---|---|---|
> | Platform | Platform Lead | `[SIMULATED]` |
> | Platform | SRE Peer Lead | `[SIMULATED]` |
> | Consumer `search-rag` | Track Lead | `[SIMULATED]` |
> | Consumer `assistant-product` | Track Lead | `[SIMULATED]` |
> | Consumer `research-sandbox` | Track Lead | `[SIMULATED]` |
>
> **Review cadence.** Quarterly, at the migration-roadmap review slot in
> §7. Mid-quarter changes require written amendment signed by the Platform
> Lead and the affected consumer-track lead. Emergency SLO downgrades
> (capacity event, hardware incident) use the incident-command path in
> Deliverable D §5.2 and auto-expire after seven days unless re-signed.
>
> **Escalation path for an SLO breach exceeding the error budget.**
> On-call primary pages the Platform Lead within 30 minutes of a sustained
> burn-rate alert (module 110 chapter 04 defines the burn-rate thresholds;
> the alert fires when 2 % of the monthly budget is consumed in under
> one hour). Platform Lead convenes a correction meeting within 24 hours
> with the consumer-track lead for the affected class, the SRE peer, and
> the on-call engineer. Output: a written correction plan with a
> 72-hour and a 30-day milestone, or an explicit SLO downgrade amendment
> as above.
>
> **TODOs tracked here.** Where the module 110 contract is incomplete:
> - tier-A cost ceiling number — replaces `[SIMULATED: $]` once measured.
> - tier-B / tier-C full cost model — `<!-- pending-verification: cost per 1M output tokens for routed long-tail workloads -->`.
> - Eval workload's "best-effort" definition — attach when the eval peer's
>   contract (exercise 05 Deliverable B) is signed.

---

## 4. Deliverable C — the change calendar

### 4.1 Template

```yaml
# change-calendar/<YYYY>/<MM>/<change_id>.yaml
change_id: CHG-<YYYY>-<nnn>
title: <short imperative sentence>
window:
  start: <YYYY-MM-DD HH:MM UTC>
  end: <YYYY-MM-DD HH:MM UTC>
  duration: <mins>
  timezone: UTC
blast_radius:
  workload_classes: [<tier-A | tier-B | tier-C | eval>]
  clusters: [<us-east-1-a | us-east-1-b | eu-west-1-a | eu-west-1-b>]
  tenant_tiers: [<T1 | T2 | T3>]
change_type: <model-swap | scheduler-flag | engine-upgrade | hardware | kv-tiering | routing-rule | security-config>
rollout_primitive: <canary-1pct | shadow | blue-green | in-place-rolling | drain-and-swap>
reversibility: <single-flag | config-revert | redeploy-old-image | stateful-migration-required>
on_call:
  primary: <handle>
  secondary: <handle>
peer_dependency:
  touches: [<ml-platform | performance | sre | security | eval>]
  signoff: <none | requested | received | waived>
freeze_check: <clear | frozen-blocked | frozen-waived-with-justification>
eval_gate: <passed | passed-with-caveats | waived-batch-only | n-a>
runbook: <pointer to module 113 ch. 03 runbook>
notes: <free-form>
```

### 4.2 Populated entries (five)

**Entry 1 — a real (realistic) model swap.**

```yaml
change_id: CHG-2026-091
title: "Swap primary tier-A model from Llama-3.1-70B-Instruct rev=abc123 to rev=def456"
window: { start: 2026-11-18 02:00 UTC, end: 2026-11-18 03:00 UTC, duration: 60, timezone: UTC }
blast_radius: { workload_classes: [tier-A], clusters: [us-east-1-a, us-east-1-b], tenant_tiers: [T1] }
change_type: model-swap
rollout_primitive: canary-1pct
reversibility: redeploy-old-image
on_call: { primary: "[SIMULATED: on-call-primary]", secondary: "[SIMULATED: on-call-secondary]" }
peer_dependency: { touches: [ml-platform, eval], signoff: received }
freeze_check: clear
eval_gate: passed
runbook: "runbooks/model-promotion.md"
notes: >
  Weights SHA pinned via intake manifest (see module 108 exercise 04 draft-lifecycle
  contract). Target this change pairs with tier-A SLO bump proposed for Q1 2027.
```

**Entry 2 — a scheduler-flag change.**

```yaml
change_id: CHG-2026-092
title: "Enable chunked-prefill for tier-A chat (vLLM --enable-chunked-prefill)"
window: { start: 2026-11-20 03:30 UTC, end: 2026-11-20 04:15 UTC, duration: 45, timezone: UTC }
blast_radius: { workload_classes: [tier-A], clusters: [us-east-1-a], tenant_tiers: [T1] }
change_type: scheduler-flag
rollout_primitive: shadow        # mirror traffic for 24 h before cut-over
reversibility: single-flag
on_call: { primary: "[SIMULATED: on-call-primary]", secondary: "[SIMULATED: on-call-secondary]" }
peer_dependency: { touches: [performance], signoff: requested }
freeze_check: clear
eval_gate: passed
runbook: "runbooks/scheduler-flag-rollout.md"
notes: >
  Expected TTFT p95 improvement 10-20 % per module 103 workshop. Performance peer
  (exercise 04 contract) consulted; opinion-returned SLA P2 (24 h).
```

**Entry 3 — a hardware event.**

```yaml
change_id: CHG-2026-093
title: "NVIDIA driver 555 → 560 rolling drain across us-east-1-a"
window: { start: 2026-11-22 00:00 UTC, end: 2026-11-22 06:00 UTC, duration: 360, timezone: UTC }
blast_radius: { workload_classes: [tier-A, tier-B, tier-C], clusters: [us-east-1-a], tenant_tiers: [T1, T2, T3] }
change_type: hardware
rollout_primitive: drain-and-swap    # one node at a time behind the autoscaler's warm pool
reversibility: redeploy-old-image    # driver downgrade possible but node-reboot cost
on_call: { primary: "[SIMULATED: on-call-primary]", secondary: "[SIMULATED: on-call-secondary]" }
peer_dependency: { touches: [sre, performance], signoff: received }
freeze_check: clear
eval_gate: n-a
runbook: "runbooks/driver-upgrade-drain.md"
notes: >
  Performance peer validated kernel compat on rev=560 against the kernel-regression
  ledger (exercise 04 Deliverable E). SRE peer owns incident command if the drain
  trips the tier-A TTFT alert.
```

**Entry 4 — a cross-track peer dependency (and a `freeze_check: waived-with-justification`).**

```yaml
change_id: CHG-2026-094
title: "Rate-limit policy change for assistant-product tier-A (burst 20 → 30 QPS)"
window: { start: 2026-12-05 15:00 UTC, end: 2026-12-05 15:30 UTC, duration: 30, timezone: UTC }
blast_radius: { workload_classes: [tier-A], clusters: [all], tenant_tiers: [T1] }
change_type: routing-rule
rollout_primitive: single-flag
reversibility: single-flag
on_call: { primary: "[SIMULATED: on-call-primary]", secondary: "[SIMULATED: on-call-secondary]" }
peer_dependency: { touches: [assistant-product, security], signoff: received }
freeze_check: frozen-waived-with-justification
eval_gate: n-a
runbook: "runbooks/rate-limit-adjustment.md"
notes: >
  Waiver justification: pre-holiday assistant-product launch window ends 2026-12-07;
  rate-limit lift cannot wait for post-freeze window on 2027-01-05 because the
  product launch depends on it. Risk accepted by sponsor and security peer. Reverts
  automatically at 2026-12-08 00:00 UTC unless an amendment ticket re-signs it.
```

**Entry 5 — a KV-tiering change.**

```yaml
change_id: CHG-2026-095
title: "Enable LMCache CPU-tier prefix cache for tier-A chat"
window: { start: 2026-12-10 02:00 UTC, end: 2026-12-10 04:00 UTC, duration: 120, timezone: UTC }
blast_radius: { workload_classes: [tier-A], clusters: [us-east-1-a], tenant_tiers: [T1] }
change_type: kv-tiering
rollout_primitive: canary-1pct
reversibility: config-revert
on_call: { primary: "[SIMULATED: on-call-primary]", secondary: "[SIMULATED: on-call-secondary]" }
peer_dependency: { touches: [performance, security], signoff: requested }
freeze_check: clear
eval_gate: passed-with-caveats
runbook: "runbooks/kv-tiering-rollout.md"
notes: >
  Cross-tenant prefix-sharing is disabled; single-tenant prefix only (see module
  104 exercise 06 cross-tenant KV safety review). Security peer sign-off required
  before widening past 1 % canary.
```

### 4.3 How the calendar is consulted

- The calendar lives at `change-calendar/` in the ops repo. Each YAML is one
  file; a weekly rollup renders at a pinned wiki page.
- A GitHub Action rejects any change-calendar PR whose `freeze_check` field is
  `frozen-blocked` and whose window overlaps a published freeze.
- At the weekly SLO review (§6) the "this week's changes" agenda item is the
  next seven days of the calendar; this closes the loop on *hidden freeze*.
- At the monthly change-calendar view, consumer-track leads can veto any entry
  whose `peer_dependency.signoff` is `none` and whose blast radius touches
  their workload class.

---

## 5. Deliverable D — the runbook set pointer + on-call rotation + paging matrix

### 5.1 Runbook set pointer

Full runbooks live in module 113 chapter 03 (`runbook top-ten`) under
`platform-infra/runbooks/`. This document names the rotation and the paging
behaviour; the runbook content itself is not duplicated. If the module 113
deliverable is incomplete, name each runbook by title and open a TODO; the
rotation and paging matrix still apply.

Runbook titles the paging matrix below assumes exist:

1. `tier-A TTFT burn-rate fired` (module 113 ch. 03)
2. `tier-A availability drop` (module 113 ch. 03)
3. `KV-cache fragmentation / OOM evict` (module 104)
4. `continuous-batching starvation` (module 103)
5. `autoscaler stuck / warm-pool exhausted` (module 109)
6. `model artefact-SHA mismatch at engine start` (module 108 exercise 04)
7. `cost per 1M tokens anomaly` (module 110)
8. `cross-cluster split-brain / routing loop` (module 111)
9. `security-edge rate-limit tier breach` (exercise 05 Deliverable D)
10. `eval regression on production slice` (module 113 ch. 05)

### 5.2 Rotation structure

| Dimension | Value |
|---|---|
| Shift length | 7 days (Mon 09:00 UTC → Mon 09:00 UTC) |
| Primary | One engineer |
| Secondary | One engineer (also called at 20 minutes if primary un-acks) |
| Rotation gap | 3 weeks minimum between primary shifts per engineer |
| Pool size | ≥ 6 engineers (anything less violates module 113 ch. 02 rest-ratio) |
| Timezone coverage | Follow-the-sun not yet implemented; EU and US pools paired with handoff at 16:00 UTC |
| Handoff doc | See §5.4 |

### 5.3 Paging matrix

| Alert class | Primary paging | Secondary paging timing | Peer escalation trigger | Peer paged |
|---|---|---|---|---|
| tier-A TTFT burn-rate (1 h window) | on-call-primary — page immediately | T + 20 min if no ack | T + 60 min with no path to green | SRE peer |
| tier-A availability drop (< 99 % 15 min) | on-call-primary — page immediately | T + 10 min if no ack | T + 30 min with no path to green | SRE peer + Platform Lead |
| KV-cache OOM / fragmentation | on-call-primary — page immediately | T + 20 min if no ack | T + 60 min if ≥ 3 engine restarts in 1 h | Performance peer |
| Continuous-batching starvation | on-call-primary — page immediately | T + 20 min | T + 60 min with no path to green | Performance peer |
| Autoscaler stuck / warm-pool exhausted | on-call-primary — page immediately | T + 20 min | T + 60 min | SRE peer |
| Artefact-SHA mismatch at engine start | on-call-primary — page immediately | T + 20 min | T + 60 min if ≥ 2 engines affected | ML-platform peer |
| Cost per 1M tokens anomaly | on-call-primary — ticket (not page) within 4 h | n/a | T + 24 h if anomaly persists | Finance partner (async) |
| Routing loop / split-brain | on-call-primary — page immediately | T + 20 min | T + 30 min | SRE peer |
| Edge rate-limit tier breach | on-call-primary — page immediately | T + 10 min | T + 30 min | Security peer (incident command) |
| Eval regression on prod slice | on-call-primary — page immediately | T + 20 min | T + 60 min with no path to green | Eval peers |

Design notes on the matrix:

- **One primary.** No "page both" rows. Common failure mode (§8.5) is a matrix
  where every alert pages four peers; the on-call teaches themselves to
  silence the pager. The matrix above sends every alert to exactly one
  responder, with a clock-based escalation to a peer only where the alert
  class genuinely crosses a boundary.
- **Cost anomaly is a ticket, not a page.** Cost alerts are high-value but
  almost never minute-urgent; paging an engineer at 3am for a cost anomaly
  trains the rotation to silence the budget. The policy exchange is: cost
  alerts arrive with 4 h and 24 h SLAs and a finance partner on the async
  thread.
- **Edge security alerts page faster than availability alerts.** A rate-limit
  tier breach may be an active attack; secondary pages at T + 10 min instead
  of T + 20 min.

### 5.4 Handoff doc template (leaving on-call)

```markdown
# On-call handoff — <YYYY-MM-DD> — outgoing: <primary>, incoming: <primary>

## Still-open
- Incident <INC-NNN>: <status, next action, owner>
- Ticket <TKT-NNN>: <status, blocker>

## Closed this shift
- <one-line summary per incident>

## Changes in flight that intersect this rotation
- CHG-YYYY-NNN at <date> UTC; blast radius; rollback plan

## Dashboards to watch this week
- <panel URL>: <why; what to do if it turns red>

## Pager tuning (what fired and shouldn't have)
- <alert name>: <recommended threshold change; ticket>

## Known gaps / "don't be surprised by"
- <subsystem X is in a degraded state; symptom; whom to call>

## For the outgoing on-call
- Hours slept: <approximate>
- Backlog: <any ticket you deferred that the next shift should pick up>
- Vibe: <one sentence on morale / rest state; the retro reads these to spot burnout>
```

### 5.5 Retrospective cadence

- **Per-shift retro:** 30 minutes, every Monday at 10:00 UTC, immediately
  after handoff. Agenda: alerts that fired, alerts that didn't fire and should
  have, runbook gaps, pager-tuning tickets.
- **Quarterly retro:** 90 minutes, aligned with the migration-roadmap review
  (§7). Agenda: rotation fairness, pager-noise trend, runbook coverage, hero
  on-call check (§8.1).

---

## 6. Deliverable E — the weekly SLO review (dry-run minutes)

### 6.1 Dry-run disclosure

> Run as a **dry-run** on 2026-11-10 10:00 UTC. The Platform Lead
> (`[SIMULATED]`) facilitated. Attending in person: the on-call primary
> `[SIMULATED]`. All other peer roles stood in by the facilitator; the
> facilitator took a different seat for each absent role and declared the
> swap aloud. Minutes below are written as if the meeting had been fully
> attended. Deliverables marked `[SIMULATED]` are those a real meeting would
> produce; here, the facilitator opened them as tickets after the dry-run.

### 6.2 Dashboard pointer (the agenda)

> `[SIMULATED]` Grafana board URL:
> `grafana.[org].internal/d/inf-platform-slo-weekly` — five panels that
> correspond one-to-one to the five agenda items in §6.3. The dashboard is
> the agenda; no slides are prepared.

### 6.3 Minutes

**Agenda item 1 — last week's burn rate (panel: *30-day SLO burn, tier A/B/C*)**

> Tier-A TTFT burned 0.9 % of the monthly 0.5 % budget on Tuesday during
> the 15:00 UTC peak — a 1.8× burn over the week. Root cause was a single
> noisy neighbour on `us-east-1-a-node-12` that saturated an engine's KV
> pool; the on-call drained the node and the burn stopped. Tier-B and
> tier-C within budget.

**Agenda item 2 — incidents this week (panel: *incident timeline*)**

> One P2: `INC-2026-114` (noisy-neighbour KV saturation above). Resolved in
> 42 minutes. No P0 or P1.

**Agenda item 3 — capacity trend (panel: *fleet utilisation and warm-pool drain rate*)**

> 7-day moving average GPU utilisation at 58 % (baseline 55 %). Warm-pool
> drain rate up 15 % WoW — the `assistant-product` track's launch traffic
> is reaching the fleet earlier than the forecast in CHG-2026-094. Capacity
> headroom remains at 25 %.

**Agenda item 4 — this week's changes (panel: *change calendar, next 7 days*)**

> Four entries scheduled (CHG-091 through CHG-094 from §4.2). Noted: CHG-094
> is a freeze waiver; the facilitator confirmed sponsor and security-peer
> sign-off is on file. Facilitator asked whether the assistant-product
> track's capacity forecast has been updated given the higher-than-forecast
> drain in item 3; action in §6.4.

**Agenda item 5 — action items from last week + new tickets**

> Three carried from last week: two closed, one still blocked on the
> upstream vLLM upgrade PR. Three new (see §6.4).

### 6.4 Tickets created out of the meeting

| Ticket | Title | Owner | Due |
|---|---|---|---|
| `[SIMULATED] TKT-2026-410` | "Update capacity forecast given 15 % higher warm-pool drain; propose topology adjustment or scale-out for us-east-1-a." | on-call-primary | 2026-11-14 |
| `[SIMULATED] TKT-2026-411` | "Add a Grafana panel for noisy-neighbour KV saturation per engine; alert threshold TBD." | SRE peer stand-in | 2026-11-17 |
| `[SIMULATED] TKT-2026-412` | "Pre-write CHG-094 revert ticket so the auto-revert at 2026-12-08 is a one-click merge, not an incident." | on-call-primary | 2026-11-24 |

### 6.5 Honest critique of this ritual

What the ritual missed:

- The dashboard does not have a **noisy-neighbour KV-saturation panel**;
  the facilitator reconstructed the Tuesday burn by cross-referencing the
  per-engine KV panel with the tenant-tier panel. The panel from TKT-411
  is the fix; until it exists, the dashboard is not the complete agenda.
- The **cost-per-1M-tokens panel** is on the dashboard but nobody looked
  at it in the meeting. Agenda item 3 should explicitly include a cost
  check; add to the dashboard's title order.
- The dry-run stand-in format works but is slow; a real meeting with
  four peers takes 45 minutes, the dry-run took 65 minutes because the
  facilitator was context-switching between roles. The dry-run itself is
  a useful artefact; it is not a long-term substitute for the real meeting.

What the dashboard didn't show that it should:

- Error budget *consumption velocity* (burn-rate derivative, not just
  cumulative burn). The Tuesday peak was only visible after-the-fact.
- Per-tenant TTFT distribution. Current panel is per-workload-class; a
  single noisy tenant inside a class is invisible.

---

## 7. Deliverable F — the quarterly migration roadmap

### 7.1 Migration ledger (in-flight)

| Migration | Stage | Owner | Next milestone | Target date |
|---|---|---|---|---|
| vLLM 0.6.x → 0.7.x on tier-A | 60 % canary | `[SIMULATED: Platform Lead]` | 100 % cut-over after 2026-11-18 model swap settles | 2026-11-25 |
| LMCache CPU prefix cache for tier-A | 1 % canary (CHG-095) | `[SIMULATED: Performance peer lead]` | 10 % canary pending security sign-off | 2026-12-15 |
| Add `eu-west-1-b` cluster for tier-B long-tail | infra provisioned, router not yet pointing | `[SIMULATED: SRE peer lead]` | router rule PR merged and shadowed for 1 week | 2026-12-01 |
| Replace in-house router with LiteLLM-based router | design review in flight | `[SIMULATED: Platform Lead]` | design approved by consumer-track leads | 2026-12-10 |

### 7.2 Ranked candidate migrations for next quarter

Scoring on three factors: `(SLO value) × (cost savings) / (risk)`.

| Rank | Candidate | Sponsor | One-line rationale |
|---|---|---|---|
| 1 | Introduce speculative decoding (draft + target) for tier-A | `[SIMULATED: Platform Lead]` | Expected 25 – 40 % TTFT and ITL improvement per module 108; draft-lifecycle contract already authored. |
| 2 | Move tier-C batch embedding to a managed vendor | `[SIMULATED: Sponsor]` | Off-peak workload saturates fleet during overnight runs; vendor TCO possibly lower — triggers an exercise-02 RFC and exercise-03 scorecard. |
| 3 | Add FP8 quantisation for secondary long-tail models | `[SIMULATED: Performance peer lead]` | Reduces footprint; enables denser packing on tier-B nodes. |
| 4 | Multi-region routing for tier-A latency | `[SIMULATED: SRE peer lead]` | Reduces US / EU cross-region tail; moderate complexity. |
| 5 | Introduce EAGLE-style self-speculative heads as a fallback to #1 | `[SIMULATED: Performance peer lead]` | Second-order; contingent on #1 outcome. |
| 6 | Cross-tenant prefix-sharing for tier-A system prompts | `[SIMULATED: Platform Lead]` | Highest potential throughput gain but fails the module 104 ch. 06 safety review in its naive form; needs scoping before it can be committed. |

### 7.3 Critical-path graph

```
  [#1 spec decoding] ──depends on──> [draft-model lifecycle contract SIGNED] (done, mod-108 ex-04)
       │
       └──> [exercise-04 performance-peer kernel review of draft+target kernels]
                 │
                 └──> [#5 EAGLE heads]  (contingent)

  [#2 managed-vendor embedding]
       └──> [exercise-02 RFC]
             └──> [exercise-03 vendor scorecard]
                     └──> [sponsor sign-off]

  [#3 FP8 long-tail]
       └──> [kernel + quant eval gate signed by eval peers]

  [#4 multi-region routing]
       └──> [#3 cluster "eu-west-1-b" fully live] (in-flight, §7.1)
       └──> [SRE peer availability contract amendment]

  [#6 cross-tenant prefix]
       └──> [security peer policy sign-off] (not yet started)
       └──> [#1 complete]  (otherwise interacts with speculation accounting)
```

### 7.4 Commit

- **Committed for Q1 2027:** #1 speculative decoding, #2 managed-vendor
  embedding, #4 multi-region routing.
- **Backlog with reasons:**
  - #3 FP8 long-tail — deferred to Q2 2027; the performance peer's capacity
    in Q1 is on #1's kernel review.
  - #5 EAGLE — explicitly contingent on #1's delivered gain; no decision until
    #1's acceptance-rate baseline is measured.
  - #6 cross-tenant prefix-sharing — paused pending security peer scoping
    conversation (expected Q2); moving it in Q1 would skip the module 104
    ch. 06 safety review.

### 7.5 Full migration plan — committed migration #1 (speculative decoding)

Using chapter 01's eight-field shape.

**1. Problem.** Tier-A TTFT p95 is at 1300 ms against the 1500 ms SLO; the
200 ms margin is fragile against seasonal traffic and the assistant-product
launch. ITL p95 is at 50 ms against the 60 ms SLO. Capacity utilisation has
no obvious slack to fund a bigger model or heavier prompt; the next lever is
speculative decoding, which trades draft-model cost for target-model
throughput.

**2. Scope.** Tier-A chat only; `us-east-1-a` and `us-east-1-b` first, EU
clusters after one stable month; vLLM only (not SGLang fallback pool, since
vLLM is primary and the fallback pool is explicitly a low-feature escape
hatch).

**3. Target state.** vLLM draft-target speculative decoding enabled on tier-A
engines with `meta-llama/Llama-3.2-1B-Instruct` as the draft, pinned to the
current primary target revision per the module 108 exercise-04 lifecycle
contract. Acceptance rate α ≥ 0.65 on the three-row prompt mix from
exercise-03 of module 108; TTFT p95 improvement ≥ 15 % versus baseline;
ITL p95 improvement ≥ 25 %.

**4. Peer touch points.**

| Peer | Touch point | Gate |
|---|---|---|
| Performance peer (ex-04 contract) | Kernel-level review of the draft-model attention path and the acceptance-sampling kernel in the vLLM revision we deploy. | Opinion-returned in P1 SLA before canary. |
| ML-platform peer (ex-05 A) | Draft artefact intake per draft-lifecycle contract; SHA pinned. | Artefact ingress signed. |
| Eval peers (ex-05 B) | Pre-flight eval on the prompt distribution at canary 1 % and 10 %. | Pre-flight eval sign-off. |
| SRE peer (ex-05 C) | SLO contract amendment if the TTFT improvement enables a tighter SLO; burn-rate thresholds must track α. | Burn-rate alert tuned. |
| Security peer (ex-05 D) | Draft-model supply-chain review (SHA, provenance, licence) per draft-lifecycle contract §2.3. | Draft intake security review signed. |

**5. Stages.**

| # | Stage | Exit criterion | Duration |
|---|---|---|---|
| 1 | Intake the draft model; sign the lifecycle manifest; measure α on the correctness fixture. | α measured, SHA pinned, security review signed. | 2 weeks |
| 2 | Deploy to one `us-east-1-a` engine behind a shadow flag; mirror 1 % of traffic; validate equivalence against baseline. | Shadow panel green; no correctness regressions. | 2 weeks |
| 3 | 1 % canary on live traffic; the exercise-03 acceptance-rate dashboard watches α. | α ≥ 0.65 sustained for 72 h; TTFT improvement visible; no ITL regression. | 1 week |
| 4 | 10 % canary. | Same thresholds, 5 days sustained. | 1 week |
| 5 | 100 % cut-over on `us-east-1-a`. | 7-day burn-rate stable or improved; α stable; no incidents attributed to speculation. | 1 week |
| 6 | 100 % cut-over on `us-east-1-b`. | Same. | 1 week |
| 7 | EU clusters. | Same. | 1 week |

**6. Rollback branch.** Rollback is cheap through the end of stage 4
(single-flag revert; canary traffic returns to baseline). **Stage 5 is the
point past which rollback becomes expensive**: once 100 % of `us-east-1-a`
is on speculation, the on-call muscle memory and the capacity model start to
assume the throughput gain. Reverting after stage 5 is a second migration
(un-pin the draft, re-plan capacity, un-tune burn-rate alerts). A hard revert
is still possible via a flag flip, but the post-revert capacity plan has to
be re-done; the plan should budget 48 hours for a stage-5-or-later rollback
rather than treating it as a 30-minute flag revert.

**7. Consumer communication plan.** `search-rag` and `assistant-product`
track leads receive:

- A pre-stage-1 email naming the target completion date and the TTFT /
  ITL improvement forecast.
- At each canary stage, a one-paragraph update with the acceptance-rate
  and latency panel screenshots.
- At stage-5 cut-over, a 24-hour-advance notice and a 7-day hyper-care
  window (on-call bumps the alert thresholds tighter).
- At EU cut-over, same cadence.

A single Slack thread (`[SIMULATED: #inf-platform-migrations-q1-2027`]`)
holds all updates so consumer leads can read a continuous story rather than
piecing it together from scattered messages.

**8. Evidence of done.**

- α measurement from stage 1 archived in the draft-model registry
  (module 108 ex-04).
- Panel screenshots from each stage archived with the migration record.
- TTFT and ITL comparison (before / 100 %) attached to the migration
  record.
- Burn-rate alert thresholds updated in the Grafana config repo.
- Lifecycle contract reviewed quarterly going forward.

---

## 8. Deliverable G — anti-pattern self-review

### 8.1 Hero on-call

**Assessment.** The rotation in §5.2 has a pool of six; the handoff doc
(§5.4) explicitly captures hours slept and backlog to make imbalance visible
at the quarterly retro. **However**, the dry-run in §6 was run by the
facilitator standing in for every peer — a hero-shaped artefact. If this
platform shipped with the current roster in a real incident, the on-call
primary would carry disproportionate load for any P0 outside business hours
because the secondary pool is EU-biased and the handoff clock loses an hour
on daylight saving boundaries.

**Fix.** At the quarterly retro, enforce a "hours-on-call per engineer"
threshold; any engineer above the threshold triggers a rotation re-balance.
If the pool cannot support the threshold, the SRE peer (ex-05 C) escalates
to the sponsor for a headcount request. Do not compensate by stretching the
on-call duration; stretched rotations become permanent heroism.

### 8.2 Dashboard-as-ritual

**Assessment.** The SLO review in §6 produced three tickets with owners and
due dates. The tickets are the output; the dashboard was the agenda, not the
deck. One risk: the tickets are routine (panel addition, dashboard tweak,
pre-wiring a revert). If quarters go by with only routine tickets, the
ritual has drifted into status-reporting even if it looks healthy on the
surface.

**Fix.** At the quarterly retro, count "tickets that survived more than one
sprint" per quarter. A quarter where every SLO-review ticket closes inside
one sprint is a signal that the ritual is under-ambitious; add a standing
agenda item for "one thing we don't understand about the dashboard" to force
non-routine discussion.

### 8.3 Hidden freeze

**Assessment.** The change calendar (§4) is enforced by a CI check. Consumer
tracks can read the calendar directly. The one entry that uses
`freeze-waived-with-justification` (CHG-094) has a sponsor and security peer
on the waiver. **However**, no test confirms consumer tracks have actually
consulted the calendar before their own launches; the only check is the
reverse direction (calendar → consumer blast radius). A consumer track can
schedule a product launch in a frozen window and only discover the freeze
when a change-calendar PR intersects.

**Fix.** Add a monthly "freeze-check from the consumer side" in the exercise-
05 cross-peer cadence (§Deliverable E of that exercise): consumer tracks
submit their next 30-day launch plan, and the platform reviews freeze
overlaps before the plan is published.

### 8.4 Migration-by-stealth

**Assessment.** Every in-flight migration in §7.1 is linked to a plan or an
RFC. The four committed candidates for Q1 2027 each have a sponsor named.
Historic risk is the LMCache rollout — the 1 % canary was scheduled before
the security peer's sign-off was formally captured; the entry (§4.2 CHG-095)
now records `peer_dependency.signoff: requested`, but the first version of
the change had `signoff: none`.

**Fix.** Make `peer_dependency.signoff: none` an explicit CI reject for any
change that touches more than one workload class or the T1 tenant tier. Keep
`signoff: waived` possible but require a justification pointer on the same
line.

### 8.5 One additional anti-pattern worth self-reviewing — paging matrix fan-out

Beyond the chapter 01 four: the paging matrix (§5.3) deliberately sends one
page per alert and clock-escalates to a peer. A prior version of this
matrix fanned every alert to all peers for "visibility"; within two weeks
the Slack channels were muted and the on-call was silencing the pager at
night. The one-primary rule is the fix; the self-review is to re-check
every quarter that no row has crept back to "page everyone."

---

## Implementation

The seven deliverables above are written artefacts, but each one lands on a
concrete enforcement point in the platform. A working implementation of this
cadence looks like:

- **Charter (§2) in Git.** The charter is stored at
  `docs/platform/charter.md` in the platform repo and reviewed at the
  quarterly contract re-signature ritual. Any PR that would expand the
  "owns" list requires a signer rotation on the file; any PR that would
  narrow the "does not own" list requires a signature from the peer whose
  territory is being absorbed.
- **SLO contract pointer (§3) wired to the dashboard.** The pointer is a
  `docs/platform/slo.md` whose links resolve to live dashboard panels; a
  CI check fails the build if a pointer URL 404s. The five SLO numbers are
  also exported as YAML (`slo.yaml`) so the alerting stack and the weekly
  review tool read from the same source.
- **Change calendar (§4) backed by a ticket query.** The calendar template is
  a saved query in the ticket tracker (label `change-window`), rendered into
  the Monday weekly review. Populated entries are tickets; the freeze-waiver
  row is a specific ticket type that demands a justification field.
- **On-call rotation + paging matrix (§5) in PagerDuty (or equivalent).**
  Rotation structure is encoded in the scheduler; the paging matrix is a
  routing policy with one primary per row. The handoff doc template is a
  required field on shift-end; retrospectives are a recurring calendar
  invite with the facilitator rotating.
- **Weekly SLO review (§6) as a scheduled meeting with a persistent agenda
  URL.** The agenda URL is the dashboard; minutes are posted to the platform
  channel; follow-up tickets are opened from the minutes template.
- **Quarterly migration roadmap (§7) as a single Git-tracked ledger.** The
  in-flight table and the ranked candidate list live in
  `docs/platform/migration-ledger.md`; the critical-path graph is rendered
  from a sibling `migration-graph.dot` file so the dependencies survive
  text search. Committed migrations get their own `plans/<name>.md` file.
- **Anti-pattern self-review (§8) as a quarterly checklist.** The
  five anti-patterns are a saved review template; each quarterly
  contract re-signature re-runs the checklist and files a ticket per
  finding.

None of these controls require bespoke tooling — a Git repo, a ticket
tracker, a paging product, and the module 110 dashboard cover the full
surface. What matters is that each artefact in §2–§8 has exactly one
enforcement point and that the point is automated or scheduled, not
remembered.

---

## 9. Validation steps for a learner submission

Reviewers should run this in order. Missing any step fails the submission.

1. **Charter is one page** and names what the platform does *not* own.
2. **SLO pointer** has a URL, five SLO numbers, named signers, a review
   cadence, and an escalation path. Signatures may be simulated but roles
   must be named.
3. **Change calendar template** has every required field; populated entries
   exercise model-swap, scheduler-flag, hardware, routing-rule, and a
   freeze-waiver with a defensible justification.
4. **Rotation and paging matrix** names shift length, pool size, and a
   paging matrix with a single-primary rule.
5. **Weekly SLO review** names the dashboard, the five agenda items, the
   tickets created out of the meeting with owners and due dates, and an
   honest critique of what the ritual missed.
6. **Migration roadmap** has a ledger, a ranked candidate list, a critical-
   path graph, a commit decision, and *one* full migration plan with all
   eight fields.
7. **Anti-pattern self-review** addresses hero on-call, dashboard-as-ritual,
   hidden freeze, and migration-by-stealth, each with an honest assessment
   and a proposed fix.
8. **Simulation disclosure.** If the artefacts are simulated, each one says
   so; sign-off names are marked `[SIMULATED]`.

## 10. Rubric / review checklist

Score each item 0 (missing), 1 (partial), 2 (fully correct). Passing ≥ 16 / 20.

| # | Item | Points |
|---|---|---|
| 1 | Charter is one page, names what the platform does and does not own, has a review cadence, is signed. | 2 |
| 2 | SLO contract pointer names location, signers, cadence, escalation path. | 2 |
| 3 | Change calendar template includes every required field and at least five populated entries that collectively exercise model-swap, flag, hardware, peer-dependency, and freeze-waiver. | 2 |
| 4 | Paging matrix has a single-primary rule and a clock-based escalation to a peer; different alert classes route differently. | 2 |
| 5 | Weekly SLO review has the dashboard-is-agenda structure, minutes in the five-item form, tickets with owners and due dates, and an honest critique. | 2 |
| 6 | Migration roadmap has a ledger, a ranked candidate list, a critical-path graph, a commit/backlog decision, and one full migration plan using chapter 01's eight-field shape. | 2 |
| 7 | Migration plan explicitly names the rollback-expensive stage. | 2 |
| 8 | Anti-pattern self-review addresses all four and proposes at least one concrete fix per anti-pattern. | 2 |
| 9 | Peer touch points named in both the change calendar and the migration plan; signatures / sign-off trail traceable. | 2 |
| 10 | Simulation clearly disclosed where applicable; no silent fabrication of signed numbers. | 2 |

## 11. Common mistakes

- **Three-page charter.** Every attempt to pack "mission" into the charter
  dilutes the one-sentence boundary that matters. Keep the charter one
  page; put mission in the sponsor memo.
- **SLO pointer with the SLO numbers duplicated as the primary artefact.**
  The pointer is a *pointer*; the contract itself lives in module 110. If
  your pointer document is longer than the one-page shape here, the
  contract will fork from the pointer and the pointer will rot.
- **Change calendar with every entry `peer_dependency: none`.** The first
  real peer-crossing change surprises a consumer track. If your five
  entries have no peer dependencies, you are not stressing the change
  calendar; invent an entry that spans tracks.
- **Freeze waiver without a time-box.** CHG-094 above auto-reverts at
  2026-12-08. A waiver that extends freeze permissions indefinitely is
  how freezes become advisory.
- **Paging matrix that pages every peer on every alert.** Within two
  weeks the pager is muted; within a month a real incident will be
  missed. The matrix in §5.3 sends every alert to exactly one primary
  and escalates on a clock.
- **SLO review as status update.** No tickets came out of the meeting.
  The review is the agenda, not the deck; if the output is slides, you
  ran the wrong ritual.
- **Migration plan without a rollback-expensive stage.** The plan commits
  to the move but never names where rollback flips from cheap to a
  second migration. Stage-5 in §7.5 is the example; every migration plan
  must name its own.
- **Anti-pattern self-review that is self-congratulatory.** "We don't
  have any of these" is not a self-review. If honest assessment turns up
  zero drift from any of the four anti-patterns, assume the reviewer is
  not being honest and re-run.
- **Hiding the simulation.** The dry-run in §6 is more useful than a
  pretend meeting that claims four peers attended. Name stand-ins
  explicitly.

## 12. References

- Google SRE Workbook, "Implementing SLOs" and "Alerting on SLOs" (public
  Google SRE books) — the burn-rate model referenced by the SLO contract
  pointer in §3 and the alerting matrix in §5.3.
  https://sre.google/workbook/alerting-on-slos/
- vLLM documentation — scheduler flags, engine configuration, chunked
  prefill, speculative decoding. https://docs.vllm.ai/
- SGLang documentation — fallback-pool reference in §1.2 and §7.5.
  https://docs.sglang.ai/ <!-- pending-verification: confirm SGLang's role in the specific team's fleet shape; the public docs cover the engine, not a specific deployment -->
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework) —
  the governance / ritual framing in this exercise maps to the RMF's
  GOVERN function (charter, cadence, signers) and MANAGE function (change
  calendar, paging matrix, migration-roadmap triggers).
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/)
  — the change calendar's model-swap and KV-tiering entries invoke this
  control set where the change touches supply chain (ML10) or output
  integrity (ML06).
- [mitre-atlas](https://atlas.mitre.org/) — the paging matrix's edge
  security route (§5.3, row 10) and the quarterly contract re-review feed
  the detection-side discipline that ATLAS catalogues.
- Module 110 (SLO engineering, cost per 1M tokens) — the SLO contract
  this pointer references and the cost column of the SLO table.
- Module 113 chapter 03 (runbook top-ten) — the runbooks the paging
  matrix in §5.3 depends on.
- Module 108 exercise 04 (draft-model lifecycle contract) — the artefact
  underlying migration #1's intake stage.
- Chapter 01 of this module — the four-artefact / five-ritual shape
  these solutions concretise.
