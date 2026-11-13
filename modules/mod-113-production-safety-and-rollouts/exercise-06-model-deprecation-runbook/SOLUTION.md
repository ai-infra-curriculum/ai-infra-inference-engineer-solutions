# Solution — Exercise 06: Model-Deprecation Runbook

> Paired learning file: [`lessons/mod-113-production-safety-and-rollouts/exercises/exercise-06-model-deprecation-runbook.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-113-production-safety-and-rollouts/exercises/exercise-06-model-deprecation-runbook.md)
> Module: `mod-113-production-safety-and-rollouts`
> Type: design / runbook

## 1. Solution overview

A model deprecation is the long tail of a model's lifecycle. It is the
process that moves users off an older model and eventually turns the
older model off. Done badly, it is the source of a steady trickle of
support tickets, broken agents, and compliance findings that linger for
months. Done well, it is a published timeline, a shrinking traffic share,
a final sunset date that holds, and an auditable last-request record.

The exercise asks the learner to author the **deprecation runbook** — not
the rollout of the replacement model (that is ex-01/ex-02), but the
coordinated shutdown of the predecessor. It overlaps with
[[exercise-02-blue-green-model-swap-walkthrough]] (the swap itself) and
with [[exercise-05-eval-gate-contract-with-peer-tracks]] (the replacement
must pass the gate before deprecation begins) but adds the long-horizon
communications, forced-migration, and sunset-day surfaces that neither
of those cover.

### Why a runbook, and why separate from the rollout

- **Users and dependent services need lead time.** Unlike the short soak
  of a canary or blue-green, deprecation cadence is measured in weeks or
  months. The platform publishes a sunset date and holds it; dependent
  services plan around it.
- **The failure mode is quiet.** Rollout failure is loud (a canary
  rolling back pages everyone). Deprecation failure is quiet — a
  dependent service that didn't migrate before sunset gets 410 responses
  at 02:00 and only a few users notice. A runbook forces the discovery
  of these services *before* sunset.
- **The decision to turn off is a governance decision.** Who authorises
  shutdown, who confirms compliance obligations have been met, and who
  retains the archived weights (for forensics and legal hold) sits
  outside day-to-day ops. A runbook captures the sign-offs so they are
  not improvised on sunset day.

### Deliverable summary

- A **phased runbook** (§2) covering announcement, soft shutdown (hard
  deprecation warnings, forced migration windows), traffic drawdown,
  sunset, and post-sunset archive.
- A **template announcement** and a **template dependent-service notice**
  (§2.6) tuned to inference-platform users.
- A **sunset-day checklist** (§2.7) that the on-call runs against.
- A **rubric** (§4) for review.

## 2. Implementation — worked answer

### 2.1 Phase 0 — Prerequisites

Before entering Phase 1, the following must be true:

- A **replacement model** is in production and has been serving for a
  pre-committed stabilisation window — typically at least two full sprint
  cycles without a related incident. The gate record from
  [[exercise-05-eval-gate-contract-with-peer-tracks]] for the replacement
  is attached.
- The **business owner** (product or platform) has signed off on the
  deprecation. Deprecating a model because "it's old" without an owner
  committing to the user-impact decision is the first failure mode.
- **Legal / Compliance hold** considerations have been reviewed. If the
  predecessor is under a hold (regulatory, customer contract, litigation
  preservation), the weights and inference logs must be retained beyond
  sunset even though the endpoint is turned off.
- A **dependent-service inventory** has been collected — either from
  request-origin metrics (who calls the endpoint with which API keys /
  tokens) or from an organisational registry. The inventory is the
  denominator for migration tracking.

### 2.2 Phase 1 — Announcement (D-90)

**Goal:** everybody who matters knows the sunset date and the migration
path.

Actions:

1. Publish the announcement document (template in §2.6). It names the
   model, the sunset date, the replacement, the migration guide, and the
   contact for migration help.
2. Send targeted notices to the top-N dependent services from the
   inventory. "Top N" means the services whose traffic share * SLA
   sensitivity is above a threshold the business owner signs off on.
3. Open a dashboard panel showing **predecessor traffic share** and
   **dependent-service migration status**. Both are public inside the
   org.
4. Freeze the predecessor: no new fine-tunes, no engine upgrades, no
   configuration changes that would change its outputs. A deprecated
   model is a stable model.

**Rollback / adjust trigger:** if the dependent-service inventory
surfaces a service whose migration is blocked by an external vendor's
timeline, adjust the sunset date before announcement goes out — do not
announce a date and then move it. Announced dates are commitments.

### 2.3 Phase 2 — Soft deprecation (D-90 → D-30)

**Goal:** dependent services migrate; those that can't are escalated.

Actions:

1. The predecessor's endpoint begins emitting deprecation warnings in
   response headers (`Deprecation: true`, `Sunset: <date>`, and a `Link`
   to the migration guide — per the IETF HTTP Deprecation and Sunset
   header conventions where supported).
2. Weekly review of the migration dashboard. Any service whose share
   has not declined is contacted directly by the Rollout Lead's
   counterpart (service owner) — not via an automated nag.
3. For any service that has not migrated by D-45, the Business Owner
   sends a direct message with an escalation path (name a specific
   decision-maker; a mail to a group is not an escalation).

**Rollback / adjust trigger:** the sunset date does not move for
convenience; it moves only for a documented external blocker and only
once. A second slip is a signal that the replacement is not actually
ready.

### 2.4 Phase 3 — Forced migration windows (D-30 → D-7)

**Goal:** make it impossible to ignore the deprecation.

Actions:

1. On a pre-announced schedule — typically one or two short windows per
   week — the predecessor returns `503 Service Unavailable` with a
   `Retry-After` and a message naming the replacement. These are
   **planned brown-outs**, announced at least seven days ahead.
2. The brown-outs make dependent services exercise their migration code
   paths. A service that falls over during a brown-out finds out at 10am
   on an announced day, not at 02:00 on sunset day.
3. Any service that cannot survive a brown-out is explicitly
   whitelisted by the Business Owner for a narrower window, with a
   written plan to resolve before sunset.

**Rollback / adjust trigger:** if a brown-out breaks a critical service
that was supposed to have migrated, the on-call ends the brown-out
window (standard incident handling) *but* the sunset date holds — the
response is to raise the service's priority, not to extend the model's
life.

### 2.5 Phase 4 — Sunset (D-0)

**Goal:** the predecessor's endpoint is turned off cleanly with a
fully-documented last-request record.

Actions:

1. Pre-sunset window (T-24h): final notice sent to anyone still calling
   the endpoint. Any request in the last 24h surfaces an explicit header
   with a stronger wording: `Sunset: <date> (in <hours>h)`.
2. At T-0, the endpoint returns `410 Gone` with a stable body that names
   the replacement and the migration guide. `410` (not `404`) is
   intentional — it tells clients the resource has been intentionally
   removed, not that it was never there.
3. The last-request log is archived: the endpoint retains the final
   seven days of request metadata (not full prompts, which may be in
   compliance scope) for forensics.
4. The inference fleet for the predecessor scales down to zero pods,
   but its deployment manifest is kept in Git under a `deprecated/`
   path for reconstruction purposes. The checkpoint is retained in cold
   storage per Phase 5's archive policy.
5. The on-call runs the sunset-day checklist in §2.7.

**Rollback / adjust trigger:** sunset has no rollback in the usual sense
— the endpoint is off. If a critical service is discovered at sunset to
still be on the predecessor, the response is incident-management, not
re-enablement. Re-enabling a sunset model is a governance event that
requires Business Owner, Rollout Lead, and Compliance sign-off; it is
not a one-button recovery.

### 2.6 Phase 5 — Post-sunset archive (D+0 → forever)

**Goal:** the predecessor's artefacts are retrievable for forensics,
compliance, or reconstruction needs.

Actions:

1. The checkpoint, tokenizer, prompt-template bundle, and the
   deprecated deployment manifest are stored in cold archive (object
   store with versioned, immutable storage).
2. The archive entry includes: SHA256 of the checkpoint, the gate
   record from the predecessor's most recent production promotion, the
   final seven days of request-metadata log reference, a pointer to
   any related post-mortems, and the sunset-day checklist outcome.
3. Compliance hold tags are applied where applicable. The archive
   policy's retention window is at least two years, longer if required
   by the Compliance track from
   [[exercise-05-eval-gate-contract-with-peer-tracks]].
4. The deployment manifest in Git under `deprecated/` is read-only; a
   CI check prevents live traffic from being routed to the deprecated
   path (even if someone revives the manifest by mistake).

### 2.7 Templates

#### 2.7.1 Announcement template

```markdown
# Model Deprecation Notice: <model-name> → <replacement-name>

**Status:** scheduled
**Sunset date:** <YYYY-MM-DD> (D-0)
**Publisher:** Inference Platform
**Contact:** inference-platform@<org> · migration-help@<org>

## What is changing

<model-name> (<revision>) will stop accepting requests on <date>. The
replacement is <replacement-name> (<revision>), which has been serving
production traffic since <date> and has passed the eval gate
(record: <link>).

## What you need to do

1. Point clients at <replacement-endpoint>. The API shape is
   <same | changed>. Migration guide: <link>.
2. Run your staging tests against the replacement before <D-30>.
3. Expect brown-outs on the predecessor between <D-30> and <D-7>; these
   are announced below.

## Timeline

| Date | Event |
|---|---|
| D-90 | This announcement (today) |
| D-60 | Response headers begin advertising deprecation |
| D-30 | Brown-out windows begin (see schedule) |
| D-7  | Final notice window |
| D-0  | Endpoint returns 410 Gone |

## Brown-out schedule

<Specific windows, each ≥ 7 days in advance.>

## Who to ask

- API shape / migration code: <migration-help> channel.
- Response-quality questions on the replacement: <model-owner>.
- Stuck on external blocker: <rollout-lead> + <business-owner>.

## What happens if we don't migrate

On D-0 your calls to <model-name> will return 410. The response body
names the replacement endpoint. There is no re-enablement path.
```

#### 2.7.2 Dependent-service direct notice (shorter, sent individually)

```markdown
Subject: Action required — migrate off <model-name> by <D-0>

Hello <service-owner>,

Your service <service-name> is calling <model-name> at a rate of
<requests/day>. That model's sunset is <YYYY-MM-DD>.

Please migrate to <replacement-name>. Guide: <link>. We expect
brown-outs starting <D-30>; your service should survive those.

If you have a blocker, please reply with the specifics so we can
sequence appropriately. The sunset date is a commitment; the blocker
response is to raise your migration priority, not to delay sunset.

Thanks,
Rollout Lead
```

#### 2.7.3 Sunset-day checklist

- [ ] T-24h: final-notice header deployed; dashboard snapshot archived.
- [ ] T-01h: on-call paged; Business Owner + Rollout Lead on call; comms
      channel open.
- [ ] T-0: endpoint flipped to `410 Gone`; response body validated
      against template.
- [ ] T+10m: predecessor fleet scaled to zero; final metrics snapshot
      taken.
- [ ] T+30m: any remaining caller IPs / API keys / tokens identified from
      the last-request log; escalation list sent to Rollout Lead.
- [ ] T+24h: last-request archive sealed; checkpoint cold-stored;
      Git manifest moved under `deprecated/<model-name>/`.
- [ ] T+7d: post-sunset review opened; action items (if any) tracked per
      [[exercise-04-mock-incident-rca-and-postmortem]].

### 2.8 Mapping the timeline to risk

The whole shape is defensive:

- **Announcement (D-90):** removes "we didn't know" as a reason.
- **Soft deprecation (D-90 → D-30):** uses headers and dashboards so
  that dependent-service owners can discover the deadline at their own
  pace.
- **Forced migration (D-30 → D-7):** reveals the services that would
  have broken on sunset day.
- **Sunset (D-0):** `410 Gone` signals intent; last-request archive
  captures the tail.
- **Archive (D+0 → forever):** NIST AI RMF **Manage** function; OWASP ML
  supply-chain integrity discipline (the checkpoint is still an artefact
  the org is responsible for even after it is not being served).

## 3. Validation steps

A reviewer checking a learner's runbook should:

1. **Phases 0–5 are all present.** A runbook that skips the archive or
   the prerequisites fails.
2. **Prerequisite gate is explicit.** The replacement must have been
   stable for a committed window and have a gate record attached. A
   runbook that starts at announcement without a prerequisite check is
   a runbook that deprecates a model before its replacement is proven.
3. **Announced dates are commitments, not suggestions.** The runbook
   says so in writing, at least once.
4. **Brown-outs are announced, not surprise.** ≥ 7 days' notice for each
   window.
5. **`410 Gone`, not `404`.** A learner who recommends `404` has not
   understood the IETF HTTP semantics for an intentionally-removed
   endpoint.
6. **Last-request archive policy defined.** Retention duration, what is
   archived (metadata, not full prompts unless Compliance says so), and
   who holds the archive.
7. **Compliance hold considered.** Even a short sentence showing the
   runbook is aware that legal or regulatory hold may keep weights alive
   past sunset is sufficient.
8. **Re-enablement is a governance event.** A runbook that treats
   re-enabling a sunset model as a one-button rollback is misunderstanding
   the whole pattern.
9. **Template announcement is user-friendly.** It contains dates, the
   replacement, the guide, and the contact. Missing any of these means
   support will carry the deprecation on its back.
10. **Links to adjacent exercises** — ex-02 (the swap that created the
    replacement), ex-04 (the post-mortem format if deprecation goes
    wrong), ex-05 (the gate the replacement passed).

## 4. Rubric / review checklist

Score each item 0 / 1 / 2. Passing: ≥ 14 / 20.

| # | Item | Points |
|---|---|---|
| 1 | Phases 0–5 present and distinct | 2 |
| 2 | Prerequisites include replacement-stability window + attached gate record | 2 |
| 3 | Announcement at D-90 or longer with full timeline | 2 |
| 4 | Response headers used for soft deprecation (`Deprecation`, `Sunset`, `Link`) | 2 |
| 5 | Forced migration windows announced in advance, with explicit dwell schedule | 2 |
| 6 | Sunset returns `410 Gone` with a body that names the replacement | 2 |
| 7 | Post-sunset archive specified: checkpoint + manifest + last-request metadata | 2 |
| 8 | Compliance hold handling considered | 2 |
| 9 | Re-enablement is governed; not a one-click recovery | 2 |
| 10 | External framework mapping (§6) non-superficial | 2 |

## 5. Common mistakes

- **No announcement window.** A sunset date revealed a week before the
  endpoint goes off is not a deprecation; it is an outage with advance
  notice.
- **Moving the sunset date.** If the date slips, dependent services
  learn that deadlines are negotiable. The commitment to not slip is the
  one lever that makes announcement credible.
- **404 instead of 410.** `404 Not Found` says the endpoint never existed.
  `410 Gone` says it existed and has been intentionally removed. The
  distinction matters to clients that distinguish retryable from
  permanent errors.
- **No brown-outs.** Without brown-outs, you discover the broken
  dependents at T-0. Brown-outs are the cheap way to pay that discovery
  cost in advance, on a weekday, when you can fix things.
- **Full-prompt last-request archive.** Full prompts are compliance-
  sensitive. The archive is request *metadata* (timestamp, API key,
  endpoint, response code, token count), not request *bodies* — unless
  Compliance explicitly requires the body.
- **No checkpoint archive.** A deprecated model whose weights are
  deleted cannot be inspected during a later incident, cannot be
  reconstructed for compliance forensics, and cannot be retrained from.
  Weights go to cold storage, not to `/dev/null`.
- **Re-enablement by configuration flip.** If a critical service missed
  sunset, the response is incident-management of that service, not
  re-enabling the deprecated model. Re-enablement is a governance event
  with sign-offs; otherwise sunset is just a long pause.
- **No dependent-service inventory before announcement.** Announcing a
  sunset without knowing who calls the endpoint is announcing an
  incident date. The inventory is Phase 0 for a reason.
- **Writing it once.** Deprecation runbooks rot — endpoints, teams, and
  compliance obligations change. Review cadence is annual at minimum, or
  on every major org change affecting the Compliance or Platform teams.

## 6. References

- IETF — *HTTP `Deprecation` header* (RFC 9745) and *HTTP `Sunset`
  header* (RFC 8594) — practitioner standards for the soft-deprecation
  phase.
- Google SRE Workbook — *Deprecation* chapter —
  https://sre.google/workbook/ — generic deprecation pattern that this
  runbook specialises for model endpoints.
- vLLM documentation — https://docs.vllm.ai/ — for the operational
  details of scaling a model fleet to zero and retaining a manifest
  under `deprecated/`.
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/):
  - **ML05 Model Theft** — the archived checkpoint is a liability if not
    stored with the same integrity controls as production models;
    cold-store it signed and SHA-pinned.
  - **ML10 Insufficient Supply Chain Security** — the archive policy's
    signatures, pinning, and retention are a supply-chain discipline
    issue.
- [mitre-atlas](https://atlas.mitre.org/):
  - **AML.T0010 ML Supply Chain Compromise** — if the deprecated
    checkpoint is later revived from cold storage, its provenance must
    be verifiable; this is why the archive stores SHA + signatures.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework):
  - **Govern** — Business Owner sign-off; re-enablement governance;
    annual runbook review.
  - **Map** — dependent-service inventory as a Phase-0 artefact.
  - **Measure** — the migration dashboard; brown-out observations.
  - **Manage** — archive retention; compliance hold interaction;
    last-request metadata.

## Appendix — control-to-framework mapping

| Control | OWASP ML | MITRE ATLAS | NIST AI RMF |
|---|---|---|---|
| Replacement gate-record attached before Phase 1 | ML06 | AML.T0010 attestation | Govern |
| Dependent-service inventory before announcement | ML-governance adjacent | n/a | Map |
| Response headers (`Deprecation`, `Sunset`, `Link`) | ML06 | n/a | Measure |
| Brown-outs in Phase 3 | ML-governance adjacent | n/a | Measure, Manage |
| `410 Gone` with replacement pointer at sunset | ML-governance adjacent | n/a | Manage |
| Checkpoint + manifest + metadata archive with SHA + signatures | ML05, ML10 | AML.T0010 | Manage |
| Compliance hold handling | ML-governance adjacent | n/a | Govern, Manage |
| Re-enablement is a governance event | ML-governance adjacent | n/a | Govern |
