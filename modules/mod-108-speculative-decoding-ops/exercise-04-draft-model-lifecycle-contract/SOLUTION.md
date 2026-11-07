# Solution — Exercise 04: Draft-Model Lifecycle Contract

> Paired learning file: [`lessons/mod-108-speculative-decoding-ops/exercises/exercise-04-draft-model-lifecycle-contract.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-108-speculative-decoding-ops/exercises/exercise-04-draft-model-lifecycle-contract.md)
> Module: `mod-108-speculative-decoding-ops`
> Type: design / governance contract

## 1. Solution overview

The draft model (or self-speculative head — Medusa / EAGLE) is a **production
trained artefact** with its own lifecycle, separate from but pinned to the
target model's lifecycle. Teams that treat drafts as "just a config flag" ship
silent regressions when the target is updated, when the draft provider changes
its checkpoint, or when a vLLM upgrade re-interprets flag semantics. This
exercise writes down the lifecycle rules as a contract — the kind of document
an inference platform PM, a model-serving engineer, and a security reviewer can
sign off on together.

The contract covers five phases of the draft's life: **selection**, **intake**,
**pinning**, **operation**, and **retirement**. For each phase the contract
specifies what must be recorded, who approves, what observability and controls
apply, and what triggers the next phase.

### Why a contract at all

Three failure modes motivate this:

1. **Target rolled, draft didn't.** Target model updates to a newer fine-tune.
   The old draft keeps serving, but its logits no longer align with the new
   target's; acceptance rate degrades silently until it crosses the alert
   threshold from [[acceptance-rate-monitoring-and-alerts]] (exercise 03).
   Latency regresses in the meantime.
2. **Draft provenance rot.** A draft was downloaded from a public hub by name,
   but the checkpoint behind that name was updated. The deployment now runs a
   subtly different draft and nobody noticed. OWASP ML05 (Model Theft) and ML04
   (Membership Inference) are adjacent, but the direct failure mode is
   **supply-chain integrity** — see MITRE ATLAS tactics on tampered or
   substituted models.
3. **Governance invisibility.** The draft can leak training-data fingerprints or
   be fine-tuned in ways that affect safety outputs. NIST AI RMF requires the
   trained artefact to be inventoried, documented, and periodically reassessed
   — the draft is a trained artefact.

### Deliverable summary

One Markdown document (§2.2 below) containing:

- A named owner per phase.
- A checklist of recorded fields at intake.
- A pinning scheme that binds draft version to target version.
- Operational signals (links to the exercise 03 dashboard and alerts).
- A retirement procedure that does not leave a dangling draft in the registry.

Plus an appendix mapping the contract's controls to the three external
frameworks in §6.

## 2. Worked answer — the contract

### 2.1 Roles

| Role | Responsibility | Example org |
|---|---|---|
| **Draft Owner** | Overall accountability for a draft's selection, upgrades, and retirement. | Senior inference engineer on the serving team. |
| **Target Owner** | Accountability for the target model that the draft shadows. Must co-sign every draft pinning. | Model-serving tech lead. |
| **Security Reviewer** | Approves new drafts / heads against OWASP ML, MITRE ATLAS, and internal supply-chain policy. | Security engineering. |
| **On-Call Engineer** | Responds to acceptance-rate alerts from exercise 03. Not expected to approve lifecycle transitions. | Platform on-call rotation. |

No two of these should be the same person for a draft in production. A review
by the same hand that selected the draft is not a review.

### 2.2 Phase 1 — Selection

**Entry trigger:** a target model is being deployed, is being upgraded, or an
existing draft is showing acceptance regression per exercise 03.

**Required record:**

- Candidate draft(s) considered (at least two).
- Model family / tokenizer match against the target (must match for
  draft-target; must match target revision for Medusa/EAGLE heads).
- Parameter count, expected cost ratio `c = T_draft / T_target` on the
  deployment hardware.
- Expected acceptance band from either: (a) the paper the method is from, (b) a
  practitioner report on a similar deployment, or (c) a prior deployment's
  measured α. The expected band becomes the baseline against which exercise 03
  alerts are set.
- Rejection rationale for the candidates not picked.

**Approver:** Draft Owner, with Target Owner co-sign if the pick changes the
target↔draft pairing versus the prior deployment.

**Exit:** a named candidate is selected and enters Intake.

### 2.3 Phase 2 — Intake

**Entry trigger:** a candidate has passed Selection.

**Required record (the intake manifest):**

| Field | Example | Why |
|---|---|---|
| Draft identifier | `meta-llama/Llama-3.2-1B-Instruct` | human name |
| Draft revision / commit | `revision=<git-sha or hub-commit>` | pinning (see §2.4) |
| Checkpoint SHA256 | `sha256=...` | tamper detection |
| Source URL | `https://...` | provenance |
| License | `Llama 3 Community License` | legal |
| Training data summary | from model card | OWASP ML / NIST AI RMF |
| Target this draft pairs with | `meta-llama/Meta-Llama-3.1-70B-Instruct @ <rev>` | pinning |
| Method | `draft-target` / `medusa` / `eagle` | engine config |
| Measured α on intake load test | α on the three-row prompt mix used by ex-01 | baseline |
| Measured `c` (draft/target cost ratio) on hardware | from timing both forward passes | break-even derivation per ex-03 |
| Security review link | URL to the signed review document | audit trail |

**Controls:**

- The checkpoint SHA256 is pinned at intake and stored in the model registry.
  At every engine start the SHA is recomputed and matched; mismatch blocks
  startup.
- The intake load test is the same methodology used by exercise 01 (greedy
  decoding, correctness check, acceptance-rate measurement, three-row latency
  table). The resulting numbers anchor the exercise 03 break-even threshold.
- The security review is non-optional, even if the draft is from a well-known
  public hub. The checkpoint can have been republished since the hub's last
  audit.

**Approver:** Security Reviewer and Draft Owner.

**Exit:** manifest is committed to the registry and the draft is Pinned.

### 2.4 Phase 3 — Pinning

**Entry trigger:** Intake complete.

**Pinning rule (the single most important rule in the contract):**

> **The draft version (or head checkpoint) is pinned to a specific target
> revision. The draft cannot be deployed with a different target revision
> without re-entering Intake.**

Operationally, this means the deployment manifest records something like:

```yaml
# deployment-manifest.yaml (fragment)
target:
  model: meta-llama/Meta-Llama-3.1-70B-Instruct
  revision: <hub-commit-or-internal-tag>
  sha256: <...>
speculative:
  method: draft-target
  model: meta-llama/Llama-3.2-1B-Instruct
  revision: <hub-commit-or-internal-tag>
  sha256: <...>
  intake_record: registry://drafts/llama-3.2-1b-instruct/<intake-id>
  valid_for_targets:
    - meta-llama/Meta-Llama-3.1-70B-Instruct@<target-rev-1>
    - meta-llama/Meta-Llama-3.1-70B-Instruct@<target-rev-2>
```

`valid_for_targets` is the enforcement point. The deployment controller refuses
to launch an engine where the running target revision is not in the list; the
alternative — launching and letting acceptance-rate monitoring catch the drift
hours later — is slower, noisier, and places the diagnosis on the on-call.

**Controls:**

- CI check on the deployment manifest: the target revision is in the draft's
  `valid_for_targets` list. Failing this blocks the deploy.
- Registry mutation of `valid_for_targets` requires a lightweight "re-pin"
  review — sign-off from Draft Owner + Target Owner — not a full new Intake,
  provided the draft itself is unchanged. The re-pin requires a correctness
  check against the new target revision (greedy outputs match).

**Approver:** Draft Owner.

**Exit:** the pinned deployment enters Operation.

### 2.5 Phase 4 — Operation

**Entry trigger:** the pinned deployment is live.

**Signals (all from exercise 03):**

- Rolling acceptance rate `α_window`.
- Draft tokens-per-second.
- p95 latency, viewed jointly with α.

**Thresholds:** break-even `α` derived at Intake plus 10 points of headroom,
per [[acceptance-rate-monitoring-and-alerts]] §2.4.

**Operational rules:**

- Acceptance-rate alerts from exercise 03 fire to on-call, not to the Draft
  Owner. The runbook (exercise 03 §2.5) names the two-way branch between
  "draft silently off" and "acceptance degraded".
- **Silent draft swap is forbidden.** A new checkpoint for the same draft
  name (e.g., an updated Llama-3.2-1B revision) requires re-entering Intake. A
  release-gate can enforce this by refusing a deployment where the pinned
  draft SHA no longer matches the checkpoint behind the pinned name.
- **Target rolls require re-pinning.** The target cannot be updated without
  either (a) adding the new target revision to the draft's `valid_for_targets`
  through re-pin review, or (b) retraining / re-selecting the draft for the
  new target. The default is (a) if correctness + α check pass against the
  new target revision; (b) only if they do not.
- **Routine health check.** At least monthly, the Draft Owner runs the
  correctness + α check from Intake against the current pinned pair and
  records the result. Decay in α over time without a workload change is a
  signal to begin Retirement for this draft.

**Approver:** On-Call Engineer for incident response; Draft Owner for routine
checks and re-pins.

**Exit:** either the draft is retired, or a new draft is selected (returning
the lifecycle to Phase 1 while this one is in Operation on the old target
revision).

### 2.6 Phase 5 — Retirement

**Entry trigger:** any of:

- Target model class has changed (e.g., Llama-3 → Llama-4).
- Measured α has sustained below the operational threshold despite re-pinning.
- Security review has revoked approval (e.g., training-data provenance issue
  surfaced after intake, OWASP ML / ATLAS advisory against the checkpoint).
- The draft's license has been revoked or materially changed.

**Required record:**

- Reason for retirement (one of the triggers above).
- New replacement draft (if any) — cross-referenced to its Selection record.
- Final observed α at retirement, for historical trend.

**Controls:**

- The retired draft's `valid_for_targets` list is emptied in the registry; any
  deployment that tries to pin it will fail CI.
- The checkpoint stays in the registry for forensics but is marked `retired`
  and non-deployable.
- If a replacement is already in place, the replacement's Pinning record must
  cover the target revisions that the retired draft was serving, or a
  coordinated target revision roll is scheduled.

**Approver:** Draft Owner, with Security Reviewer sign-off if retirement is
security-driven.

**Exit:** the draft is marked retired in the registry.

## Implementation

The contract is text, but the controls behind it are concrete artefacts. A
working implementation of this contract on an inference platform looks like:

- **Registry schema.** The draft registry stores one record per draft with
  fields matching the Intake manifest table in §2.3 (identifier, revision,
  SHA256, source URL, license, training-data summary, paired target revision,
  method, measured α, measured `c`, security-review link). The record also
  carries `status ∈ {intake, pinned, operational, retired}` and the
  `valid_for_targets` list from §2.4. The registry is the single source of
  truth; the deployment manifest only references registry IDs.
- **Deploy-time CI check.** A pipeline step reads the deployment manifest,
  resolves the pinned draft record from the registry, and fails the deploy if
  the running target revision is not in `valid_for_targets`, or if the
  registry `status` is `retired`. This is the enforcement for the pinning
  rule in §2.4 and the retirement rule in §2.6.
- **Engine-start integrity check.** On vLLM startup, the engine recomputes
  the checkpoint SHA256 of the draft (and head, for Medusa/EAGLE) and matches
  it against the registry. Mismatch blocks startup — this is what makes a
  silent draft swap (§2.5) impossible rather than just discouraged.
- **Operation telemetry.** The exercise 03 dashboard and alerts
  ([[acceptance-rate-monitoring-and-alerts]]) are the Operation phase's
  observable surface — no new dashboards. Alerts route to on-call; the
  runbook branches on "draft silently off" vs "acceptance degraded".
- **Scheduled health check.** A monthly job re-runs the Intake correctness +
  α test against the current pinned pair and writes the result to the
  registry record. A decaying α without a workload change opens a Retirement
  review automatically.
- **Re-pin workflow.** A lightweight review tool lets the Draft Owner +
  Target Owner add a new target revision to `valid_for_targets` after a
  passing correctness + α check. The same tool emits an auditable record
  tying the new revision to the two signers.

None of the above requires bespoke tooling — the registry can be a Git-backed
YAML directory, the CI check a small script, and the engine-start SHA check a
few lines in the engine's startup path. What matters is that each of the
controls in §2.1–§2.6 has exactly one enforcement point and that the point is
automated.

## 3. Validation steps

A reviewer validating a learner submission should do the following, in order.

1. **All five phases named, with entry trigger / required records / controls /
   approver / exit for each.** Missing any one phase fails.
2. **Roles separated.** Draft Owner, Target Owner, Security Reviewer, On-Call
   Engineer are distinct roles with distinct responsibilities. A submission that
   collapses roles ("the on-call owns everything") fails on separation-of-duties.
3. **Pinning rule explicit.** Phase 3 names the one rule — draft pinned to
   target revision; silent swap forbidden — and shows a manifest fragment or
   equivalent enforcement artefact.
4. **Operation phase references exercise 03.** The signals, thresholds, and
   runbook from [[acceptance-rate-monitoring-and-alerts]] are cited, not
   re-invented. A contract that lists a new ad-hoc dashboard here fails.
5. **Retirement is a phase, not a comment.** A submission that treats
   retirement as "we'll figure it out when it happens" fails on lifecycle
   completeness.
6. **Mapping to external frameworks.** The appendix / cross-reference maps the
   controls to OWASP ML Top 10, MITRE ATLAS, and NIST AI RMF. Vague gestures
   ("supports OWASP") are not enough; a submission names at least one item per
   framework and connects it to a control.
7. **Self-speculative variant.** The contract explicitly addresses whether the
   same rules apply to Medusa / EAGLE heads (they do, with a note that the
   "draft" is a head checkpoint rather than a separate model, and that the
   head is more tightly target-coupled than a separate draft).

## 4. Rubric / review checklist

Score each item 0 (missing), 1 (partial), 2 (fully correct). Passing: ≥ 10 / 14.

| # | Item | Points |
|---|---|---|
| 1 | Five phases named with entry/exit, records, controls, approver | 2 |
| 2 | Roles defined with separation-of-duties enforced across phases | 2 |
| 3 | Intake manifest captures SHA256, revision, license, training-data notes, measured α, measured `c` | 2 |
| 4 | Pinning rule binds draft version to target revision and is enforced by CI | 2 |
| 5 | Operation phase references exercise 03 dashboard, alerts, and runbook | 2 |
| 6 | Retirement is a distinct phase with triggers and residual-checkpoint handling | 2 |
| 7 | Controls mapped to OWASP ML Top 10, MITRE ATLAS, NIST AI RMF with at least one item each | 2 |

## 5. Common mistakes

- **Treating the draft as a tunable, not a trained artefact.** The draft has
  parameters, training data, license, and lineage. A contract that governs
  "the flag" instead of "the artefact" fails on every framework.
- **Pinning the draft to the target *name* instead of the target *revision*.**
  Model names stay stable while weights change (fine-tunes land under the same
  hub name). Pin to revision / commit / SHA, not to display name.
- **Making the on-call the Draft Owner.** The on-call responds to alerts; they
  do not own lifecycle decisions. Making them accountable for upgrades creates
  pressure to ignore alerts.
- **Letting re-pinning become routine without a correctness + α check.** The
  entire point of re-pinning is to catch the case where the new target
  revision silently breaks acceptance. Skipping the check makes the pin
  meaningless.
- **Forgetting retirement.** Retired drafts left in the registry get
  accidentally re-selected by a copy-paste deploy months later. The registry
  must mark them non-deployable.
- **Hand-waving the security review.** "Downloaded from the official hub" is
  not a review. The review looks at: training-data licensing / provenance,
  checkpoint integrity (SHA, signatures if available), supply-chain controls
  (does the hub tag correspond to a reproducible build?), and OWASP ML /
  ATLAS advisories.
- **Writing the contract as a one-time doc.** The contract must have an owner
  and a review cadence (at least annually, plus on any major vLLM upgrade or
  target class change). A frozen contract rots the same way frozen pinned
  dependencies rot.
- **Treating Medusa / EAGLE heads as "not a draft".** Heads are trained
  artefacts. Omitting them from the contract leaves the self-speculative
  deployments ungoverned.
- **Collapsing the pinning rule into a Prometheus alert.** Alerts are detection,
  not prevention. The deploy-time CI check is the enforcement; the alert is a
  safety net.

## 6. References

- Leviathan, Y., Kalman, M., Matias, Y. *Fast Inference from Transformers via
  Speculative Decoding.* ICML 2023. https://arxiv.org/abs/2211.17192 — the
  method whose draft this contract governs.
- Cai, T. et al. *Medusa: Simple LLM Inference Acceleration Framework with
  Multiple Decoding Heads.* ICML 2024. https://arxiv.org/abs/2401.10774 —
  self-speculative heads that this contract also covers.
- Li, Y. et al. *EAGLE: Speculative Sampling Requires Rethinking Feature
  Uncertainty.* ICML 2024. https://arxiv.org/abs/2401.15077 — feature-level
  self-speculative draft; same lifecycle rules apply.
- vLLM documentation. *Speculative Decoding.* https://docs.vllm.ai/ —
  practitioner implementation; informs the deployment-manifest schema in §2.4.
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/)
  — governance framing for the Intake and Retirement phases. Items most
  directly applicable:
  - ML05 Model Theft / tampered-artefact handling (SHA pinning, signed
    checkpoints).
  - ML06 Output Integrity (silent-fallback speculation is an integrity-adjacent
    failure mode).
  - ML10 Insufficient Supply Chain Security (checkpoint provenance, hub
    republishing risk).
- [mitre-atlas](https://atlas.mitre.org/) — tactics map:
  - Supply-chain compromise of a third-party model (ATLAS AML.T0010).
  - Model substitution / tampered-artefact detection by SHA checks.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework) — the
  four functions map onto this contract:
  - **Govern:** roles and separation-of-duties in §2.1; approver gates in each
    phase.
  - **Map:** Selection and Intake records characterize the artefact.
  - **Measure:** Operation phase signals from exercise 03 and the monthly
    health check.
  - **Manage:** Retirement procedure and the registry-level non-deployable
    marking.
- Sigstore / Cosign (if available in the deployment stack) — a reasonable
  implementation of checkpoint signing for the Intake SHA field; verify
  compatibility with the project's registry.

## Appendix — control-to-framework mapping

| Control | OWASP ML | MITRE ATLAS | NIST AI RMF |
|---|---|---|---|
| SHA256 pinned at Intake, re-checked on every engine start | ML05, ML10 | AML.T0010 (Supply-chain compromise) | Map, Manage |
| `valid_for_targets` enforced at deploy | ML06 | AML.T0015 (Unauthorized use / substitution pattern) | Govern, Manage |
| Separation of roles (Draft / Target / Security / On-Call) | ML-governance adjacent | n/a | Govern |
| Security review at Intake and at Retirement | ML10 | ATLAS case studies on tampered models | Govern, Map |
| Monthly correctness + α health check | ML06 | Detection discipline | Measure |
| Retired drafts marked non-deployable in registry | ML05 | Prevents stale-artefact revival | Manage |
