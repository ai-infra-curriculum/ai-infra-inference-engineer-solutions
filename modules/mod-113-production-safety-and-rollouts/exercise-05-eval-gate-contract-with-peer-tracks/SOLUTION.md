# Solution — Exercise 05: Eval-Gate Contract with Peer Tracks

> Paired learning file: [`lessons/mod-113-production-safety-and-rollouts/exercises/exercise-05-eval-gate-contract-with-peer-tracks.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-113-production-safety-and-rollouts/exercises/exercise-05-eval-gate-contract-with-peer-tracks.md)
> Module: `mod-113-production-safety-and-rollouts`
> Type: design / governance contract

## 1. Solution overview

An **eval gate** is the automated promotion criterion that sits between
"model owner says this candidate is ready" and "rollout controller is
allowed to canary or blue-green the candidate." It is the most powerful
control on an inference platform because it is the one line of defence
that runs *before* any live user is exposed.

The problem the exercise targets: eval gates built by model teams in
isolation share one failure mode — they measure what the model team cares
about and are silent on what everybody else cares about. The fine-tune
team scores a candidate on helpfulness; the safety team cares about
refusal quality; the platform team cares about tokens-per-request; the
compliance team cares about data-handling behaviour. "The eval gate
passed" means very different things to each.

The exercise's deliverable is an **eval-gate contract with peer tracks**:
a document that names a small set of **peer tracks** (model quality,
safety, cost, compliance, structural / schema), assigns each track an
owning team, requires each track to publish its own pass/fail criteria,
and requires *all* tracks to pass before promotion. Each track is autonomous
inside its scope — the model team does not review the safety suite, and
vice versa — but no track can veto by inaction, and no track can promote
unilaterally.

This is the control that the post-mortem in
[[exercise-04-mock-incident-rca-and-postmortem]] found inadequate when it
had been scoped too narrowly. It is also the precondition the blue-green
walkthrough in [[exercise-02-blue-green-model-swap-walkthrough]] names as
load-bearing, and the "quality signal is precomputed" source for the
canary policy in
[[exercise-01-canary-with-response-diffing-and-auto-rollback]].

### Why peer tracks (and not one big suite)

Three reasons:

1. **Scope separation of concerns matches team ownership.** The safety
   team should not have to review a quality evaluator's rubric to veto a
   candidate; the quality team should not need to understand jailbreak
   taxonomy to contribute. Peer tracks let each team ship a suite in its
   own language.
2. **Independent failure.** If the quality track is red, the safety track
   should still run. An all-in-one suite's single flake halts the whole
   gate; peer tracks let the on-call Model Owner see exactly which tracks
   failed and which passed.
3. **Audibility.** NIST AI RMF's Govern function and OWASP ML's governance
   adjacencies both expect sign-offs to be attributable. A peer-tracked
   gate makes "the safety team signed off" a distinct assertion from
   "the quality team signed off."

### Deliverable summary

- The **contract document** (§2.1): roles, tracks, pass/fail rules,
  record-keeping requirements, review cadence.
- A **manifest schema** (§2.2) that CI consumes to determine whether a
  candidate has passed every peer track.
- A **minimal CI implementation** (§2.3): a short script that reads the
  manifest, confirms each track's record is present and PASS, and returns
  a single exit code the rollout controller gates on.
- A **rubric** (§4) for reviewing a learner's contract.

## 2. Worked answer

### 2.1 The contract

#### 2.1.1 Roles

| Role | Responsibility |
|---|---|
| **Gate Owner** | Owns the contract itself; schedules reviews; adjudicates track disputes. |
| **Track Owner** (one per track) | Authors and maintains the track's suite; signs off on each candidate's track result. |
| **Model Owner** | Submits candidates; accountable for the candidate's intent. Cannot be a Track Owner on the same candidate's run. |
| **Rollout Lead** | Consumer of the gate: cannot promote without a signed gate record. |

Gate Owner and any single Track Owner must be different people. Model
Owner and any Track Owner on the same candidate must be different people
(no self-signoff). Disputes within a track are resolved by the Gate Owner,
but the Gate Owner cannot flip a track's verdict — they can only extend
the timeline, require re-run, or escalate.

#### 2.1.2 The five peer tracks

The contract names five tracks. Each track has an owner, a scope, a set
of required measurements, and a published pass/fail rule.

| Track | Owner | Scope | Representative measurements | Pass rule (illustrative) |
|---|---|---|---|---|
| **Quality** | Model team | Task-level helpfulness / correctness on the deployed workload | Task-specific benchmark deltas vs current production | No measured regression > X% on any blocking axis |
| **Safety / abuse** | Trust & Safety team | Refusal behaviour, jailbreak resistance, PII handling in outputs | Red-team suite results; standardised prompt batteries | No new high-severity findings; regression metrics within ε of baseline |
| **Structural / schema** | Platform team | Response shape under agent / tool-call / structured-output use | Schema-conformance rate on parser-relevant turns | Shape-conformance rate ≥ baseline on all audited parsers |
| **Cost / latency** | Platform team | Tokens per request, cost per request, latency distribution | Token spend delta vs current; p95 latency delta | Cost per request ≤ baseline × (1 + budget); p95 latency within SLO |
| **Compliance** | Compliance / Legal | Data-handling behaviour where applicable (PII leakage, licence terms) | Standardised compliance probes | No findings in blocking category |

The illustrative pass rules are **starting points** — each track's owner
publishes its own binding rule in a track-manifest file under its team's
ownership. The gate's job is not to know the rule; the gate's job is to
know which file to read for each track and to require that the file
reports PASS.

#### 2.1.3 Pass / fail rule for the gate as a whole

> The gate passes only if every peer track produces a signed PASS record
> for this candidate within a pre-committed timeout. Any FAIL, any
> TIMEOUT, or any absent track record is a gate-level FAIL.

The gate does not weigh tracks against one another. A compliance FAIL is
not overridable by a quality PASS, and vice versa. If a track wants to
make its own rule probabilistic ("regression up to 2% on axis A is
acceptable"), that goes into the track's rule — not the gate's.

#### 2.1.4 Record-keeping (the gate's output)

The gate emits a single **gate record** per candidate:

- Candidate identifier (model + revision + engine version + tokenizer).
- Per-track entry: owner, suite version, PASS/FAIL, timestamp, signer,
  link to the track's own result artefact.
- Overall PASS/FAIL.
- Signature of the Gate Owner.

The gate record is attached to the rollout ticket consumed by
[[exercise-02-blue-green-model-swap-walkthrough]] and
[[exercise-01-canary-with-response-diffing-and-auto-rollback]]. A rollout
ticket without an attached gate record is rejected by the rollout
controller — this is the single enforcement point that keeps the contract
load-bearing.

#### 2.1.5 Review cadence

- The contract itself is reviewed quarterly by the Gate Owner and all
  Track Owners jointly. Changes to the set of tracks require Gate Owner +
  Model Owner + Rollout Lead sign-off.
- Each track's rule is reviewed by its own team at least every two
  releases of the relevant suite; stale suites are a common cause of
  gate-rot.
- The post-mortem in
  [[exercise-04-mock-incident-rca-and-postmortem]] can file a P0 action
  item to add a new measurement or a new track. The action item is
  completed when the Gate Owner acknowledges and the Track Owner's
  suite updates ship.

### 2.2 Gate manifest schema

A single YAML file per candidate, written to the rollout-tickets
repository. The schema is intentionally shallow so that any CI runner can
consume it; the depth lives in each track's own linked artefact.

```yaml
# gate-record-<candidate-id>.yaml
version: 1
candidate:
  model: meta-llama/Meta-Llama-3.1-70B-Instruct
  revision: <hub-commit-or-internal-tag>
  checkpoint_sha256: <sha256>
  engine: vllm
  engine_version: <semver>
  tokenizer_revision: <hash or name@rev>

gate:
  owner: inference-platform/gate-owner
  opened_at: 2026-11-05T19:12:03Z
  closes_at: 2026-11-07T19:12:03Z   # pre-committed timeout
  overall: null                      # PASS | FAIL | PENDING

tracks:
  - name: quality
    owner: team/model-chat
    suite_version: chat-eval@v4.2.1
    result: PASS                     # PASS | FAIL | PENDING
    signed_by: j.model-owner
    signed_at: 2026-11-06T14:22:10Z
    artefact: https://evals.internal/runs/qual-7781
  - name: safety
    owner: team/trust-and-safety
    suite_version: ts-eval@v2.9.0
    result: PASS
    signed_by: k.safety-reviewer
    signed_at: 2026-11-06T16:01:44Z
    artefact: https://evals.internal/runs/safety-4412
  - name: structural
    owner: team/inference-platform
    suite_version: shape-eval@v1.3.0
    result: PASS
    signed_by: a.platform-eng
    signed_at: 2026-11-06T17:11:08Z
    artefact: https://evals.internal/runs/shape-2201
  - name: cost
    owner: team/inference-platform
    suite_version: cost-eval@v1.1.2
    result: PASS
    signed_by: a.platform-eng
    signed_at: 2026-11-06T17:28:55Z
    artefact: https://evals.internal/runs/cost-2202
  - name: compliance
    owner: team/compliance
    suite_version: comp-eval@v0.6.3
    result: PASS
    signed_by: c.compliance-reviewer
    signed_at: 2026-11-07T09:04:12Z
    artefact: https://evals.internal/runs/comp-1118
```

Two schema invariants the CI enforces:

- Every track named in the contract is present.
- `signed_by` for a track is not the same identity as `candidate.model`'s
  Model Owner (no self-signoff).

### 2.3 Minimal CI implementation

A small Python script wraps the manifest, makes the single pass/fail
decision, and exits 0 or non-zero for the rollout controller to gate on.

```python
"""Gate evaluator.

Reads gate-record-<candidate-id>.yaml and returns exit code 0 iff every
required track has result == PASS, every track is signed, and the gate's
pre-committed timeout has not elapsed. Prints a structured report to
stdout for the rollout ticket.
"""

from __future__ import annotations
import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import yaml  # dependency carried by the rollout-tickets repo's CI image

REQUIRED_TRACKS = {"quality", "safety", "structural", "cost", "compliance"}


def _now_utc() -> _dt.datetime:
    # Caller injects the clock in tests; in CI we accept wall clock.
    return _dt.datetime.now(tz=_dt.timezone.utc)


def _parse_time(s: str) -> _dt.datetime:
    # Accept Z or offset forms.
    s = s.replace("Z", "+00:00")
    return _dt.datetime.fromisoformat(s)


def evaluate(manifest: dict, now: _dt.datetime) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    gate = manifest.get("gate", {})
    closes_at = _parse_time(gate["closes_at"])
    if now > closes_at:
        reasons.append(f"gate timeout: closed at {gate['closes_at']}")

    tracks_in = {t["name"]: t for t in manifest.get("tracks", [])}
    missing = REQUIRED_TRACKS - tracks_in.keys()
    if missing:
        reasons.append(f"missing tracks: {sorted(missing)}")

    model_owner = manifest.get("candidate", {}).get("model_owner", "")
    for name in sorted(REQUIRED_TRACKS & tracks_in.keys()):
        t = tracks_in[name]
        if t.get("result") != "PASS":
            reasons.append(f"track {name}: result={t.get('result')}")
            continue
        if not t.get("signed_by"):
            reasons.append(f"track {name}: unsigned")
            continue
        if model_owner and t.get("signed_by") == model_owner:
            reasons.append(f"track {name}: self-signoff by model owner")

    return (not reasons, reasons)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    args = ap.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text())
    ok, reasons = evaluate(manifest, _now_utc())
    report = {
        "candidate": manifest.get("candidate", {}),
        "gate_overall": "PASS" if ok else "FAIL",
        "reasons": reasons,
    }
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

Deliberate design choices:

- The script knows the track **names** but not the rules. Rules live
  inside the tracks; this script would be identical if a new track were
  added and `REQUIRED_TRACKS` updated.
- The self-signoff check depends on the candidate recording its Model
  Owner identity. This is a small addition to the schema that pays off
  every release.
- The timeout is pre-committed in the manifest, not inferred from "when
  did the gate open." A rollout that drags past its timeout is explicit
  about doing so.

### 2.4 Worked example — how the gate blocks a bad candidate

Return to the mock incident in
[[exercise-04-mock-incident-rca-and-postmortem]]: a fine-tune that broke
the downstream tool-call parser. The post-mortem's AI-1 is "add a tool-
call-shape suite to the eval gate."

Under the peer-track contract, that suite is a Platform-owned addition to
the **structural** track. After AI-1 is complete, the next candidate's
`shape-eval` includes tool-call-shape conformance. If the new fine-tune
regresses on that axis, `tracks[structural].result = FAIL` and the gate's
overall result is FAIL regardless of the other tracks' verdicts. The
rollout controller refuses to canary the candidate.

This is the shape of control the gate is supposed to provide: not that
every possible regression is caught, but that any regression a team has
written a suite for is caught, and that the gate's PASS actually reflects
the aggregated verdict of the tracks that exist.

## 3. Validation steps

A reviewer checking a learner's contract should:

1. **At least four tracks are named** (quality, safety, and at least two
   of structural / cost / compliance). A contract with one track is not a
   peer-track contract.
2. **Each track has an owner that is a different team from the Model Owner.**
   Reject contracts where the Model team owns the quality *and* structural
   *and* cost tracks — that is not peer separation.
3. **Pass / fail rule at the gate level is the AND of all tracks.** Reject
   any contract that lets a quality PASS override a compliance FAIL.
4. **No self-signoff.** The submission should state and the CI should
   enforce that no track's signer may be the candidate's Model Owner.
5. **Manifest schema is minimal and consumable.** A schema with free-form
   fields for each track's rule is a sign the author is trying to put
   track-specific logic into the gate.
6. **Pre-committed timeout.** The gate has a `closes_at`; breaching it is
   a FAIL. A gate that "stays open until all tracks pass" is a gate that
   silently blocks releases.
7. **Review cadence stated.** The contract itself is reviewed quarterly;
   track suites by their own teams per release.
8. **Post-mortem integration.** The contract explicitly describes how an
   RCA action item (ex-04 style) enters the gate (as a track addition or a
   track-suite update). A contract with no feedback loop from incidents
   rots quickly.
9. **Rollout-controller enforcement.** The submission shows the single
   point where the controller checks for the gate record. Without this,
   the contract is an advisory.

## 4. Rubric / review checklist

Score each item 0 / 1 / 2. Passing: ≥ 14 / 20.

| # | Item | Points |
|---|---|---|
| 1 | Four or more tracks named, each with a distinct owning team | 2 |
| 2 | Each track publishes its own pass/fail rule; gate does not override | 2 |
| 3 | Gate is strict AND across tracks; no weighting between tracks | 2 |
| 4 | No self-signoff (Model Owner ≠ track signer) enforced in schema + CI | 2 |
| 5 | Pre-committed timeout with explicit FAIL on breach | 2 |
| 6 | Manifest schema minimal enough that adding a track is a one-line change | 2 |
| 7 | Gate record attached as precondition at the rollout controller | 2 |
| 8 | Post-mortem feedback loop (ex-04 action items feed the gate) | 2 |
| 9 | Review cadence for contract and track suites explicit | 2 |
| 10 | External framework mapping (§6) non-superficial | 2 |

## 5. Common mistakes

- **One big suite labelled "the eval gate".** The failure mode this
  exercise exists to prevent. If every evaluator is in one repo owned by
  the Model team, the gate measures what the Model team measures. Peer
  tracks force the ownership split.
- **Gate owner can flip track verdicts.** The Gate Owner adjudicates
  procedure and timeouts; they cannot overrule a FAIL from a Track Owner.
  A contract that lets the Gate Owner override a compliance FAIL is a
  contract that will produce a compliance incident.
- **Model Owner signs the quality track.** The single most common
  self-signoff shape in practice. The quality track's signer is a peer on
  the Model team, not the Model Owner for this candidate.
- **No timeout.** Gates without timeouts become permanent WIP states that
  silently prevent releases and then get bypassed. Commit to a timeout;
  if it breaches, write a post-mortem.
- **Track rules in the gate manifest.** A manifest that encodes "regression
  up to 2% acceptable on axis A" leaks the track's logic into the gate.
  Put the rule inside the track's own artefact; the manifest records
  only the verdict and a link.
- **Rollout controller consults the manifest but doesn't require it.** A
  "required-when-present" gate is not a gate. The controller must fail
  closed when the manifest is absent.
- **No feedback loop from post-mortems.** Gates that don't grow after
  incidents fossilise. The RCA → action-item → new track-suite update
  path is the liveness condition.
- **Track suites not versioned.** A gate PASS is only meaningful if you
  know which version of the suite produced it. Pin `suite_version` in
  every track entry.

## 6. References

- Google SRE Workbook — *Release Engineering* and *Managing Risk* —
  https://sre.google/workbook/ — the generic release-gate pattern this
  exercise specialises for model artefacts.
- Anthropic / OpenAI / DeepMind public writing on model release processes
  — treat as background for the peer-track separation of concerns;
  specific claims from these sources need citation at the track-suite
  level, not at the contract level.
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/):
  - **ML06 Output Integrity Attack** — the Quality and Structural tracks
    together cover the direct detection surface.
  - **ML04 Membership Inference / ML08 Data Leakage** (framing varies) —
    the Compliance track addresses this surface.
  - **ML10 Insufficient Supply Chain Security** — the manifest's
    candidate pinning (revision + SHA + engine version) is the source of
    truth the gate record ties to.
- [mitre-atlas](https://atlas.mitre.org/):
  - **AML.T0010 ML Supply Chain Compromise** — the gate record is the
    attestation that the candidate pinned in the manifest is the
    candidate that was evaluated.
  - **AML.T0043 Craft Adversarial Data / regression detection** — the
    Quality and Safety tracks together are the pre-production detection
    surface.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework):
  - **Govern** — peer-track ownership, no self-signoff, review cadence.
  - **Map** — Compliance track's data-handling probes characterise the
    candidate.
  - **Measure** — all tracks' measurements.
  - **Manage** — gate-record attachment at the rollout controller;
    feedback loop from ex-04 post-mortems.

## Appendix — control-to-framework mapping

| Control | OWASP ML | MITRE ATLAS | NIST AI RMF |
|---|---|---|---|
| Peer tracks with distinct owning teams | ML-governance adjacent | n/a | Govern |
| Strict AND across tracks (no weighting) | ML06, ML10 | AML.T0010 attestation | Govern, Manage |
| No self-signoff (Model Owner ≠ signer) | ML-governance adjacent | n/a | Govern |
| Pre-committed timeout with explicit FAIL | ML06 | n/a | Govern, Manage |
| Gate record required at rollout controller | ML06, ML10 | AML.T0010 | Manage |
| RCA action items feed new track suites | ML06 | AML.T0043 | Govern, Measure |
