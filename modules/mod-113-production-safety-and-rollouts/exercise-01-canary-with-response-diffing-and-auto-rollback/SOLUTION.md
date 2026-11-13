# Solution — Exercise 01: Canary with Response Diffing and Auto-Rollback

> Paired learning file: [`lessons/mod-113-production-safety-and-rollouts/exercises/exercise-01-canary-with-response-diffing-and-auto-rollback.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-113-production-safety-and-rollouts/exercises/exercise-01-canary-with-response-diffing-and-auto-rollback.md)
> Module: `mod-113-production-safety-and-rollouts`
> Type: design + implementation

## 1. Solution overview

A canary for an inference service is not just "route 5% of traffic to the new
replica." That shape of canary catches **infrastructure** failures (the new
pod crashloops, latency doubles, error rate climbs) but is blind to the one
failure that only inference has: the new model, or the new sampler setting, or
the new stop-token list, produces **different outputs on the same inputs** —
and those outputs may be worse without being wrong-shaped.

This exercise wires a canary whose decision loop watches three signals at
once:

1. **Infra health** — p95 latency, 5xx rate, saturation — on both baseline
   and canary.
2. **Response-diff signal** — a side-by-side comparison of baseline and
   canary responses to a mirrored subset of live traffic, scored with cheap
   deterministic diffs plus (optionally) a judge model.
3. **Business / quality signal** — any downstream feedback the service
   already has (thumb-up/down, ticket deflection, retrieval-reranker
   agreement), measured over the canary cohort.

If any of the three crosses a pre-committed threshold, the controller
auto-rolls back; promotion requires all three to stay green through a
pre-committed soak window.

The deliverable is a runnable canary controller (Python + a small
Prometheus-style metrics interface) plus the policy document that names
thresholds, soak times, and the escalation path. The policy document is the
load-bearing artefact — the controller only enforces a policy that has
already been debated and signed.

### Why response diffing specifically

Shadow traffic alone catches "is the new engine up?" but not "is the new
engine's output degraded?". A response-diff canary compares the two engines'
outputs on identical inputs, which is the only cheap way to detect the
model-regression class of failure before a human notices it. The same
machinery is what MITRE ATLAS tactics on **model-level evasion / regression
detection** assume a defender has. OWASP ML06 (Output Integrity Attack) and
NIST AI RMF's **Measure** function both expect the platform to observe
outputs continuously, not only at release-gate time.

### Deliverable summary

- `canary_controller.py` — the control loop, with pluggable signal sources.
- `diff_scorer.py` — the response-diff scorer (token-level + optional
  judge-model hook).
- `policy.yaml` — thresholds, soak window, escalation path, kill switch.
- `README.md` (inline in §2 here) — how to run, how to tune, how to roll back
  manually.
- Rubric and review checklist in §4 for the design review.

## 2. Worked implementation

### 2.1 Topology

```
                 ┌──────────┐
   client ─────▶ │  router  │
                 └────┬─────┘
                      │ 95% ───▶ baseline engine (e.g., vLLM, model v1)
                      │ 5%  ───▶ canary engine   (e.g., vLLM, model v2)
                      │
                      │ mirror ─▶ diff queue ─▶ diff worker ─┐
                      │                                      ▼
                      │                              diff metrics + alerts
                      ▼
               infra metrics
               (p95, 5xx, sat)
                      │
                      ▼
               canary controller ─▶ decision: hold / promote / rollback
```

Key properties:

- The router splits by **cohort** (stable hash on user_id or request_id), not
  per-request random — so a given user sees the same engine for a session and
  the quality signal attaches to a stable identity.
- The mirror is a **copy**, not a split. The client's response is always the
  baseline's; the canary's response flows only to the diff queue for the
  mirrored subset. This matters: it means the response-diff signal does not
  itself expose users to the canary, so we can afford to mirror more traffic
  than we route to the canary.
- Rollback is a router flag change, not a redeploy. The canary engine stays
  running after a rollback so forensics can replay failing prompts.

### 2.2 Policy (the load-bearing artefact)

`policy.yaml`:

```yaml
# Canary policy for model vN -> vN+1 rollout.
version: 1

cohorts:
  canary_traffic_percent: 5         # router split
  mirror_percent: 25                # subset diff-compared; superset of canary_traffic_percent

soak:
  min_samples_for_promote: 5000     # minimum diff-scored samples before any promotion decision
  min_wall_clock_minutes: 60        # even if we hit the sample count fast, don't short-cut

promotion_requires_all_of:
  - infra_canary_p95_regression_ratio_lte: 1.10   # canary p95 <= 1.10 * baseline p95
  - infra_canary_5xx_rate_lte: 0.001              # canary error rate <= 0.1%
  - diff_mean_score_gte: 0.85                     # higher is better; 1.0 = identical
  - diff_bad_fraction_lte: 0.02                   # <= 2% of diffed samples flagged "bad"
  - quality_signal_regression_lte: 0.00           # no measurable regression in downstream signal

rollback_any_of:
  - infra_canary_5xx_rate_gt: 0.01                # hard: >= 1% errors
  - infra_canary_p95_regression_ratio_gt: 1.50    # hard: 50% latency regression
  - diff_bad_fraction_gt: 0.05                    # hard: >= 5% bad diffs
  - diff_score_floor_breached_count_ge: 50        # 50 samples below floor within window
  - manual_kill_switch: true

kill_switch:
  path: /ops/canary/kill        # file flag; presence => rollback immediately
  authority: on_call_engineer   # anyone paged may flip it

escalation:
  on_rollback:
    - page: rollout_lead
    - notify: inference_platform_channel
    - open: incident_ticket_template://canary-rollback
  on_promote:
    - notify: inference_platform_channel
    - update: deployment_manifest_pr

diff_scorer:
  method: token_overlap_plus_judge
  judge_model: ""              # empty => skip judge; "openai://..." / "internal://..." if enabled
  judge_sample_rate: 0.1       # fraction of diffs sent to judge
  bad_threshold: 0.60          # per-sample diff score below this is "bad"
```

**Thresholds are pre-committed.** The controller refuses to run if the
policy has been edited inside the active rollout window; a mid-flight
threshold change is a sign that someone is moving the goalposts.

### 2.3 Diff scorer

`diff_scorer.py` (reference implementation):

```python
"""Response-diff scorer for canary comparison.

A "score" is a number in [0, 1] where 1 means baseline and canary responses
are effectively identical and 0 means they disagree completely. We combine a
cheap deterministic signal (token overlap + length ratio) with an optional
judge model that is only invoked on a sampled fraction (expensive).
"""

from __future__ import annotations
import difflib
from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class DiffResult:
    deterministic_score: float   # in [0, 1]
    judge_score: Optional[float] # in [0, 1] or None if not judged
    is_bad: bool                 # below policy's bad_threshold


class Judge(Protocol):
    def score(self, prompt: str, a: str, b: str) -> float: ...


def _token_overlap(a: str, b: str) -> float:
    """Ratio of matched tokens; simple whitespace tokenisation keeps this
    dependency-free for the exercise. Production variants use the model's
    tokenizer so that scoring reflects the deployed vocabulary."""
    ta, tb = a.split(), b.split()
    if not ta and not tb:
        return 1.0
    matcher = difflib.SequenceMatcher(a=ta, b=tb, autojunk=False)
    return matcher.ratio()


def _length_ratio(a: str, b: str) -> float:
    """Punish large length divergence; common regression sign is 'canary
    now truncates at a different stop sequence'."""
    la, lb = max(len(a), 1), max(len(b), 1)
    r = min(la, lb) / max(la, lb)
    return r


def score_pair(
    prompt: str,
    baseline: str,
    canary: str,
    judge: Optional[Judge] = None,
    judge_sample: bool = False,
    bad_threshold: float = 0.60,
) -> DiffResult:
    det = 0.5 * _token_overlap(baseline, canary) + 0.5 * _length_ratio(baseline, canary)
    judge_score = None
    if judge is not None and judge_sample:
        judge_score = max(0.0, min(1.0, judge.score(prompt, baseline, canary)))
    combined = det if judge_score is None else 0.5 * det + 0.5 * judge_score
    return DiffResult(
        deterministic_score=det,
        judge_score=judge_score,
        is_bad=combined < bad_threshold,
    )
```

Two scoring choices worth calling out:

- **Token overlap plus length ratio** is deliberately simple. It catches
  truncation, stop-token changes, and gross divergence. It does not catch
  "same shape, worse content" — that is what the judge sample is for.
- **Judge sampling is a cost knob, not a quality knob.** Set
  `judge_sample_rate` low enough that the canary doesn't add material LLM
  spend, and treat any period where the judge disagrees with the
  deterministic score as a signal to tighten the deterministic score.

### 2.4 Controller

`canary_controller.py` (reference, elided for length; the core loop):

```python
"""Canary controller.

Reads policy.yaml, consumes three signals, decides hold / promote / rollback.
No service dependencies; everything plugs in via the Signals protocol so that
this file is testable without Prometheus."""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

log = logging.getLogger("canary.controller")


class Decision(Enum):
    HOLD = "hold"
    PROMOTE = "promote"
    ROLLBACK = "rollback"


@dataclass
class InfraSnapshot:
    baseline_p95_ms: float
    canary_p95_ms: float
    canary_5xx_rate: float


@dataclass
class DiffSnapshot:
    mean_score: float
    bad_fraction: float
    sample_count: int
    floor_breaches: int   # samples whose combined score fell below a hard floor


@dataclass
class QualitySnapshot:
    baseline_signal: float
    canary_signal: float   # same scale as baseline; delta is baseline - canary


class Signals(Protocol):
    def infra(self) -> InfraSnapshot: ...
    def diffs(self) -> DiffSnapshot: ...
    def quality(self) -> QualitySnapshot: ...
    def kill_switch_set(self) -> bool: ...


@dataclass
class Policy:
    canary_traffic_percent: float
    min_samples_for_promote: int
    min_wall_clock_minutes: int
    # promotion thresholds
    infra_canary_p95_regression_ratio_lte: float
    infra_canary_5xx_rate_lte: float
    diff_mean_score_gte: float
    diff_bad_fraction_lte: float
    quality_regression_lte: float
    # rollback thresholds
    infra_canary_5xx_rate_gt: float
    infra_canary_p95_regression_ratio_gt: float
    diff_bad_fraction_gt: float
    diff_score_floor_breached_count_ge: int


def decide(policy: Policy, signals: Signals, started_at: float, now: float) -> Decision:
    if signals.kill_switch_set():
        log.warning("kill switch set; rollback")
        return Decision.ROLLBACK

    infra = signals.infra()
    diffs = signals.diffs()
    quality = signals.quality()
    elapsed_min = (now - started_at) / 60.0

    # Rollback: any hard condition trips, we leave.
    p95_ratio = infra.canary_p95_ms / max(infra.baseline_p95_ms, 1e-9)
    if infra.canary_5xx_rate > policy.infra_canary_5xx_rate_gt:
        log.warning("5xx hard breach: %.4f", infra.canary_5xx_rate)
        return Decision.ROLLBACK
    if p95_ratio > policy.infra_canary_p95_regression_ratio_gt:
        log.warning("p95 hard breach: %.2fx baseline", p95_ratio)
        return Decision.ROLLBACK
    if diffs.bad_fraction > policy.diff_bad_fraction_gt:
        log.warning("diff bad-fraction hard breach: %.3f", diffs.bad_fraction)
        return Decision.ROLLBACK
    if diffs.floor_breaches >= policy.diff_score_floor_breached_count_ge:
        log.warning("diff floor breaches: %d", diffs.floor_breaches)
        return Decision.ROLLBACK

    # Promotion: all soft conditions hold AND we have earned the right to promote.
    if diffs.sample_count < policy.min_samples_for_promote:
        return Decision.HOLD
    if elapsed_min < policy.min_wall_clock_minutes:
        return Decision.HOLD

    promote_ok = (
        p95_ratio <= policy.infra_canary_p95_regression_ratio_lte
        and infra.canary_5xx_rate <= policy.infra_canary_5xx_rate_lte
        and diffs.mean_score >= policy.diff_mean_score_gte
        and diffs.bad_fraction <= policy.diff_bad_fraction_lte
        and (quality.baseline_signal - quality.canary_signal) <= policy.quality_regression_lte
    )
    return Decision.PROMOTE if promote_ok else Decision.HOLD


def run(policy: Policy, signals: Signals, *, poll_interval_s: float = 30.0) -> Decision:
    started_at = time.time()
    while True:
        d = decide(policy, signals, started_at=started_at, now=time.time())
        if d is not Decision.HOLD:
            return d
        time.sleep(poll_interval_s)
```

The point of this shape is that `decide()` is a pure function of the policy
and the signal snapshot — which means it is exhaustively unit-testable, and
in particular the rollback path is testable without standing up a real
canary.

### 2.5 How this is wired at the router

The router runs a stable-hash cohort split:

- `cohort = sha256(user_id)` mod 100.
- `cohort < canary_traffic_percent` → canary.
- `cohort < mirror_percent` → also enqueue to diff queue (superset,
  including canary cohort itself — diff queue carries canary + mirrored
  baseline responses for these users).

On rollback, the router config is updated to `canary_traffic_percent: 0`;
the diff queue continues to drain for forensics. On promotion, the canary
build becomes the new baseline and a fresh deployment manifest PR is opened
to pin the new revision.

### 2.6 Minimum harness run

```bash
# Smoke-test decide() in both directions.
python -m canary_controller.tests.test_decide

# Run against a staging stack with a replayed trace:
python canary_controller.py \
  --policy policy.yaml \
  --signals staging.yaml   # names the Prometheus scrape targets and diff queue
```

The staging run should (a) promote cleanly when the canary is identical to
the baseline and (b) roll back within one decision tick when the canary is
swapped to a known-bad model. These are the two acceptance tests.

## 3. Validation steps

1. **Policy-first review.** Reject any submission where the thresholds are
   named inline in code rather than in `policy.yaml`. The split exists so
   that review of the rollout criteria happens separately from review of the
   controller code.
2. **`decide()` is pure.** The reviewer must be able to unit-test the
   decision function with hand-constructed snapshots. If the controller mixes
   signal-fetching with decision logic, call it out.
3. **Rollback path dominates promotion path.** Any hard rollback trigger
   wins over any promotion check — check the test matrix contains a case
   where promotion thresholds are all satisfied *and* a rollback threshold
   is tripped, and confirm the decision is `ROLLBACK`.
4. **Kill switch is trivially accessible.** The on-call engineer must be
   able to force a rollback without editing YAML or redeploying; a file flag
   or a single API call is sufficient. A kill switch gated on CI is a
   non-starter.
5. **Mirror superset of canary cohort.** The mirror cohort must include the
   canary cohort; otherwise the diff signal does not see the users whose
   responses actually reached production.
6. **Diff scorer honest about limits.** The submission should explicitly
   state what the scorer cannot catch (same shape, worse content) and name
   the judge-sample or offline-eval that covers the gap.
7. **Soak window present.** `min_samples_for_promote` and
   `min_wall_clock_minutes` both exist; neither can be zero. Promotion
   without a soak is a "canary" in name only.
8. **Rollback is not a redeploy.** Rollback flips a router flag; it does not
   require rebuilding the baseline engine. The canary engine keeps running
   for forensics.

## 4. Rubric / review checklist

Score each item 0 (missing) / 1 (partial) / 2 (fully correct). Passing:
≥ 14 / 20.

| # | Item | Points |
|---|---|---|
| 1 | Three independent signals (infra, response-diff, quality) wired | 2 |
| 2 | Policy expressed as data, not code; versioned alongside controller | 2 |
| 3 | `decide()` is a pure function with unit tests covering promote / hold / rollback | 2 |
| 4 | Rollback thresholds strictly separate from promotion thresholds; hard rollback wins | 2 |
| 5 | Stable-hash cohort split with mirror ⊇ canary | 2 |
| 6 | Kill switch exists, is single-action, documented on-call-path | 2 |
| 7 | Soak window (samples + wall clock) enforced before any promotion | 2 |
| 8 | Diff scorer has deterministic floor + sampled judge; honest about blind spots | 2 |
| 9 | Escalation path spelled out (page, channel, incident template) for rollback case | 2 |
| 10 | Mapping of controls to OWASP ML, MITRE ATLAS, and NIST AI RMF (§6) | 2 |

## 5. Common mistakes

- **Treating "latency didn't regress" as "the canary is fine."** Infra
  health is necessary and insufficient. A model that silently truncates every
  answer at 30 tokens has excellent p95; its users have a terrible day.
- **Comparing live baseline and canary at different times.** The mirror has
  to compare *paired* responses on the *same prompt*. Comparing aggregates
  (mean score of canary responses vs mean score of baseline responses) will
  hide the regressions the mirror is supposed to catch.
- **Random-per-request cohort split.** A given user bounces between engines
  every turn and the quality signal is useless. Hash by user or session.
- **Judge model inside the request path.** The judge is a sampled offline
  scorer, not a gate on live response latency. Adding it inline makes the
  canary slower than baseline for reasons that have nothing to do with the
  canary.
- **Rollback by redeploy.** A rollout controller that requires a redeploy to
  roll back hasn't built a canary — it's built a release pipeline with
  monitoring. Router-flag rollback is the point.
- **Mid-flight threshold editing.** Once the rollout starts, the policy is
  locked. A submission that lets operators lower
  `diff_mean_score_gte` to rescue a struggling canary has built an anti-
  safety control.
- **No explicit "cannot catch" statement.** A canary scorer that claims to
  catch every regression class is wrong. The submission must name what the
  scorer misses and route it to a different control (offline eval, red team,
  thumb feedback).
- **Rolling back silently.** Rollback must page a human and open an
  incident ticket. A rollback that nobody notices becomes the new permanent
  state and the question of why the canary failed never gets answered.

## 6. References

- vLLM documentation — *OpenAI-Compatible Server* and *Metrics* —
  https://docs.vllm.ai/ — practitioner source for the Prometheus metrics the
  controller scrapes.
- SGLang documentation — https://docs.sglang.ai/ — alternative serving
  stack; the controller is engine-agnostic.
- Google SRE Workbook — *Canarying Releases* —
  https://sre.google/workbook/canarying-releases/ — the non-ML canary
  pattern this exercise extends with a response-diff signal.
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/):
  - **ML06 Output Integrity Attack** — the response-diff signal is the
    direct detection control; a policy with no output-level signal cannot
    see ML06.
  - **ML09 Output Manipulation / Insufficient Monitoring** (framing varies
    by release of the list) — the three-signal policy addresses the
    "insufficient monitoring" failure mode.
- [mitre-atlas](https://atlas.mitre.org/):
  - **AML.T0043 Craft Adversarial Data** and adjacent evasion tactics — the
    diff signal is one of the few detections that fires on model-level
    regressions, including a canary that was fine-tuned on tainted data.
  - **AML.T0010 ML Supply Chain Compromise** — if the canary pulled a
    tampered checkpoint, the response-diff signal is the last line of
    defence before promotion.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework):
  - **Govern** — the policy document, the kill-switch authority, and the
    escalation path.
  - **Measure** — the three signals during the rollout.
  - **Manage** — the rollback and forensic-replay procedures.

## Appendix — control-to-framework mapping

| Control | OWASP ML | MITRE ATLAS | NIST AI RMF |
|---|---|---|---|
| Response-diff scorer on mirrored cohort | ML06 | AML.T0043 regression detection | Measure |
| Pre-committed, versioned policy | ML-governance adjacent | n/a | Govern |
| Kill switch with single-action authority | ML06 | ATLAS incident response patterns | Manage |
| Soak window (samples + wall clock) | ML-governance adjacent | n/a | Govern, Measure |
| Forensic-replay from retained diff queue | ML06 | ATLAS post-incident forensics | Manage |
