# Solution — Exercise 04: Mock Incident — RCA and Post-Mortem

> Paired learning file: [`lessons/mod-113-production-safety-and-rollouts/exercises/exercise-04-mock-incident-rca-and-postmortem.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-113-production-safety-and-rollouts/exercises/exercise-04-mock-incident-rca-and-postmortem.md)
> Module: `mod-113-production-safety-and-rollouts`
> Type: design / writing exercise

## 1. Solution overview

A post-mortem is not a blame assignment and not a status update; it is a
document the next on-call reads to avoid the same incident. For inference
platforms, the dominant anti-pattern is an RCA that stops at "the model
regressed" — which is the inference-world equivalent of "the server
returned 500." The real root causes usually sit upstream of the model: an
eval-gate that doesn't cover the failing axis, a prompt-template rotation
that invalidated the prefix cache, a draft model that slipped out of
pinning, a client upgrade that changed the request shape.

This exercise authors a post-mortem for a mock inference incident, using
the SRE post-mortem shape adapted to inference specifics. The learner
receives a mock timeline (described in §2.1 below as the reference
scenario, since the learning file hands one to the student); they produce
a post-mortem following §2's template.

The point is **structural**: a good post-mortem names the causal chain,
distinguishes contributing factors from the trigger, assigns owners to
action items, and files them in a way that a future on-call can verify
were done.

### Why this specific shape

- **Blameless** — the post-mortem names systems, defaults, and decisions,
  not the person who clicked. SRE-wide convention and non-controversial.
- **Causal-chain explicit** — "model regressed" is a symptom, not a cause.
  The chain backs up to the eval-gate miss and/or the rollout-pattern
  mismatch.
- **Action items are specific and owned** — every action item has an
  owner, a due date, and a done-criterion. Unowned action items are the
  reason the same incident happens a second time.
- **Mapped to external frameworks** — the post-mortem names which NIST AI
  RMF function the fix lives under and whether any OWASP ML / MITRE ATLAS
  item is adjacent. The mapping forces the author to distinguish "we had
  no control" from "the control existed and didn't fire."

## 2. Implementation — worked answer (post-mortem template plus a filled example)

### 2.1 Reference mock scenario

Scenario (as handed to the learner in the paired lesson, used here as the
worked example's substrate). Treat as **local exercise context**, not as a
real incident.

- Service: a chat inference endpoint serving a 70B instruct model, behind
  a rollout controller patterned on
  [[exercise-01-canary-with-response-diffing-and-auto-rollback]].
- Change under rollout: a new fine-tune of the model, same tokenizer, same
  engine version. The canary policy promoted on all three signals (infra,
  response-diff mean, downstream feedback) within soak.
- Post-promotion, over the next 90 minutes, a growing fraction of chat
  sessions begin failing a downstream tool-call parser. The parser expects
  a specific JSON shape for function arguments; the new fine-tune
  intermittently emits the arguments as a code-fenced block. Response-diff
  mean remained high because the natural-language parts agreed; the parser
  class of failure was not in any eval-gate suite.
- On-call pages at 90 minutes; rollback via router reversal completes in
  under 60 seconds; incident lasts 97 minutes end-to-end.

### 2.2 Post-mortem template

```markdown
# Post-Mortem: <incident short name> (<YYYY-MM-DD>)

**Status**: draft | review | final
**Severity**: S1 / S2 / S3
**Author**: <name>  **Reviewers**: <names>  **Signoff**: <name, role>
**Incident window**: <start UTC> → <end UTC>  (<duration>)

## 1. TL;DR

<Three sentences. What broke, how long, what's the one thing we'll change.>

## 2. Impact

- **User-facing**: <who was affected, in what way, over what window>
- **SLO budget burn**: <error-budget consumed>
- **Cost**: <wasted tokens, incremental GPU hours, support load>
- **Reputational**: <escalations received; if none, state so>

## 3. Timeline (UTC)

<One line per event. Who, what, what signal they used.>

- 2026-10-14 13:02 — Rollout Lead promotes canary to 100% (ex-01 policy met).
- 2026-10-14 13:07 — Downstream parser begins throwing on 0.3% of chat turns.
- ...
- 2026-10-14 14:37 — Rollback executed; chat turns return to baseline shape.
- 2026-10-14 14:39 — Incident closed.

## 4. Trigger and root cause

**Trigger (the thing that made it fail *now*)**: <one line>

**Root cause (the condition that made it possible)**: <one paragraph>

**Why the system did not catch it**: <one paragraph, naming the specific
control that was absent or inadequate>

## 5. Contributing factors

<Three to seven bullets. Name each as a factor, not a person.>

## 6. What went well

<Three bullets. Keep honest, keep brief.>

## 7. What went badly

<Three bullets.>

## 8. Where we got lucky

<One or two bullets. The most important section for forecasting the next
incident. If the author claims no luck was involved, push back — luck is
always a factor.>

## 9. Action items

| ID | Owner | Due | Severity | Done-criterion |
|---|---|---|---|---|
| AI-1 | <name/role> | <YYYY-MM-DD> | P0/P1/P2 | <how we'll know it's done> |
| ... |

## 10. Framework mapping

| Factor | OWASP ML | MITRE ATLAS | NIST AI RMF | Action item |
|---|---|---|---|---|

## 11. References

- Incident ticket: <link>
- Related runbooks: <list from exercise 03>
- Related rollout / eval-gate documents: <list>
```

### 2.3 Worked example — the scenario in §2.1 rendered

```markdown
# Post-Mortem: Tool-Call Parser Regression After Fine-Tune Promotion (2026-10-14)

**Status**: final
**Severity**: S2
**Author**: J. Rollout-Lead   **Reviewers**: Platform-TL, Model-Owner-Chat, On-Call-Lead
**Signoff**: Inference Platform Director
**Incident window**: 2026-10-14 13:02 UTC → 14:39 UTC  (1h 37m)

## 1. TL;DR

A fine-tune of the chat model promoted cleanly through canary but emitted
tool-call arguments in a code-fenced shape on ~0.3% of turns, breaking a
downstream parser. We rolled back via router reversal in 97 minutes. We
are adding a tool-call-shape suite to the eval gate so the next fine-tune
cannot promote without it.

## 2. Impact

- **User-facing**: approximately 11,400 chat turns across 3,800 users saw
  tool-call failures; downstream agent workflows failed open.
- **SLO budget burn**: 1h 37m of 0.3% tool-call-success degradation; 42%
  of the monthly tool-call SLO budget consumed in one incident.
- **Cost**: incremental retries by the agent framework added ~5% token
  spend over the window.
- **Reputational**: two external escalations; no customer departures
  recorded.

## 3. Timeline (UTC)

- 13:02 — Canary policy (ex-01) met; Rollout Lead promotes to 100%.
- 13:07 — Downstream `agent-exec` begins throwing `MalformedToolArguments`
  at 0.3% rate. No inference alert; this metric is owned by the agent
  team, not inference.
- 13:42 — Agent team pages downstream on-call; investigation points at a
  change in response shape.
- 13:55 — Agent on-call correlates to the chat promotion window; pages
  inference on-call.
- 14:02 — Inference on-call pulls the canary dashboard; response-diff
  mean score was 0.93 at promotion, above the 0.85 gate.
- 14:18 — Rollback dry-run confirmed against staging; Rollout Lead
  authorises production rollback.
- 14:37 — Router reversal applied; blue (prior fine-tune) at 100%.
- 14:39 — Downstream error rate returns to baseline; incident closed.

## 4. Trigger and root cause

**Trigger**: the new fine-tune's conditional probability of emitting
tool-call arguments inside a code-fenced block was higher than the prior
fine-tune's, concentrated on a small set of prompt patterns.

**Root cause**: the eval gate (ex-05) had no suite that exercised the
tool-call shape. Response-diff (ex-01) scored on the natural-language
portion of responses and gave the tool-call-fenced variant a high score
because the user-facing text agreed. The canary's three-signal policy had
no signal that could see the tool-call shape.

**Why the system did not catch it**: the response-diff scorer is text-
level; code-fence structure is not within its vocabulary. The eval gate's
tool-call coverage was implicit — tool-call turns existed in the
evaluation set but were scored on natural-language similarity, not on
whether a parser downstream would accept them. The downstream parser's
error rate is not an inference-fleet metric, so it was not in the canary
policy at all.

## 5. Contributing factors

- **Eval-gate coverage gap**: no tool-call-shape suite; coverage matrix in
  ex-05 did not flag this axis.
- **Signal selection on the canary policy**: the "quality signal" in
  ex-01's policy was a user-feedback proxy, which had no latency on this
  failure class.
- **Cross-service metric ownership**: the downstream parser error rate
  sits with the agent team; the inference rollout policy had no hook into
  it.
- **Response-diff scorer vocabulary**: a text-level scorer cannot see
  structural equivalence; this is a known limit of the ex-01 diff signal
  and was not compensated for.
- **Soak window was wall-clock-sufficient but sample-insufficient for
  low-base-rate tool-call turns** (0.3%): the soak saw ~15 affected turns
  before promotion, which is below the statistical threshold to detect a
  shape regression.

## 6. What went well

- Rollback took 97 minutes end-to-end and under 60 seconds once
  authorised; the router-reversal pattern from ex-02 worked as designed.
- Downstream on-call paged inference on-call directly (no triage dead-end).
- The forensic replay of in-flight requests from the canary diff queue
  made the shape-regression hypothesis checkable within five minutes.

## 7. What went badly

- Thirty-five minutes elapsed between downstream error and inference-side
  paging. This is not a technical issue — the metric correlation should
  not have taken that long.
- The eval gate's "we cover tool calls" claim was true-but-shallow and
  nobody had reviewed it against the shape class.
- The canary promotion decision was made on three signals, all of which
  were green by their own definition. "All signals green" was treated as
  "nothing is wrong." The ex-05 eval-gate was supposed to be the
  compensation for ex-01's scorer limits; it wasn't.

## 8. Where we got lucky

- The downstream parser failed loudly (threw an exception) rather than
  silently returning degraded outputs. A silent downstream parser would
  have stretched this incident into days.
- The rollback target (the prior fine-tune) was still running at full
  capacity because this had been a canary promotion, not a blue-green
  swap. Rollback was cheap because the rollback target was warm.

## 9. Action items

| ID | Owner | Due | Severity | Done-criterion |
|---|---|---|---|---|
| AI-1 | Model-Owner-Chat | 2026-10-28 | P0 | Tool-call-shape suite added to ex-05 eval gate; coverage matrix updated |
| AI-2 | Platform | 2026-11-04 | P1 | Downstream parser error rate added as a signal in ex-01 canary policy for chat endpoints |
| AI-3 | Platform | 2026-11-11 | P1 | Diff scorer extended with structural-equivalence check for fenced-code segments |
| AI-4 | Inference TL | 2026-10-21 | P1 | ex-01 policy requires minimum *affected-subcohort* sample count, not just total count, before promotion |
| AI-5 | Agent-Team + Inference TL | 2026-11-11 | P2 | Cross-team paging runbook: downstream parser alerts auto-correlate to recent inference promotions |

## 10. Framework mapping

| Factor | OWASP ML | MITRE ATLAS | NIST AI RMF | Action item |
|---|---|---|---|---|
| Eval-gate coverage gap on tool-call shape | ML06 Output Integrity | AML.T0043 (regression detection) | Measure | AI-1 |
| Canary policy missing downstream signal | ML06 | n/a | Measure | AI-2 |
| Diff scorer blind to structural change | ML06 | AML.T0043 | Measure | AI-3 |
| Soak sample count insufficient for low-base-rate class | ML-governance adjacent | n/a | Measure | AI-4 |
| Cross-team page latency | ML-governance adjacent | n/a | Govern | AI-5 |

## 11. References

- Incident ticket: INC-2026-10-14-01 (link)
- Related runbooks: ex-03 #8 (canary regression), ex-03 #9 (downstream
  consumer errors post-swap)
- Related documents: ex-01 canary policy v1 (archived), ex-05 eval-gate
  manifest at the time of promotion.
```

### 2.4 How this template is used in practice

- The template lives in the repo next to the rollout tooling, not in a
  wiki. A wiki post-mortem is a post-mortem whose action items are not
  enforceable.
- Post-mortem ownership is the Incident Commander's for the first week;
  ownership of each action item is tracked individually after that.
- The post-mortem is **published** inside the org after signoff; external
  summary (if any) is the Communications team's job and is a separate
  artefact.
- Action items are tracked to closure in the issue tracker; the post-
  mortem is not considered final until all action items are either
  completed or explicitly accepted-as-residual-risk by the signer in §10.
- A quarterly review re-opens every post-mortem that still has open
  action items older than one quarter.

## 3. Validation steps

A reviewer should check the learner's submission against the following:

1. **The TL;DR is three sentences and names the "one thing we'll change."**
   A TL;DR that opens with "there was an incident" fails on length and
   purpose.
2. **The timeline is in UTC**, one line per event, each tied to a signal
   (who saw it on what dashboard). Timelines without signals are
   narratives, not evidence.
3. **Trigger and root cause are distinguished.** A submission that
   collapses them fails the central skill this exercise builds.
4. **"Why the system did not catch it" is specific.** Name the exact
   control (ex-01 scorer, ex-05 suite, ex-02 pre-check) that was absent or
   inadequate.
5. **Contributing factors are structural, not personal.** Reject any
   factor framed as "<person> did X"; reframe as "<decision/default> made
   X possible."
6. **"Where we got lucky" is non-empty.** Every incident has luck; a
   submission that claims no luck was involved is not thinking.
7. **Action items are owned, dated, and have done-criteria.** Reject any
   row missing one of the four fields.
8. **Framework mapping ties each factor to at least one of OWASP ML /
   MITRE ATLAS / NIST AI RMF**, or explicitly marks a factor as
   out-of-scope with a reason.
9. **Blameless.** A submission that names an individual as a causal factor
   fails. Even if a person took an action, the question is why the system
   permitted or defaulted them to that action.
10. **Post-mortem is linked to runbooks (ex-03) and to the eval-gate
    (ex-05).** An inference post-mortem that lives in isolation is a lost
    opportunity.

## 4. Rubric / review checklist

Score each item 0 / 1 / 2. Passing: ≥ 14 / 20.

| # | Item | Points |
|---|---|---|
| 1 | Template filled for all required sections | 2 |
| 2 | Trigger and root cause distinguished; "why the system missed it" specific | 2 |
| 3 | Timeline in UTC with per-event signals | 2 |
| 4 | Contributing factors structural, not personal | 2 |
| 5 | "Where we got lucky" non-empty and plausible | 2 |
| 6 | Action items are owned, dated, and have done-criteria | 2 |
| 7 | Framework mapping present and non-superficial | 2 |
| 8 | Blameless: no individual named as cause | 2 |
| 9 | Post-mortem links to relevant runbooks (ex-03) and eval-gate (ex-05) | 2 |
| 10 | Review cadence / action-item tracking mechanism named | 2 |

## 5. Common mistakes

- **Trigger-as-cause.** "The model regressed" is the trigger; the cause is
  the eval gate or the policy that let the regression through. A post-
  mortem that stops at the trigger is not a post-mortem.
- **Blame.** A single named individual as a causal factor is the fastest
  way to destroy the culture that makes future post-mortems honest. If
  someone clicked the wrong button, the post-mortem asks why the system
  presented that button next to the right one.
- **"We'll do better next time" action items.** Vague action items don't
  get done. "Add a tool-call-shape eval suite" with an owner and a date is
  the standard; "improve eval coverage" is not.
- **No done-criterion.** An action item without a verifiable done-
  criterion is a backlog item that will be closed as "won't fix" in a
  quarter.
- **Missing "where we got lucky."** If the author thinks they got no luck,
  they will miss the near-miss that fires next week in a different shape.
- **No framework mapping.** Mapping forces the author to say which control
  failed, not just that the system failed. It also makes the post-mortem
  readable to a security reviewer.
- **Post-mortem in a wiki.** Action items must live in a tracker that
  enforces closure. A wiki page with a to-do list is a hope, not a plan.
- **Collapsing cross-team latency into a single "we'll page faster" bullet.**
  Cross-team paging is a routing and ownership design problem; name it
  that way.

## 6. References

- Google SRE Book — *Postmortem Culture: Learning from Failure* —
  https://sre.google/sre-book/postmortem-culture/ — the blameless post-
  mortem pattern this template specialises.
- Google SRE Workbook — *Managing Incidents* and *Post-Mortem Templates* —
  https://sre.google/workbook/ — the shape of the template.
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/):
  - **ML06 Output Integrity Attack** — the direct framework item for the
    tool-call shape regression class; a post-mortem of an output-level
    regression should map to ML06.
  - **ML10 Insufficient Supply Chain Security** — if the trigger were a
    tampered checkpoint, this would be the mapping.
- [mitre-atlas](https://atlas.mitre.org/):
  - **AML.T0043 Craft Adversarial Data / regression detection** — the
    discipline the post-mortem evaluates against when it asks "why did
    we not detect this?".
  - Case studies on model regression / output shape change — framing for
    the Contributing-Factors section.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework):
  - **Govern** — ownership of action items; review cadence.
  - **Measure** — the signals that did and didn't fire; the eval-gate
    coverage gap.
  - **Manage** — the rollback executed and the action items that close
    the residual risk.

## Appendix — control-to-framework mapping

| Control / Action item | OWASP ML | MITRE ATLAS | NIST AI RMF |
|---|---|---|---|
| Eval-gate tool-call suite added (AI-1) | ML06 | AML.T0043 | Measure, Govern |
| Canary policy adds downstream signal (AI-2) | ML06 | n/a | Measure |
| Diff scorer structural check (AI-3) | ML06 | AML.T0043 | Measure |
| Subcohort sample floor in soak (AI-4) | ML-governance adjacent | n/a | Govern, Measure |
| Cross-team paging runbook (AI-5) | ML-governance adjacent | n/a | Govern |
