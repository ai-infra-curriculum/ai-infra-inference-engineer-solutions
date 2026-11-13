# Solution — Exercise 02: Blue-Green Model Swap Walkthrough

> Paired learning file: [`lessons/mod-113-production-safety-and-rollouts/exercises/exercise-02-blue-green-model-swap-walkthrough.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-113-production-safety-and-rollouts/exercises/exercise-02-blue-green-model-swap-walkthrough.md)
> Module: `mod-113-production-safety-and-rollouts`
> Type: design walkthrough

## 1. Solution overview

A **blue-green model swap** is the rollout pattern for a change too large to
canary safely: a new model family (Llama-3 → Llama-4), a new engine version
with changed request semantics (e.g., a new default sampler, a breaking
tool-call schema), or a tokenizer or prompt-template change that would make
response-diffing meaningless. Instead of splitting traffic for a long
window, you stand up a parallel "green" fleet at full capacity, cut traffic
over in one or more fast steps once green has passed the gate, and keep
"blue" alive as a one-command rollback until confidence is earned.

The exercise asks the learner to walk through a concrete swap: what to
provision, when, in what order, what the rollback trigger is at each phase,
and what breaks if you skip a step. The deliverable is a phase-by-phase
runbook plus a sequence diagram, written as if you were briefing the next
on-call rotation.

Blue-green is **not** a replacement for the canary pattern from
[[exercise-01-canary-with-response-diffing-and-auto-rollback]]. It is the
pattern you reach for when the change is too large for response-diffing to
be a trustworthy signal — because the two models' outputs are *supposed* to
differ. You still run offline evals (gated per
[[exercise-05-eval-gate-contract-with-peer-tracks]]) before the swap; blue-
green is how you execute the swap safely once the offline gate has passed.

### Why blue-green for a model swap

- **The two models disagree on purpose.** A response-diff canary flags
  divergence as risk; a blue-green swap accepts divergence as the point of
  the change. The decision signals shift from "do outputs match?" to "are
  infra, SLOs, and downstream quality signals holding?".
- **Capacity is pre-paid.** The swap window is a small number of minutes,
  not hours. The green fleet is already warmed, caches are already
  populated, KV-cache memory is already reserved. A canary that creeps from
  5 → 100% over a day is a terrible fit for "we just need to flip the
  tokenizer".
- **Rollback is one command.** The blue fleet keeps running through and
  after the swap. Rollback is a router reversal, not a cold redeploy.

### Deliverable summary

- A phased runbook (§2) with provisioning, pre-swap checks, cutover steps,
  post-swap validation, and retirement.
- A sequence diagram of the swap moment.
- A decision table for "what signal would cause me to roll back at each
  phase."
- A rubric (§4) for reviewing a learner's walkthrough.

## 2. Implementation — worked walkthrough

The walkthrough below assumes the common inference stack: router (envoy /
nginx / custom), two inference fleets each behind a Kubernetes service,
Prometheus metrics, a deployment manifest in Git, and the canary signal
plumbing from exercise 01.

### 2.1 Roles

| Role | Responsibility |
|---|---|
| **Rollout Lead** | Owns the swap end-to-end; makes the final cutover and rollback calls. |
| **Model Owner** | Signed the offline eval gate that authorised this swap. |
| **On-Call Engineer** | Watches dashboards during the swap window; holds the kill switch. |
| **Platform Reviewer** | Confirms green capacity, warm state, and manifest integrity before cutover. |

The Rollout Lead and On-Call Engineer must be different people. The Rollout
Lead is accountable for the plan; the On-Call Engineer is accountable for
the decision to stop.

### 2.2 Phase 1 — Pre-provision (T-24h → T-1h)

1. **Green fleet manifest merged to Git**, pinned to specific model
   revision, engine version, tokenizer, and any tool-call / template schema
   changes. Manifest references the eval-gate record from
   [[exercise-05-eval-gate-contract-with-peer-tracks]].
2. **Green fleet provisioned at full baseline capacity** — same replica
   count, same resource shapes as blue. Under-provisioned green fleets
   become failed cutovers.
3. **Warm-up load** played at low RPS: the green fleet serves synthetic
   traffic drawn from a sanitised production prompt sample until KV-cache
   hit rates, p95 TTFT, and per-replica saturation stabilise. Cold fleets
   look fine in isolation and collapse under real traffic.
4. **Eval-gate pass archived.** Offline eval results, red-team sign-off, and
   any peer-track approvals (ex-05) are attached to the rollout ticket. A
   swap without these cannot enter the cutover phase.
5. **Rollback drill.** The Rollout Lead confirms, in a lower environment
   that mirrors production topology, that a one-command router reversal
   actually sends traffic back to blue. If the drill fails, the swap is
   postponed.

**Rollback trigger in this phase:** any of the above not passing. Rollback
here is "we don't cut over today."

### 2.3 Phase 2 — Pre-swap health check (T-1h → T-0)

1. **Both fleets green on the dashboard**: blue's p95 at baseline, green's
   p95 at baseline *under its warm-up load*. Compare apples to apples —
   green under warm-up RPS, not green under zero traffic.
2. **Dependency fanout check.** If the new model changes outputs in ways
   that cascade (different tool-call shapes, different embedding
   dimensions, different structured-output schemas), the downstream
   consumers must have already been updated and themselves verified. The
   model swap is *not* the moment to discover that the eval service parses
   a field that moved.
3. **Pre-committed cutover plan.** The plan — one shot or stepped, at what
   percentages, with what dwell between steps — is written down in the
   ticket and not changed during the window. A mid-flight plan edit is a
   sign the pre-swap health check missed something.

**Rollback trigger in this phase:** any precondition fails; defer to the
next window. There is no "we'll fix it during cutover."

### 2.4 Phase 3 — Cutover (T-0 → T+5m)

Two common shapes, both acceptable:

- **One-shot swap.** Router config change: 100% of live traffic routed to
  green in a single step. Appropriate when (a) green has passed warm-up at
  realistic RPS, (b) the eval gate has high coverage, (c) rollback is
  verified single-command. This is the default for a modern blue-green.
- **Stepped swap.** 10% → 50% → 100% with ≥ 1 minute dwell at each step.
  Appropriate when green warm-up could not hit full production RPS (e.g.,
  GPU budget), or when the downstream fanout has any uncertainty.

Sequence diagram (one-shot shape, which is what the walkthrough defaults
to):

```
time   actor               action                                   observable signal
----   -----------------   --------------------------------------   --------------------------------
T-10s  Rollout Lead        "Cutover in 10 seconds" announcement     paging/channel notice
T-5s   Platform Reviewer   final glance at blue and green p95       dashboard
T-0    Rollout Lead        apply router manifest: blue=0, green=100 router reload metrics
T+10s  On-Call             watch: 5xx, p95, saturation on green     Prometheus + diff queue (quiet)
T+30s  On-Call             watch: downstream consumer error rates   downstream dashboards
T+60s  Rollout Lead        "Green holding; cutover complete"        channel notice
T+5m   Rollout Lead        begin soak (Phase 4)                     ticket update
```

**Rollback trigger in this phase (hard):**

- `5xx` rate on green > 1% sustained > 30 seconds.
- Green p95 > 1.5 × blue baseline sustained > 60 seconds.
- Downstream consumer error rate > 1% sustained > 60 seconds.
- Kill switch set by on-call.

**Rollback procedure:** apply the inverse router manifest (blue=100,
green=0). The command is prepared and dry-run tested *before* T-0. The
Rollout Lead has it on-screen during the cutover. If paged during cutover,
the on-call applies it unilaterally — the Rollout Lead is informed, not
asked.

### 2.5 Phase 4 — Soak (T+5m → T+2h)

This is the phase that catches the slow-boil regressions.

Signals watched continuously:

- **Infra signals**: p95 latency, 5xx, GPU saturation, memory headroom,
  token throughput per request type.
- **Quality signals**: any downstream feedback the service already captures
  — thumb-up/down, ticket deflection, retrieval-reranker agreement, batch
  job success rates.
- **Shape signals**: distribution of response length, stop-reason, tool-
  call frequency. A model that is answering shorter on average than blue
  did, holding all else equal, is often the first visible sign of a
  regression that the offline eval missed.
- **Cost signals**: tokens per request, cost per request. A new model that
  passes quality but doubles token spend is a rollback candidate even if no
  user complains — the cost signal is a hard SLO from
  [[exercise-05-eval-gate-contract-with-peer-tracks]].

**Rollback trigger in this phase (soft):**

- Any signal crosses its pre-committed warning line for longer than its
  dwell threshold. The Rollout Lead and Model Owner jointly decide whether
  to roll back, continue soaking, or stop traffic to green pending
  investigation.

Blue stays running and drained of live traffic, retained as a one-command
rollback for the full soak. Removing blue earlier than the soak completes
removes the safety net the whole pattern depends on.

### 2.6 Phase 5 — Retirement (T+2h → T+N)

1. **Mark blue inactive** in the registry. Keep the fleet running at
   reduced replica count for a pre-committed grace window (24–72h is
   typical; longer for regulated workloads).
2. **Teardown gated on a pre-committed condition** — e.g., 24h of soak with
   no incidents, next business day for financial / compliance workloads.
   Teardown is not time-based alone; the grace window plus an explicit
   signal-based condition.
3. **Archive the rollout ticket** with: what was swapped, who approved,
   final metrics, whether rollback was armed and how often. The next person
   doing a swap reads this ticket.
4. **Update the deployment manifest** to remove the blue entry and bump
   `current` to green's revision.

**Rollback trigger in this phase:** rollback is still armed through the
grace window; on-call retains kill-switch authority. After teardown, the
only path back is a fresh rollout in the reverse direction, which is
treated as a new swap with its own eval gate.

### 2.7 Decision table

| Phase | Hardest trigger | Response time | Who decides |
|---|---|---|---|
| Pre-provision | Any precondition failing | — | Rollout Lead |
| Pre-swap | Any precondition failing | — | Rollout Lead |
| Cutover | 5xx > 1% / 30s · p95 > 1.5x / 60s · kill switch | seconds | On-call (unilateral) |
| Soak | Soft threshold sustained past dwell | minutes | Rollout Lead + Model Owner |
| Retirement | Any regression surfaced during grace window | minutes | Rollout Lead |

## 3. Validation steps

A reviewer validating a learner submission should check:

1. **Five phases named, in order, with entry and exit criteria.** Any
   compressed-to-"swap + watch" submission fails.
2. **Capacity is pre-paid.** The green fleet is at full capacity at
   cutover, not scaled up afterwards. "We'll autoscale green as traffic
   lands" is not blue-green, it is a fast ramp.
3. **Rollback is a single router command, dry-run tested before cutover.**
   Any submission that implies rollback requires redeploy, image rebuild,
   or multiple sequential manifest edits fails.
4. **On-call has unilateral authority during cutover.** If the submission
   makes rollback contingent on paging the Rollout Lead, it is wrong — the
   Lead is informed in parallel.
5. **The pre-swap health check includes downstream consumers**, not just
   the inference fleet.
6. **Soak phase watches shape and cost signals, not only infra.** A
   submission that only lists p95 and 5xx misses the regressions blue-green
   is uniquely positioned to catch.
7. **Retirement is gated on both time and signal.** Teardown on a timer
   alone is a classic post-rollout outage pattern.
8. **Pattern choice is justified.** The submission explains *why* this
   change was blue-green and not canary, naming the property of the change
   that made response-diffing untrustworthy (new tokenizer, new output
   schema, new model family).

## 4. Rubric / review checklist

Score each item 0 / 1 / 2. Passing: ≥ 14 / 20.

| # | Item | Points |
|---|---|---|
| 1 | Pattern rationale: why blue-green and not canary | 2 |
| 2 | Five phases with explicit entry / exit per phase | 2 |
| 3 | Pre-provisioned green fleet at full capacity with warm-up | 2 |
| 4 | Rollback single-command, dry-run tested pre-cutover | 2 |
| 5 | On-call holds unilateral rollback authority in cutover phase | 2 |
| 6 | Downstream consumer check in pre-swap | 2 |
| 7 | Soak watches shape and cost signals, not only infra | 2 |
| 8 | Retirement gated on time *and* signal; grace window documented | 2 |
| 9 | Decision table ties each phase to its hardest rollback trigger | 2 |
| 10 | External framework mapping (§6) | 2 |

## 5. Common mistakes

- **Treating blue-green as "scale up green, scale down blue."** Scaling
  overlaps the fleets and makes rollback a slow redeploy. Blue-green is a
  router cutover; the fleets live side by side at full capacity.
- **Skipping the warm-up at production RPS.** A green fleet that passed
  warm-up at 10% load will discover its KV-cache fragmentation bug at
  cutover, which is exactly the moment you least want to discover it.
- **Making rollback a redeploy.** The whole value proposition is one-
  command rollback. If rollback means pushing an image and waiting for
  pods, blue-green has been built as a release pipeline with extra steps.
- **Not updating downstream consumers in advance.** If the new model
  changes a tool-call schema or an embedding dimension, every consumer
  needs to be updated and verified *before* cutover, not during soak.
- **Removing blue too early.** The grace window exists for the regressions
  that take hours to surface. Blue torn down at T+10m is a decision that
  will cost you the next outage.
- **Collapsing the Rollout Lead and On-Call Engineer.** The Lead is
  invested in the swap succeeding; the On-Call is invested in the service
  staying up. These incentives differ on purpose.
- **Blue-green without an offline eval gate.** Blue-green runs the change
  live without response diffing; offline evaluation is the only line of
  defence against "the new model is subtly worse." Skipping the eval gate
  (ex-05) makes blue-green a controlled regression deployment.
- **Mid-flight plan editing.** The cutover plan is pre-committed in the
  ticket. If midway through someone proposes a new split, the right answer
  is to abort, re-plan, and schedule a new window.

## 6. References

- vLLM documentation — https://docs.vllm.ai/ — practitioner source for
  engine warm-up and prefix-cache behaviour that informs the pre-swap
  warm-up phase.
- SGLang documentation — https://docs.sglang.ai/ — alternative serving
  stack; same phased pattern applies.
- Google SRE Workbook — *Non-abstract Large System Design (Deployment
  patterns)* — https://sre.google/workbook/ — the generic blue-green
  pattern this exercise specialises to models.
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/):
  - **ML05 Model Theft / tampered artefact** — the pre-swap manifest
    integrity check (revision + SHA) is the direct control.
  - **ML06 Output Integrity Attack** — the offline eval gate (ex-05)
    required before cutover is the control; blue-green itself does not
    detect this class and acknowledges that.
  - **ML10 Insufficient Supply Chain Security** — the green fleet pulls
    exactly the pinned revision; drift blocks the swap.
- [mitre-atlas](https://atlas.mitre.org/):
  - **AML.T0010 ML Supply Chain Compromise** — manifest + SHA pinning at
    provisioning.
  - **AML.T0012 Supply Chain Compromise (Model)** — the retirement phase
    retains blue as the rollback target; a compromised green does not
    strand the service.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework):
  - **Govern** — roles (Rollout Lead, Model Owner, On-Call, Reviewer) and
    pre-committed cutover plan.
  - **Map** — pre-swap check characterises green's operating envelope.
  - **Measure** — soak-phase signals.
  - **Manage** — rollback procedure, retirement gating, archived ticket.

## Appendix — control-to-framework mapping

| Control | OWASP ML | MITRE ATLAS | NIST AI RMF |
|---|---|---|---|
| Manifest pinned to revision + SHA, verified at provision | ML05, ML10 | AML.T0010 | Map, Manage |
| Dry-run tested one-command rollback before cutover | ML06 | ATLAS incident response | Govern, Manage |
| Soak phase monitoring (infra + shape + cost) | ML06 | Detection discipline | Measure |
| Retirement gated on time *and* signal | ML-governance adjacent | n/a | Govern, Manage |
| Eval-gate record attached as swap precondition | ML06 | n/a | Govern |
