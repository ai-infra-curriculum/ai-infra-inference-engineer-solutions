# Solution — Exercise 05: Cross-Peer Interface Contracts

> Paired learning file: [`lessons/mod-114-cross-functional-inference-leadership/exercises/exercise-05-cross-peer-interface-contracts.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-114-cross-functional-inference-leadership/exercises/exercise-05-cross-peer-interface-contracts.md)
> Module: `mod-114-cross-functional-inference-leadership`
> Type: interface contract set / governance artefact bundle

## 1. Solution overview

Exercise 04 wrote the full interface contract for **one** peer — the
performance peer — because that boundary carries the densest 20 % shared
surface. Exercise 05 extends the same shape to the four remaining peers
that every inference-platform engineer governs an edge with:

- **A.** ML-platform peer — model training / fine-tuning → serving handoff.
- **B.** Eval peers — pre-flight evaluation, gate authoring, regression
  slices on production traffic.
- **C.** SRE peer — SLO contract co-ownership, incident command, pager
  policy, capacity escalation.
- **D.** Security peer — supply-chain review, tenancy / abuse policy,
  edge rate-limit tiers, incident-response playbooks.
- **E.** Cross-peer cadence — the joint rituals that stitch all five
  interfaces (the four above plus exercise 04's performance peer) back
  into the exercise-01 program.

Every signature in this deliverable is of the form `[SIMULATED: role]`.
Every ledger entry marked `[SIMULATED]` is a plausible-hypothetical
derived from the exercise-01 reference fleet and the public standards
listed in §7. Any claim that would need validation in a real org is
tagged `<!-- pending-verification: ... -->`.

### 1.1 Reference fleet

Inherits exercise 01's §1.2: 40 H100-SXM5 nodes, Llama-3.1-70B-Instruct
FP16 primary on tier A, 20 long-tail models on tier B, batch embedding
on tier C, dual-region US / EU, vLLM primary + SGLang fallback. Workload
classes `tier-A chat` / `tier-B long-tail` / `tier-C batch embedding` /
`internal eval-runner`.

### 1.2 Document map

| § | Deliverable | Pairs with learning file |
|---|---|---|
| §2 | A. ML-platform peer interface contract | chapter 02 / 05 |
| §3 | B. Eval peers interface contract | chapter 02 / 05 + module 113 ch. 05 |
| §4 | C. SRE peer interface contract | chapter 02 / 05 + module 113 ch. 03 |
| §5 | D. Security peer interface contract | chapter 02 / 05 |
| §6 | E. Cross-peer cadence | chapter 03 |
| §7 | References, validation, rubric, common mistakes | — |

### 1.3 Contract shape (shared with exercise 04)

Every one of the four peer contracts (A–D) uses the same seven-part
structure that exercise 04 wrote for the performance peer. For brevity
and non-duplication, this solution writes each peer at the depth that
teaches the shape without copying exercise 04's contract body wholesale:

1. **Boundary statement** (one page) — what each side owns, named in
   artefacts, ending with a one-sentence rule.
2. **Ticket template** — the structured form filed across the boundary;
   fields tuned to the peer's domain.
3. **Response SLA table** — P0 / P1 / P2 / P3 with ack / opinion /
   resolution and a mapping to the org severity scale.
4. **Return-shape taxonomy** — the small set of concrete things the
   peer returns; symmetric reverse direction.
5. **Peer-specific ledger** — the long-lived state object (artefact
   registry, regression ledger, SLO ledger, policy ledger); schema plus
   seeded entries.
6. **Worked example** — one incident end-to-end through every artefact.
7. **Three dry-run tickets** — two forward, one reverse.

The contract plugs back into exercise 01 Deliverable C (change
calendar), Deliverable D (paging matrix), and Deliverable F (migration
roadmap) at the identified points.

---

## 2. Deliverable A — ML-platform peer interface contract

### 2.1 Boundary statement (one page)

> **`[SIMULATED]` Inference Platform ↔ ML-Platform Peer Interface
> Contract — v1.0 — effective 2026-Q4**
>
> **The inference-platform role owns the production serving system
> downstream of a signed model handoff.** Specifically:
>
> - Model artefact *ingress* (manifest validation, SHA pin, registry
>   write once the handoff is signed).
> - Lifecycle of the artefact inside production (promotion stages,
>   canary, rollback, sunset).
> - Serving-time configuration (engine args, KV allocator parameters,
>   topology, autoscaler policy).
> - Capacity and cost for the serving fleet.
> - Production incident ownership for anything that fires after the
>   artefact is in the registry.
> - Draft-model lifecycle (module 108 ex-04) — the ML-platform peer
>   delivers the draft; this role governs it the same way it governs
>   the target.
>
> **The ML-platform peer owns everything upstream of the handoff.**
> Specifically:
>
> - Training / fine-tuning datasets and their lineage.
> - Training code, training clusters, training compute budget.
> - Model-weight files as *produced artefacts* (format, tokenizer,
>   config, preprocessor, licence provenance).
> - Pre-handoff evaluation (loss curves, capability evals on the
>   training frontier; application evals are shared with the eval peer
>   — see §3).
> - Choice of base model and fine-tune approach (LoRA / QLoRA / full
>   SFT / RLHF / DPO / merge).
> - Draft-model training for speculative decoding (module 108).
> - Model card authorship and attestation.
>
> **The 20 % shared surface.**
>
> - **Handoff manifest** — the structured artefact the peer signs and
>   this role ingests. Named in §2.5.
> - **Quantisation plan** — the peer owns quantisation *algorithm*
>   choice at training time; the performance peer (ex-04) owns the
>   kernel path; this role owns the deployment and the accuracy watch.
> - **Draft-target pairing** — the peer trains or selects the draft;
>   this role runs the acceptance-rate measurement (module 108 ex-03)
>   and the draft-lifecycle contract (module 108 ex-04).
> - **Prompt-format and chat template** — the peer defines the
>   training-time template; this role deploys it and validates with
>   the eval peer (§3).
> - **Pre-handoff eval → application eval handoff** — the peer owns
>   training-frontier evals; the eval peer owns application evals;
>   this role is the broker that guarantees both have run.
>
> **The boundary rule (one sentence).**
>
> > The ML-platform peer owns the model as a produced artefact; this
> > role owns the artefact as a running system from signed handoff
> > onward.
>
> **Signed.** `[SIMULATED: Platform Lead]`, `[SIMULATED: ML-Platform
> Peer Lead]`, 2026-10-30. Re-signed at every quarterly cadence
> review (exercise 01 Deliverable F).

### 2.2 Ticket template (handoff + ongoing)

Two distinct ticket shapes travel this boundary: **handoff tickets**
(peer → this role, one per model promotion) and **operational
tickets** (either direction, routine).

```yaml
# handoff/<HANDOFF-YYYY-nnn>.yaml
handoff_id: HANDOFF-YYYY-NNN
title: <model name + training lineage + imperative>
intended_workload_class: <tier-A | tier-B | tier-C | draft-for-speculation>
artefact:
  name: <HF handle or internal name>
  revision: <hub commit or internal tag or SHA>
  dtype: <FP16 | BF16 | FP8 | INT8 | AWQ-INT4 | GPTQ-INT4>
  size_bytes: <int>
  tokenizer_sha: <SHA>
  config_sha: <SHA>
  preprocessor_sha: <SHA or n/a>
  licence: <licence id + attribution pointer>
training_lineage:
  base_model: <handle + revision>
  fine_tune_type: <LoRA | QLoRA | SFT | RLHF | DPO | merge | n/a>
  training_dataset_pointer: <path or URI; lineage record>
  training_commit: <SHA of training repo>
  wandb_or_mlflow_run: <URL or n/a>
pre_handoff_evals:
  - name: <eval name>
    dataset_sha: <SHA>
    metric: <value>
    pass_threshold: <value>
    pass: <true | false>
    artefact: <report URL>
model_card: <URL — must exist and be reviewable>
intended_prompt_format: <chat template id + pointer>
quantisation_plan:
  stage: <trained-native | post-train-quantised | pre-handoff-quantised>
  algorithm: <GPTQ | AWQ | bitsandbytes | NF4 | TRT-LLM calibration | n/a>
  calibration_dataset_sha: <SHA or n/a>
  expected_accuracy_delta: <percent or n/a>
draft_target_pair:                 # only for draft artefacts
  target_model: <handle + revision>
  draft_acceptance_rate_baseline: <alpha measured by peer on the baseline fixture>
  pair_contract_pointer: <module 108 ex-04 contract URL>
supply_chain:
  provenance: <upstream source; download URL or internal path>
  sha_verified_by: <named engineer or automated step>
  licence_cleared_by: <legal / security peer pointer>
intended_sunset_criterion: <criterion that triggers decommission>
peer_signatures:
  ml_platform_lead: "[SIMULATED]"
  training_owner: "[SIMULATED]"
  date: <YYYY-MM-DD>
```

```yaml
# ops/<TKT-YYYY-nnn>.yaml   (either direction)
ticket_id: TKT-YYYY-NNN
direction: <platform-to-ml | ml-to-platform>
title: <short imperative>
touches_handoff: <handoff id or n/a>
artefact_pinned: <sha>
workload_class: <...>
axis: <accuracy | drift | prompt-format | tokenizer | preprocessor | quantisation-regression>
reproduction: <pointer>
expected_return: <one of §2.4>
priority: <P0 | P1 | P2 | P3>
sla: <row from §2.3>
notes: <free-form>
```

### 2.3 Response SLA table

| Priority | Description | Ack | Opinion returned | Resolution target | Org severity |
|---|---|---|---|---|---|
| P0 | Signed artefact fails a correctness check post-handoff; tier-A model accuracy regression identified in production; tokenizer / chat-template mismatch causing user-visible garbling. | 30 min (24/7) | 4 h | best-effort within 48 h; acknowledged workaround in 4 h | sev-1 |
| P1 | Pre-handoff eval disagrees with application eval on the same slice; drift surfaced by the eval peer's ongoing regression; draft-target acceptance rate below the module 108 ex-04 baseline. | 2 h business / 4 h off-hours | 1 business day | 7 days | sev-2 |
| P2 | Quantisation plan question; licence provenance query; training-lineage reproducibility request. | 1 business day | 3 business days | 30 days | sev-3 |
| P3 | Model-card improvements; non-blocking observability requests; future-handoff scoping conversations. | 3 business days | 2 weeks | next quarter | ticket-only |

Reuses the org severity scale per exercise 04 §4.3; no peer-specific
severity invented.

### 2.4 Return-shape taxonomy

**Forward direction (peer → this role; artefact handoff).**

1. **Signed handoff manifest** — the structured artefact above with
   peer signatures. This role ingests and promotes per the module 108
   ex-04 lifecycle discipline.
2. **Revised handoff manifest** — the peer re-issues after an in-flight
   fix (missing field, updated eval numbers, revised quantisation). The
   prior handoff ID is superseded.
3. **"Not ready for handoff"** — the peer declines to sign because a
   pre-handoff eval failed or a provenance gap remains. The handoff
   ticket closes with the gating reason; this role does not ingest.
4. **Ad-hoc ownership query** — rare; a response that clarifies who
   owns a specific edge case (e.g., a preprocessor artefact that
   sits ambiguously between the model config and the prompt pipeline).
   Resolved by naming the artefact in the next contract re-sign.

**Reverse direction (this role → peer; operational).**

- **Promotion-stage report** — "artefact HANDOFF-2026-012 is at 10 %
  canary; accuracy delta on prod slice is -0.3 % vs. baseline;
  proceeding to 100 % barring eval-peer veto."
- **Drift ticket** — "we are seeing regression on the `search-rag`
  eval slice; is this a known limitation of the training frontier or
  is production drift?"
- **Sunset proposal** — "model HANDOFF-2025-003 sits at 2 % traffic;
  proposing sunset at 2027-02-01 per original sunset criterion."
- **Reproducibility request** — "please attach the training-run log
  for HANDOFF-2026-005; we are reproducing the quantisation step for
  a security peer review."

### 2.5 Artefact registry (ledger)

```yaml
# registry/<handoff-id>.yaml
handoff_id: HANDOFF-YYYY-NNN
artefact_sha: <sha>
workload_class: <...>
traffic_share: <percent>
promoted_at: <date>
pin_expiry: <date or n/a>
accuracy_watch_metric: <axis + threshold>
sunset_criterion: <criterion>
sunset_target_date: <date or n/a>
related_regressions: [<ticket ids>]
last_reviewed: <date>
status: <staged | canary | 100pct | sunsetting | sunset>
```

**Review cadence.** Weekly staged / canary scan by this role; monthly
joint scan with the ML-platform peer at the cross-peer cadence (§6);
quarterly reconciliation at exercise 01 Deliverable F.

**Seeded entries.**

```yaml
handoff_id: HANDOFF-2026-012
artefact_sha: "[SIMULATED: sha256:...]"
workload_class: tier-A
traffic_share: 100
promoted_at: 2026-07-20
pin_expiry: n/a
accuracy_watch_metric: "search-rag-eval-f1 >= 0.82"
sunset_criterion: "replaced by next primary promotion or held until 2027-07-01"
sunset_target_date: 2027-07-01
related_regressions: []
last_reviewed: 2026-11-04
status: 100pct
```

```yaml
handoff_id: HANDOFF-2026-018
artefact_sha: "[SIMULATED]"
workload_class: draft-for-speculation
traffic_share: 0                 # pre-canary
promoted_at: n/a
pin_expiry: 2026-12-15
accuracy_watch_metric: "acceptance-rate >= 0.65 per module 108 ex-04"
sunset_criterion: "superseded when target model rev changes"
related_regressions: [KREG-2026-003]   # cross-ref from exercise 04 ledger
last_reviewed: 2026-11-10
status: staged
```

### 2.6 Worked example — tokenizer drift after a fine-tune promotion

**Setting.** The ML-platform peer signs HANDOFF-2026-015, a LoRA
fine-tune of Llama-3.1-70B-Instruct intended to replace the tier-A
primary. The manifest declares the tokenizer SHA matches the base
model. At 3 % canary the eval peer's ongoing regression slice detects
a 1.5 pp F1 drop on `search-rag`.

**Rule-outs performed by this role before filing to the peer.**

- Artefact SHA pin check passes.
- Deployment manifest unchanged against the HANDOFF-2026-012 baseline.
- Workload shape unchanged.
- Quantisation path unchanged (both artefacts FP16).
- Prompt format unchanged (same chat template id).

Rule-outs land the ticket in handoff territory, not kernel or runtime.

**Filed ticket (operational, platform → ML-platform peer).**

```yaml
ticket_id: TKT-2026-501
direction: platform-to-ml
title: "search-rag F1 regresses 1.5 pp at 3 % canary for HANDOFF-2026-015"
touches_handoff: HANDOFF-2026-015
artefact_pinned: "[SIMULATED]"
workload_class: tier-A
axis: accuracy
reproduction: repro/TKT-2026-501/   # eval-peer slice + runner script
expected_return: diagnostic opinion → either revised handoff manifest or confirmation that drift is within the training-frontier tolerance
priority: P1
sla: ack 2 h business / opinion 1 bd / resolution 7 d
notes: >
  Eval-peer slice fired at 3 % canary. Rule-outs exercised; no runtime
  cause. Suspect a tokenizer special-token mismatch after LoRA merge;
  the fine-tune's special-token list extends the base's by one token,
  which should be compatible but has been a prior-year incident
  pattern in similar fine-tunes.
```

**Peer response.** Within the P1 opinion SLA: "Confirmed. LoRA merge
did not propagate the added special token through the tokenizer vocab;
production deployment is using the base tokenizer, which silently
drops the token. Re-issuing HANDOFF-2026-015 as HANDOFF-2026-015.r2
with the correct tokenizer SHA. Prior-year pattern matched."

**Follow-up.** This role (a) halts the canary, (b) reverts traffic to
HANDOFF-2026-012, (c) files a change-calendar entry for the re-canary
once HANDOFF-2026-015.r2 lands, (d) extends the handoff-manifest
schema's `tokenizer_sha` field documentation to call out LoRA-merge
compatibility explicitly.

**Ledger impact.** HANDOFF-2026-015 is marked `status: sunset` with
the gating reason. HANDOFF-2026-015.r2 is a new registry entry that
inherits the traffic-share-watch from its predecessor.

### 2.7 Three dry-run tickets

**Ticket 1 (forward; operational).**

```yaml
ticket_id: TKT-2026-502 (dry-run)
direction: platform-to-ml
title: "Reproducibility: attach training-run log for HANDOFF-2026-012 (security peer request)"
touches_handoff: HANDOFF-2026-012
workload_class: tier-A
axis: supply-chain / provenance
expected_return: diagnostic opinion → attached wandb run
priority: P2
sla: ack 1 bd / opinion 3 bd
notes: "Needed for the §5.6 security-peer supply-chain review."
```

**Ticket 2 (forward; handoff-blocking).**

```yaml
ticket_id: TKT-2026-503 (dry-run)
direction: platform-to-ml
title: "Draft-target acceptance rate below the module 108 ex-04 baseline on HANDOFF-2026-018"
touches_handoff: HANDOFF-2026-018
workload_class: draft-for-speculation
axis: acceptance-rate
expected_return: revised handoff manifest with retrained draft
priority: P1
sla: ack 2 h / opinion 1 bd / resolution 7 d
notes: >
  Pre-canary α measured 0.58 on the fixture; baseline from the ex-04
  contract is 0.65. Migration #1 in ex-01 §7.5 is blocked until this
  resolves.
```

**Ticket 3 (reverse).**

```yaml
ticket_id: TKT-2026-504 (dry-run)
direction: ml-to-platform
title: "Please schedule a 1 % shadow canary for HANDOFF-2026-020 (safety-fine-tuned Llama-3.1-70B)"
touches_handoff: HANDOFF-2026-020
expected_return: canary-scheduled; accuracy delta report at end of shadow
priority: P2
sla: ack 1 bd / deliverable 10 bd
notes: "Shadow only; no live traffic. Needed to validate the fine-tune before promotion."
```

---

## 3. Deliverable B — Eval peers interface contract

### 3.1 Boundary statement (one page)

> **`[SIMULATED]` Inference Platform ↔ Eval Peers Interface Contract —
> v1.0 — effective 2026-Q4**
>
> **This role owns the production serving system.** Specifically:
>
> - Running the inference engine that evals call.
> - Capacity for eval runs (the `internal eval-runner` workload class).
> - Scheduling eval runs against production replicas without perturbing
>   consumer-track SLOs.
> - Ingesting eval sign-off as a change-calendar gate (exercise 01
>   Deliverable C `eval_gate` field).
> - Attaching eval regression slices to the SLO contract (module 110).
>
> **The eval peers own evaluation methodology.** Specifically:
>
> - Capability / benchmark eval authorship (MMLU-style, harness-based,
>   leaderboards) — "does this model class work at all?"
> - Application eval authorship (`search-rag` slice, `assistant-product`
>   slice, task-specific slices) — "does this model work for our users?"
> - Dataset curation, versioning, dataset-lineage records.
> - Metric definitions and their ongoing validity.
> - Eval harness code and its reproducibility.
> - Red-team / adversarial eval authorship (coordinated with the
>   security peer; see §5).
>
> **The 20 % shared surface.**
>
> - **Eval gate on changes.** The change calendar's `eval_gate` field
>   is a joint artefact: the eval peers define pass thresholds; this
>   role runs the gate on the change artefact.
> - **Regression slices on production traffic.** The eval peer defines
>   the slice; this role instruments production, samples traffic,
>   routes the sample to the harness, and surfaces the result on the
>   weekly SLO review dashboard (exercise 01 §6).
> - **Pre-flight evals for canary stages.** The migration plan's
>   canary stages (exercise 01 Deliverable F) name eval gates at 1 %,
>   10 %, and 100 %; this role sequences, the eval peer validates.
> - **Eval runs as a workload class.** Capacity, prioritisation, and
>   quota for `internal eval-runner` is a joint artefact.
> - **Prompt-format and chat-template changes.** The eval peer catches
>   regressions; the ML-platform peer (§2) authors the template; this
>   role deploys. All three sign.
>
> **The boundary rule (one sentence).**
>
> > The eval peers own what "working" means; this role owns running
> > the thing being evaluated and ingesting the eval sign-off as a
> > deploy gate.
>
> **Signed.** `[SIMULATED: Platform Lead]`, `[SIMULATED: Capability
> Eval Lead]`, `[SIMULATED: Application Eval Lead]`, 2026-10-30.

Two eval leads are named because capability and application evals are
often owned by different teams (the capability eval peer frequently
lives with the ML-platform org; the application eval peer frequently
lives with the consumer tracks).

### 3.2 Ticket template

```yaml
# eval-gate/<EG-YYYY-nnn>.yaml
gate_id: EG-YYYY-NNN
direction: <platform-to-eval | eval-to-platform>
title: <short imperative>
change_ref: <CHG-id or handoff-id or migration-id>
eval_scope:
  capability_evals: [<eval name + dataset sha + threshold>]
  application_evals: [<slice name + dataset sha + threshold>]
  red_team: <included | not applicable>
measurement_window:
  type: <pre-deploy | canary-1pct | canary-10pct | 100pct | 7-day-post>
  duration: <hours or days>
  sample_rate: <float or absolute count>
pass_criteria:
  per_eval:
    - name: <eval name>
      metric: <value>
      threshold: <value>
      pass: <true | false | pending>
  aggregate: <all-must-pass | k-of-n | weighted-score>
deploy_gate: <blocking | advisory>
priority: <P0 | P1 | P2 | P3>
sla: <row from §3.3>
notes: <free-form>
peer_signatures:
  capability_eval_lead: "[SIMULATED]"
  application_eval_lead: "[SIMULATED]"
  platform_lead: "[SIMULATED]"
  date: <YYYY-MM-DD>
```

### 3.3 Response SLA table

| Priority | Description | Ack | Opinion returned | Resolution target | Org severity |
|---|---|---|---|---|---|
| P0 | Production regression slice fires; tier-A eval metric moves > 3 pp vs. baseline; red-team signal of a safety regression. | 30 min (24/7) | 2 h | eval verdict in 6 h; deploy halt in 30 min if verdict is "stop" | sev-1 |
| P1 | Pre-canary gate fails on a committed migration; canary stage result ambiguous; drift on a non-tier-A slice. | 1 h business / 2 h off | 1 bd | 5 bd | sev-2 |
| P2 | New eval slice proposal; dataset-version bump; methodology question on an existing slice. | 1 bd | 3 bd | 30 d | sev-3 |
| P3 | Eval-harness performance / reproducibility work; backlog. | 3 bd | 2 w | next Q | ticket-only |

### 3.4 Return-shape taxonomy

**Forward (platform → eval).**

- **Eval-run request** — scheduling a run for a gate; this role
  provides capacity and the artefact under test.
- **Production-slice attach request** — "please add a regression slice
  on this new eval; we will instrument."
- **Reproducibility request** — "please attach the dataset SHA and
  harness commit for this previous eval."

**Reverse (eval → platform).**

1. **Gate pass** — all pass criteria met; deploy may proceed. Signed
   by the relevant eval lead(s).
2. **Gate pass with caveats** — metric moved within tolerance but a
   specific slice is worth watching; deploy may proceed with an
   accuracy-watch task filed.
3. **Gate fail** — pass criteria not met; deploy must halt. The eval
   peer attaches the failing slice evidence; this role halts the
   change-calendar entry.
4. **Verdict pending** — measurement window not yet elapsed; deploy
   must wait. Common on 7-day-post-100pct slices.
5. **Methodology escalation** — the eval peer flags that the metric
   or dataset no longer represents the production distribution; a
   methodology ticket is opened. Deploy decision deferred pending
   methodology fix or an explicit waiver.

### 3.5 Eval ledger

```yaml
# eval-ledger/<SLICE-id>.yaml
slice_id: <id>
title: <slice name>
owner_eval_peer: <capability | application | both>
dataset_sha: <sha>
dataset_version_date: <date>
metric: <metric id>
baseline_value: <value>
threshold: <value>
production_sample_rate: <float>
last_firing: <date or n/a>
last_reviewed: <date>
status: <active | muted | retired>
notes: <free-form>
```

**Seeded entries (two).**

```yaml
slice_id: SLICE-search-rag-f1
title: "search-rag retrieval-and-answer F1 on the consumer-track eval set"
owner_eval_peer: application
dataset_sha: "[SIMULATED]"
dataset_version_date: 2026-09-01
metric: f1
baseline_value: 0.84
threshold: 0.82      # 2 pp drop fires
production_sample_rate: 0.01
last_firing: 2026-11-04        # the §2.6 worked example
last_reviewed: 2026-11-10
status: active
```

```yaml
slice_id: SLICE-assistant-safety-refusal
title: "assistant-product refusal-rate on the red-team battery"
owner_eval_peer: capability
dataset_sha: "[SIMULATED]"
dataset_version_date: 2026-10-01
metric: refusal-rate
baseline_value: 0.97
threshold: 0.95
production_sample_rate: 0.005
last_firing: n/a
last_reviewed: 2026-11-10
status: active
notes: "Co-owned with the security peer; see §5."
```

### 3.6 Worked example — pre-canary gate on a 70B promotion

See §2.6 for the companion story: the ML-platform peer's
HANDOFF-2026-015 fine-tune is halted because this contract's
`SLICE-search-rag-f1` fires at 3 % canary. The eval peer returns a
**gate-fail** verdict within the P1 SLA (1 bd opinion, 5 bd
resolution). This role halts CHG-2026-100 (the canary change-calendar
entry); the ML-platform peer re-issues the handoff (§2.6); the
eval peer re-runs the gate on HANDOFF-2026-015.r2 at the same 3 %
canary; verdict is **gate pass**; CHG-2026-100 re-opens as
CHG-2026-100.r2.

The eval-peer verdict is what transitions the change-calendar entry's
`eval_gate` field from `pending` through `passed` or `failed`. The
gate IS the contract's return shape.

### 3.7 Three dry-run tickets

**Ticket 1 (forward; gate request).**

```yaml
gate_id: EG-2026-061 (dry-run)
direction: platform-to-eval
title: "Please gate the speculative-decoding 1 % canary (migration #1 in ex-01 §7.5)"
change_ref: migration #1 stage 3
eval_scope: { capability_evals: [], application_evals: [SLICE-search-rag-f1, SLICE-assistant-safety-refusal], red_team: not-applicable }
measurement_window: { type: canary-1pct, duration: 72h, sample_rate: 1.0 of the 1% }
pass_criteria: { aggregate: all-must-pass, per_eval: [{name: SLICE-search-rag-f1, threshold: 0.82}, {name: SLICE-assistant-safety-refusal, threshold: 0.95}] }
deploy_gate: blocking
priority: P1
sla: ack 1 bd / opinion 3 bd / resolution 5 bd
```

**Ticket 2 (reverse; methodology escalation).**

```yaml
gate_id: EG-2026-062 (dry-run)
direction: eval-to-platform
title: "search-rag F1 dataset drift — propose dataset-version bump before next quarterly eval refresh"
change_ref: SLICE-search-rag-f1
expected_return: dataset-version bump; re-baseline accepted
priority: P2
sla: ack 1 bd / opinion 3 bd
notes: >
  Production distribution has shifted (more code-adjacent queries) since the
  2026-09-01 dataset snapshot. Proposing a 2026-12-01 refresh; this role to
  schedule the re-baseline as a change-calendar entry.
```

**Ticket 3 (forward; production-slice attach).**

```yaml
gate_id: EG-2026-063 (dry-run)
direction: platform-to-eval
title: "Attach a per-tenant TTFT fairness slice to the weekly SLO review dashboard"
eval_scope: { application_evals: [SLICE-per-tenant-ttft-fairness] }
measurement_window: { type: 7-day-post, duration: 168h, sample_rate: 1.0 }
deploy_gate: advisory
priority: P3
sla: next Q
notes: >
  Follow-up from exercise 01 §6.5 — the dry-run SLO review identified a
  per-tenant TTFT gap that the current dashboard can't see.
```

---

## 4. Deliverable C — SRE peer interface contract

### 4.1 Boundary statement (one page)

> **`[SIMULATED]` Inference Platform ↔ SRE Peer Interface Contract —
> v1.0 — effective 2026-Q4**
>
> **This role owns the inference-specific production system.**
> Specifically:
>
> - vLLM / SGLang engine serving, the KV allocator, the autoscaler,
>   the router, the warm pool.
> - Inference-specific SLO numbers (TTFT, ITL, E2E, availability,
>   cost per 1M tokens).
> - Inference-specific runbooks (module 113 ch. 03).
> - Inference-specific paging matrix (exercise 01 Deliverable D).
> - Inference-specific change calendar (exercise 01 Deliverable C).
>
> **The SRE peer owns the shared production substrate.**
> Specifically:
>
> - Kubernetes / cluster orchestration primitives this platform rides
>   on.
> - Node lifecycle (provisioning, draining, decommissioning, imaging).
> - Hardware failure handling at the SRE-platform layer.
> - Networking, load-balancing, DNS, service mesh.
> - Logging / metrics / traces infrastructure (not the inference
>   dashboards themselves; those are this role's).
> - Incident command *framework* (channel setup, timeline capture,
>   RCA template) — this role runs the incident, the SRE peer owns
>   the framework.
> - Availability SLOs *as a platform-wide floor* — "availability shall
>   be ≥ 99 % at every service" — this role owns the inference-specific
>   tighter number (99.5 % tier-A) on top.
>
> **The 20 % shared surface.**
>
> - **SLO contract co-signature.** The SRE peer signs the availability
>   column of the SLO contract (module 110 / ex-01 Deliverable B)
>   because that column rides on the substrate the SRE peer owns.
> - **Pager-policy joint ownership.** The paging matrix in exercise 01
>   §5.3 routes incidents to the SRE peer on five of its ten rows; the
>   peer's on-call rota is part of the escalation chain.
> - **Change-calendar co-ownership for hardware events.** Driver
>   upgrades, kernel rolls, hardware refresh waves go on the inference
>   change calendar (because they can perturb SLO) AND on the SRE
>   peer's substrate calendar.
> - **Capacity escalation.** The migration roadmap's "add nodes"
>   candidates (exercise 01 Deliverable F) are negotiated with the SRE
>   peer because capacity is a substrate concern.
> - **Incident command for cross-boundary incidents.** The SRE peer
>   runs incident command when an incident spans multiple services;
>   this role runs it for inference-contained incidents.
>
> **The boundary rule (one sentence).**
>
> > The SRE peer owns the substrate and the incident-command framework;
> > this role owns the inference stack that rides on it and the
> > inference-specific SLO and runbooks.
>
> **Signed.** `[SIMULATED: Platform Lead]`, `[SIMULATED: SRE Peer
> Lead]`, 2026-10-30.

### 4.2 Ticket template

Three distinct shapes travel this boundary: **joint change-calendar
entries**, **capacity-escalation tickets**, and **incident-command
handoff tickets**.

```yaml
# joint-change/<JCHG-id>.yaml
joint_change_id: JCHG-YYYY-NNN
title: <short imperative>
primary_owner: <platform | sre>
inference_change_ref: <CHG-id on inference calendar>
sre_substrate_change_ref: <peer calendar id or n/a>
window: { start: ..., end: ..., duration: ..., timezone: UTC }
blast_radius:
  inference_workload_classes: [...]
  substrate_services_affected: [...]
reversibility: <per exercise 01 §4>
rollback_coordination: <who flips which flag when>
peer_signatures:
  platform_lead: "[SIMULATED]"
  sre_peer_lead: "[SIMULATED]"
notes: <free-form>
```

```yaml
# capacity/<CAP-id>.yaml
capacity_id: CAP-YYYY-NNN
title: <short imperative>
workload_class: <...>
current_headroom: <percent>
projected_breach_date: <date>
growth_driver: <forecast pointer>
ask: <N nodes of SKU X, by date>
tco_delta: <per-month $ delta>
alternate: <what we would do if the ask is declined>
priority: <P0 | P1 | P2 | P3>
sla: <row from §4.3>
```

```yaml
# ic-handoff/<ICH-id>.yaml
handoff_id: ICH-YYYY-NNN
incident_ref: <INC-id>
direction: <platform-to-sre | sre-to-platform>
reason: <scope expanded | scope narrowed | on-call exhaustion | expertise>
state_snapshot: <timeline link + current actions + unknowns>
new_ic: <named engineer>
ack_within: <minutes>
```

### 4.3 Response SLA table

| Priority | Description | Ack | Opinion returned | Resolution target | Org severity |
|---|---|---|---|---|---|
| P0 | Cross-boundary incident (inference + substrate); availability floor at risk; node-level pattern of failure that threatens tier-A SLO. | 15 min (24/7) | 2 h | best-effort within 24 h | sev-1 |
| P1 | Capacity breach projected within 10 days; joint change-calendar entry needs scheduling; substrate change proposed that touches inference. | 1 h business / 2 h off | 1 bd | 5 bd | sev-2 |
| P2 | Routine joint-change coordination; capacity escalation > 10 d out; dashboard / metric infra request. | 1 bd | 3 bd | 30 d | sev-3 |
| P3 | Postmortem actions from a shared incident; longer-horizon capacity planning. | 3 bd | 2 w | next Q | ticket-only |

### 4.4 Return-shape taxonomy

**Forward (platform → SRE).**

- **Joint change-calendar signoff request** — co-signs an entry that
  spans inference and substrate.
- **Capacity escalation** — formal ask for node / hardware / network
  capacity with a date and a TCO delta.
- **Substrate-side incident handoff** — "incident scope has grown
  beyond inference; please take IC."

**Reverse (SRE → platform).**

- **Substrate change notification** — "we are rolling driver 555 → 556
  across the fleet in this window; please sign the inference-side
  calendar entry."
- **Hardware event escalation** — "node us-east-1-a-node-12 has failed
  SMI for the third time this week; recommending decommission; please
  signal if that perturbs inference capacity."
- **Capacity delivery confirmation** — the SRE peer closes the loop on
  a prior capacity ticket with provisioning dates.
- **Inference-side incident handoff** — "the paging matrix sent this to
  SRE first, but on triage it is inference-contained; handing IC back."

### 4.5 SLO / availability ledger

```yaml
# slo-events/<SEV-id>.yaml
event_id: SEV-YYYY-NNN
date: <date>
workload_class: <...>
metric: <ttft | itl | e2e | availability | cost>
baseline: <value>
observed: <value>
budget_burn: <percent of monthly budget consumed>
root_cause_class: <kernel | config | substrate | capacity | workload-shift | deploy-induced | upstream>
incident_ref: <INC-id>
runbook_fired: <runbook id>
correction_plan: <pointer to the ex-01 §3.1 correction plan or amendment>
last_reviewed: <date>
status: <open | burned-within-budget | correction-in-flight | correction-closed | amendment-signed>
```

**Seeded entries.**

```yaml
event_id: SEV-2026-114
date: 2026-11-08
workload_class: tier-A
metric: ttft
baseline: 1.2s
observed: 1.65s
budget_burn: 0.9
root_cause_class: kernel         # driver-induced via KREG-2026-001 (ex-04 §6)
incident_ref: INC-2026-114
runbook_fired: tier-a-ttft-burn
correction_plan: CHG-2026-096 (the pin) + follow-up on engine upgrade
last_reviewed: 2026-11-14
status: correction-in-flight
```

```yaml
event_id: SEV-2026-098
date: 2026-10-22
workload_class: tier-B
metric: availability
baseline: 99.0
observed: 98.4
budget_burn: 2.1
root_cause_class: substrate     # us-east-1-b-node-17 NIC flap
incident_ref: INC-2026-098
runbook_fired: tier-b-availability
correction_plan: SRE-peer-owned substrate NIC refresh wave
last_reviewed: 2026-11-04
status: correction-in-flight
```

### 4.6 Worked example — assistant-product launch capacity collision

**Setting.** The exercise 01 §6 weekly review flags that warm-pool
drain is up 15 % WoW because the `assistant-product` track's launch
traffic arrived earlier than forecast. Tier-A headroom is at 25 %;
projected breach of the capacity ceiling is 2026-12-18 without action.

**Capacity escalation filed.**

```yaml
capacity_id: CAP-2026-022
title: "Add 6 H100 nodes to us-east-1-a by 2026-12-10 for the assistant-product launch window"
workload_class: tier-A
current_headroom: 25
projected_breach_date: 2026-12-18
growth_driver: assistant-product capacity forecast rev-2
ask: "6 × H100-SXM5 nodes × 8 GPU = 48 GPUs in us-east-1-a by 2026-12-10"
tco_delta: "[SIMULATED: $450k/yr incremental; prorated to Q1 2027: $112k]"
alternate: "throttle research-sandbox T2 to 50% of its current allocation and recover 15% headroom; not sufficient standalone"
priority: P1
sla: ack 1 bd / opinion 1 bd / resolution 5 bd
```

**Peer response.** SRE peer confirms 4 of 6 nodes can be provisioned
from existing substrate pool by 2026-12-05; 2 require a hardware
purchase order with a 20-business-day lead time, delivered 2027-01-12.

**Joint negotiation (recorded in the ticket).** This role proposes a
hybrid: 4 nodes from pool by 2026-12-05 (covers through 2026-12-24),
plus a planned research-sandbox throttle starting 2026-12-24 if the
remaining 2 nodes haven't landed. SRE peer accepts; filed as
JCHG-2026-033 on both calendars.

**SLO-ledger impact.** Opens SEV-2026-150 pre-emptively with
`status: correction-in-flight`; the correction plan is JCHG-2026-033.
Closes to `correction-closed` when 2026-12-05 provisioning completes
and panels confirm headroom returns above 25 %.

### 4.7 Three dry-run tickets

**Ticket 1 (forward; joint change).**

```yaml
joint_change_id: JCHG-2026-033 (dry-run)
title: "Provision 4 H100 nodes in us-east-1-a for assistant-product launch"
primary_owner: sre
inference_change_ref: CHG-2026-110
sre_substrate_change_ref: "[SIMULATED: SRE-CAL-2026-442]"
window: { start: 2026-12-01 02:00 UTC, end: 2026-12-05 06:00 UTC, duration: 100h, timezone: UTC }
blast_radius:
  inference_workload_classes: [tier-A]
  substrate_services_affected: [cluster us-east-1-a node pool]
reversibility: node-decommission within 10 bd if capacity forecast reverses
rollback_coordination: "SRE drains before any flag flip; platform re-balances warm pool before traffic admit"
```

**Ticket 2 (reverse; substrate notification).**

```yaml
capacity_id: CAP-2026-023 (dry-run; reverse)
title: "Driver 555 → 556 roll across us-east-1-a — please co-sign the inference-side CHG"
inference_change_ref: CHG-2026-115 (pending co-sign)
notes: >
  Known-good for 24h in us-east-1-b at the substrate layer. Need the
  inference-side calendar entry and the ex-04 performance-peer kernel
  compat sign on KREG-2026-001.
priority: P1
sla: ack 1 bd / opinion 1 bd
```

**Ticket 3 (forward; incident handoff).**

```yaml
handoff_id: ICH-2026-011 (dry-run)
incident_ref: INC-2026-152
direction: platform-to-sre
reason: "scope expanded — the TTFT burn traces to a network-plane flap on the service mesh; outside inference"
state_snapshot: "timeline attached; current action: throttle traffic 20%; unknowns: mesh team root cause"
new_ic: "[SIMULATED: SRE on-call primary]"
ack_within: 10 min
```

---

## 5. Deliverable D — security peer interface contract

### 5.1 Boundary statement (one page)

> **`[SIMULATED]` Inference Platform ↔ Security Peer Interface
> Contract — v1.0 — effective 2026-Q4**
>
> **This role owns the inference serving plane and its running
> configuration.** Specifically:
>
> - Engine CLI / args and runtime config.
> - Deploy ritual and change calendar.
> - Serving-side rate-limit *mechanisms* (token-bucket implementation,
>   per-tenant counters, 429 behaviour).
> - Model artefact ingress (with the ML-platform peer §2) — SHA
>   verification, licence recording, registry hygiene.
> - KV-cache tenancy and prefix-sharing configuration (module 104
>   ex-06 cross-tenant safety review).
> - Log / trace emission on the serving path.
>
> **The security peer owns security policy, review, and incident
> response for the organisation.** Specifically:
>
> - Supply-chain policy authorship (what sources are allowed for model
>   weights, what SHA-pin discipline is required, what licence classes
>   are allowed).
> - Abuse / jailbreak policy authorship (which prompts get refused,
>   which behaviours trip audit).
> - Red-team programme (co-ordinated with eval peers §3) — the
>   security peer owns the adversarial-testing programme; the eval
>   peer owns the harness that runs it.
> - Rate-limit *tiers and thresholds* (the policy numbers; not the
>   mechanism).
> - Incident command for security incidents (data exfiltration,
>   unauthorised access, abuse-at-scale, credential compromise).
> - Compliance audit response.
> - External vulnerability disclosure response for inference-adjacent
>   CVEs (engine vulns, model-weight supply-chain vulns, dependency
>   vulns).
>
> **The 20 % shared surface.**
>
> - **Supply-chain review for new model handoffs.** Named in the
>   handoff manifest §2.2 `supply_chain.licence_cleared_by`. The
>   security peer signs supply-chain review on every new artefact
>   ingested.
> - **Rate-limit tier assignment and threshold tuning.** Policy from
>   the security peer; implementation from this role.
> - **KV cross-tenant policy.** Any prefix-sharing or cross-tenant
>   KV-cache configuration requires security peer sign-off
>   (reference: module 104 ex-06).
> - **Deploy gate for security-touching changes.** Model-swap changes
>   that cross a supply-chain boundary (new vendor source, new
>   licence class) go through a security-peer review on the
>   change-calendar entry.
> - **Audit-log emission.** The security peer dictates what must be
>   logged (per-request tenant id, model id, prompt-hash where
>   permitted, tooling policy); this role emits.
> - **Incident response for cross-boundary security incidents.**
>   Security peer is IC; this role is on the technical bridge.
>
> **The boundary rule (one sentence).**
>
> > The security peer owns policy and incident command for security
> > issues; this role owns the serving plane and the mechanisms that
> > implement the policy, with sign-off from the security peer on
> > every policy-touching change.
>
> **Signed.** `[SIMULATED: Platform Lead]`, `[SIMULATED: Security Peer
> Lead]`, 2026-10-30. Framework references: [owasp-ml-top-10],
> [mitre-atlas], [nist-ai-rmf] — governance mapping in §5.4.

### 5.2 Ticket template

```yaml
# sec/<SEC-YYYY-nnn>.yaml
sec_id: SEC-YYYY-NNN
direction: <platform-to-sec | sec-to-platform>
category: <supply-chain | rate-limit-tier | kv-tenancy | audit-log | incident | policy-interpretation | vuln>
title: <short imperative>
touches:
  change_ref: <CHG-id or handoff-id>
  workload_class: <...>
  tenant_tier: <T1 | T2 | T3 | all>
threat_mapping:
  owasp_ml_top_10: [ML01..ML10]       # list all applicable
  mitre_atlas_techniques: [...]       # e.g. T0021 (model weight exfiltration)
  nist_ai_rmf_functions: [GOVERN | MAP | MEASURE | MANAGE]
evidence:
  logs: <pointer>
  traces: <pointer>
  benchmark_or_eval: <pointer>
proposed_control: <description + where it is enforced>
proposed_rollback: <how we revert if control is wrong>
priority: <P0 | P1 | P2 | P3>
sla: <row from §5.3>
peer_signatures:
  security_peer_lead: "[SIMULATED]"
  platform_lead: "[SIMULATED]"
  date: <YYYY-MM-DD>
notes: <free-form>
```

### 5.3 Response SLA table

| Priority | Description | Ack | Opinion returned | Resolution target | Org severity |
|---|---|---|---|---|---|
| P0 | Active abuse at scale; credential or data exfiltration signal; CVE in a running engine; deploy that silently widened a tenancy boundary. | 15 min (24/7) | 1 h | mitigation in place within 4 h | sev-1 |
| P1 | Non-active signal of policy drift (rate-limit tier lift active, KV prefix-share enabled past approval, missing audit-log emission). | 1 h business / 2 h off | 1 bd | 5 bd | sev-2 |
| P2 | Supply-chain review for routine handoff; rate-limit threshold tuning request; audit-log scope change. | 1 bd | 3 bd | 30 d | sev-3 |
| P3 | Policy interpretation question; red-team programme coordination; longer-horizon control authorship. | 3 bd | 2 w | next Q | ticket-only |

### 5.4 Return-shape taxonomy

**Forward (platform → security).**

- **Supply-chain review request** — for a new model handoff, driver
  revision, engine version bump, or dependency upgrade.
- **Policy clarification** — "does the current rate-limit policy
  intend to cap per-tenant WHOLE-REQUEST tokens or per-tenant
  concurrent requests?"
- **Deploy gate review** — for a security-touching change.
- **Incident technical-bridge join** — this role attaches to a
  security incident as technical lead; security peer is IC.

**Reverse (security → platform).**

1. **Policy sign** — control approved; this role proceeds with deploy.
2. **Policy sign with constraint** — approved only with named
   constraint (e.g., "KV prefix-sharing within single tenant only; no
   cross-tenant"). This role records the constraint in the change
   entry and in the policy ledger §5.5.
3. **Policy veto** — not approved; this role halts.
4. **New control authored** — the security peer issues a new policy;
   this role implements by the named deadline.
5. **Incident tasking** — during an incident, "halt tier-A ingestion;
   block tenant X; rotate key Y." This role executes within the
   stated timing.

Framework mapping — each return shape ties to one or more external
frameworks ([owasp-ml-top-10], [mitre-atlas], [nist-ai-rmf]). The
security peer records the mapping on the sign to keep the mapping
externally verifiable at audit time. The ticket schema (§5.2)
includes the mapping fields for that reason.

### 5.5 Security-policy ledger

```yaml
# sec-policy/<POL-id>.yaml
policy_id: POL-YYYY-NNN
title: <short imperative>
category: <supply-chain | rate-limit-tier | kv-tenancy | audit-log | incident-playbook>
owner_security: "[SIMULATED: role]"
owner_platform: "[SIMULATED: role]"
framework_mapping:
  owasp_ml_top_10: [...]
  mitre_atlas: [...]
  nist_ai_rmf: [...]
mechanism: <where in the serving plane this is enforced>
threshold_or_policy: <value or rule>
effective_from: <date>
review_cadence: <quarterly | semiannual>
last_reviewed: <date>
status: <active | suspended | deprecated>
notes: <free-form>
```

**Seeded entries.**

```yaml
policy_id: POL-2026-001
title: "Model-weight supply-chain policy — SHA-pin and licence attestation required at ingest"
category: supply-chain
owner_security: "[SIMULATED: Security Peer Lead]"
owner_platform: "[SIMULATED: Platform Lead]"
framework_mapping:
  owasp_ml_top_10: [ML05 Model Theft, ML10 Model Supply-Chain Compromise]
  mitre_atlas: ["AML.T0010 ML Supply Chain Compromise"]   # <!-- pending-verification: verify current technique ID on atlas.mitre.org -->
  nist_ai_rmf: [GOVERN, MAP]
mechanism: "handoff-manifest ingest step + registry write; artefact SHA verified against upstream source; licence recorded"
threshold_or_policy: "No artefact ingested without a verified SHA and a cleared licence; no silent pin changes without a security review on the CHG entry"
effective_from: 2026-07-01
review_cadence: quarterly
last_reviewed: 2026-10-14
status: active
```

```yaml
policy_id: POL-2026-002
title: "KV-cache tenancy — cross-tenant prefix-sharing requires security review"
category: kv-tenancy
owner_security: "[SIMULATED]"
owner_platform: "[SIMULATED]"
framework_mapping:
  owasp_ml_top_10: [ML06 Output Integrity, ML04 Membership Inference]
  mitre_atlas: ["AML.T0024 Exfiltration via ML Inference API"]   # <!-- pending-verification -->
  nist_ai_rmf: [MANAGE]
mechanism: "KV allocator prefix-sharing flag; per-tenant isolation in the block-pool configuration"
threshold_or_policy: "Single-tenant prefix only by default; cross-tenant prefix enabled only with signed exemption + per-tenant opt-in; no system-prompt prefix-sharing across tier-A and tier-B"
effective_from: 2026-09-01
review_cadence: quarterly
last_reviewed: 2026-11-04
status: active
notes: "Gates migration candidate #6 in ex-01 §7.2 (cross-tenant prefix-sharing)."
```

```yaml
policy_id: POL-2026-003
title: "Edge rate-limit tiers"
category: rate-limit-tier
owner_security: "[SIMULATED]"
owner_platform: "[SIMULATED]"
framework_mapping:
  owasp_ml_top_10: [ML02 Data Poisoning via API, ML09 Model Denial of Service]
  mitre_atlas: ["AML.T0029 Denial of ML Service"]   # <!-- pending-verification -->
  nist_ai_rmf: [MANAGE]
mechanism: "edge gateway token-bucket per API key + per-tenant aggregate"
threshold_or_policy:
  T1_per_key_tps: 30
  T2_per_key_tps: 10
  T3_per_key_tps: 2
  T1_per_tenant_tps: 500
  T2_per_tenant_tps: 100
  T3_per_tenant_tps: 20
  burst_allowance: "2x sustained for 60 s"
effective_from: 2026-04-01
review_cadence: quarterly
last_reviewed: 2026-11-04
status: active
```

### 5.6 Worked example — rate-limit tier lift during launch window

**Setting.** Exercise 01 §4.2 CHG-2026-094 — lift assistant-product
tier-A burst rate from 20 to 30 QPS during the pre-launch window,
with freeze-waiver from the sponsor.

**Filed ticket.**

```yaml
sec_id: SEC-2026-044
direction: platform-to-sec
category: rate-limit-tier
title: "Approve burst-rate lift 20 → 30 QPS for assistant-product tier-A, 2026-12-05 through 2026-12-08 auto-revert"
touches: { change_ref: CHG-2026-094, workload_class: tier-A, tenant_tier: T1 }
threat_mapping:
  owasp_ml_top_10: [ML09 Model Denial of Service]
  mitre_atlas: ["AML.T0029"]
  nist_ai_rmf: [MANAGE]
evidence: { logs: grafana.../rate-limit-429, benchmark_or_eval: capacity forecast rev-2 }
proposed_control: "edge gateway per-key bucket lifted; per-tenant aggregate unchanged; auto-revert flag set"
proposed_rollback: "revert bucket config; auto-trigger at 2026-12-08 00:00 UTC if not re-signed"
priority: P2
sla: ack 1 bd / opinion 3 bd
```

**Peer response.** `Policy sign with constraint` within the P2 SLA:
"Approved. Constraint: per-tenant aggregate stays at 500 TPS; the
auto-revert must fire regardless of launch state; if sustained 429
rate exceeds 0.5 % during the window, this role to notify security
peer within 1 h."

**Follow-up.** This role (a) implements the two constraints in the
edge config, (b) updates POL-2026-003's notes with a pointer to
SEC-2026-044, (c) files a pre-authored revert ticket per exercise 01
§6.4 TKT-2026-412 so the 2026-12-08 auto-revert is a one-click
merge, (d) signals the constraint to the on-call so the extra 1-h
notification path is in the handoff doc.

### 5.7 Three dry-run tickets

**Ticket 1 (forward; supply-chain review).**

```yaml
sec_id: SEC-2026-045 (dry-run)
direction: platform-to-sec
category: supply-chain
title: "Supply-chain review for HANDOFF-2026-018 draft model (speculative-decoding migration #1)"
touches: { change_ref: HANDOFF-2026-018, workload_class: draft-for-speculation, tenant_tier: T1 }
threat_mapping:
  owasp_ml_top_10: [ML10 Model Supply-Chain Compromise]
  mitre_atlas: ["AML.T0010"]
  nist_ai_rmf: [GOVERN, MAP]
proposed_control: "SHA-pin at ingest; licence cleared; manifest signed by ML-platform peer"
priority: P1
sla: ack 2 h / opinion 1 bd / resolution 7 d
```

**Ticket 2 (reverse; incident tasking).**

```yaml
sec_id: SEC-2026-046 (dry-run)
direction: sec-to-platform
category: incident
title: "Suspend tenant T2-acct-033 ingest; abuse signal on red-team battery"
touches: { tenant_tier: T2 }
threat_mapping:
  owasp_ml_top_10: [ML02, ML09]
  mitre_atlas: ["AML.T0043 Craft Adversarial Data"]   # <!-- pending-verification -->
  nist_ai_rmf: [MANAGE]
proposed_control: "edge ACL block on tenant key; preserve logs"
priority: P0
sla: ack 15 min / mitigation 4 h
```

**Ticket 3 (forward; policy clarification).**

```yaml
sec_id: SEC-2026-047 (dry-run)
direction: platform-to-sec
category: policy-interpretation
title: "Clarify POL-2026-002 scope for the LMCache CPU prefix tier (ex-01 CHG-2026-095)"
touches: { change_ref: CHG-2026-095, workload_class: tier-A, tenant_tier: T1 }
threat_mapping:
  owasp_ml_top_10: [ML04, ML06]
  nist_ai_rmf: [MANAGE]
proposed_control: "single-tenant prefix only at CPU tier; cross-tenant disabled at both GPU and CPU tiers"
priority: P2
sla: ack 1 bd / opinion 3 bd
```

---

## 6. Deliverable E — cross-peer cadence

### 6.1 Joint rituals

Five rituals stitch the four peer contracts (plus exercise 04's
performance peer contract) back to the platform cadence from
exercise 01.

| Ritual | Frequency | Attendees | Output |
|---|---|---|---|
| **R1 — Weekly SLO review (platform-led)** | Weekly | Platform Lead, on-call primary + secondary, SRE peer on-call rep, performance peer rep. ML-platform / eval / security peers on async digest. | Tickets with owners + due dates (ex-01 §6). |
| **R2 — Cross-peer monthly** | Monthly, 60 min | Platform Lead + one lead per peer contract (perf, ML-platform, eval, SRE, security). | Ledger reconciliation across the five peer ledgers; freeze-check from the consumer side (next 30-day launch plans reviewed against the change calendar). |
| **R3 — Quarterly migration-and-contract review** | Quarterly, 90 min | All peer leads + sponsor. Aligned with ex-01 Deliverable F. | Re-signed charter, re-signed SLO contract, re-signed peer contracts; migration-roadmap commit; stale ledger entries closed or reaffirmed. |
| **R4 — Per-incident cross-peer debrief** | As triggered | Peers touched by the incident. | Postmortem additions; policy / contract amendments if the incident revealed a contract gap. |
| **R5 — Consumer-track launch review** | Monthly, first week | Platform Lead + consumer-track leads + eval peer + security peer. | Launch plans for the next 30 days; freeze overlaps surfaced before they collide with the change calendar (fix for ex-01 §8.3 "hidden freeze"). |

### 6.2 Joint artefacts refreshed at R3

- **Charter** (ex-01 §2) — consumer list, boundary changes.
- **SLO contract** (ex-01 §3, SRE §4.5) — numbers, signers,
  escalation path.
- **Change calendar template** (ex-01 §4) — new fields proposed from
  the quarter's incidents.
- **Paging matrix** (ex-01 §5.3) — rows re-evaluated; any pager-fan-out
  creep caught.
- **Performance peer contract** (ex-04) — regression ledger review.
- **ML-platform peer contract** (§2) — artefact registry review; any
  stale entries marked for sunset.
- **Eval peers contract** (§3) — slice ledger review; dataset
  refreshes scheduled.
- **SRE peer contract** (§4) — SLO ledger review; capacity trend
  review; joint-change posture.
- **Security peer contract** (§5) — policy ledger review; framework
  mapping audit; any new CVE postings on engines or dependencies.

### 6.3 Signatures required at R3

Re-signed each quarter. Minimum signers for a re-signed quarter:

- Platform Lead.
- Sponsor (VP Infrastructure or equivalent).
- Each peer lead named in a contract (perf, ML-platform ×1, eval ×2
  — capability + application, SRE, security).
- One consumer-track lead for each T1 consumer.

A missing signature blocks the quarter's commit; the cadence re-opens
R3 within 10 business days.

### 6.4 Handling cross-boundary incidents

The paging matrix (ex-01 §5.3) sends every alert to one primary and
escalates to one peer. Some incidents cross more than two boundaries
in the first hour. The cross-peer cadence authors an **incident-bridge
template**:

```markdown
# Incident bridge — <INC-id> — <YYYY-MM-DD HH:MM UTC>

## Facts (updated by IC every 10 min)
- Symptom: <>
- Blast radius: <inference workload classes / substrate services / tenants>
- Current IC: <role + named engineer>
- Severity: <sev-1 | sev-2 | sev-3>

## Peers on bridge
- Platform on-call: <named + status>
- SRE on-call: <named + status or "not called">
- Performance peer: <named + status or "not called">
- ML-platform peer: <named + status or "not called">
- Eval peer: <named + status or "not called">
- Security peer: <named + status or "not called">

## Running actions
- <HH:MM UTC — named engineer — action — status>

## Running decisions
- <HH:MM UTC — decision — owner — reverts if ...>

## Handoff log
- <HH:MM UTC — IC passes from X to Y — reason>
```

The bridge is opened by the primary; once two peers are on bridge, IC
responsibility is explicit (named in the "Current IC" line); every
decision is captured; the bridge becomes the postmortem seed.

### 6.5 Contract-amendment protocol

A contract (any of ex-04, §2, §3, §4, §5) can be amended between
quarterly re-signs. The protocol:

1. The proposing side files a `ticket_id` of type
   `contract-amendment` on the relevant peer boundary.
2. The amendment states what artefact changes, what text is added or
   removed, and what is unchanged.
3. The affected peer lead signs or declines within the P2 SLA on that
   boundary.
4. If signed, the amendment is appended to the contract with a date
   and a pointer to the ticket; the full re-sign happens at the next
   R3.
5. If declined, the proposing side either escalates to R3 for a joint
   decision or withdraws.

### 6.6 How the cadence closes the exercise-01 anti-patterns

| ex-01 §8 anti-pattern | How this cadence closes it |
|---|---|
| **Hero on-call** | R3 enforces on-call-hours threshold; cross-peer visibility prevents a single peer absorbing all of a quarter's load. |
| **Dashboard-as-ritual** | R2's ledger reconciliation plus R3's "tickets that survived more than one sprint" count makes trivial-ticket drift visible across peers, not just inside platform. |
| **Hidden freeze** | R5's monthly launch review runs the consumer-side freeze check — the specific fix named in ex-01 §8.3. |
| **Migration-by-stealth** | R3's migration-roadmap review requires every in-flight migration to show a signed peer-contract trail; any gap stops the migration. |
| **Pager fan-out creep (ex-04 §8.5)** | R3's paging-matrix row re-evaluation catches creep before it goes a quarter unchallenged. |

### 6.7 Signatures (cadence itself)

> **`[SIMULATED]` Cross-Peer Cadence — v1.0 — effective 2026-Q4**
>
> **Signers.** `[SIMULATED: Platform Lead]`, `[SIMULATED: Sponsor]`,
> `[SIMULATED: Performance Peer Lead]`, `[SIMULATED: ML-Platform Peer
> Lead]`, `[SIMULATED: Capability Eval Lead]`, `[SIMULATED: Application
> Eval Lead]`, `[SIMULATED: SRE Peer Lead]`, `[SIMULATED: Security Peer
> Lead]`, 2026-10-30.
>
> Re-signed at R3 each quarter.

---

## Implementation

The four peer contracts and the cadence above are text, but they only
hold if each deliverable lands on a concrete control. A working
implementation across the four peer boundaries looks like:

- **Boundary statements in Git, one per peer.** §2.1 / §3.1 / §4.1 /
  §5.1 land at `docs/peers/ml-platform.md`,
  `docs/peers/evals.md`, `docs/peers/sre.md`, and
  `docs/peers/security.md`. PRs that would move an artefact across a
  boundary require sign-off from the named owner pair; files are
  re-read at R3 each quarter.
- **Four ticket templates as issue-tracker forms.** §2.2 / §3.2 /
  §4.2 / §5.2 are saved issue-form schemas, each with its own
  required-field set. The ML-platform intake checklist, the eval
  gate schema, the SRE handoff fields, and the security policy
  reference are form fields the tracker rejects submission without.
- **Four response SLA tables as routing policies.** §2.3 / §3.3 /
  §4.3 / §5.3 are encoded in the on-call paging product as
  per-peer routing rules keyed on severity. Cross-boundary incidents
  (§6.4) route to a joint channel via a multi-recipient rule.
- **Four return-shape taxonomies as PR labels.** §2.4 / §3.4 / §4.4 /
  §5.4 become closed label sets on the respective peer repos; a
  ticket cannot be closed without exactly one return label, and the
  per-peer ledgers in §2.5 / §3.5 / §4.5 / §5.5 are populated from
  label metadata.
- **Four Git-tracked ledgers.** `ledger/ml-platform/`,
  `ledger/evals/`, `ledger/sre/`, and `ledger/security/` each hold
  one YAML file per artefact (fine-tune, eval, capacity plan,
  security policy). The joint artefacts refreshed at R3 (§6.2) are
  rendered views across these ledgers.
- **Worked examples and seeded tickets as living fixtures.** The
  four worked examples (§2.6, §3.6, §4.6, §5.6) and the twelve
  dry-run tickets (§2.7, §3.7, §4.7, §5.7) are copied into
  peer-onboarding docs; a new on-call engineer re-reads them before
  the first shift at each boundary.
- **Cadence as scheduled meetings with persistent agenda URLs.**
  R1–R4 in §6.1 are recurring calendar invites; each has a saved
  agenda doc that reads from the ledgers above. R3 signatures are
  recorded in `docs/peers/r3-signoff-<YYYYQN>.md`; the
  contract-amendment protocol in §6.5 is a labelled ticket type
  that routes to all affected peer leads.
- **Anti-pattern closures as CI rejects.** §6.6's mapping from
  exercise-01 anti-patterns to the cadence's structural fixes is
  enforced where possible: hero on-call is prevented by the
  rotation policy on each peer; dashboard-as-ritual is prevented by
  the joint-review requirement in §6.1; hidden freeze and
  migration-by-stealth are prevented by the change-calendar and
  cross-peer sign-off checks; pager fan-out creep is caught at R3's
  paging-matrix re-evaluation.

The contracts and cadence are the agreement; the list above is what
makes the four boundaries enforceable, auditable, and visible on
cadence rather than re-negotiated per incident.

---

## 7. Validation, rubric, common mistakes, and references

### 7.1 Validation steps for a learner submission

Reviewers should run this in order. Missing any step fails the
submission.

1. **Four peer boundaries** named at the specificity of exercise 04
   (artefacts, not abstractions), each ending with a one-sentence
   boundary rule and each signed (dry).
2. **Ticket templates** present per peer; fields tuned to the peer
   (handoff fields for ML-platform; eval-gate fields for eval; joint-
   change / capacity / IC-handoff fields for SRE; framework-mapping
   fields for security).
3. **SLA tables** present per peer; P0–P3 with ack / opinion /
   resolution; each maps to the org severity scale, not a
   peer-specific one.
4. **Return-shape taxonomies** present per peer; forward and reverse.
5. **Peer-specific ledgers** have a schema plus at least two seeded
   entries per peer (or three for security given the framework-mapping
   discipline).
6. **At least one worked example per peer** carries one artefact or
   incident through every section of the contract.
7. **Three dry-run tickets per peer** (two forward, one reverse).
8. **Cross-peer cadence (Deliverable E)** names at least four rituals,
   maps rituals to the exercise-01 anti-patterns, and defines a
   contract-amendment protocol.
9. **Framework mapping in the security contract** uses
   [owasp-ml-top-10], [mitre-atlas], and [nist-ai-rmf] by name; where
   the specific technique ID is not verified, the entry is marked
   `<!-- pending-verification: ... -->` rather than invented.
10. **Simulation disclosure** present; sign-offs marked `[SIMULATED]`.

### 7.2 Rubric / review checklist

Score each item 0 (missing), 1 (partial), 2 (fully correct). Passing
≥ 16 / 20.

| # | Item | Points |
|---|---|---|
| 1 | Four peer boundaries at artefact-specificity; each ends with a one-sentence rule. | 2 |
| 2 | Each peer has a ticket template with peer-tuned fields. | 2 |
| 3 | Each peer has an SLA table mapped to the org severity scale, not a peer-invented scale. | 2 |
| 4 | Each peer has a return-shape taxonomy, forward and reverse. | 2 |
| 5 | Each peer has a ledger with schema and ≥ 2 seeded entries. | 2 |
| 6 | At least one worked example per peer carries one incident / artefact through every section. | 2 |
| 7 | Three dry-run tickets per peer (two forward, one reverse). | 2 |
| 8 | Cross-peer cadence (Deliverable E) names ≥ 4 rituals, maps to ex-01 anti-patterns, defines a contract-amendment protocol. | 2 |
| 9 | Security contract includes a framework mapping to [owasp-ml-top-10], [mitre-atlas], and [nist-ai-rmf]; unverified IDs are tagged for research rather than invented. | 2 |
| 10 | Every deliverable plugs back into ex-01 (change calendar, paging matrix, migration roadmap) at a named point; simulation disclosure present. | 2 |

### 7.3 Common mistakes

- **One contract reused across all peers.** The ticket fields a
  security review needs (framework mapping, proposed control) are
  not the fields an ML-platform handoff needs (tokenizer SHA,
  training lineage). Writing one template for all four peers loses
  the point of the exercise.
- **Severity scale per peer.** Inventing "P-sec-0 / P-perf-0" per
  peer loses the pager mapping. Reuse the org's severity scale and
  map the peer P into it.
- **Security peer treated as a sign-off step, not an interface.**
  The security peer has its own ledger, its own return shapes, and
  a reverse-direction flow (incident tasking) that this role must
  execute within a stated time. A contract that only names
  "security sign-off" misses the reverse direction.
- **Eval contract without a reverse direction.** Eval → platform
  flow (dataset drift, methodology escalation, red-team signal)
  institutionalises the eval peer as a dependency of this role, not
  a service desk.
- **SRE contract that re-authors availability.** The availability
  SLO is co-signed; the inference-specific tighter number sits on
  top of the substrate floor. Writing a brand-new availability
  number invites a two-way disagreement at every incident.
- **ML-platform contract without a handoff manifest.** A boundary
  with no structured handoff turns into a slack-message exchange at
  promotion time; the next tokenizer-mismatch incident becomes a
  surprise.
- **Framework mapping invented.** Writing specific MITRE ATLAS
  technique IDs without checking them against
  `https://atlas.mitre.org/` is worse than writing none. Mark
  unverified IDs for research.
- **Cross-peer cadence that duplicates the weekly SLO review.** R2
  should add something — specifically the ledger reconciliation
  across peers and the consumer-side freeze check — not re-run the
  same five agenda items.
- **Contract-amendment protocol missing.** A contract with no
  amendment path drifts silently; the quarterly re-sign surfaces
  the drift too late.
- **Simulation hidden.** Writing simulated signatures as if real
  loses the trust signal the exercise is teaching.

### 7.4 References

- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/)
  — the framework the security contract's policy-ledger mapping
  anchors on. ML01–ML10 appear in the ticket schema §5.2 and in the
  seeded policies §5.5.
- [mitre-atlas](https://atlas.mitre.org/) — the detection-side
  framework the security contract cites; specific technique IDs in
  the seeded policies are marked `<!-- pending-verification -->` where the
  current ATLAS page should be confirmed before use in a real
  policy document.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework)
  — the GOVERN / MAP / MEASURE / MANAGE functions the ticket schema
  tags. The cross-peer cadence (§6) is a GOVERN artefact; the
  ledgers are MEASURE artefacts; incident response is MANAGE.
- vLLM documentation — scheduler flags, tokenizer and chat-template
  handling. Referenced by §2 (ML-platform handoff) and §4 (SRE
  substrate changes). https://docs.vllm.ai/
- Google SRE Workbook, "Implementing SLOs" and "Alerting on SLOs" —
  the SLO-contract pattern the SRE contract co-signs.
  https://sre.google/workbook/
- Module 102 (engine internals) — the surface §2's quantisation and
  prompt-format artefacts sit against.
- Module 104 exercise 06 (cross-tenant KV safety) — the policy
  POL-2026-002 in §5.5 is the surface this contract governs.
- Module 108 exercises 03 and 04 (acceptance-rate / draft-lifecycle)
  — the ML-platform contract §2.5 draft registry entry references
  this directly; ex-05 §2 is the generalisation of ex-04's
  draft-lifecycle shape across all model handoffs.
- Module 110 (SLO engineering) — the SLO contract the SRE contract
  co-signs and the burn-rate thresholds the SLA tables import.
- Module 113 chapter 03 (runbook top-ten) — the runbook ledger the
  SRE and performance contracts' worked examples rely on.
- Module 113 chapter 05 (production evals) — the eval contract §3
  concretises that chapter's gate / slice / regression pattern.
- Exercise 01 — the change calendar, paging matrix, and migration
  roadmap every contract in this exercise plugs back into.
- Exercise 04 — the performance peer contract this exercise extends
  to four more peers; §1.3 inherits exercise 04's seven-part shape.
- Chapters 02, 03, 04, 05 of this module — the four-peer taxonomy
  these four contracts concretise.
