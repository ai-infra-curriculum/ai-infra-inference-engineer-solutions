# Solution — Exercise 04: Performance Peer Interface Contract

> Paired learning file: [`lessons/mod-114-cross-functional-inference-leadership/exercises/exercise-04-performance-peer-interface-contract.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-114-cross-functional-inference-leadership/exercises/exercise-04-performance-peer-interface-contract.md)
> Module: `mod-114-cross-functional-inference-leadership`
> Type: interface contract / governance artefact

## 1. Solution overview

The performance peer owns the kernel surface; the inference-platform
engineer owns the production system. The 20 % shared surface — which flag
to flip, which backend to pick, how a quantisation change gets deployed,
how a kernel regression gets filed — is the slice that generates all the
expensive incidents. This exercise produces the written contract that
keeps that slice governed.

The contract has seven parts (A–G). Every signature is of the form
`[SIMULATED: role]`; every ledger entry marked `[SIMULATED]` is a
plausible-hypothetical derived from module 102 / 107 / 113's runbook
catalogues.

### 1.1 Reference fleet

Same as exercise 01: 40 H100 nodes, Llama-3.1-70B-FP16 primary on tier A,
long-tail routed on tier B, batch embedding on tier C, dual-region US / EU.
vLLM primary, SGLang fallback.

### 1.2 Document map

| § | Deliverable |
|---|---|
| §2 | A. Boundary statement |
| §3 | B. Ticket template |
| §4 | C. Response SLA table |
| §5 | D. Return-shape taxonomy |
| §6 | E. Kernel-regression ledger |
| §7 | F. Worked example |
| §8 | G. Three live or dry-run tickets |

---

## 2. Deliverable A — the boundary statement

### 2.1 Boundary (one page)

> **`[SIMULATED]` Inference Platform ↔ Performance Peer Interface
> Contract — v1.0 — effective 2026-Q4**
>
> **The inference-platform role owns the production serving system.**
> Specifically:
>
> - vLLM / SGLang engine *configuration* (CLI flags, `EngineArgs`, the
>   deployment manifest).
> - KV allocator *parameters* as a runtime object (block size, pool
>   size, prefix caching flag, tiering configuration — not the allocator
>   implementation).
> - Fleet shape (node count, topology choice, autoscaler policy, warm
>   pool).
> - SLO contract (TTFT, ITL, E2E, availability, cost per 1M tokens).
> - Deploy rituals (change calendar entries, canary stages, rollback).
> - Model artefact *ingress and lifecycle* once the ML-platform peer
>   has signed the handoff.
> - On-call rotation and the paging matrix (exercise 01 Deliverable D).
> - Production runbooks (module 113 chapter 03).
>
> **The performance peer owns the kernel surface.** Specifically:
>
> - Kernel source files (attention kernels, GEMM kernels, custom ops,
>   paged-attention block scatter/gather, speculative-decoding verify
>   kernels, quantisation kernels, moe routing kernels).
> - Attention-backend *selection logic inside the engine* (which
>   kernel is chosen for a given shape — the codepath, not the flag
>   that enables it).
> - Quantisation *algorithms* (which calibration method; which weight
>   layout; fused weight+scale paths).
> - Nsight / SASS / `ncu` profiling and the interpretation thereof.
> - torch.compile pass pipelines and custom lowering passes.
> - Kernel-level regression benchmarks (microbenchmarks that pin
>   per-kernel performance to a baseline).
> - Upstream engine contributions and bug reports (vLLM / SGLang /
>   TensorRT-LLM issue tracking).
>
> **The 20 % shared surface.**
>
> - **Which flag to flip on a regression.** Example: TTFT regression
>   suggests chunked-prefill might help; the performance peer decides
>   whether that is the right lever; this role decides when and how to
>   deploy it.
> - **Quantisation deployment.** The performance peer builds the
>   quantised weights and the kernel path; this role owns the
>   deployment (artefact ingress, canary, accuracy watch).
> - **Speculative decoding.** The performance peer owns the draft /
>   verify kernel correctness and performance; this role owns the
>   draft-lifecycle contract (module 108 ex-04), the canary, and the
>   acceptance-rate monitoring (module 108 ex-03).
> - **Distributed-inference topology.** The performance peer owns the
>   collective kernels and their efficiency on specific hardware; this
>   role owns the topology choice for a workload class (TP / PP / DP /
>   EP / hybrid, per module 107 ex-01).
>
> **The boundary rule (one sentence).**
>
> > The performance peer owns kernels and the choice logic inside the
> > engine; this role owns the running system and the flag-flip that
> > selects one kernel over another in production.
>
> **Signed.** `[SIMULATED: Platform Lead]`, `[SIMULATED: Performance
> Peer Lead]`, 2026-10-30. Re-signed at every quarterly cadence
> review (exercise 01 Deliverable F).

### 2.2 Why name artefacts, not abstractions

Chapter 04's common failure mode "they do kernels, we do serving"
is the archetypal unusable boundary. Naming artefacts — "kernel source
files" and "deployment manifest" and "KV block size as a runtime
parameter" — makes every edge case resolvable without re-opening the
contract: a change to a flag goes to this role; a change to the kernel
goes to the peer.

---

## 3. Deliverable B — the ticket template

### 3.1 Template (markdown form; a JSON / issue-tracker form follows in §3.3)

```markdown
# Performance Peer Ticket — <short imperative title>

- **title:** <short imperative sentence; one line>
- **workload_class:** <tier-A | tier-B | tier-C | internal-eval>
- **axis:** <TTFT | ITL | TPOT | E2E | availability | cost-per-1M | acceptance-rate | throughput-tokens-s>
- **slo_impact:**
    - <baseline value, measured SLO, current value, burn rate, SLO at risk by when>
- **engine_and_version:**
    - name: <vLLM | SGLang | TensorRT-LLM | ...>
    - version: <release tag or git SHA>
- **model_and_version:**
    - name: <HF handle or internal name>
    - revision: <hub commit or internal tag>
    - dtype: <FP16 | FP8 | INT8 | AWQ-INT4 | GPTQ-INT4 | ...>
- **hardware_class:**
    - sku: <H100-SXM5 | A100-80G | L40S | ...>
    - driver: <rev>
    - cuda: <rev>
- **topology:**
    - TP / PP / DP / EP / hybrid; nodes × GPUs
    - routing: <engine replicas; disaggregated flag>
- **reproduction:**
    - prompt_set: <pointer to fixture; synthetic or anonymised real>
    - knob_set: <deployment manifest frag>
    - command: <the exact reproducer; a script in `repro/<ticket-id>/`>
- **expected_return:**
    - <one of the five shapes from Deliverable D; what the filer expects back>
- **priority:** <P0 | P1 | P2 | P3>
- **sla:** <ack / opinion / resolution target per §4>
- **metric_overlays:** <Grafana URL(s); dashboard snapshot links>
- **trace_sample:** <Chrome trace / Nsight trace / torch profile pointer>
- **prior_runbook:** <module 113 ch. 03 runbook title that got us here>
- **related_regressions:** <pointers to ledger IDs (§6)>
- **rule_outs_performed:**
    - [ ] fleet hardware health check (SMI, dmesg, driver state)
    - [ ] config drift check vs. last known-good manifest
    - [ ] workload drift check (QPS, prompt-length distribution shift)
    - [ ] cold-cache / warm-pool exhaustion check
    - [ ] noisy-neighbour / per-tenant check
    - [ ] model-revision pin check (artefact SHA vs. registry)
    - [ ] driver / firmware recent change check (last 7 d)
    - [ ] known regressions in ledger consulted (§6)
    - [ ] runbook top-ten consulted (module 113 ch. 03)
- **notes:** <free form>
```

### 3.2 One-sentence explanation per field

| Field | What it is for |
|---|---|
| title | Grep-friendly one-liner so the peer can scan a queue. |
| workload_class | So the peer knows which SLO is at risk and whether this is tier-A urgency. |
| axis | So the peer knows which metric to reproduce — different kernels, different fixtures. |
| slo_impact | Quantifies the ticket; a ticket without an SLO impact is a research question, not a P0. |
| engine_and_version | Reproducibility; the peer builds against the same engine SHA. |
| model_and_version | Pinning; artefact SHA matters for all performance work. |
| hardware_class | SKU-specific kernels differ; driver / CUDA revisions move independently. |
| topology | Same engine, different topology — different kernels fire. |
| reproduction | The one artefact the peer always asks for; without it, the ticket stalls. |
| expected_return | Picks one of the five return shapes so both sides know when "done" is reached. |
| priority | Maps to SLA table. |
| sla | The time contract; echo the row from §4 so the ticket is self-contained. |
| metric_overlays | Dashboards that show the symptom; confirm the ticket is about what the filer thinks. |
| trace_sample | A captured trace shortcuts the peer's reproduction. |
| prior_runbook | Which runbook led here — informs whether the issue is runbook-fixable or kernel-territory. |
| related_regressions | Avoids duplicating open tickets. |
| rule_outs_performed | The gatekeeper. Without it, a non-kernel symptom gets filed to the peer and the next "not a kernel issue" return costs a day. |
| notes | Free form; everything else. |

### 3.3 JSON / issue-tracker shape

```json
{
  "title": "...",
  "workload_class": "tier-A",
  "axis": "TTFT",
  "slo_impact": {
    "baseline_p95_ms": 1200,
    "current_p95_ms": 1650,
    "slo_p95_ms": 1500,
    "burn_rate_pct_per_hour": 1.8,
    "projected_breach_at": "2026-11-23T14:00:00Z"
  },
  "engine_and_version": { "name": "vLLM", "version": "0.7.1+sha=abc123" },
  "model_and_version": { "name": "...", "revision": "...", "dtype": "FP16" },
  "hardware_class": { "sku": "H100-SXM5", "driver": "555.xx", "cuda": "12.4" },
  "topology": { "tp": 8, "pp": 1, "dp": 2, "disaggregated": false },
  "reproduction": {
    "prompt_set": "repro/TKT-2026-421/prompts.jsonl",
    "knob_set": "repro/TKT-2026-421/deployment.yaml",
    "command": "repro/TKT-2026-421/run.sh"
  },
  "expected_return": "engine-config-suggestion | kernel-patch | diagnostic-opinion",
  "priority": "P1",
  "sla": { "ack": "1h", "opinion": "1d", "resolution": "7d" },
  "metric_overlays": ["grafana.internal/d/tier-a-slo"],
  "trace_sample": "traces/TKT-2026-421/vllm-2026-11-22.json",
  "prior_runbook": "runbooks/tier-a-ttft-burn.md",
  "related_regressions": ["KREG-2026-008"],
  "rule_outs_performed": {
    "fleet_health": true,
    "config_drift": true,
    "workload_drift": true,
    "cold_cache": true,
    "noisy_neighbour": true,
    "artefact_sha_pin": true,
    "driver_firmware_recent": true,
    "ledger_consulted": true,
    "runbook_top_ten": true
  },
  "notes": "..."
}
```

---

## 4. Deliverable C — the response SLA table

### 4.1 Table

| Priority | Description | Ack time | Opinion returned | Resolution target | Burn-rate mapping (module 110) |
|---|---|---|---|---|---|
| P0 | tier-A SLO actively breaching; or any workload class availability < 99 % within 15 min; or a running engine's kernel crashing more than 1× / h. | 15 min (24/7) | 2 h | best-effort within 24 h; acknowledged workaround within 2 h | 14.4× monthly budget in 1 h |
| P1 | tier-A SLO degraded but not breaching; burn rate projected to exhaust monthly budget in < 10 days. | 1 h (business) / 2 h (off-hours) | 1 business day | 7 days | 3× monthly budget in 6 h |
| P2 | tier-B / tier-C SLO at risk; tier-A long-tail regression; kernel-level observation that is not currently causing burn. | 1 business day | 3 business days | 30 days | 1× monthly budget in 3 d |
| P3 | Backlog; investigations; kernel-level improvements not tied to a current regression. | 3 business days | 2 weeks | next quarter | n/a |

### 4.2 Pager mapping

Priorities map to the exercise-01 Deliverable D paging matrix as
follows. The peer is **not the pager** for alerts; alerts go to the
on-call (ex-01 §5.3). Tickets are filed to the peer by the on-call once
the rule-outs are exercised.

- **P0.** On-call pages the Performance Peer Lead through the same pager
  channel used for SRE-peer escalation, at the escalation trigger times
  in the ex-01 matrix. The lead's 15-minute ack is the ack entered on
  the ticket.
- **P1.** On-call opens the ticket during the same incident; no peer
  page; peer lead gets a Slack ping and acknowledges within 1 h
  business / 2 h off-hours.
- **P2 / P3.** Routine ticket flow; no paging.

### 4.3 Relationship to the organisation's severity map

The organisation almost certainly has an existing severity map
(e.g., sev-1 / sev-2 / sev-3 that maps to a pager tier). This SLA table
*extends* that, it does not replace it. The mapping to organisation
severity:

| Perf-peer P | Org sev |
|---|---|
| P0 | sev-1 |
| P1 | sev-2 |
| P2 | sev-3 |
| P3 | ticket-only (no sev) |

Any team that tries to assign its own severity scale per peer creates
confusion at 03:00 UTC. Reuse the org's map; this table only governs
response timings for the specific peer.

---

## 5. Deliverable D — the return-shape taxonomy

### 5.1 Five return shapes

**1. Diagnostic opinion.** The peer reproduces the symptom and returns a
written assessment: what the kernel is doing, whether the engine is
behaving correctly, what the knob space looks like, whether the issue is
likely to recur. No code change.

- **Artefacts delivered.** A short write-up (1–2 pages), trace
  attachments, panel screenshots at the reproduction site.
- **This role does.** Reads the opinion; updates the runbook (module 113
  ch. 03) with the new symptom-to-kernel mapping if novel; tunes the
  burn-rate alert thresholds; files a follow-up ticket if the opinion
  points at a flag / config / topology change.
- **Ledger entry.** Opens a new entry under `status: diagnosis` if the
  opinion names a specific kernel or codepath; otherwise closes the
  ticket without a ledger entry.

**2. Kernel patch.** The peer authors a change to the kernel source (or
to the engine's kernel-selection logic) that fixes the regression.

- **Artefacts delivered.** A PR against the engine's source (upstream if
  possible; internal fork if the upstream review cycle is slower than
  the SLA). A regression fixture that would have caught the issue. A
  before / after microbenchmark.
- **This role does.** Picks up the resulting engine build; deploys via
  canary (exercise 01 Deliverable C change-calendar entry); monitors the
  axis that regressed.
- **Ledger entry.** `status: patched`; records the PR URL, the fixture,
  the fix commit, the kernel-regression fixture that now covers it.

**3. Engine config suggestion.** The peer points to a flag / config /
topology change that mitigates without a kernel change.

- **Artefacts delivered.** A one-paragraph recommendation and a cited
  justification (benchmark data, trace reading). Named flags with
  recommended values.
- **This role does.** Pre-sign the change-calendar entry; stage the
  rollout; monitor.
- **Ledger entry.** `status: workaround-only` with the flag name and the
  deploy identifier; links to the deploy ticket.

**4. Upstream engine ticket.** The peer concludes the issue is in the
engine's upstream, not in our configuration, and files / cites an
upstream issue.

- **Artefacts delivered.** A pointer to the upstream issue (or a filed
  new one) with our reproducer attached.
- **This role does.** Picks a workaround for the duration of the
  upstream fix; files a change-calendar entry for the version bump when
  the upstream fix lands.
- **Ledger entry.** `status: upstream-pending`; records the upstream
  issue URL, the workaround, and the expected upstream release.

**5. "Not a kernel issue."** The peer reproduces and concludes the
issue is in configuration, workload, infra, or a runbook already
covers it.

- **Artefacts delivered.** A one-paragraph explanation of why the issue
  is not kernel territory.
- **This role does.** Updates `rule_outs_performed` in the ticket
  template (§3.1) if a new rule-out would catch this class of
  misfiling; updates the runbook that *should* have caught it.
- **Ledger entry.** No new entry; the ticket closes with a pointer to
  the runbook. However, if the misfile recurs in the same shape, the
  entry gets opened against the misfile pattern itself ("recurrent
  misfiled as kernel") so the ticket template can be hardened.

### 5.2 Reverse direction

Tickets from the performance peer to this role.

- **Trace capture.** "Please capture a vLLM profiler trace for 5 min at
  15:00 UTC on tier-A us-east-1-a and attach." This role runs the
  trace, attaches, closes.
- **Canary slot.** "Please schedule a 1 % canary slot for engine
  version `0.7.2-rc1` with the attached kernel patch." This role files
  a change-calendar entry; stages the canary; reports back on the
  measured axes.
- **SLO data slice.** "Please provide last-30-day TTFT p95 per topology
  variant across us-east-1-a." This role queries the Prometheus /
  Grafana surface and returns.
- **Fleet-shape change proposal.** "I've found evidence that a TP-4 ×
  PP-2 split would improve TTFT by ~10 %; please run an RFC." This
  role triggers an RFC under the exercise-02 shape.

Reverse-direction response SLAs mirror the forward direction (§4.1),
with the role flipped. The symmetry is the point.

---

## 6. Deliverable E — the kernel-regression ledger

### 6.1 Ledger schema

```yaml
# ledger/<KREG-YYYY-nnn>.yaml
id: KREG-YYYY-NNN
title: <short imperative>
first_seen: <date>
status: <open | diagnosis | workaround-only | patched | upstream-pending | closed>
workaround: <one-paragraph description of the active workaround>
upstream_ticket: <URL or n/a>
fix_commit: <commit SHA or PR URL or n/a>
regression_fixture: <path to the fixture that catches this regression>
owner: <role or named engineer>
last_reviewed: <date>
notes: <free-form>
```

### 6.2 Review cadence

- **Weekly:** the Performance Peer Lead scans `status: open` and
  `status: diagnosis` entries for drift.
- **Quarterly:** the ledger is reviewed at the exercise 01 Deliverable F
  migration-roadmap review; stale entries (`last_reviewed` > 6 months)
  are either closed, reaffirmed, or escalated.
- **On engine upgrade:** every `status: upstream-pending` entry is
  re-checked against the new engine release's changelog.

### 6.3 Seeded ledger entries

Three entries plausible from the module 102 / 107 / 113 runbook
catalogues. Each is marked `[SIMULATED]` but uses the kind of symptom
real teams catalogue.

**Entry 1 — KREG-2026-001**

```yaml
id: KREG-2026-001
title: "FlashAttention variant selection regresses on H100 for short prompts after driver 555 upgrade"
first_seen: 2026-07-14
status: patched
workaround: "Pin attention backend to FA2 via --attention-backend=fa2 for engines on driver >= 555 until fix propagated."
upstream_ticket: "[SIMULATED: vLLM/issues/NNNN]"    # TODO link when ticket exists
fix_commit: "[SIMULATED: vllm-sha=def456]"
regression_fixture: "bench/attention-backend-short-prompt-h100.sh"
owner: "[SIMULATED: Performance Peer Lead]"
last_reviewed: 2026-10-15
notes: >
  Symptom: tier-A TTFT p95 +25 % on short (< 500 token) prompts after driver 555
  rollout. Root cause: selection heuristic chose a kernel variant that is slower
  for short-prompt shapes on H100 under new driver. Forward fix in engine release
  0.7.0; backport to 0.6.5 landed 2026-08-01. Workaround on older engines kept
  for one quarter pending full upgrade.
```

**Entry 2 — KREG-2026-002**

```yaml
id: KREG-2026-002
title: "PagedAttention block scatter underperforms on tier-C long-batch embedding after FP16 → FP8 experiments"
first_seen: 2026-09-20
status: open
workaround: "Keep tier-C embeddings on FP16 for now; do not enable FP8 for embedding workloads."
upstream_ticket: n/a
fix_commit: n/a
regression_fixture: "bench/paged-attn-scatter-fp8-embed.sh"
owner: "[SIMULATED: Performance Peer Lead]"
last_reviewed: 2026-11-10
notes: >
  Observed during the FP8 long-tail investigation (migration candidate #3 in
  ex-01 §7.2). FP8 path is faster in theory but a dimension-specific scatter
  slows at the embedding workload's shape; needs investigation. Gated the FP8
  migration on this entry closing.
```

**Entry 3 — KREG-2026-003**

```yaml
id: KREG-2026-003
title: "Speculative decoding verify kernel slower than microbenchmark baseline on 70B target"
first_seen: 2026-10-05
status: diagnosis
workaround: "Speculative decoding disabled on tier-A chat pending diagnosis; migration candidate #1 in ex-01 §7.5 depends on this entry closing or being downgraded to workaround-only."
upstream_ticket: n/a
fix_commit: n/a
regression_fixture: "bench/spec-verify-70b.sh"
owner: "[SIMULATED: Performance Peer Lead]"
last_reviewed: 2026-11-12
notes: >
  Verify kernel running ~15 % slower than microbenchmark predicts on tier-A target
  model. Could be a cache-interference effect with the primary attention path.
  Opinion expected per the P2 SLA; speculative decoding migration gated on either
  a kernel patch (status → patched) or a diagnostic opinion that the gap does
  not translate to end-to-end regression (status → workaround-only).
```

---

## 7. Deliverable F — a worked example

### 7.1 Scenario (tier-A TTFT regression)

**Setting.** Tier-A chat on vLLM 0.7.1 on H100-SXM5, Llama-3.1-70B-Instruct
FP16. An axis-set alert from module 110 fires at 15:40 UTC on 2026-11-22:
`tier-A-TTFT-p95-burn-rate = 1.8× / h`.

### 7.2 On-call symptom and triage

On-call primary pages at 15:40 UTC. The ex-01 Deliverable D paging
matrix sends the alert to one primary; no peer paged yet.

### 7.3 The runbook step that pointed at kernel territory

Runbook `runbooks/tier-a-ttft-burn.md` (module 113 ch. 03) walks the
on-call through standard rule-outs. By 16:15 UTC the on-call has
confirmed:

- Fleet SMI / dmesg clean, no hardware events.
- Deployment manifest unchanged in last 24 h.
- Workload shape unchanged (prompt-length distribution normal).
- Warm pool nominal.
- Not a noisy-neighbour or per-tenant issue (per-tenant TTFT panel is
  uniform across tenants).
- Model-revision SHA matches the registry.
- One recent driver event: `nvidia-driver` rolled from 554 to 555 at
  12:00 UTC (CHG-2026-093 from ex-01 §4.2). Timing lines up.

Runbook's "kernel territory if timing correlates with a driver / firmware
event" branch fires. The on-call prepares a ticket.

### 7.4 Rule-outs performed (before filing)

The on-call ticks each `rule_outs_performed` box in the template (§3.1).
Each is a one-line note in the ticket; attaching evidence takes ~15 min.

### 7.5 The filed ticket (fully populated)

```markdown
# Performance Peer Ticket — TKT-2026-421

- **title:** "Tier-A TTFT p95 regresses after driver 554 → 555 upgrade; short-prompt shape affected"
- **workload_class:** tier-A
- **axis:** TTFT
- **slo_impact:**
    - baseline p95: 1 200 ms
    - current p95:  1 650 ms
    - SLO p95:      1 500 ms
    - burn rate:    1.8× / h
    - projected breach of monthly budget: 2026-11-23 14:00 UTC
- **engine_and_version:**
    - name: vLLM
    - version: 0.7.1
- **model_and_version:**
    - name: meta-llama/Meta-Llama-3.1-70B-Instruct
    - revision: `[SIMULATED: hub-sha=...]`
    - dtype: FP16
- **hardware_class:**
    - sku: H100-SXM5
    - driver: 555 (recently from 554)
    - cuda: 12.4
- **topology:**
    - TP = 8, PP = 1, DP = 2, disaggregated = false
- **reproduction:**
    - prompt_set: repro/TKT-2026-421/prompts-shortprompt.jsonl
    - knob_set:   repro/TKT-2026-421/deployment.yaml (identical to prod)
    - command:    repro/TKT-2026-421/run.sh
- **expected_return:**
    - diagnostic-opinion or engine-config-suggestion (flag-side mitigation)
      so tier-A can return to SLO within the 7-day P1 resolution window.
- **priority:** P1
- **sla:** ack 1 h / opinion 1 business day / resolution 7 days (§4)
- **metric_overlays:** grafana.internal/d/tier-a-slo
- **trace_sample:** traces/TKT-2026-421/vllm-prof.json (captured 16:20 UTC)
- **prior_runbook:** runbooks/tier-a-ttft-burn.md
- **related_regressions:** KREG-2026-001 — "FlashAttention variant selection regresses on H100 for short prompts after driver upgrade"
- **rule_outs_performed:**
    - [x] fleet hardware health check
    - [x] config drift check
    - [x] workload drift check
    - [x] cold-cache / warm-pool check
    - [x] noisy-neighbour / per-tenant check
    - [x] artefact-SHA pin check
    - [x] driver / firmware recent change check  ← timing correlates
    - [x] ledger consulted; KREG-2026-001 noted
    - [x] runbook top-ten consulted
- **notes:** >
  The KREG-2026-001 pattern matches: short-prompt shape, H100, post-driver.
  Prior workaround was `--attention-backend=fa2` pin. Requesting peer
  confirmation before applying the pin broadly, plus guidance on whether
  the engine upgrade to 0.7.2 should be prioritised.
```

### 7.6 Performance-peer response

Within the P1 opinion-returned SLA (1 business day), the Performance Peer
Lead posts:

> **Return shape: engine-config suggestion.** Confirmed: driver 555
> changes the attention-backend selection heuristic's cost model on
> H100; short-prompt shapes re-select a slower variant. Pin
> `--attention-backend=fa2` on all engines where driver ≥ 555 until
> the engine upgrade to 0.7.2 propagates (vLLM has merged the fix;
> release ETA `[SIMULATED: 2026-12-02]`). Microbenchmark attached.
> KREG-2026-001 extended with this incident; no new ledger entry
> needed.

### 7.7 Follow-up deploy on this side

The on-call files CHG-2026-096 in the change calendar:

```yaml
change_id: CHG-2026-096
title: "Pin --attention-backend=fa2 on tier-A engines (driver 555 mitigation)"
window: { start: 2026-11-22 18:00 UTC, end: 2026-11-22 19:00 UTC, duration: 60, timezone: UTC }
blast_radius: { workload_classes: [tier-A], clusters: [us-east-1-a, us-east-1-b], tenant_tiers: [T1] }
change_type: scheduler-flag
rollout_primitive: canary-1pct
reversibility: single-flag
on_call: { primary: "[SIMULATED]", secondary: "[SIMULATED]" }
peer_dependency: { touches: [performance], signoff: received }
freeze_check: clear
eval_gate: n-a
runbook: "runbooks/scheduler-flag-rollout.md"
notes: "Mitigation for KREG-2026-001 recurrence; sunset at engine upgrade 0.7.2."
```

At 18:30 UTC the flag is on 1 %; p95 drops to 1 220 ms on the canary
engines. By 19:30 UTC full rollout; p95 fleet-wide 1 210 ms; burn rate
reset to baseline.

### 7.8 Ledger entry (extended)

The existing KREG-2026-001 entry gains an update block:

```yaml
updates:
  - date: 2026-11-22
    event: recurrence after driver 554 → 555
    ticket: TKT-2026-421
    mitigation_deployed: CHG-2026-096 (pin --attention-backend=fa2)
    engine_fix_eta: "[SIMULATED: 2026-12-02 vLLM 0.7.2 release]"
    action_item: "Remove pin and verify after vLLM 0.7.2 propagation; close update block."
last_reviewed: 2026-11-22
```

### 7.9 What makes this worked example useful

- Carries through to the ledger entry; the loop closes.
- Every artefact in §3–§6 appears in the right place.
- The runbook step that pointed at kernel territory is explicit;
  without it, this ticket is a guess.
- The `rule_outs_performed` list explains why this ticket is not
  "not a kernel issue."
- The follow-up deploy uses the exercise-01 change calendar; the
  contract plugs back into the cadence.

---

## 8. Deliverable G — three live or dry-run tickets

### 8.1 Disclosure

These are **dry-run tickets**. They are written as if filed and routed
under the SLA table in §4; no real peer response was received. The
section below records expected responses based on the taxonomy in §5.

### 8.2 Ticket 1 (from this role to the peer) — tier-A ITL regression

```markdown
# Performance Peer Ticket — TKT-2026-422 (dry-run)

- title: "Tier-A ITL p95 regresses 15 % after chunked-prefill shadow enable"
- workload_class: tier-A
- axis: ITL
- slo_impact:
    baseline: 48 ms; current: 56 ms; SLO: 60 ms; burn: 0.4× / h (not currently breaching)
- engine_and_version: vLLM 0.7.1
- model_and_version: Llama-3.1-70B-Instruct FP16
- hardware_class: H100-SXM5, driver 555, CUDA 12.4
- topology: TP=8, PP=1, DP=2
- reproduction: repro/TKT-2026-422/
- expected_return: diagnostic-opinion
- priority: P2
- sla: ack 1 bd / opinion 3 bd / resolution 30 d
- metric_overlays: grafana.internal/d/tier-a-slo
- trace_sample: traces/TKT-2026-422/vllm-prof.json
- prior_runbook: runbooks/scheduler-flag-rollout.md
- related_regressions: n/a
- rule_outs_performed: all boxes ticked; see ticket
- notes: >
  Observed in the chunked-prefill shadow (CHG-2026-092). Within SLO
  margin. Filing P2 because we want the diagnosis before promoting the
  flag past shadow.
```

**Expected response.** Diagnostic opinion. Likely root cause is an
attention-key-scatter path that interacts with chunked prefill's
intermediate buffering. Peer either (a) confirms the regression is
real and safe to live with for a tier-A burn-rate headroom benefit
(net TTFT improvement > ITL cost), or (b) proposes a flag combination
(`--chunked-prefill-size`) that mitigates.

### 8.3 Ticket 2 (from this role to the peer) — speculative decoding verify kernel

```markdown
# Performance Peer Ticket — TKT-2026-423 (dry-run)

- title: "Please deep-dive the verify-kernel 15 % gap on 70B target before we canary speculative decoding"
- workload_class: tier-A
- axis: throughput-tokens-s
- slo_impact:
    migration-blocking: committed migration #1 in ex-01 §7.5 cannot start canary until this closes or downgrades.
- engine_and_version: vLLM 0.7.1
- model_and_version: Llama-3.1-70B-Instruct FP16 target + Llama-3.2-1B-Instruct draft
- hardware_class: H100-SXM5
- topology: TP=8, PP=1
- reproduction: repro/TKT-2026-423/
- expected_return: diagnostic-opinion → either kernel-patch or workaround-only
- priority: P2
- sla: 3 bd opinion / 30 d resolution
- prior_runbook: n/a (pre-deploy investigation)
- related_regressions: KREG-2026-003
- rule_outs_performed:
    this is a pre-deploy investigation; rule-outs are "is the microbench
    fixture realistic" rather than the production rule-outs
- notes: >
  KREG-2026-003 is in diagnosis; this ticket is the tracking for that
  diagnosis to resolve so migration #1 can start canary.
```

**Expected response.** Diagnostic opinion. Most-likely outcome is
"workaround-only": the microbench overstates the gap because of a
cache-interference effect that does not reproduce end-to-end. If the
end-to-end benefit holds, downgrade KREG-2026-003 to `status:
workaround-only` and unblock migration #1.

### 8.4 Ticket 3 (reverse — peer to this role) — trace capture request

```markdown
# Platform Ticket (reverse) — TKT-2026-424 (dry-run)

- title: "Please capture a vLLM profiler trace for 5 min at 15:00 UTC on tier-A us-east-1-a"
- filed by: [SIMULATED: Performance Peer Lead]
- requested artefact:
    vLLM profiler trace (torch.profiler + nvtx annotations) for 5 min
    starting 2026-11-26 15:00 UTC on us-east-1-a-node-03 engine, during
    the EU peak overlap window.
- purpose:
    needed for the investigation of KREG-2026-002 (FP8 scatter underperformance).
- priority: P2
- sla: ack 1 bd / deliverable 3 bd
- target drop path: traces/KREG-2026-002/us-east-1-a-node-03-2026-11-26.json
- notes: >
  Low-traffic time preferred; if the peak window is operationally
  sensitive, propose an alternate.
```

**Expected response.** This role acknowledges within the business-day
ack; schedules the trace capture for the requested window; files a
change-calendar entry for the brief profiler enable; drops the trace
file at the agreed path; closes the ticket. Mirror of §5.1's "reverse
direction" flow.

---

## Implementation

The contract above is text, but the boundary only holds if each of the
deliverables lands on a concrete control. A working implementation on an
inference platform looks like:

- **Boundary statement in Git.** The one-page statement from §2.1 lives
  at `docs/peers/performance.md`; any PR that would move an artefact
  across the boundary requires sign-off from both named owners. The
  file is re-read at every quarterly contract re-signature ritual
  (§7 of exercise 01).
- **Ticket template as an issue-tracker form.** The JSON shape from
  §3.3 is the saved issue-form schema on the peer repo; the markdown
  template from §3.1 is its rendered preview. The
  `rule_outs_performed` block is a required checklist field — the form
  rejects submission if any item is missing.
- **Response SLA as a routing policy.** §4's table is encoded in the
  on-call paging product as a routing rule keyed on severity: `sev-1`
  pages the performance primary within 15 minutes, `sev-2` pages
  within the business-day window, `sev-3` emails the queue, `sev-4` is
  a scheduled review. The clock-escalation path mirrors §5.3 of
  exercise 01.
- **Return-shape taxonomy as PR labels.** The five return shapes from
  §5.1 are a closed label set on the peer repo (`return:fix`,
  `return:config`, `return:runbook`, `return:no-repro`,
  `return:won't-fix`). A ticket cannot be closed without exactly one
  return label; the ledger is populated from label metadata.
- **Kernel-regression ledger as a Git-tracked YAML directory.** §6.1's
  schema is `ledger/kernel-regressions/<id>.yaml`; the monthly review
  cadence from §6.2 is a scheduled meeting that reads the directory
  and triages stale entries. Seeded entries from §6.3 ship in the
  initial directory.
- **Worked example and seeded tickets as living fixtures.** §7's
  worked example and §8's three dry-run tickets are copied into the
  peer-onboarding doc; a new on-call engineer re-reads them before
  their first shift. The tickets themselves live in the issue tracker
  with a `fixture:` prefix so they are not confused with live work.
- **Boundary violation as a CI reject.** A pre-merge check on the
  engine repo blocks PRs that touch fused-kernel sources without a
  linked performance-peer ticket; the inverse check on the peer repo
  blocks PRs that touch routing or scheduler code without a linked
  inference-platform ticket. This is the enforcement for the "name
  artefacts, not abstractions" rule in §2.2.

The contract is the agreement; the list above is what makes the
boundary enforceable, auditable, and visible on cadence rather than
re-negotiated per incident.

---

## 9. Validation steps for a learner submission

1. **Boundary statement** lists what each side owns in specific
   artefacts (not abstractions), names the shared 20 %, and ends with
   a one-sentence boundary rule.
2. **Ticket template** contains every required field, each with a
   one-sentence explanation; the `rule_outs_performed` checklist is
   present and explicit.
3. **Response SLA table** has P0 / P1 / P2 / P3 rows with ack, opinion,
   resolution times, and a mapping to burn rate and to the
   organisation's existing severity map.
4. **Return-shape taxonomy** covers all five shapes plus the reverse
   direction.
5. **Kernel-regression ledger** has a schema plus at least three seeded
   entries with every required field populated; review cadence stated.
6. **Worked example** walks through one incident from axis-alert
   through ticket through peer response to deploy to ledger update,
   with every artefact in the right place.
7. **Three tickets** filed or dry-run; one is a reverse-direction
   ticket. Each uses the full template.
8. **Signatures** present for both sides' leads; dry if necessary but
   roles and dates named.

## 10. Rubric / review checklist

Score each item 0 / 1 / 2. Passing ≥ 16 / 20.

| # | Item | Points |
|---|---|---|
| 1 | Boundary statement names artefacts, not abstractions; ends with a specific one-sentence rule. | 2 |
| 2 | Ticket template has every required field and an explanation per field; `rule_outs_performed` is a checklist. | 2 |
| 3 | SLA table has P0–P3 with ack / opinion / resolution and a mapping to the org's severity map. | 2 |
| 4 | All five return shapes are described concretely with artefacts delivered and ledger impact. | 2 |
| 5 | Reverse-direction tickets are described with artefacts expected and a response flow. | 2 |
| 6 | Ledger has a schema plus ≥ 3 seeded entries with all fields populated; review cadence stated. | 2 |
| 7 | Worked example carries one incident end-to-end through every artefact. | 2 |
| 8 | Three tickets filed or dry-run; one is reverse-direction. | 2 |
| 9 | Contract signatures present (real or dry) with named signers and dates; review cadence in body. | 2 |
| 10 | Contract plugs back into the exercise-01 change calendar and runbook set. | 2 |

## 11. Common mistakes

- **Boundary statement in abstract.** "They do kernels, we do serving"
  is not specific enough. Name artefacts.
- **Ticket template without `rule_outs_performed`.** The next three
  tickets to the peer will be KV-cache tenancy issues, not kernel
  issues; the "not a kernel issue" return costs a day each. Build the
  checklist.
- **SLA table with no pager mapping.** "P0 is 1 hour" means nothing
  if the on-call doesn't know to escalate at 1 hour. Map to the
  paging matrix in exercise 01 Deliverable D.
- **Ledger with one entry.** The ledger gets maintenance only as it is
  used. Seed with three realistic entries so the maintenance habit
  sticks and the shape is tested.
- **Worked example incomplete.** A worked example that stops at
  "filed the ticket" is half the lesson. Carry it through the
  peer response, the deploy, and the ledger entry.
- **Reverse direction missing.** A contract that flows one way
  institutionalises the peer as a service desk. Invert it; name the
  reverse-direction tickets explicitly.
- **Severity scale per peer.** Peer-specific severity invented ("P-perf-0")
  loses the pager mapping. Reuse the org's severity map.
- **Contract signed once and forgotten.** The chapter 01 quarterly
  cadence re-reviews it; add the re-signature date in the contract body.

## 12. References

- vLLM documentation — kernel selection, attention backends, scheduler
  flags, profiler instrumentation. https://docs.vllm.ai/
- SGLang documentation — fallback engine; relevant where the ticket
  reproducer runs across both engines. https://docs.sglang.ai/
- Module 102 (engine internals) — the surface the boundary in §2
  partitions.
- Module 107 (distributed inference topologies) — topology choices
  referenced in the ticket template and the ledger entries.
- Module 108 (speculative decoding ops) — the draft-lifecycle contract
  this interface references; migration candidate #1 in the worked
  example depends on it.
- Module 110 (SLO engineering) — the burn-rate thresholds in the SLA
  table and the ticket's `slo_impact` field.
- Module 113 chapter 03 (runbook top-ten) — the runbooks the worked
  example's rule-outs rely on.
- Module 113 chapter 04 (incident RCA) — the method the P0 flow
  imports.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework)
  — the interface is a GOVERN artefact (named roles, separation of
  duties, review cadence) and a MEASURE artefact (ledger as the
  measurement over time).
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/)
  — the ledger's `fix_commit` and upstream-ticket fields support the
  supply-chain discipline (ML10) by keeping the attribution chain
  visible.
- [mitre-atlas](https://atlas.mitre.org/) — the ledger is a detection
  artefact; ATLAS's detection discipline for tampered or drift-prone
  models maps to the ledger's quarterly review cadence.
- Chapter 04 of this module — the interface shape this solution
  concretises.
- Exercise 01 of this module — the cadence this interface plugs into.
- Exercise 05 of this module — the same shape applied to four more
  peers.
