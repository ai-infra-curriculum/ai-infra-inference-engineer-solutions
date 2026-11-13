# Solution — Exercise 02: Build-vs-Buy RFC Authoring

> Paired learning file: [`lessons/mod-114-cross-functional-inference-leadership/exercises/exercise-02-build-vs-buy-rfc-authoring.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-114-cross-functional-inference-leadership/exercises/exercise-02-build-vs-buy-rfc-authoring.md)
> Module: `mod-114-cross-functional-inference-leadership`
> Type: RFC / decision document

## 1. Solution overview

The deliverable is a defensible build-vs-buy RFC for an inference-platform
decision. A good RFC does four things that a vibe-check memo does not:

1. **Decomposes the decision.** The real choice is rarely "self-host vs.
   vendor across the whole fleet" — it is "which cells of the
   workload × tenant × region matrix go where."
2. **Prices the full TCO per option.** Not just `$ / 1M tokens`; also
   underutilised capacity, exit cost, renewal lock-in erosion, incident
   ownership, people cost, egress.
3. **Lists risks that could reverse the decision.** With likelihood,
   impact, owner, and mitigation.
4. **States triggers that reopen the RFC.** An RFC with no triggers is
   stale on the day it ships.

This worked answer runs the full RFC for a plausible decision — moving a
**tier-C batch-embedding workload** from self-host to a managed provider
(or hybrid), continuing the exercise-01 fleet. Numbers are clearly labelled
`[SIMULATED]`; TCO arithmetic uses publicly posted vendor pricing where
possible and is marked `<!-- pending-verification: ... -->` where a real
procurement would want a signed quote.

### 1.1 Simulation preamble

> Every dollar value in this RFC is **simulated** using publicly posted
> on-demand vendor prices and a plausible fleet shape derived from the
> exercise-01 simulation (320 H100 GPUs, 40 nodes, ≈3.5 B output tokens /
> month). The simulation preserves the arithmetic of the RFC so a real
> team can swap in signed quotes and a measured fleet without changing the
> structure.

### 1.2 RFC-document map

| § | RFC section | Deliverable in the learning file |
|---|---|---|
| §2 | Problem statement | A |
| §3 | Current state | B |
| §4 | Options under consideration | C |
| §5 | Decomposition | D |
| §6 | TCO models (arithmetic) | E |
| §7 | Risk register | F |
| §8 | Recommendation | G |
| §9 | Migration path | H |
| §10 | Open questions | I |
| §11 | Sign-off | J |
| §12 | Appendix (cost model + exhibits) | Appendix |

---

## 2. Deliverable A — problem statement

> **Forcing function.** Tier-C batch embedding (an internal retrieval index
> rebuild workload running nightly on open-weights embedding models)
> currently saturates 25 % of the tier-A H100 fleet in the 02:00–06:00 UTC
> window, blocking the EU peak-interactive window on `eu-west-1-a` and
> `eu-west-1-b`. The assistant-product launch forecast (Deliverable C of
> exercise 01) adds EU morning load that will collide with this window.
> The 2027 capacity plan either adds 8 nodes to the fleet (≈ `[SIMULATED:
> $600 k/yr]` amortised) or relocates the batch-embedding workload. The
> `platform-program-review` forum of 2026-11-03 asked for an RFC on the
> second option.

The forcing function is a specific capacity collision and a specific
budget commitment that would be avoided if the embedding workload moves.
"We should consider managed inference" would not be a forcing function.

---

## 3. Deliverable B — current state

### 3.1 Fleet shape

Inherits §1.2 of the exercise-01 solution. Relevant numbers:

```
fleet:
  hardware: 40 H100-SXM5 nodes × 8 GPU = 320 H100
  regions: [us-east-1 ×2 clusters, eu-west-1 ×2 clusters]
  utilisation: average 55 %, peak 85 %
  warm_pool: per module 109 — ~3 nodes held hot
  engines: [vLLM primary, SGLang fallback]
  router: in-house LiteLLM-shape
```

### 3.2 Workload inventory (just the relevant classes)

| Workload class | Model | Pattern | Monthly token / embedding volume | Current `$ / 1M tokens` | Notes |
|---|---|---|---|---|---|
| tier-A chat | Llama-3.1-70B FP16 | interactive | ≈ 2.5 B output tokens | `[SIMULATED: $2.40 / 1M output]` <!-- pending-verification: replace with measured per-1M cost from module 110 ch. 04 --> | SLO covered |
| tier-B long-tail | routed open-weights | interactive, long-tail | ≈ 0.6 B output tokens | `[SIMULATED: $4.10 / 1M output]` | SLO covered, higher per-unit due to under-utilisation |
| **tier-C batch embedding** | open-weights `bge-m3`-class embeddings | nightly batch | ≈ 2 B input tokens / month → ≈ 2 B embeddings | `[SIMULATED: $0.08 / 1M embed-tokens]` <!-- pending-verification: measured --> | **Subject of this RFC.** Runs 02:00–06:00 UTC nightly, saturates EU GPUs. |
| internal eval | mixed | scheduled | ≈ 0.2 B output tokens | `[SIMULATED: $3.00 / 1M output]` | best-effort |

### 3.3 Current `$ / 1M tokens` derivation for tier-C (source of the baseline)

From module 110 chapter 04's shape:

```
monthly_amortised_hardware_cost_attributable_to_batch_embed  ≈  $120 000   # 8-node slice × H100 amortised monthly × overnight-only = ~25 % of 8 nodes
monthly_embed_tokens_processed                                ≈  2 000 M
$ / 1M embed-tokens                                           ≈  $120 000 / 2 000  =  $60 / 1 M      # internal attribution; cost of embedding token is embedding-unit, not the output-generation shape
```

**Important restatement.** Embeddings are usually priced *per 1M input
tokens* on vendor platforms. The `$60 / 1M embed-tokens` figure above is
the **fleet-attribution cost per 1M embedded input tokens** — not the vendor
unit. For a fair comparison in §6, both self-host and vendor numbers are
normalised to `$ / 1M input tokens embedded`.

### 3.4 SLOs per workload class (extracted from Deliverable B of exercise 01)

| Workload | TTFT p95 | E2E p95 | Availability |
|---|---|---|---|
| tier-A chat | 1500 ms | 7500 ms | 99.5 % |
| tier-B long-tail | 2500 ms | 15 000 ms | 99.0 % |
| tier-C batch embedding | n/a | 60 s per 1 k batch | 99.0 % |

---

## 4. Deliverable C — options under consideration

### 4.1 Option A — self-host, scale up

Add 8 H100 nodes (4 US, 4 EU) specifically to carry the batch-embedding
window without colliding with the assistant-product EU morning. Continue
self-hosting on vLLM / SGLang.

> **What wins here.** Keeps all data in our environment; no new vendor
> procurement cycle; full control over scheduling. Loses on capital cost
> and because the added nodes are sized for the tier-C peak but will sit
> idle 20 h of every day.

### 4.2 Option B — managed inference (vendor-only), Fireworks AI

Move tier-C batch embedding to Fireworks AI serverless embedding endpoints
(`bge-m3`-compatible model). Keep tier-A and tier-B self-hosted.

> **What wins here.** Fireworks' serverless embedding pricing is published
> publicly (see https://fireworks.ai/pricing — the price per 1M input
> tokens for embedding models is posted; verify the current number against
> the vendor's page at RFC-signing time). Loses on egress cost, data
> boundary (embeddings cross a vendor boundary nightly), and lock-in.

### 4.3 Option C — managed inference (vendor-only), Together AI

Move tier-C batch embedding to Together AI's serverless embedding
endpoints.

> **What wins here.** Second managed-inference option with a public
> pricing page (see https://www.together.ai/pricing). Loses on the same
> axes as B; included for comparison so the RFC is not a sole-source
> argument.

### 4.4 Option D — hybrid, per-region

Keep US tier-C on self-host; move EU tier-C to a managed vendor during EU
morning only. Nightly window keeps overflow routed to vendor if the
in-house capacity trips a threshold.

> **What wins here.** Isolates the specific capacity collision in EU
> while avoiding a full vendor migration. Loses on split discipline — two
> paths to debug, two observability integrations, and a router that must
> know how to route embeddings across providers correctly.

### 4.5 Excluded options (and why)

- **Groq / Cerebras.** Current public embedding endpoints are not a
  match for the workload shape as of RFC authoring date; verify against
  vendor landing pages before signing.
  <!-- pending-verification: confirm Groq / Cerebras embedding product availability at RFC-signing time; include in a future scorecard if added -->
- **Hyperscaler inference offerings (Bedrock, Vertex AI).** Reasonable
  candidates but priced differently; exercise-03's scorecard will cover
  them if the sponsor requests a widened shortlist.

---

## 5. Deliverable D — decomposition

Axes used: `workload class × region × time window`. Three axes are enough;
tenant tier is uniform (T1 internal workloads only) and model class is
uniform (embedding model) for this decision.

```
cells in-scope for the RFC decision:
  [tier-C embedding  × us-east-1 × batch window 02-06 UTC]
  [tier-C embedding  × eu-west-1 × batch window 02-06 UTC]
  [tier-C embedding  × eu-west-1 × overflow outside batch window]     # new, caused by assistant-product
cells out-of-scope for this RFC:
  [tier-A chat × *]                      — not moving; capacity owner argument lives in a separate RFC
  [tier-B long-tail × *]                 — stays self-hosted for now; may be revisited in Q3 2027
  [internal eval × *]                    — best-effort workload; not worth the migration cost
```

Explicit out-of-scope cells carry a sentence each so a reader doesn't
infer the decision from absence.

---

## 6. Deliverable E — TCO model (the arithmetic)

### 6.1 Normalisation unit

`$ / 1M input tokens embedded`. All four options report their 12-month TCO
in this unit and as a total-dollar figure at the current forecast volume
(2 B input tokens / month embedded = 24 B / year).

### 6.2 Option A — self-host, scale up

| Line item | Monthly | Annual | Notes |
|---|---|---|---|
| Hardware amortised (8 new nodes, 3-year amortisation, reserved pricing) | `[SIMULATED: $60 000]` | `[SIMULATED: $720 000]` | 8 × H100-SXM5 node reserved rate; <!-- pending-verification: replace with signed hardware quote --> |
| Underutilised capacity (new nodes idle 20 h/day) | `[SIMULATED: $50 000]` | `[SIMULATED: $600 000]` | 20/24 of hardware cost not doing useful work |
| Warm pool (module 109 reservation for autoscaler) | `[SIMULATED: $8 000]` | `[SIMULATED: $96 000]` | ~1 extra node held warm |
| Model-weights storage (object store for embed model + versions) | `[SIMULATED: $500]` | `[SIMULATED: $6 000]` | small |
| Observability (Prometheus, Grafana, tracing for the new nodes) | `[SIMULATED: $1 500]` | `[SIMULATED: $18 000]` | proportional to fleet growth |
| Egress (fleet → object store → fleet) | `[SIMULATED: $800]` | `[SIMULATED: $9 600]` | internal; low |
| Software licences | $0 | $0 | vLLM / SGLang open-source |
| People cost (incremental on-call + ops for 8 nodes) | `[SIMULATED: $6 000]` | `[SIMULATED: $72 000]` | ≈ 10 % of one FTE allocated to the expanded fleet |
| On-call overhead (incremental pager load) | included above | | |
| Opportunity cost (capacity *not* used for assistant-product growth) | qualitative | qualitative | note in prose |
| Exit cost | `[SIMULATED: $30 000]` | one-time | re-sell / de-provision 8 nodes if the decision reverses |

**Option A annual TCO: `[SIMULATED: ~$1 521 600]` (excluding one-time
exit cost) → ~$63 / 1M embed input tokens.**

The arithmetic repeats the module 110 chapter 04 pattern: hardware
amortised + idle + warm pool + ops + observability. The large line item
is *underutilised capacity* — buying 8 nodes to carry a 4-hour-per-day
workload bakes 83 % idle time into the TCO.

### 6.3 Option B — Fireworks AI, serverless embedding

| Line item | Monthly | Annual | Notes |
|---|---|---|---|
| Usage billed (forecast traffic × posted per-1M embed-input-token rate) | `[SIMULATED: $20 000]` | `[SIMULATED: $240 000]` | 2 B tokens / month × posted vendor rate; <!-- pending-verification: capture the exact posted rate per 1M input tokens at RFC-signing time from https://fireworks.ai/pricing --> |
| Overage buffer (forecast-error allowance, +20 %) | `[SIMULATED: $4 000]` | `[SIMULATED: $48 000]` | first-year forecast is noisy |
| Minimum commits | `[SIMULATED: $0]` | `[SIMULATED: $0]` | serverless tier, no minimum per public docs <!-- pending-verification: confirm against enterprise contract --> |
| Data transit (fleet → vendor → fleet) | `[SIMULATED: $3 000]` | `[SIMULATED: $36 000]` | 2 B tokens worth of input + vectors returned; AWS egress applies |
| Observability integration | `[SIMULATED: $1 000]` | `[SIMULATED: $12 000]` | pulling vendor metrics into our Grafana |
| Security and compliance overhead | `[SIMULATED: $2 000]` | `[SIMULATED: $24 000]` | DPA review, vendor security review per module 113 ch. 05 shape |
| Integration people cost | one-time `[SIMULATED: $25 000]` | amortised `[SIMULATED: $8 400]` | router rule, retry logic, failure mode handling |
| Incident ownership premium (support tier) | `[SIMULATED: $1 500]` | `[SIMULATED: $18 000]` | assumed "business" tier; "enterprise" tier priced separately if needed |
| **Lock-in discount erosion** (model the renewal) | `[SIMULATED: year-2 +15 %, year-3 +25 %]` | year-1 above, year-3 ≈ `[SIMULATED: $300 000]` | first-year serverless pricing is usually promotional; posted prices can move at the vendor's discretion <!-- pending-verification: contractual price lock period --> |
| Exit cost | `[SIMULATED: $40 000]` | one-time | 2 weeks' engineering to rebuild self-host path if reversed |

**Option B annual TCO (year 1): `[SIMULATED: ~$386 400]` → ~$16 / 1M
embed input tokens.** Year 3 annualised, with renewal erosion:
`[SIMULATED: ~$450 000]` → ~$19 / 1M embed input tokens.

Two important caveats baked into the number:

- **The vendor's posted price per 1M input tokens is the headline,** but
  the TCO adds data transit, observability integration, DPA review, and
  the renewal erosion. A "it's $0.02 / 1M tokens on their page" argument
  loses 20–30 % of its headline advantage once these are added.
- **Exit cost is non-zero.** The RFC must explicitly budget exit; if it
  doesn't, year-3 renewal negotiations have a hidden floor.

### 6.4 Option C — Together AI, serverless embedding

Structure identical to Option B. Sub-dimension numbers differ by vendor;
the RFC's appendix carries the per-vendor arithmetic. **Option C annual
TCO (year 1): `[SIMULATED: ~$405 000]` → ~$17 / 1M embed input tokens.**
<!-- pending-verification: measured against vendor-posted pricing at RFC signing -->

Option B and Option C come out within `[SIMULATED: ~5 %]` of each other
on year-1 TCO. The scorecard (exercise 03) is where the non-cost axes
pick between them.

### 6.5 Option D — hybrid, per-region

Per-cell:

| Cell | Option | Attributed annual TCO | Rationale |
|---|---|---|---|
| tier-C × us-east-1 × batch | self-host (status quo) | `[SIMULATED: ~$300 000]` | uses existing capacity, no new nodes |
| tier-C × eu-west-1 × batch | hybrid overflow to vendor | `[SIMULATED: ~$180 000]` | base self-host + ~30 % overflow to vendor |
| tier-C × eu-west-1 × overflow outside batch | vendor | `[SIMULATED: ~$60 000]` | new volume from assistant-product forecast |
| Router cost (incremental) | shared | `[SIMULATED: ~$24 000]` | added conditional-routing rule and observability |
| Cross-option observability | shared | `[SIMULATED: ~$18 000]` | two-path dashboards |
| Failover rehearsal cost | shared | `[SIMULATED: ~$12 000]` | quarterly drill |

**Option D annual TCO: `[SIMULATED: ~$594 000]` → ~$25 / 1M embed input
tokens.**

Option D sits between A and B. The hybrid overhead (router,
observability, rehearsal) is real but not large; the trade is split
discipline for the ability to pull the embedding workload back in-house
without a vendor migration if the renewal pricing in year 3 escalates.

### 6.6 Summary

| Option | Year-1 TCO | Year-1 `$ / 1M embed input tokens` | Year-3 renewal exposure |
|---|---|---|---|
| A. Self-host, scale up | `[SIMULATED: $1 521 600]` | ≈ $63 | n/a; cost is capex-amortised |
| B. Fireworks | `[SIMULATED: $386 400]` | ≈ $16 | +15 to +25 % |
| C. Together | `[SIMULATED: $405 000]` | ≈ $17 | +15 to +25 % |
| D. Hybrid (EU overflow) | `[SIMULATED: $594 000]` | ≈ $25 | partial exposure |

A detailed line-item spreadsheet accompanies the RFC as appendix exhibit 1
(`cost-model.yaml`).

---

## 7. Deliverable F — risk register

Scoring: Likelihood `L / M / H`, Impact quantified where possible.

### 7.1 Option A (self-host, scale up) — 7 risks

| # | Risk | Likelihood | Impact | Owner | Mitigation |
|---|---|---|---|---|---|
| A-1 | Nodes sit idle 20 h/day, inflating TCO; sustained low utilisation prompts a budget conversation | H | `[SIMULATED: $600 000 / yr]` under-utilisation cost | Platform Lead | time-share the new nodes onto tier-B overflow during the day; module 109 autoscaling |
| A-2 | Driver / firmware regression on new nodes delays batch window | M | `[SIMULATED: 1 missed batch window = 24-h staleness in retrieval index]` | SRE peer | stage rollouts; kernel-regression ledger (ex-04 Deliverable E) |
| A-3 | Hardware supply lead-time slips RFC's capacity date | M | missed assistant-product launch window | Sponsor + procurement | place order ahead; keep Option B/D as hot plan-B |
| A-4 | On-call load from expanded fleet pushes rotation above sustainable threshold | M | hero on-call (ex-01 §8.1) | SRE peer | pool size review before the capacity bump |
| A-5 | Embedding model upgrade (bge-m3 → bge-m4 class) forces a fleet-side re-deploy | L | `[SIMULATED: 2 weeks engineering]` | Platform Lead | lifecycle contract per module 108 ex-04 shape |
| A-6 | Spot / on-demand price volatility exceeds amortisation plan | L | `[SIMULATED: +10 % TCO]` | Finance partner | reserved pricing; quarterly check |
| A-7 | Opportunity cost — the new nodes could have gone to tier-A chat growth | M | qualitative | Sponsor | explicit annual capacity review |

### 7.2 Option B (Fireworks) — 7 risks

| # | Risk | Likelihood | Impact | Owner | Mitigation |
|---|---|---|---|---|---|
| B-1 | Vendor SLA credit schedule does not cover a batch-window outage; rebuild missed, retrieval index stale | M | 24-h data staleness during recovery | Platform Lead | contract red-line; batch-window-aware SLA language |
| B-2 | Per-1M-token price moves at renewal | M | `[SIMULATED: +25 % at year 3]` | Sponsor + Finance | model renewal in TCO (done, §6.3); keep Option D as plan-B |
| B-3 | Feature parity — embedding quality divergence from self-hosted model | L | downstream retrieval quality regression | Eval peers | benchmark on representative traffic (exercise 03 Deliverable C) |
| B-4 | Data boundary — DPA review exposes a subprocessor concern | M | `[SIMULATED: 2 weeks delay]` | Security peer | DPA review before contract (exercise 03 Deliverable F) |
| B-5 | Exit cost materially higher than modelled (rebuild self-host path) | L | `[SIMULATED: +$60 000 one-time]` | Platform Lead | document the self-host path as a dormant runbook; rehearse annually |
| B-6 | Lock-in by API-shape divergence (vendor-specific response schema) | M | `[SIMULATED: 1-2 months to re-port]` | Platform Lead | thin adapter layer; keep an API-shape test harness |
| B-7 | Vendor incident cascades into our product (vendor outage during nightly rebuild) | M | 24-h staleness | Platform Lead | retry to Option D vendor; consider hybrid as default |

### 7.3 Option C (Together) — 7 risks

Same shape as B. Not duplicated in full; the vendor-specific risks below
are the ones that differ materially.

- C-2 renewal exposure is ≈ B-2.
- C-4 subprocessor list differs; check against security peer's catalogue.
- C-7 incident history is a different distribution — see exercise 03
  scorecard's historical uptime extraction.

### 7.4 Option D (hybrid) — 8 risks

| # | Risk | Likelihood | Impact | Owner | Mitigation |
|---|---|---|---|---|---|
| D-1 | Split discipline — two paths to debug; a bug in self-host path can be masked by vendor overflow | H | increased MTTR | Platform Lead | two-path dashboards (observability line item is funded) |
| D-2 | Observability gap — vendor metrics and self-host metrics not comparable | M | missed regressions | SRE peer | shared panel contract in ex-03 scorecard |
| D-3 | Failover drift — rehearsal lapses, vendor path goes unexercised, real outage reveals it | M | 24-h staleness in an incident | Platform Lead | quarterly failover rehearsal (funded in §6.5) |
| D-4 | Router rule complexity — decision between vendor and self-host becomes a source of its own bugs | M | low-grade incidents | Platform Lead | module 111 API design; keep the rule simple |
| D-5 | Vendor renewal exposure still applies to the vendor-serviced share | M | partial `[SIMULATED: +$30 000]` at year 3 | Sponsor + Finance | model, re-RFC before renewal |
| D-6 | Hardware supply lead-time still applies for the self-host share | L | partial Option A exposure | Procurement | same mitigation as A-3 |
| D-7 | Data-boundary complexity — only EU share crosses the boundary; GDPR posture still needs DPA | M | `[SIMULATED: 2 weeks delay]` | Security peer | DPA review before contract |
| D-8 | People cost for running two paths | M | `[SIMULATED: +10 % ops load]` | Platform Lead | funded in §6.5 |

---

## 8. Deliverable G — recommendation

### 8.1 Per-cell recommendation

| Cell | Decision | Rationale |
|---|---|---|
| tier-C × us-east-1 × batch | **Self-host, status quo.** No change. | Existing capacity carries this cell; no collision in US. |
| tier-C × eu-west-1 × batch | **Move to managed vendor (Option B — Fireworks, pending the exercise-03 scorecard outcome).** | EU capacity collision is the forcing function; vendor TCO beats the nodes-added TCO by `[SIMULATED: ~$1.1M / yr]`. |
| tier-C × eu-west-1 × overflow outside batch | **Managed vendor (same contract).** | Hybrid routing rule adds marginal complexity; natural fit. |

The overall posture is **Option D (hybrid)** — not Option B in full.
Pure Option B would also move the US batch workload, which already has
capacity; moving it for symmetry would double exposure to vendor lock-in
with no capacity benefit. The hybrid shape keeps US on self-host (zero
exit cost there) and uses the vendor only where the capacity collision
is real.

### 8.2 TCO delta

`[SIMULATED: ~$927 000 / yr]` saved vs. Option A at the year-1 price.
Year-3 with renewal erosion modelled: `[SIMULATED: ~$850 000 / yr]`
saved. See appendix exhibit 1 for the line-item derivation.

### 8.3 Decision reverses if

- Vendor posts a renewal price increase beyond the modelled +25 %
  ceiling.
- Vendor's embedding quality on the retrieval benchmark (exercise 03
  Deliverable C) drops by more than `[SIMULATED: 3 %]` cosine-average
  against the self-host baseline.
- Vendor availability on the batch window drops below the 99.0 % SLO
  for two consecutive quarters.
- A DPA term changes materially at renewal (e.g., subprocessor region
  shift, retention default change).
- Assistant-product traffic forecast is revised such that the US batch
  window is itself threatened — reopens the US decision.
- Groq / Cerebras / hyperscaler enters the embedding market with a
  materially different price or capability shape (reopens the shortlist
  in a scorecard refresh).

Each trigger owns a specific review that reopens this RFC; the
exercise-01 quarterly migration-roadmap review carries the action.

---

## 9. Deliverable H — migration path (chapter 01's eight-field shape)

**1. Problem.** EU batch-embedding workload saturates the EU H100 fleet
during the 02:00–06:00 UTC window and will collide with the assistant-
product morning launch forecast. The next capacity option is an 8-node
expansion at `[SIMULATED: ~$1.5M / yr]`; this plan replaces that with a
managed vendor for the EU share.

**2. Scope.** EU batch embedding + EU overflow outside batch. US batch
remains self-hosted.

**3. Target state.** Embeddings for EU workloads flow through a routing
rule that sends them to the managed vendor's endpoint; retrieval pipeline
reads the vendor's embedding output in the same schema as self-host
output; US retains self-host as-is.

**4. Peer touch points.**

| Peer | Touch point | Gate |
|---|---|---|
| ML-platform peer | Embedding-model equivalence check; vendor model-card captured in model registry. | Artefact-shape parity signed. |
| Eval peers | Retrieval-benchmark comparison between self-host and vendor embeddings. | Benchmark signed. |
| SRE peer | SLO contract pointer updated; availability SLO for EU batch amended to reflect vendor path. | Pointer signed. |
| Security peer | DPA review; data-boundary review; subprocessor list vetted. | Security sign-off on file. |
| Finance partner | Budget code; AP integration; renewal calendar. | PO signed. |

**5. Stages.**

| # | Stage | Exit criterion | Duration |
|---|---|---|---|
| 1 | Sign vendor contract; DPA review complete; API keys provisioned. | Contract + DPA on file. | 3 weeks |
| 2 | Build the routing rule; shadow EU embedding traffic to the vendor for 1 week; compare quality. | ≥ `[SIMULATED: 97 %]` cosine agreement on retrieval benchmark. | 2 weeks |
| 3 | Canary EU batch at 1 % for one nightly window. | Nightly rebuild completes within SLO. | 1 week |
| 4 | Canary EU batch at 25 %. | Same; cost reporting pipeline functional. | 1 week |
| 5 | 100 % EU batch cut-over. | Nightly SLO held for 7 days. | 1 week |
| 6 | Enable EU overflow routing for outside-batch window. | No incidents for 7 days; cost reporting shows expected delta. | 1 week |
| 7 | Decommission attributable EU self-host capacity for this workload (return nodes to the shared pool). | Node reclaim ticket closed. | 1 week |

**6. Rollback branch.** Rollback is cheap through stage 4 (revert the
routing rule, EU traffic returns to self-host). **Stage 5 is the point
past which rollback becomes expensive:** once the EU nightly is 100 % on
vendor, the overnight self-host capacity has been re-allocated or scaled
in, and reverting requires either waking the capacity back up (lead time:
days) or accepting a staleness in the retrieval index (operational cost).
Stage 7 makes rollback a *second migration*: nodes have been reclaimed;
re-provisioning them is the forward cost all over again. A rollback in
Q2–Q3 is still possible, but budget 2–3 weeks and an explicit Platform
Lead sign-off.

**7. Consumer communication plan.** `search-rag` is the only consumer
track whose retrieval index depends on this workload. Communication:

- Pre-stage-1: email to `search-rag` lead with the migration plan, the
  equivalence-benchmark plan, and the rollback branch.
- At stage 2 (shadow): results of the equivalence benchmark shared.
- At stage 3 and 4: weekly update, including cost and SLO panel
  screenshots.
- At stage 5 (100 %): 24-hour-advance notice and a 7-day hyper-care
  window.
- At stage 7 (decommission): confirmation that the rollback branch is now
  a second migration, with expected cost if invoked.

Single Slack thread
(`[SIMULATED: #inf-tier-c-embedding-migration-q1-2027]`) carries all
updates.

**8. Evidence of done.**

- Signed DPA on file.
- Equivalence benchmark archived.
- Cost delta measured for one full billing cycle, matching §6's model
  within `[SIMULATED: ±10 %]`.
- Nightly SLO panel green for 30 consecutive days post cut-over.
- Rollback runbook written and rehearsed once (ex-01 Deliverable D
  §5 runbook format).

---

## 10. Deliverable I — open questions

| # | Question | Owner | Due |
|---|---|---|---|
| 1 | What is the vendor's posted embedding price per 1M input tokens at RFC-signing time, and what is the contractual price-lock period? | Finance partner + Platform Lead | before stage 1 |
| 2 | What is the vendor's historical uptime on the batch-window shape (not the marketing "99.9 %"; the extracted incident history)? | Platform Lead (exercise 03 Deliverable C does this extraction) | stage 1 |
| 3 | What subprocessors handle vector data for the vendor's EU region? | Security peer | stage 1 |
| 4 | Does the retrieval benchmark show quality parity with the self-host embedding? | Eval peers (benchmark in exercise 03 Deliverable C) | stage 2 |
| 5 | What exit cost is realistic if we reverse in year 3? (Beyond the one-time `[SIMULATED: $40 000]` estimate — are there contractual egress / de-indexing charges?) | Platform Lead + Finance | before stage 1; revise at year-2 review |

---

## 11. Deliverable J — sign-off

| Role | Signer | Date | Notes |
|---|---|---|---|
| Platform Lead | `[SIMULATED: name]` | 2026-11-20 | |
| Sponsor — VP Infrastructure | `[SIMULATED: name]` | 2026-11-22 | gates budget |
| SRE Peer Lead | `[SIMULATED: name]` | 2026-11-20 | gates SLO contract pointer update |
| ML-platform Peer Lead | `[SIMULATED: name]` | 2026-11-20 | gates artefact ingress |
| Security Peer Lead | `[SIMULATED: name]` | 2026-11-22 | gates DPA |
| Eval Peer Lead (model-evaluation) | `[SIMULATED: name]` | 2026-11-21 | gates retrieval benchmark |
| Consumer Track Lead — `search-rag` | `[SIMULATED: name]` | 2026-11-23 | gates stage-2 shadow |
| Finance Partner | `[SIMULATED: name]` | 2026-11-24 | gates contract |

Signatures are simulated; dates reflect the plan rather than history. In a
real RFC, each signer initials the section they gate.

---

## 12. Appendix

### 12.1 `cost-model.yaml` (appendix exhibit 1)

```yaml
# cost-model.yaml — Tier-C Embedding RFC — SIMULATED numbers
options:
  A_self_host_scale_up:
    monthly:
      hardware_amortised: 60000     # 8 new H100 nodes reserved
      underutilised_capacity: 50000 # 20h/24h idle on new nodes
      warm_pool: 8000
      model_weights_storage: 500
      observability: 1500
      egress: 800
      software_licences: 0
      people_cost_incremental: 6000
      on_call_overhead: included-in-people
      opportunity_cost: qualitative
    one_time:
      exit_cost: 30000
    annual_total: 1521600          # sum of 12 months

  B_fireworks:
    monthly:
      usage_billed: 20000          # 2B tokens * posted rate; TODO verify rate
      overage_buffer_20pct: 4000
      minimum_commits: 0           # serverless
      data_transit: 3000
      observability_integration: 1000
      security_compliance_overhead: 2000
      integration_people_amortised: 2080
      incident_ownership_premium: 1500
    one_time:
      integration_people: 25000
      exit_cost: 40000
    renewal:
      year_2_multiplier: 1.15
      year_3_multiplier: 1.25
    annual_total_year_1: 386400

  C_together:
    # same structure; sub-items differ per vendor pricing page
    annual_total_year_1: 405000

  D_hybrid_eu_overflow:
    cells:
      us_batch_self_host: 300000
      eu_batch_hybrid: 180000
      eu_overflow_vendor: 60000
    shared:
      router_incremental: 24000
      cross_option_observability: 18000
      failover_rehearsal: 12000
    annual_total_year_1: 594000

forecast_volume:
  annual_input_tokens_embedded: 24000000000   # 2B/month * 12

normalised_cost_per_million_input_tokens:
  A: 63
  B: 16
  C: 17
  D: 25
```

### 12.2 Supporting exhibits

- Exhibit 2: utilisation measurement (where `55 %` comes from). Would be
  a Grafana screenshot in the real RFC;
  <!-- pending-verification: attach panel from module 110 dashboard once available -->.
- Exhibit 3: assistant-product traffic forecast collision (why the EU
  morning collides with the batch window). <!-- pending-verification: attach
  from assistant-product track's forecast doc -->.
- Exhibit 4: vendor-pricing page snapshots at RFC signing time (Fireworks,
  Together). The TCO model depends on these; archive them to prove the
  arithmetic was priced at a point in time.
- Exhibit 5: scorecard pointer. Exercise 03 produces the scorecard that
  selects between vendors B and C; attach once completed.
- Exhibit 6: DPA notes (private file; not published with the RFC).

---

## Implementation

The RFC itself is prose and arithmetic, but once signed it lands on concrete
platform controls. A working implementation of the recommendation in §8
(fully adopt option B — Fireworks AI serverless embedding for tier-C batch
embedding) looks like:

- **Router config change.** The in-house router gains a new backend entry
  `tier-c-embed-fireworks` pointing at the vendor endpoint, gated behind a
  canary flag. The flag is flipped from off → 1 % → 10 % → 100 % over the
  migration windows named in §9 (Deliverable H — migration path).
- **Per-call cost accounting.** The router emits a `cost_usd` label per
  tier-C request derived from the posted per-1M input-token rate captured
  at RFC signing (recorded alongside the exhibit snapshot from §12.2).
  This feeds the module 110 cost panel and the quarterly contract
  re-signature ritual (§7 of exercise 01).
- **Capacity decommission.** The 8 reserved H100-SXM5 nodes (§6.2) are
  moved from the tier-C batch pool back to the tier-A/B shared pool when
  the canary reaches 100 %; the move is tracked as a change-calendar
  entry (§4 of exercise 01) with explicit rollback criteria.
- **Fallback path.** The option-A self-hosted path remains built but
  idle for one quarter after cutover — the migration ledger carries a
  "revert ready" flag and the engine artefacts are pinned in the model
  registry. The decision to retire the fallback is a separate ticket
  reviewed at the next quarterly contract re-signature.
- **Risk register as recurring agenda.** The risks in §7.2 are copied
  into the weekly SLO review template (§6 of exercise 01) for the first
  month after cutover; the "SLA breach without compensation" and "data
  boundary" risks carry explicit owner and review-date fields.
- **Reverse-decision triggers wired to alerts.** The "decision reverses
  if" rows in §8.3 become named alert rules on the module 110 dashboard
  (vendor p95 latency, vendor error rate, sustained posted-rate
  increase); each rule routes to the inference platform PM for
  re-opening the RFC, not to on-call.

The RFC is the decision; the list above is what makes the decision
auditable, reversible, and visible on cadence.

---

## 13. Validation steps for a learner submission

Reviewers should run these in order. Missing any step fails the
submission.

1. **Problem statement names a concrete forcing function**; not "we
   should consider this."
2. **Current state** has per-workload-class SLOs and a per-workload-class
   `$ / 1M tokens` (real or clearly-simulated).
3. **Options** evaluate at least two named managed vendors plus
   self-host plus at least one hybrid variant; excluded options named
   with reasons.
4. **Decomposition** has at least two axes and some cells are explicitly
   out of scope with reasons.
5. **TCO model** has line-item cost per option; renewal priced for
   managed; exit priced for every option; hybrid overhead priced; an
   appendix exhibit contains the arithmetic.
6. **Risk register** has at least six risks per option, scored with
   likelihood / impact / owner / mitigation; covers capacity, people,
   hardware, software supply chain, driver/firmware for self-host; SLA,
   price, feature parity, data boundary, exit cost, lock-in for managed;
   split discipline, observability gap, failover drift for hybrid.
7. **Recommendation** is per-cell, not a binary verdict.
8. **Triggers** that reopen the RFC are explicit and specific
   (thresholds, dates, or named events).
9. **Migration path** has all eight fields, including the rollback-
   expensive stage.
10. **Open questions** name 2–5 items with owners and due dates.
11. **Sign-off** has roles named; signatures real or clearly marked
    simulated; dates present.
12. **Appendix** has the cost spreadsheet / YAML and supporting
    exhibits (even if any individual exhibit is stubbed with a
    `<!-- pending-verification --> `).

## 14. Rubric / review checklist

Score each item 0 / 1 / 2. Passing ≥ 24 / 30.

| # | Item | Points |
|---|---|---|
| 1 | Concrete forcing function stated; no "we should consider" phrasing. | 2 |
| 2 | Current state has per-workload SLOs and `$ / 1M tokens`. | 2 |
| 3 | At least four options evaluated: self-host, ≥ 2 vendors, ≥ 1 hybrid. | 2 |
| 4 | Decomposition on ≥ 2 axes; out-of-scope cells named with reasons. | 2 |
| 5 | Line-item TCO per option; renewal, exit, and hybrid overhead priced. | 2 |
| 6 | Risk register ≥ 6 risks per option with full scoring. | 2 |
| 7 | Per-cell recommendation rather than a binary verdict. | 2 |
| 8 | Reopen-triggers specific (thresholds / dates / events). | 2 |
| 9 | Migration path uses chapter 01's eight-field shape; rollback-expensive stage named. | 2 |
| 10 | Open questions have owners and due dates. | 2 |
| 11 | Sign-off: roles named, signatures real or simulated, dates. | 2 |
| 12 | Appendix contains cost model / YAML and at least two supporting exhibits (even if stubbed). | 2 |
| 13 | Simulation clearly disclosed where applicable; `<!-- pending-verification: ... -->` used for unverifiable claims. | 2 |
| 14 | References cover the official source policy (NIST AI RMF for governance, OWASP ML for vendor supply-chain where relevant, MITRE ATLAS for edge risk) and the two vendor public pricing pages that the TCO depends on. | 2 |
| 15 | Numeric arithmetic in §6 is reproducible from the appendix; a reader can replace a cell and re-derive the totals. | 2 |

## 15. Common mistakes

- **The unit-cost shootout.** Comparing `$ / 1M tokens` between options
  on one axis and declaring a winner. Price the rest of the TCO
  (idle capacity, exit, people, observability).
- **Promotional pricing treated as the long-run number.** Vendor
  serverless pricing often has implicit year-1 promo rates. Model year-2
  and year-3 explicitly.
- **Exit cost of zero.** Even a great decision needs an exit estimate.
  "Unknown; recommend rehearsed exit by Q3 2027" is a valid placeholder,
  but zero is not.
- **Binary pick without decomposition.** If the recommendation is a
  single verdict for the whole fleet, almost certainly a cell-level
  recommendation would differ somewhere. Force at least one cell to a
  different placement.
- **Risk register as three bland risks.** Six per option, scored, with
  owners and mitigations. Bland registers are the hallmark of an
  undefended RFC.
- **Marketing SLA scored.** Vendor marketing uptime copied as fact.
  Extract historical uptime (exercise 03 Deliverable C does this) and
  note the credit schedule.
- **No triggers that reopen the RFC.** Decisions without triggers are
  stale on day one.
- **Simulation hidden.** Numbers that look signed but were invented are
  a worse failure than numbers that are clearly labelled simulated.
  `[SIMULATED]` and `<!-- pending-verification: ... -->` are both low-friction
  and preserve the review trail.
- **Migration plan without rollback branch.** Every migration plan has
  a stage past which rollback becomes a second migration; name it.

## 16. References

- vLLM documentation. https://docs.vllm.ai/
- SGLang documentation. https://docs.sglang.ai/
- Fireworks AI pricing (public). https://fireworks.ai/pricing
  <!-- pending-verification: capture the current posted rate per 1M input tokens at RFC-signing time; the number drives §6.3 directly -->
- Together AI pricing (public). https://www.together.ai/pricing
  <!-- pending-verification: capture the current posted rate at RFC-signing time; drives §6.4 -->
- Baseten pricing (public). https://www.baseten.co/pricing
  <!-- pending-verification: include if shortlist widens -->
- Modal pricing (public). https://modal.com/pricing
  <!-- pending-verification: include if shortlist widens -->
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework)
  — the Govern and Map functions cover the governance of a build-vs-buy
  decision: documented context of use (which workloads, which tenants,
  which regions), scored risks, owners, triggers.
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/)
  — ML10 (Insufficient Supply Chain Security) covers the vendor
  intake and DPA posture; ML05 (Model Theft) and ML06 (Output Integrity)
  cover residual risk when an external vendor is in the serving path.
- [mitre-atlas](https://atlas.mitre.org/) — the risk register for
  managed vendors picks up the ATLAS supply-chain tactics (AML.T0010).
- Module 110 chapter 04 — cost per 1M tokens; the arithmetic in §3 and
  §6 reuses it directly.
- Module 113 chapter 05 — eval-gate contract; the migration path's
  stage-2 equivalence benchmark leans on this.
- Chapter 02 of this module — the RFC shape this solution concretises.
