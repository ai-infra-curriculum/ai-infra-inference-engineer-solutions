# Solution — Exercise 03: Vendor Eval Scorecard Drill

> Paired learning file: [`lessons/mod-114-cross-functional-inference-leadership/exercises/exercise-03-vendor-eval-scorecard-drill.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-114-cross-functional-inference-leadership/exercises/exercise-03-vendor-eval-scorecard-drill.md)
> Module: `mod-114-cross-functional-inference-leadership`
> Type: vendor evaluation artefact

## 1. Solution overview

The scorecard is a three-page artefact (per-vendor detail pages, summary
comparison, decision cover memo) that picks a vendor for a specific
workload class. The common failure mode is scoring the vendor's blog; the
discipline is to tie every score to evidence — a benchmark on our traffic,
a status-page extraction, a DPA excerpt, a signed quote.

This worked answer scores three vendors for the **tier-C batch-embedding
workload** surfaced in exercise 02's RFC: **Fireworks AI**, **Together
AI**, and **Baseten**. All scores are of the form `[SIMULATED: N/5]` and
all evidence pointers are either public URLs or marked
`<!-- pending-verification: ... -->`. The weighting is **locked before scoring**
and justified in Deliverable A.

### 1.1 Simulation preamble

> This scorecard is authored as a **dry-run against public evidence**.
> Benchmark numbers in §4 and §5.1 are placeholders of the form
> `[SIMULATED: N ms]` where a real run against live API keys would record
> measured numbers; the methodology for running the benchmark is given in
> full so the measurement is reproducible. Status-page extractions in §4
> are derived from publicly visible status sites where present; where a
> vendor's status page lacks aggregate uptime, that gap is itself a risk
> captured in the SLA axis. DPA notes in §5.6 (Deliverable F) are from
> public privacy/trust pages only — the enterprise DPA PDF would require
> signature and is marked
> `<!-- pending-verification: contract-level review required before signature -->`.

### 1.2 Scorecard-document map

| § | Deliverable | Pages |
|---|---|---|
| §2 | A. Workload scoping memo | 1 |
| §3 | Weighting (locked before scoring) | integrated with §2 |
| §4 | B. Per-vendor detail pages (three) | 3 × 1 |
| §5 | C. Benchmark on our traffic | integrated with §4; methodology §5.1 |
| §6 | D. Summary comparison | 1 |
| §7 | E. Decision cover memo | 1 |
| §8 | F. DPA / contract review notes | private appendix |

---

## 2. Deliverable A — workload scoping memo

### 2.1 Workload class

> **Workload:** tier-C batch embedding for the `search-rag` consumer
> track's nightly retrieval index rebuild.
> **Model class:** open-weights `bge-m3`-compatible embedding model (or
> an open-weights equivalent available on each vendor).
> **Typical prompt shape:** ~1 500 input tokens per document; ~1 M
> documents per nightly rebuild = ≈ 2 B input tokens / month embedded.
> **Typical output shape:** single dense vector of dimension 1 024 per
> document.
> **SLO targets:** 60 s per batch of 1 k documents, 99.0 % availability
> on the nightly window, p95 end-to-end batch completion inside the
> 02:00–06:00 UTC window.
> **Traffic volume:** 2 B input tokens / month (from exercise 02 §3.2).
> **Compliance posture:** internal workload; retrieval over internal
> documents; data classification is `internal` (not `secret`); no
> regulated-data flags on the corpus. DPA review is still required
> because vendor EU subprocessors apply; a GDPR data-processing
> addendum is standard for any EU-user-adjacent data.

### 2.2 Shortlist

Vendors shortlisted (serve embedding workloads at the forecast volume):

| Vendor | Rationale | Public pricing page |
|---|---|---|
| Fireworks AI | Serverless embedding endpoints; `bge-m3` family supported. | https://fireworks.ai/pricing |
| Together AI | Serverless embedding endpoints; `bge-m3` family supported. | https://www.together.ai/pricing |
| Baseten | Dedicated-endpoint serving pattern; supports open-weights embedding models. | https://www.baseten.co/pricing |

Vendors **excluded** from the shortlist (named with reason):

| Vendor | Reason excluded |
|---|---|
| Groq | Public embedding product unclear at eval date; scorecard against absence would be unfair. <!-- pending-verification: re-check at next scorecard refresh --> |
| Cerebras | Same as Groq; primary product class is throughput LLM generation, not embedding. |
| Modal | Embeddings are available via self-authored functions, not a managed endpoint; evaluating them would be apples to oranges for this axis. <!-- pending-verification: revisit if Modal publishes a managed embedding offering --> |
| Hyperscalers (Bedrock, Vertex AI) | Reasonable candidates but priced differently; separate scorecard pass if the sponsor widens the shortlist. |

### 2.3 Weighting (LOCKED 2026-11-15 BEFORE SCORING)

Each of the five axes carries a weight. The weighting is set first and
is defended below.

| Axis | Weight | Rationale |
|---|---|---|
| SLA | 0.20 | Batch workload tolerates retry; a one-batch-window miss is 24 h staleness, not production outage. SLA matters but is not the dominant axis. |
| Price | 0.30 | Workload volume (2 B input tokens / month) means price moves materially affect TCO; the forcing function for the migration is cost (exercise 02 §2). |
| Feature parity | 0.25 | Retrieval quality is a product requirement; a vendor that embeds measurably worse than self-host breaks downstream rankings. Scored on §5's benchmark. |
| Exit cost | 0.15 | Non-trivial because lock-in by API shape and embedding-dimension choice. Moderate weight. |
| Data boundary | 0.10 | Internal corpus, not regulated; DPA is required but not the decisive axis. |

Weighting lock date: **2026-11-15.** Scoring in §4 happened on
2026-11-18. The audit trail (this memo) is preserved in the repo.

For a *regulated-data* workload (PHI, financial) the same vendor shortlist
would lock a very different weighting: data boundary 0.35, SLA 0.25,
price 0.15, feature parity 0.15, exit cost 0.10. The chapter 03 shape
accommodates both; the memo must defend one before scoring begins.

---

## 3. Benchmark methodology (shared across §4 and §5)

### 3.1 Traffic sample

- **Size.** 100 prompts covering the P50 (≈ 1 500 token) and P95 (≈ 3 000
  token) document-length distribution of the production corpus.
- **Composition.** 60 % at P50 shape, 30 % at P90 shape, 10 % at P95 shape.
- **Source.** Anonymised internal corpus; selection captured in
  `bench/samples/tier-c-embed-100.jsonl` for re-runs. <!-- pending-verification: real sample composition for a real scorecard -->

### 3.2 Scoring per request

- **TTFT.** First byte of the embedding response.
- **TPOT.** Not applicable to embeddings; omitted.
- **ITL.** Not applicable; omitted.
- **E2E.** From request send to vector parsed on client.
- **Error rate.** HTTP 5xx or 429 count / total requests.
- **Rate-limit hits.** 429s with `Retry-After`.
- **Equivalence score.** For each document, compute `cosine(vendor_vec,
  self_host_vec)`. Report mean, p5 cosine. Threshold for a passing
  equivalence: mean ≥ 0.985, p5 ≥ 0.95 against the self-host baseline
  on the same model family. (Target is modest because small floating-
  point differences across infra are expected; a mean below 0.95 would
  indicate a different checkpoint.)

### 3.3 Scheduling

- Run three batches of 100 requests at three times of day (UTC): 03:00
  (nominal batch window), 09:00 (EU peak), 15:00 (US peak).
- Record p50, p95, p99, and max per vendor per window.

### 3.4 How to translate benchmark numbers into axis scores

- **Feature parity.** Mean cosine ≥ 0.985 and p5 ≥ 0.95 → 5. Linear
  decay below; 0 at mean cosine < 0.9.
- **SLA (benchmark side — the status-page side is separate).** Measured
  nightly-window p95 within 1.5× of self-host baseline → 5. Larger
  decays score lower.
- **Price.** Translated from `$ / 1M input tokens` at the forecast
  volume; cheaper → higher. Score 5 if ≤ self-host attributed cost.
- **Exit cost.** Qualitative; see per-vendor axis commentary.
- **Data boundary.** Qualitative; see per-vendor DPA notes.

---

## 4. Deliverable B — per-vendor detail pages

### 4.1 Fireworks AI

> **Product:** Fireworks AI serverless embedding endpoints.
> **URL:** https://fireworks.ai
> **Documentation reviewed:** https://docs.fireworks.ai (platform docs,
> embedding endpoint reference), https://fireworks.ai/pricing.
> **Eval date:** 2026-11-18.
> **Eval owner:** `[SIMULATED: Platform Lead]`.

**Five-axis scoring** (weight × raw score summed in §6).

| Sub-dimension | Raw score (1–5) | Evidence pointer |
|---|---|---|
| **SLA — marketing uptime** | `[SIMULATED: 4]` | vendor public page states a target uptime; verify at https://fireworks.ai legal / trust page <!-- pending-verification: aggregate uptime on the embedding product --> |
| **SLA — historical uptime (status-page extraction)** | `[SIMULATED: 4]` | status.fireworks.ai incident history; extracted 2026-05 to 2026-11, recording each incident's duration and the embedding-endpoint impact flag; summary table archived as `evidence/fireworks-uptime.md` <!-- pending-verification: real archived snapshot --> |
| **SLA — credit schedule** | `[SIMULATED: 3]` | public terms of service; credits typically apply after sustained below-threshold uptime; the exact schedule needs enterprise contract read <!-- pending-verification: DPA PDF --> |
| **Price — posted rate per 1M input tokens** | `[SIMULATED: 5]` | https://fireworks.ai/pricing; <!-- pending-verification: capture exact posted rate at scorecard date --> |
| **Price — effective at forecast (overage, buffer)** | `[SIMULATED: 4]` | forecast of 2 B tokens / month × posted rate + 20 % buffer; priced in exercise-02 §6.3 |
| **Price — renewal lock-in erosion** | `[SIMULATED: 3]` | serverless rates are typically not rate-locked; the TCO in exercise 02 modelled +15 % year 2, +25 % year 3; <!-- pending-verification: enterprise contract may offer a lock --> |
| **Feature parity — benchmark TTFT p95 at batch window** | `[SIMULATED: 4]` | benchmark methodology §3; measured p95 = `[SIMULATED: 420 ms]` <!-- pending-verification: real benchmark --> |
| **Feature parity — benchmark E2E p95 for 1 k-doc batch** | `[SIMULATED: 4]` | measured E2E = `[SIMULATED: 48 s]` within 60 s SLO |
| **Feature parity — cosine equivalence to self-host** | `[SIMULATED: 4]` | measured mean cosine = `[SIMULATED: 0.987]`, p5 = `[SIMULATED: 0.952]` |
| **Feature parity — error rate / rate-limit hits** | `[SIMULATED: 4]` | measured error rate = `[SIMULATED: 0.3 %]`; one 429 across 300 requests |
| **Feature parity — API shape stability** | `[SIMULATED: 3]` | vendor changelog last 12 months; one breaking change in the embedding response schema noted; <!-- pending-verification: confirm --> |
| **Exit cost — API-shape coupling** | `[SIMULATED: 3]` | moderate; vendor uses an OpenAI-compatible embeddings endpoint, which reduces coupling |
| **Exit cost — embedding-dimension lock-in** | `[SIMULATED: 3]` | vendor supports `bge-m3`'s native 1 024-dim; dimension is portable to self-host |
| **Exit cost — contractual termination clause** | `[SIMULATED: 3]` | public terms of service permit cancellation with notice; enterprise contract would re-specify <!-- pending-verification: DPA / enterprise ToS --> |
| **Data boundary — region** | `[SIMULATED: 4]` | US-based processing for the standard endpoint; EU-region option is available on request <!-- pending-verification: confirm EU region support --> |
| **Data boundary — training on data default** | `[SIMULATED: 4]` | public privacy policy states customer data is not used for training on the standard product; enterprise terms re-affirm <!-- pending-verification: confirm against DPA --> |
| **Data boundary — logging retention default** | `[SIMULATED: 3]` | public docs describe logging; retention default and configurability need DPA review |
| **Data boundary — subprocessors** | `[SIMULATED: 3]` | subprocessor list is on the public trust page or in the DPA; <!-- pending-verification: extract list and map against our security peer's catalogue --> |

**Benchmark panel — Fireworks (§5.1 methodology)**

| Window | TTFT p50 | TTFT p95 | E2E p95 for 1k batch | Error rate | Cosine mean (vs. self-host) |
|---|---|---|---|---|---|
| 03:00 UTC | `[SIMULATED: 180 ms]` | `[SIMULATED: 420 ms]` | `[SIMULATED: 48 s]` | `[SIMULATED: 0.3 %]` | `[SIMULATED: 0.987]` |
| 09:00 UTC | `[SIMULATED: 210 ms]` | `[SIMULATED: 530 ms]` | `[SIMULATED: 55 s]` | `[SIMULATED: 0.5 %]` | `[SIMULATED: 0.987]` |
| 15:00 UTC | `[SIMULATED: 220 ms]` | `[SIMULATED: 560 ms]` | `[SIMULATED: 58 s]` | `[SIMULATED: 0.4 %]` | `[SIMULATED: 0.987]` |

**Open questions for Fireworks sales / support**

1. Enterprise price lock period and renewal structure.
2. EU-region processing availability for the embedding product.
3. Retention default and configurable retention on embedding requests.
4. Subprocessor list (signed version in DPA).
5. SLA credit schedule specifics; whether batch-window outages trigger
   credits the same as sustained outages.

### 4.2 Together AI

> **Product:** Together AI serverless embedding endpoints.
> **URL:** https://www.together.ai
> **Documentation reviewed:** https://docs.together.ai, https://www.together.ai/pricing
> **Eval date:** 2026-11-18.
> **Eval owner:** `[SIMULATED: Platform Lead]`.

**Scoring highlights** (abbreviated vs. Fireworks; full table in the
repo evidence directory).

| Sub-dimension | Raw score | Notes |
|---|---|---|
| SLA — marketing uptime | `[SIMULATED: 4]` | vendor public page target |
| SLA — historical uptime | `[SIMULATED: 4]` | status.together.ai archived incident history <!-- pending-verification --> |
| Price — posted rate | `[SIMULATED: 5]` | https://www.together.ai/pricing; <!-- pending-verification: capture exact rate --> |
| Price — renewal | `[SIMULATED: 3]` | same shape as Fireworks on serverless |
| Feature parity — benchmark TTFT p95 | `[SIMULATED: 4]` | `[SIMULATED: 450 ms]` |
| Feature parity — benchmark E2E p95 | `[SIMULATED: 4]` | `[SIMULATED: 51 s]` within SLO |
| Feature parity — cosine | `[SIMULATED: 4]` | `[SIMULATED: 0.986]` mean; `[SIMULATED: 0.950]` p5 |
| Feature parity — error rate | `[SIMULATED: 4]` | `[SIMULATED: 0.4 %]` |
| Feature parity — API shape stability | `[SIMULATED: 3]` | OpenAI-compatible shape; one stated deprecation in the 12-month window <!-- pending-verification --> |
| Exit cost | `[SIMULATED: 3]` | similar to Fireworks |
| Data boundary — region | `[SIMULATED: 4]` | US primary; EU option mentioned <!-- pending-verification --> |
| Data boundary — training default | `[SIMULATED: 4]` | opt-out of training per public docs <!-- pending-verification --> |
| Data boundary — subprocessors | `[SIMULATED: 3]` | public trust page lists subprocessors; cross-check against security peer <!-- pending-verification --> |

**Benchmark panel — Together**

| Window | TTFT p95 | E2E p95 | Error rate | Cosine mean |
|---|---|---|---|---|
| 03:00 UTC | `[SIMULATED: 450 ms]` | `[SIMULATED: 51 s]` | `[SIMULATED: 0.4 %]` | `[SIMULATED: 0.986]` |
| 09:00 UTC | `[SIMULATED: 530 ms]` | `[SIMULATED: 58 s]` | `[SIMULATED: 0.5 %]` | `[SIMULATED: 0.986]` |
| 15:00 UTC | `[SIMULATED: 550 ms]` | `[SIMULATED: 59 s]` | `[SIMULATED: 0.4 %]` | `[SIMULATED: 0.986]` |

**Open questions for Together sales / support**

1. Enterprise price-lock terms.
2. EU-region processing for embedding endpoint.
3. Data retention default.
4. SLA credit schedule.
5. Model-version upgrade cadence — does an upstream `bge-m3` → `bge-m4`
   upgrade on the vendor side trigger a notice window?

### 4.3 Baseten

> **Product:** Baseten dedicated-endpoint serving (own open-weights
> model deployment).
> **URL:** https://www.baseten.co
> **Documentation reviewed:** https://docs.baseten.co, https://www.baseten.co/pricing
> **Eval date:** 2026-11-18.
> **Eval owner:** `[SIMULATED: Platform Lead]`.

Baseten is structurally different: rather than a shared serverless
endpoint, a dedicated endpoint for the embedding model is deployed and
billed by compute. That changes the scoring:

| Sub-dimension | Raw score | Notes |
|---|---|---|
| SLA — uptime | `[SIMULATED: 4]` | status.baseten.co <!-- pending-verification --> |
| SLA — dedicated-endpoint failure mode | `[SIMULATED: 3]` | failure mode differs from serverless: cold starts, instance replacement |
| Price — compute-based | `[SIMULATED: 3]` | `[SIMULATED: $ / GPU-hour]` posted; cost at this volume comparable to serverless if right-sized, higher if idle; <!-- pending-verification: measure right-sizing --> |
| Price — scale-to-zero availability | `[SIMULATED: 4]` | dedicated-endpoint scale-to-zero reduces idle; batch workload shape fits |
| Feature parity — TTFT p95 (dedicated, warm) | `[SIMULATED: 4]` | `[SIMULATED: 350 ms]` with dedicated warm endpoint |
| Feature parity — TTFT p95 cold start | `[SIMULATED: 2]` | `[SIMULATED: several seconds first request]`; relevant if scale-to-zero used |
| Feature parity — E2E p95 at batch window | `[SIMULATED: 4]` | `[SIMULATED: 45 s]` with warm endpoint |
| Feature parity — cosine | `[SIMULATED: 5]` | identical model checkpoint deployable; cosine ≈ 1 in theory <!-- pending-verification: confirm against real run --> |
| Exit cost — model portability | `[SIMULATED: 4]` | deploying the same checkpoint elsewhere is straightforward |
| Exit cost — API coupling | `[SIMULATED: 3]` | Baseten's API shape is Baseten-specific; adapter required for exit |
| Data boundary — region | `[SIMULATED: 4]` | US-based; EU region availability varies <!-- pending-verification --> |
| Data boundary — training default | `[SIMULATED: 5]` | dedicated-endpoint model; vendor does not have shared access to customer traffic |
| Data boundary — subprocessors | `[SIMULATED: 4]` | smaller subprocessor footprint than serverless vendors |

**Benchmark panel — Baseten (warm endpoint)**

| Window | TTFT p95 | E2E p95 | Error rate | Cosine mean |
|---|---|---|---|---|
| 03:00 UTC | `[SIMULATED: 350 ms]` | `[SIMULATED: 45 s]` | `[SIMULATED: 0.2 %]` | `[SIMULATED: 1.000]` |
| 09:00 UTC | `[SIMULATED: 370 ms]` | `[SIMULATED: 47 s]` | `[SIMULATED: 0.3 %]` | `[SIMULATED: 1.000]` |
| 15:00 UTC | `[SIMULATED: 380 ms]` | `[SIMULATED: 48 s]` | `[SIMULATED: 0.3 %]` | `[SIMULATED: 1.000]` |

**Open questions for Baseten sales / support**

1. Dedicated-endpoint right-sizing recommendation for 2 B tokens / month
   nightly batch pattern.
2. Cold-start behaviour if scale-to-zero is enabled; whether the batch
   window's first request pays the cold penalty.
3. EU region availability on dedicated endpoints.
4. Model artefact handling and key-management for the deployed checkpoint.
5. SLA language for dedicated endpoints vs. a shared serverless SLA.

---

## 5. Deliverable C — benchmark on our traffic

### 5.1 Methodology

Reused from §3 above; the benchmark produced one sampled panel per
vendor per time-of-day window (nine runs total, 300 requests each).

### 5.2 Reproducibility

All three vendors are hit with the same 100-prompt sample
(`bench/samples/tier-c-embed-100.jsonl`) with the same request shape
through a shared script (`bench/run-vendor-embed.sh`). The script:

1. Reads prompts from the sample file.
2. Sends them concurrently at concurrency = 10 to match expected batch
   fan-out.
3. Records TTFT, E2E, response body, HTTP status.
4. Writes raw JSON to `evidence/vendor-bench-<vendor>-<window>.json`.
5. A second script (`bench/score-equivalence.py`) computes cosine against
   the self-host baseline captured on the same prompts the day before.

The evidence files are archived next to this SOLUTION.md's rendered
version. Benchmark re-runs at the next quarterly review are cheap
because the sample, scripts, and directory layout are fixed.
<!-- pending-verification: scripts and sample files are referenced but not committed in this learning artefact; instantiate them in the real scorecard repo. -->

### 5.3 Panel comparison

The comparison panel brings §4's per-vendor tables onto one page.

| Metric | Fireworks | Together | Baseten (warm) |
|---|---|---|---|
| TTFT p95 (03:00 UTC) | `[SIMULATED: 420 ms]` | `[SIMULATED: 450 ms]` | `[SIMULATED: 350 ms]` |
| E2E p95 (1 k-doc batch) | `[SIMULATED: 48 s]` | `[SIMULATED: 51 s]` | `[SIMULATED: 45 s]` |
| Error rate | `[SIMULATED: 0.3 %]` | `[SIMULATED: 0.4 %]` | `[SIMULATED: 0.2 %]` |
| Cosine mean vs. self-host | `[SIMULATED: 0.987]` | `[SIMULATED: 0.986]` | `[SIMULATED: 1.000]` |
| Cosine p5 vs. self-host | `[SIMULATED: 0.952]` | `[SIMULATED: 0.950]` | `[SIMULATED: 0.999]` |

Observations for the cover memo:

- All three meet the 60-s batch SLO on the simulated traffic.
- Baseten's dedicated endpoint shows highest cosine equivalence (same
  checkpoint); serverless vendors show the expected floating-point /
  kernel-difference drift at the fourth decimal.
- Baseten's cold-start risk is not visible on a warm panel; it has to be
  called out separately (§4.3).

---

## 6. Deliverable D — summary comparison

### 6.1 Summary table

Weighted totals use the weights locked in §2.3: SLA 0.20, price 0.30,
feature parity 0.25, exit cost 0.15, data boundary 0.10. Per-axis scores
below are the weighted average of the axis's sub-dimensions.

| Vendor | SLA (0.20) | Price (0.30) | Feature parity (0.25) | Exit cost (0.15) | Data boundary (0.10) | Weighted total | Rec. |
|---|---|---|---|---|---|---|---|
| Fireworks | `[SIMULATED: 3.7]` | `[SIMULATED: 4.0]` | `[SIMULATED: 3.8]` | `[SIMULATED: 3.0]` | `[SIMULATED: 3.5]` | `[SIMULATED: 3.70]` | shortlist for primary |
| Together | `[SIMULATED: 4.0]` | `[SIMULATED: 4.0]` | `[SIMULATED: 3.8]` | `[SIMULATED: 3.0]` | `[SIMULATED: 3.7]` | `[SIMULATED: 3.77]` | shortlist for primary |
| Baseten | `[SIMULATED: 3.5]` | `[SIMULATED: 3.5]` | `[SIMULATED: 3.7]` | `[SIMULATED: 3.5]` | `[SIMULATED: 4.3]` | `[SIMULATED: 3.70]` | fallback / secondary |

### 6.2 Per-axis commentary

- **SLA.** All three vendors have public status pages; none publishes a
  single aggregate uptime that cleanly covers the embedding endpoint at
  the granularity we need. Together's extracted historical uptime
  slightly edges Fireworks and Baseten on the batch window; at this
  scoring resolution the three are effectively tied.
- **Price.** Fireworks and Together are the closest-matched serverless
  options; Baseten's dedicated-endpoint price depends on right-sizing
  and on scale-to-zero. All three beat self-host materially (exercise 02
  §6.6) at year 1; year-3 renewal exposure is similar for Fireworks and
  Together.
- **Feature parity.** Baseten runs an identical checkpoint, so cosine is
  highest. Fireworks and Together show expected minor drift at the
  fourth decimal; both pass the 0.985 / 0.95 threshold in §3.2. The
  benchmark's cold-start caveat for Baseten is handled by right-sizing
  the dedicated endpoint, not by scale-to-zero, during the batch window.
- **Exit cost.** Baseten scores higher here: the model is the same
  checkpoint and is portable. API-shape coupling is comparable across
  the three.
- **Data boundary.** Baseten's dedicated-endpoint shape minimises
  shared-infrastructure exposure; its lower subprocessor footprint
  modestly outscores the serverless vendors.

---

## 7. Deliverable E — decision cover memo

### 7.1 Cover memo

> **Scorecard decision — tier-C batch embedding — 2026-11-20**
>
> **Workload cell this scorecard serves** (from exercise 02 RFC):
> `tier-C embedding × eu-west-1 × batch window` and `tier-C embedding
> × eu-west-1 × overflow outside batch window`.
>
> **Ranked vendors (weighted total):**
>
> 1. Together AI — 3.77
> 2. Fireworks AI — 3.70 (tied with Baseten on total)
> 2. Baseten — 3.70 (tied)
>
> **Specific placement proposed.** Primary vendor for the EU cells:
> **Together AI**. Baseten listed as the **secondary / failover** vendor
> for the same cells, held under a thin adapter so a cut-over does not
> require a second migration. Fireworks remains in the shortlist for a
> re-evaluation at the next quarterly review.
>
> **Rationale.** Together and Fireworks are within 2 % of each other on
> weighted total; the tie-breaker is (a) Together's slightly higher
> extracted historical uptime on the batch-window window and (b) data-
> boundary posture (subprocessor list). Baseten is a strong technical
> option but its dedicated-endpoint pricing is cost-competitive only at
> right-sizing; it is held as the secondary for failover and for
> dedicated-endpoint use cases if a regulated workload emerges.
>
> **Sponsor sign-off.** `[SIMULATED: Sponsor name]`, 2026-11-22.
>
> **When does this change? Triggers that reopen the scorecard:**
>
> - Together's renewal pricing moves beyond the +25 % year-3 ceiling
>   modelled in exercise-02 §6.3.
> - Historical uptime on Together's batch window drops below 99.0 % on
>   the trailing 90 days.
> - A DPA term changes materially at renewal (subprocessor region,
>   retention default, training-on-data default).
> - Cosine equivalence on the retrieval benchmark drops by > 3 % mean
>   or > 5 % p5 on a quarterly re-run.
> - A vendor outside the current shortlist (Groq, Cerebras, hyperscaler
>   inference offerings, Modal) enters the embedding market with a
>   materially different price or capability shape.
> - The workload class changes (regulated-data flag added, volume
>   doubles / halves).
>
> **Scorecard refresh cadence.** Quarterly at the exercise-01 migration-
> roadmap review; refresh is cheap because the benchmark scripts and
> sample are durable.

---

## 8. Deliverable F — DPA / contract review notes

### 8.1 Why this deliverable is a private appendix

The notes live alongside the scorecard but are **not published** with
the three-page artefact because they sometimes carry quoted language
from vendor private documents (enterprise DPA, master service
agreement). They are referenced here so the review trail exists.

### 8.2 Per-vendor DPA notes (from public documents only)

**Fireworks AI**

- **Logging default.** Public privacy policy describes request logging
  for debugging; retention period and configurability require DPA.
  <!-- pending-verification: DPA / enterprise ToS -->
- **Training-on-data default.** Public privacy policy states customer
  data is not used for model training on the standard product.
- **Named subprocessors.** Subprocessor list expected on the public
  trust page; cross-check against the security peer's catalogue.
- **Data region flexibility.** US primary; EU option on request.
- **Termination clauses.** Public ToS permits termination with notice;
  enterprise contract would re-specify.
- **Red-line targets for a real contract review:** price-lock duration,
  SLA credit schedule, subprocessor change notice, retention defaults,
  data residency options.

**Together AI**

- **Logging default.** Public privacy policy describes request logging.
- **Training-on-data default.** Opt-out of training referenced in public
  docs; enterprise ToS re-affirms.
- **Named subprocessors.** On public trust page; cross-check.
- **Data region flexibility.** US primary; EU option mentioned.
- **Termination clauses.** Standard commercial ToS; see enterprise.
- **Red-line targets:** same shape as Fireworks.

**Baseten**

- **Logging default.** Dedicated-endpoint deployment; logging is
  largely a customer-configurable surface.
- **Training-on-data default.** N/A — vendor does not have shared
  access to the dedicated model's traffic by default.
- **Named subprocessors.** Smaller subprocessor footprint; cross-check
  against the security peer's catalogue.
- **Data region flexibility.** Varies by deployment target.
- **Termination clauses.** Standard commercial ToS; see enterprise.
- **Red-line targets:** instance right-sizing recommendation, cold-start
  SLA on the batch window.

### 8.3 Marker

> `<!-- pending-verification: contract-level review required before contract
> signature; this evaluation reads the public terms only. Red-line
> targets above are the inputs to that review. -->`

---

## Implementation

The scorecard is a decision artefact, but turning the decision in §7 (adopt
Fireworks AI as primary, hold Together AI as standby) into a running
platform looks like:

- **Benchmark harness in the platform repo.** The traffic sample
  (`bench/samples/tier-c-embed-100.jsonl`), scoring rules (§3.2), and
  scheduling policy (§3.3) are checked in as a replayable harness
  (`bench/run-scorecard.py`) so a refresh at the next quarterly
  contract re-signature is a one-command re-run rather than a bespoke
  effort. Results write to `bench/results/<YYYY-MM-DD>-<vendor>.json`
  for auditability.
- **Evidence directory under version control.** Each per-vendor page
  (§4) points at `evidence/<vendor>/<artefact>.md` — status-page
  extraction, posted-rate screenshot, DPA note. Point-in-time snapshots
  capture that prices and SLAs were as stated on the scoring date; the
  weights lock in §2.3 is recorded alongside.
- **Primary + standby routing.** The in-house router gains two tier-C
  backends (`tier-c-embed-fireworks-primary`,
  `tier-c-embed-together-standby`) with health-check driven failover.
  The standby is exercised monthly to keep the integration warm so a
  decision reversal is an operational rather than a project event.
- **Scorecard refresh cadence.** The quarterly contract re-signature
  ritual (§7 of exercise 01) carries a "re-run scorecard" ticket that
  re-executes the harness, re-captures evidence, and either reaffirms
  the ranking or triggers the reverse-decision path from §7.1 of
  exercise 02.
- **DPA review handoff.** §8's red-line targets per vendor are a
  pre-populated review brief for the security peer; the brief is
  attached to a security-contract-review ticket (Deliverable D of
  exercise 05) at scorecard sign-off, not later.
- **Dashboard wiring for reverse triggers.** The per-axis red-flags
  identified in §6.2 (vendor p95 TTFT, error rate, sustained posted-rate
  increase) become named alert rules on the module 110 dashboard; each
  routes to the inference platform PM for re-opening the scorecard.

The scorecard is the written decision; the list above is what makes the
decision reproducible, refreshable, and visible on cadence.

---

## 9. Validation steps for a learner submission

1. **Workload scoping memo** names the workload class, typical prompt
   and output shapes, SLO targets, volume, compliance posture, the
   weighting *locked before scoring*, the shortlist, and the excluded
   vendors with reasons.
2. **Per-vendor detail pages** have at least three vendors; each page
   has the five-axis scoring table, sub-dimensions, evidence pointers,
   and open questions for the vendor.
3. **Benchmark on our traffic** has a methodology (sample, scheduling,
   scoring rules), a per-vendor panel with TTFT / E2E / error rate /
   equivalence, and reproducible scripts (even if stubbed).
4. **Summary comparison** has a single table of vendors × axes +
   weighted total + recommendation column, and a short per-axis
   commentary paragraph.
5. **Decision cover memo** names the workload cells served, ranks
   vendors, proposes a specific placement, has a sponsor sign-off line
   (simulated acceptable), and names triggers that reopen the scorecard.
6. **DPA / contract review notes** cover logging default, training-on-
   data default, named subprocessors, data region, termination for each
   vendor; the limitation of public-docs-only is explicitly marked.
7. **Weighting lock date** is explicit and precedes the scoring date in
   §4.
8. **Evidence pointers**: every score has one; no score defended by
   marketing alone.

## 10. Rubric / review checklist

Score each item 0 / 1 / 2. Passing ≥ 16 / 20.

| # | Item | Points |
|---|---|---|
| 1 | Workload scoping memo scopes the workload and locks the weighting before scoring; excluded vendors named with reasons. | 2 |
| 2 | At least three real vendors; per-vendor pages have all five axes with evidence pointers. | 2 |
| 3 | Benchmark methodology is reproducible (sample, scoring rules, scripts named). | 2 |
| 4 | Per-vendor benchmark panel records TTFT / E2E / error rate / equivalence at ≥ 2 time windows. | 2 |
| 5 | Summary table has weighted totals computed with the locked weighting; per-axis paragraph explains weighting effect. | 2 |
| 6 | Decision cover memo has ranked vendors, per-cell placement, sign-off, and triggers to reopen. | 2 |
| 7 | DPA notes cover logging / training / subprocessors / region / termination per vendor; limitations of public-docs-only marked. | 2 |
| 8 | Weighting lock date precedes scoring date; audit trail preserved. | 2 |
| 9 | Simulation disclosed where applicable; no silent fabrication; `<!-- pending-verification: ... -->` markers used appropriately. | 2 |
| 10 | Benchmark scripts and sample files referenced (even if stubbed) so the scorecard refresh is cheap. | 2 |

## 11. Common mistakes

- **Scoring the vendor's blog.** Every score should be tied to a
  benchmark run, a status-page extract, a DPA excerpt, or a signed
  quote. "They say they're great at this" is 1-star evidence.
- **Marketing uptime scored as SLA.** The marketing number is the
  vendor's self-image; the SLA sub-score should be historical uptime +
  credit schedule + response SLA.
- **Weighting after scoring.** Unconsciously picks the preferred answer.
  Lock the weighting date in the artefact.
- **Shortlist too large.** Six vendors is six 1-page detail pages; some
  will be weakly evaluated. Three is enough for most workload classes.
- **First-year pricing locked in.** Managed inference almost always has
  a promotional year-1 rate. Price year-2 and year-3 explicitly.
- **DPA unread; privacy page substituted.** The privacy page is public;
  the DPA is contractual. The DPA is where logging retention and
  subprocessor-change notice live. Mark public-docs-only.
- **Benchmark on vendor-provided chart.** Benchmark on *our* traffic.
  The vendor's chart is pre-filtered.
- **Cover memo without triggers.** "Our recommendation is X" with no
  "we revisit if Y" is a scorecard that is stale before signing.
- **Excluded vendors not named.** Omitting a plausible vendor without a
  reason invites the sponsor to ask "did you consider Groq?" after the
  decision is made. Say so in the shortlist.

## 12. References

- Vendor public pricing / trust pages (captured at eval time; links are
  pointers, not quoted material):
  - https://fireworks.ai/pricing
  - https://www.together.ai/pricing
  - https://www.baseten.co/pricing
  <!-- pending-verification: capture point-in-time snapshots for the evidence directory; prices move -->
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework)
  — the scorecard is a MAP and MEASURE artefact in the RMF language:
  it characterises the vendor option (MAP) and measures evidence
  against each axis (MEASURE).
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/)
  — ML10 (Insufficient Supply Chain Security) is the axis the DPA notes
  and the subprocessor review directly address.
- [mitre-atlas](https://atlas.mitre.org/) — AML.T0010
  (supply-chain compromise of a third-party model) is why the SHA / artefact
  provenance review from module 108 ex-04 extends to vendor-hosted
  models; the scorecard's data-boundary and exit-cost axes contribute
  to the overall supply-chain posture.
- Module 110 chapter 04 — cost per 1M tokens; the arithmetic in the
  price axis uses it.
- Module 113 chapter 01 — the response-diff pattern (cosine, refusal-
  rate, schema conformance) used in §3 and §5.
- Chapter 03 of this module — the five-axis scorecard shape this
  solution concretises.
- Exercise 02 of this module — the RFC whose vendor shortlist this
  scorecard serves.
