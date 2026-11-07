# Exercise 05 — Platform Charter Authoring (Reference Solution)

Paired exercise: [`exercise-05-platform-charter-authoring.md`](../../../../ai-infra-inference-engineer-learning/lessons/mod-101-inference-engineering-position/exercises/exercise-05-platform-charter-authoring.md). Anchor chapter: [Chapter 05 — Platform Charter](../../../../ai-infra-inference-engineer-learning/lessons/mod-101-inference-engineering-position/05-platform-charter.md). Companion artefact: Exercise 01 deferral table.

## 1. Solution overview

The exercise asks learners to author the inference-platform charter described in Chapter 05 against a synthetic (or real) organisation shape, then resolve three ambiguous week-one tickets by citing the charter. The charter has seven required sections and three resolution appendices. Target length is ≤ 2 pages of Markdown.

Below is the charter in full, followed by the three ticket resolutions, the validation steps, the rubric, and the common mistakes.

The target organisation shape (from the exercise spec):

- One ML Engineering team shipping ~15 open-weight models per year.
- Three AI Engineering product teams (chatbots, agents, RAG).
- A separate SRE org and a separate security org.
- Two cloud regions, mixed Hopper and Blackwell fleet, with a managed-vendor fallback on the roadmap.
- A board / finance review asking for cost per 1 M tokens and per business unit.

Any SLO target that depends on the actual workload is marked **`org-set`** rather than filled with a fabricated number. Those slots are placeholders the team fills during its first SLO review. The charter is only pin-worthy once every `org-set` is resolved.

## 2. Implementation — Worked answer: the charter

### 2.1 Mission

The Inference Platform runs the production GPU-serving fleet for all models consumed by this organisation's AI products across two cloud regions. We provide a managed OpenAI-compatible API over a curated catalogue of models, with published SLOs for TTFT, TPOT, availability, and cost per 1 M tokens, and with change-management discipline for model and engine upgrades. We do not design applications, train or fine-tune models, or decide which models are permitted; we deliver the serving plane those decisions assume. A product or infra peer reading this paragraph should know within three sentences whether to open a ticket with us or with a sibling team.

### 2.2 Services we provide

- Managed OpenAI-compatible endpoints for the curated model catalogue (chat, completions, embeddings).
- Streaming responses with documented backpressure, SSE boundaries, and client-disconnect handling.
- Structured-output enforcement (JSON Schema, regex, grammar) and OpenAI-compatible tool-calling.
- Multi-adapter LoRA serving against a shared base model.
- Per-tenant rate limits, token budgets, idempotency keys, and request-id propagation.
- Canary / shadow / blue-green rollout of model and engine version changes with SLO-based automatic rollback.
- Fleet autoscaling against inference-shaped custom metrics (queue depth, prefill-tokens-per-second, scheduler utilisation).
- Observability stack specific to inference signals (TTFT / TPOT / ITL / E2E / prefix-cache hit rate / preemption rate / cost per 1 M tokens).
- Managed-vendor fall-back for overflow traffic (planned; triggered only under published burst conditions).

### 2.3 Non-services

The following are explicitly *not* services we provide. Each row names the owner.

| Not our service | Owner |
|---|---|
| CUDA / fused-kernel implementation | `ai-infra-performance` (peer, L35, AI Infra) |
| Model training, fine-tuning, or distillation | ML Engineering (`fine-tuning-engineer`, L30; `pretraining-engineer`, L35) |
| Model artefact registry, lineage, or signing | `ai-infra-ml-platform` (peer, L30, AI Infra) |
| Application-layer eval and release-gate verdict (model behaviour) | `ai-eval-engineer` (peer, L30, AI Engineering) |
| Benchmark / accuracy methodology for model releases | `model-evaluation-engineer` (peer, L30, ML Engineering) |
| Prompt design, agent-loop logic, RAG pipeline construction | AI Engineering family consumers (L25) |
| Adversarial defence at the API edge (prompt injection, model extraction, weight supply-chain) | `ai-infra-security` (peer, L35, AI Infra) |
| SRE methodology, incident-command discipline, error-budget math | `ai-infra-sre` (peer, L30, AI Infra) |
| Generic CI/CD plumbing (GitHub Actions, Argo, Flux) | `ai-infra-mlops` (peer, L25, AI Infra) |
| Kubernetes, Terraform, cloud base platform | `ai-infra-engineer` (prerequisite, L20) |
| Legal, compliance, data-residency, regulatory sign-off | AI Governance family (out-of-scope for inference) |
| Business decisions about product scope, pricing, roadmap | AI Startup family / product leadership (out-of-scope) |

### 2.4 SLOs we commit to

| SLO | Target | Measurement method | Window |
|---|---|---|---|
| Availability (per endpoint) | `org-set` (e.g., ≥ 99.9 % successful-response ratio) | Ratio of 2xx responses to total requests at the API gateway | 30-day rolling |
| TTFT P99 (per model class + prompt-length class) | `org-set` (e.g., ≤ 400 ms P99 for the chat class at ≤ 4 k input tokens) | Histogram `inference_ttft_ms_bucket` recorded at the API gateway; rollup per model class | 7-day rolling |
| TPOT P99 (per model class) | `org-set` (e.g., ≤ 40 ms per output token P99 for the chat class) | Histogram `inference_tpot_ms_per_token_bucket`; per-request mean-over-output-tokens | 7-day rolling |
| E2E P99 (per model class + output-length class) | `org-set` | Histogram `inference_e2e_latency_ms_bucket` | 7-day rolling |
| Cost per 1 M input tokens (per model class) | `org-set` | Cost model divides GPU-hour + supporting-infra cost by rolled-up input tokens served | Monthly |
| Cost per 1 M output tokens (per model class) | `org-set` | Same cost model, applied to output tokens | Monthly |
| Canary rollback latency | ≤ 10 minutes from auto-detected regression to full rollback | Time between SLO-breach signal and completion of blue-green or canary-ramp reversal | Per incident |
| Managed-vendor failover activation latency | `org-set` (e.g., ≤ 5 minutes) | Time from burst condition met to fallback-pool serving traffic | Per incident |

**SLOs we refuse.** The inference platform does *not* own any of the following; attempting to make these inference SLOs is a scoping error.

- Model-behaviour quality: factuality, hallucination rate, toxicity, bias. Owner: `ai-eval-engineer` for application eval; `model-evaluation-engineer` for benchmark methodology.
- Application-task completion: correct tool call count, correct citation rate, correct booking outcome. Owner: AI Engineering family consumers.
- Prompt-injection detection rate at the API edge. Owner: `ai-infra-security`.
- End-user business metrics: conversion, retention, satisfaction scores. Owner: product leadership.

### 2.5 Request-shape and protocol contract

- **Endpoints:** `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` (OpenAI-compatible). Vendor-specific endpoints require an exception and a published deprecation window.
- **Streaming:** Server-Sent Events primary. WebSocket is not offered; backpressure and client-disconnect handling are documented at the SSE level.
- **Structured output:** JSON Schema, regex, and grammar guides supported via the engine's constrained-decoding backend (Outlines / XGrammar). Guarantee: when `response_format` is set and the schema parses, the output is a valid instance of the schema; invalid-output events are counted and alerted on.
- **Tool calling:** OpenAI-compatible shape. Parallel tool calls supported. Streaming during tool-call phases follows the OpenAI reference.
- **Rate limits, backpressure, retries:** Per-tenant RPS and TPS caps with published 429 semantics; idempotency keys supported on all mutating endpoints.
- **Authentication and tenant isolation:** Ingress authentication via the organisation's auth layer. Tenant isolation enforced at the inference layer — KV-cache scope, prefix-cache scope, and adapter-routing scope are per-tenant.
- **Request-id propagation:** All requests carry a request-id header that is written to logs, traces, and all downstream engine calls.

### 2.6 Change-management and rollout discipline

- **Model-version changes:** canary ramp against TTFT / TPOT / availability SLOs with shadow-traffic diffing where feasible; blue-green for models over a stated parameter threshold; eval-engineer release-gate verdict required before canary promotion.
- **Engine-version changes (vLLM / SGLang / TRT-LLM / Dynamo):** canary with SLO gate; always with a prepared rollback runbook; a chunked-prefill or scheduler-knob change is treated as an engine-version change.
- **Hardware / firmware / driver changes (DCGM, GPU operator, driver pin):** maintenance window + canary; never a full-fleet rolling restart without SLO gating.
- **Model deprecation:** published grace period (`org-set`, e.g., 30 days minimum), a migration guide for each consumer, and an on-call rota to assist the migration.
- **Emergency rollback authority:** the inference on-call engineer can trigger a full rollback within 10 minutes without a sign-off chain. The emergency rollback path is tested at least quarterly.

### 2.7 Peer interface contracts

| Peer | They hand us | We hand them | Escalation path |
|---|---|---|---|
| `ai-infra-performance` (L35) | Engine releases, kernel changes, performance regression analyses | Production repros, SLO comparisons across engine versions, rollback plans | Joint on-call bridge during kernel-induced regressions; `ai-infra-architect` (L40) above that |
| `ai-infra-ml-platform` (L30) | Signed model artefacts with resource-profile metadata and lineage | Deployment feedback (did the artefact serve under SLO), post-deployment resource profiles | ml-platform team-lead; `ai-infra-architect` (L40) for cross-team |
| `ai-infra-sre` (L30) | SRE methodology, incident templates, error-budget framework | Inference-signal SLIs (TTFT / TPOT / ITL / cache-hit / preemption) plugged into the common framework | Shared incident commander on cross-cutting outages |
| `ai-infra-security` (L35) | WAF rules, weight-signing policy, enclave requirements, prompt-injection defence at the gateway | Traffic shape, admission logs, attack-surface diagrams, tenant-isolation configuration | Joint on-call bridge during security incidents touching serving |
| `ai-infra-mlops` (L25) | Generic CI/CD plumbing (Argo / Flux / Actions) | vLLM Helm chart, canary runbook, rollback runbook | mlops team-lead; `ai-infra-architect` (L40) |
| `ai-eval-engineer` (L30) | Release-gate verdict with evidence | Serving-level canary that respects the verdict | Shared release channel |
| `model-evaluation-engineer` (L30) | Benchmark methodology and benchmark-delta releases | Production reconciliation of benchmark deltas against serving metrics | ML Engineering release manager |
| AI Engineering family consumers (L25) | Traffic shape, latency budget, model-mix requirements | Endpoints with documented SLAs, cost telemetry, incident status | Shared consumer-impact on-call rota |

## 3. Ticket-resolution appendices

### 3.1 Ticket 1 — "We want you to own response-quality SLOs for the chatbot — just TTFT isn't enough."

- **Owning role:** `ai-eval-engineer` (L30, AI Engineering), with us owning the serving-level canary that respects their verdict.
- **Charter citation:** Non-services table — row "Application-layer eval and release-gate verdict (model behaviour)." SLOs-we-refuse list — "Model-behaviour quality."
- **Response:** "We own serving SLOs (TTFT, TPOT, availability, cost); the application-quality SLO for the chatbot is owned by `ai-eval-engineer`. We will wire a canary that gates promotion on their verdict, and we will add the verdict-pass/fail signal to the rollout dashboard. For the application SLO target itself, please open a ticket with the ai-eval team; we'll co-own the canary wiring."

### 3.2 Ticket 2 — "Do you own the WAF rules that reject prompt-injection patterns, or does security?"

- **Owning role:** `ai-infra-security` (L35, AI Infra).
- **Charter citation:** Non-services table — row "Adversarial defence at the API edge (prompt injection, model extraction, weight supply-chain)." Peer-interface row for `ai-infra-security`: "They hand us: WAF rules, weight-signing policy, enclave requirements."
- **Response:** "`ai-infra-security` owns the WAF rules. We will carry telemetry on whatever rule they publish (rule-hit rate, false-positive-rate proxy where feasible) and will participate in a joint on-call bridge during any injection-related incident. If the rule needs to be enforced *inside* the engine (e.g., per-tenant KV-cache scoping to prevent a cross-tenant leak), that is ours — reach out when the scope crosses the engine boundary."

### 3.3 Ticket 3 — "Finance wants a cost-per-conversation breakdown by business unit."

- **Owning role (as currently written):** out-of-scope — finance / product.
- **Charter gap identified:** the charter commits to `cost per 1 M input tokens` and `cost per 1 M output tokens` per model class, but not cost per *conversation* and not per business unit. Mapping tokens to conversations is a product-scope decision (what counts as a conversation varies per product); mapping to business units requires product-provided tagging.
- **Charter edit (applied):** Append to the Services section the following row — "**Token-level cost telemetry with tenant and tag dimensions.** The platform emits `inference_cost_usd_per_request` and `inference_input_tokens` / `inference_output_tokens` dimensioned by tenant and by product-supplied tags; aggregation to cost-per-conversation and per-business-unit is the consumer's responsibility." And update the Non-services table with a new row — "**Product-level cost attribution (per conversation, per BU, per campaign).** Owner: consumer / product team."
- **Response:** "We can provide per-request and per-token cost telemetry tagged by tenant and by any product-supplied tag (e.g., business-unit). Aggregating tokens into a conversation and rolling up to business units is a product-side computation. If finance will give us the tag mapping they want (BU → tag), we can ship the per-tag dashboard; the per-conversation roll-up still lives in the product team."

## 4. Validation steps

1. **Seven-section audit.** Open the charter. Count the sections. If any of Mission / Services / Non-services / SLOs / Request-shape / Change-management / Peer interface is missing, the charter is not ready.
2. **Non-service peer sign-off.** Walk each non-services row with the named peer. If they reject the row for their track, negotiate the wording and re-pin.
3. **SLO target fill-in.** Every `org-set` must be replaced with a concrete number before the charter is pinned. Numbers are produced by the first SLO review with product, SRE, and finance in the room.
4. **SLOs-we-refuse pressure test.** During the first product review, offer each refused SLO class to the correct peer explicitly. Confirm the peer accepts ownership in writing.
5. **Two-page fit test.** Export to PDF. If the charter exceeds two pages, cut adjectives and aspirational language; do not cut rows.
6. **Sibling-team five-minute test.** Hand the charter to a peer who has not read module 101. They must self-route a ticket from their inbox against the charter in ≤ 5 minutes.
7. **Ticket re-run.** Pick three ambiguous tickets from the team's last quarter. Confirm the charter resolves each without re-opening Chapter 05. Any ticket that cannot be resolved is a gap; add the row or section.

## 5. Rubric / review checklist

| # | Criterion | Pass signal |
|---|---|---|
| 1 | All seven sections present | Section-count = 7 |
| 2 | Non-services table resolves ≥ 2 of 3 week-one tickets without further edits | Verified in 3.1–3.3 |
| 3 | SLO table has concrete targets (numbers, not placeholders) and a measurement method per row | All `org-set` replaced before pin |
| 4 | SLOs-we-refuse sub-section declines ≥ 1 SLO class peers might otherwise assign to the platform | Four classes refused above |
| 5 | Peer-interface contracts table has a non-empty escalation path for every row | Row-by-row check |
| 6 | Three ticket-resolution appendices each name an owner and cite a specific section | Done in 3.1–3.3 |
| 7 | Sibling-team peer can self-route a ticket in ≤ 5 minutes | Dry-run |
| 8 | No aspirational language ("we will", "we aim to") remains in the charter | Scan complete |

## 6. Common mistakes

- **Writing Mission as marketing copy.** The mission must let a peer decide in three sentences whether to open a ticket with you. Drop adjectives.
- **Leaving `org-set` placeholders in the pinned version.** The charter without numbers is a template, not a contract. Run the SLO review before pinning.
- **Owning model-quality SLOs.** The single most common week-one trap. The SLOs-we-refuse section exists specifically to keep model-behaviour SLOs out of inference ownership.
- **Owning WAF rules.** The API gateway is security's territory; tenant isolation inside the engine is yours. Distinguish the two explicitly.
- **Attempting cost-per-conversation inside the platform.** Conversations are a product-scope concept; business units are a product-supplied tag. The platform emits token-level telemetry with dimensions; the roll-up belongs to the consumer.
- **No escalation path in the peer-interface table.** A one-way deferral without a wired escalation is a scoping wish, not a contract. Every row needs a bridge, a channel, or a named role.
- **Charter that only covers happy-path change management.** Emergency rollback authority and timing is where the charter earns its keep during a Sev-1.
- **Not re-negotiating the non-services table with peers.** The non-services table is only load-bearing if the named owners agree with their row. Walk each row with its owner.

## 7. References

- Chapter 05 — Platform Charter: `../../../../ai-infra-inference-engineer-learning/lessons/mod-101-inference-engineering-position/05-platform-charter.md`
- Chapter 01 — Role Scope on the Ladder: `../../../../ai-infra-inference-engineer-learning/lessons/mod-101-inference-engineering-position/01-role-scope-on-the-ladder.md`
- Exercise 01 solution (deferral table companion): `../exercise-01-role-scope-and-deferral-contract/SOLUTION.md`
- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> (anchors the API-edge and weight-supply-chain non-service rows).
- MITRE ATLAS — <https://atlas.mitre.org/> (taxonomy backing the prompt-injection / model-extraction row in non-services).
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> (frames the legal / compliance out-of-scope row and the cost-telemetry handoff to finance).
