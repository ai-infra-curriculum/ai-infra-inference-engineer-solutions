# Exercise 06 — Cross-Tenant KV-Sharing Safety Review

> Module: `mod-104-kv-cache-management`
> Exercise: `exercise-06-cross-tenant-kv-sharing-safety-review`
> Type: security / design review

## 1. Solution overview

The exercise asks the learner to **review a proposal (hypothetical or real) to share
KV-cache entries across tenants in a multi-tenant inference service** and produce a
safety recommendation. A "yes with these controls" or "no, and here is why" with
documented reasoning is the deliverable — both can be valid outcomes depending on
tenancy model, regulatory posture, and the workload's actual sharing potential.

Prefix caching across requests is an uncontroversial engineering win when all
requests belong to one trust domain (one tenant). Pushing that sharing *across*
tenants is attractive because system prompts and embedded context tend to repeat,
but it raises several concrete risks:

1. **Confidentiality leakage via cache contents.** KV values encode the exact tokens
   of the prompt. If a tenant can observe cached KV blocks (directly or via oracles
   like timing), another tenant's prompt text has leaked.
2. **Oracle via hit-rate / timing side channels.** Even without reading KV blocks,
   the time-to-first-token (TTFT) difference between a hit and a miss is observable.
   A malicious tenant can probe system-prompt or document content token by token by
   crafting prompts and observing TTFT.
3. **Cross-tenant cache poisoning.** If a shared cache stores "assistant" KV from
   prior turns, one tenant's content in those KV positions is used as retrieved
   context by another tenant's generation. Depending on how the engine stitches
   KV, this can shape another tenant's output.
4. **Policy drift.** Sharing couples deletion/retention semantics across tenants; a
   right-to-erasure request from one tenant cannot be met by purging only its
   own entries if those entries are identical to another tenant's.
5. **Regulatory mismatch.** Data residency, HIPAA-like, or contractual data-sharing
   restrictions may forbid any form of commingling, irrespective of the cryptographic
   properties of the cache.

These threats map to published catalogues as follows (names from the sources listed
at the top of this packet):

- **OWASP Machine Learning Security Top 10.** ML02 (Data Poisoning) and ML06
  (AI Supply Chain Attacks) are the closest; the classic enterprise OWASP Top 10
  categories A01 (Broken Access Control) and A04 (Insecure Design) also apply
  directly to multi-tenant resource sharing.
- **MITRE ATLAS.** Techniques around *Inference-time inference* and
  *Side-channel-based inference* describe the class of attack a timing oracle on a
  prefix cache enables; the ATLAS matrix is the right place to look for current
  technique IDs and case studies.
- **NIST AI RMF (AI 100-1).** The MANAGE and MAP functions require explicit
  documentation of context of use (tenancy model, data sensitivity) and of risks
  tolerated/mitigated — the design review itself is an RMF-aligned artifact.

Respect the source policy from the work packet: every factual claim below is tied
either to the named official sources, to clearly labeled local engine documentation
(vLLM, SGLang, LMCache, Mooncake), or to a flagged open question.

## 2. Implementation — worked review

### 2.1 Proposal under review (illustrative)

> "To reduce TTFT for our RAG product, the inference platform team proposes enabling
> the engine's prefix cache **globally, not partitioned by tenant**. System prompts
> are the same across tenants, and retrieved passages often overlap because tenants
> query a shared corpus. We estimate a 2× TTFT reduction and a 30% capacity
> increase."

A learner should structure the review in the five sections below; §2.2–§2.6 walk
through what each section should contain, with a model answer.

### 2.2 Threat model

Document:

- **Who is a tenant?** A single organization/customer, a single end-user within a
  customer, a background job? The answer determines what "cross-tenant" means for
  this proposal.
- **Trust boundary.** Which actors can issue arbitrary prompts? Which can observe
  response timing? Which can read engine logs?
- **Data classes.** For each tenant, label the sensitivity of (a) system prompts,
  (b) user prompts, (c) retrieved context, (d) model responses.
- **Threats in scope.** At minimum: confidentiality leakage via content; timing
  oracles; poisoning; regulatory violation; incident forensics (can you tell which
  tenant's data a cached block came from after the fact?).
- **Threats explicitly out of scope.** Model-weight inversion, prompt-injection of
  tool calls, etc., unless they interact with caching specifically.

Reference frameworks the write-up should cite by name, not by paraphrase:

- NIST AI RMF 1.0 — GOVERN and MAP functions for context of use.
- MITRE ATLAS — technique catalogue relevant to inference-time attacks.
- OWASP ML Top 10 — the list is maintained at the URL in §6; reviewers should cite
  the current list rather than quote specific category text from memory.

### 2.3 Risk table

Model answer (sensitivity tags are illustrative; replace with the actual data
labels of the deployment under review):

| Risk | Attacker capability assumed | Likelihood | Impact | Severity |
|---|---|---|---|---|
| Timing oracle on TTFT reveals cached prompt tokens | Any authenticated tenant with ability to issue arbitrary prompts and observe TTFT | High | Medium–High (depending on what sits in the cache) | High |
| Response content leak from mistakenly stitched KV | Shared KV crosses conversational turns from different tenants | Low with correct implementation; Medium if the cache is keyed only by token hash and a collision/path reuse occurs | High | High |
| Deletion/retention breach | Tenant asserts right-to-erasure, data physically still present in another tenant's cached path | Medium | Medium | Medium |
| Compliance / contractual breach from commingling | Regulated tenant class exists in the deployment | Deployment-specific | High when triggered | Varies |
| Reduced incident-response fidelity | Logs show "a tenant" fetched a cached block rather than "which tenant" | High if logs are not augmented | Medium | Medium |

The timing oracle is the headline risk because it is realistic with an attacker who
has nothing but normal API access. Bai et al. and subsequent work have reported
cache-timing side channels against LLM inference systems; the review should cite the
current literature at review time rather than a date-stamped list from here.
Reviewers should pull the current public citations and CVEs for prefix-cache timing
side channels in open LLM serving stacks at the time of writing and attach them to
the ADR produced in §3.

### 2.4 Control options and residual risk

| Control | What it changes | Residual risk |
|---|---|---|
| **Per-tenant partitioned cache** (tenant ID folded into the cache key) | Eliminates cross-tenant content leak and most of the timing oracle (hit only when the same tenant repeats) | Cache has no cross-tenant sharing — same as not sharing |
| **Shared-prefix-only cache** (cache limited to a strictly controlled, platform-owned "shared prefix" like a public system prompt; anything beyond it goes through a per-tenant path) | Retains the biggest TTFT win (shared system prompt) with no leak of tenant data | Requires the engine to support splitting system prompt from the rest of the key; review the implementation |
| **Content-salted keys** (per-tenant salt mixed into the key hash) | Prevents accidental collisions; does not help the timing oracle | Timing oracle unchanged — a hit is still faster than a miss |
| **Timing normalization** (constant-time admission regardless of hit/miss, with cached-result delivery delayed to the miss latency) | Closes the TTFT oracle | Wipes out the latency benefit that motivated the proposal; effectively equivalent to not caching from the user's perspective |
| **Rate limiting + anomaly detection on prompt-probe patterns** | Raises the cost of a timing probe attack | Mitigates but does not eliminate; detection depends on attacker's query shape |
| **Tenant-scoped tiered KV namespace** (Exercise 04 store partitioned by tenant) | Keeps tiered reuse within a tenant | Same as per-tenant partitioning in the shared-cache layer |
| **Explicit opt-in sharing** (tenants sign a consent-to-share for non-sensitive workloads; defaults to isolated) | Preserves shared caching for tenants who accept it | Operationally heavy; depends on contract flow |

### 2.5 Recommendation

For the illustrative proposal in §2.1, a defensible recommendation is **reject the
proposal as stated** and counter-propose a layered design:

1. **Default: per-tenant partitioned cache.** The cache key is `(tenant_id,
   token_sequence_hash)`. No tenant can observe another tenant's hit through either
   content or timing, because no key can match across tenants.
2. **Opt-in shared-prefix layer** for a *platform-owned, public* system prompt.
   Everything after the system prompt is tenant-partitioned. The TTFT benefit on the
   system prompt (which is often most of the prefill cost in short-prompt RAG) is
   retained.
3. **Tenant-scoped tiered KV store** (if Exercise 04 applies).
4. **Logging augmented with tenant ID on every cache event** to support
   right-to-erasure and incident forensics.
5. **Quarterly review** against OWASP ML Top 10, MITRE ATLAS, and the NIST AI RMF to
   catch new attack techniques and framework updates.

If the business pushback is severe and the data class is low-sensitivity public text,
a narrower variant of Option 1 — shared cache with content-salted keys, timing
normalization, and a tenant consent flow — may be defensible. The review should
force that pushback to go on the record, not be assumed.

### 2.6 Open questions for the architect

A high-quality review surfaces questions for the proposing team rather than making
assumptions:

- What is the data class of the retrieved context across tenants? Is any of it
  regulated (PII, PHI, PCI, export-controlled)?
- Are tenants aware of and contractually okay with cross-tenant reuse of system
  prompts? Of retrieved passages?
- What is the engine's current cache keying implementation? (Pointer to the engine
  source and version.)
- What is the acceptable TTFT regression if timing normalization is required?
- How is the cache partitioned across the tiered KV store (Exercise 04) today?

## 3. Validation steps

1. **Threat-model completeness check.** Every risk category in §2.3 has: an
   attacker capability description, a likelihood, an impact, and at least one
   proposed control.
2. **Timing-oracle demonstration (optional, for a lab).** With consent from the
   operator, construct two prompts where one matches a known cached prefix and one
   does not, drive them at matched concurrency, and show that TTFT differs. This is
   the empirical foundation for the timing-oracle risk and is worth writing up once
   per course cycle so the risk is not an abstract claim.
3. **Control implementability check.** For each recommended control, point to the
   specific engine flag, config field, or code path that implements it (e.g., vLLM's
   per-request cache keying, LMCache's namespace separation; cite docs for the
   version under review, not from memory).
4. **Framework alignment check.** The review names NIST AI RMF, OWASP ML Top 10, and
   MITRE ATLAS *by current identifier* at review time. Ad-hoc paraphrases of these
   frameworks should be replaced with current language.
5. **Decision-record output.** Produce a one-page Architecture Decision Record
   (ADR) capturing the proposal, the decision, the reasoning, and the review date.
   This is the artifact that lives past the review meeting.

## 4. Rubric / review checklist

Score 0–2; passing ≥ 14 / 20.

| # | Item | Pts |
|---|---|---|
| 1 | Threat model defines tenant, trust boundary, and data classes | 2 |
| 2 | Timing side channel on TTFT explicitly listed as a risk | 2 |
| 3 | Content stitching / collision risk explicitly listed as a risk | 2 |
| 4 | Right-to-erasure and retention implications called out | 2 |
| 5 | Risk table scores likelihood and impact independently | 2 |
| 6 | At least four control options evaluated, not just a single recommendation | 2 |
| 7 | Recommendation is a clear accept/reject/conditional with reasoning | 2 |
| 8 | Named references to OWASP ML Top 10, MITRE ATLAS, NIST AI RMF | 2 |
| 9 | Open questions surfaced for the architect; assumptions not silently made | 2 |
| 10 | ADR or equivalent decision record produced | 2 |

## 5. Common mistakes

- **"Hashes are opaque, so sharing is safe."** The cache contents are the KV
  activations of the exact input tokens; an attacker who can detect that *some*
  content is cached (via timing) can infer what that content is by constructing
  probing prompts. Opacity of the key does not protect the content.
- **Collapsing "prefix cache" into "prompt cache."** Prefix caching is about any
  shared leading sequence, not just whole prompts. A review that only considers
  whole-prompt reuse underestimates the timing-oracle surface.
- **Missing the regulatory axis.** Even if every technical control holds, contract
  or regulation may forbid commingling. A review that is all technical and no legal
  is incomplete.
- **Treating per-tenant partitioning as "defeating the point."** The dominant
  TTFT win in practice is the shared *system prompt*, which can be kept in a
  platform-owned opt-in layer without exposing tenant data.
- **No logging strategy.** Cache events need tenant attribution for incident
  response and right-to-erasure.
- **Confusing KV encryption with KV isolation.** Encryption at rest helps against
  a different threat model (disk theft). It does not stop a tenant inside the
  service from exploiting a timing oracle.
- **Framework name-dropping without citation.** Listing OWASP ML Top 10 or NIST AI
  RMF without naming the specific category or function is not a control — it is
  decoration.

## 6. References

Official sources (as listed in the work packet):

- OWASP Machine Learning Security Top 10 — [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/).
- MITRE ATLAS — [mitre-atlas](https://atlas.mitre.org/).
- NIST AI Risk Management Framework — [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework).

Engine documentation (practitioner implementations):

- vLLM — "Automatic Prefix Caching", configuration reference —
  https://docs.vllm.ai/.
- SGLang — "RadixAttention", server config —
  https://docs.sglang.ai/.
- LMCache — namespace / tenant separation — https://docs.lmcache.ai/.

Related primary literature (referenced for mechanisms, not for the safety claim):

- Kwon, W. et al. *Efficient Memory Management for Large Language Model Serving with
  PagedAttention.* SOSP 2023. arXiv:2309.06180.
- Zheng, L. et al. *SGLang: Efficient Execution of Structured Language Model
  Programs.* NeurIPS 2024. arXiv:2312.07104.

Open items the reviewer is expected to pin down at review time (the frameworks
below are versioned; cite the exact release in use):

- Current OWASP ML Top 10 category identifiers (the list is versioned; cite the
  exact release used).
- Current MITRE ATLAS technique IDs for inference-time side-channel attacks.
- Specific public incidents or CVEs on cache-timing side channels in open LLM
  serving stacks to cite in the risk table.
