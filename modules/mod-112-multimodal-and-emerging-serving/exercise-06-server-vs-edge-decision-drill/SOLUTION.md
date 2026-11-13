# Solution — Exercise 06: Server-vs-Edge Decision Drill

> Paired learning file: [`lessons/mod-112-multimodal-and-emerging-serving/exercises/exercise-06-server-vs-edge-decision-drill.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-112-multimodal-and-emerging-serving/exercises/exercise-06-server-vs-edge-decision-drill.md)
> Module: `mod-112-multimodal-and-emerging-serving`
> Type: design / decision drill

## 1. Solution overview

The learner receives three product scenarios and must, for each, decide whether
inference should run **server-side** (cloud GPUs, centralized fleet), **on the
edge** (on-device, in-browser, in-app), or in a **hybrid** split, and justify
the choice against a fixed set of decision factors. The deliverable is a
decision memo per scenario plus a one-paragraph meta-commentary identifying
which factor dominated in each case. This exercise does not have "one right
answer" — it has defensible and indefensible reasoning.

Three artefacts:

| Artefact | Form | Pass bar |
|---|---|---|
| A — Decision framework | 1-page writeup | Names the decision factors, their relative weights for *this* exercise, and the operational cost of each deployment shape |
| B — Three per-scenario decisions | 3 × short memos | Each memo names the recommendation, the dominant factor, the two next-most-important factors, and at least one real objection the author considered and rejected |
| C — Meta-commentary | 1 paragraph | Identifies which decision factor dominated across the three scenarios; discusses why "just put it in the cloud" and "just ship it on-device" are both wrong defaults |

The exercise is graded on **reasoning quality**, not on matching an answer key.

### The decision factors

Six factors recur across real server-vs-edge decisions. Each scenario below
must weigh them explicitly:

| Factor | What it asks | Where it pushes |
|---|---|---|
| **Latency budget** | p95 end-to-end from user action to result | Edge if < ~200 ms and network-RTT is not bounded; server if model size dominates |
| **Data residency / privacy** | Does the input need to stay on the user's device? | Edge if yes (medical, PII, screen contents, personal photos); server tolerable if ingress is contractually bounded |
| **Model size / capability** | Does a device-capable model meet quality bar? | Server if quality requires a frontier-scale model; edge if a 1–8 B class model suffices |
| **Connectivity** | Can we assume a working network? | Edge for offline-first (field tools, aviation, maritime); server tolerable for always-online products |
| **Unit economics** | Who pays the compute cost — vendor per-call or user per-device? | Edge amortizes to zero marginal vendor cost but requires acceptable device hardware; server scales with usage but is metered |
| **Update cadence / governance** | How often must the model change? | Server if daily/weekly refresh is required (abuse response, product iteration); edge if monthly+ is acceptable and signed-distribution channels exist |

An honest decision names which two or three of these dominate for the scenario
and acknowledges the others.

## 2. Implementation — worked walkthrough across three scenarios

### 2.1 Scenario A — Mobile keyboard "next-word" prediction

**Context (given to the learner).** A mobile keyboard app for Android / iOS.
User-visible feature: predicts the next 1–3 tokens of what the user is about to
type. SLO: < 50 ms from the last key-press to prediction appearing above the
keyboard. Target quality: at or above the current heuristic baseline. Device
base: majority on 2+ year-old mid-range phones.

**Recommendation.** **Edge-only.** A small (≤ 100 M parameter) language model
compiled to the mobile NPU, served entirely on-device.

**Dominant factor — latency budget.** Server-side cannot meet a 50 ms p95 SLO
once mobile-network RTT is accounted for. Mean 4G RTT alone sits around
30–50 ms for well-covered geographies and the tail is substantially worse;
weak-signal and rural users would see hundreds of ms. Even a "fast" server
cannot help because the network dominates.

**Next-most — data residency.** Keystrokes are extremely sensitive. Shipping
every keystroke to a vendor server is a regulatory and reputational liability
even where legally permissible. Edge-only avoids the question entirely.

**Also significant — unit economics.** The feature fires on every keypress; at
hundreds of millions of users typing continuously, per-call server costs
dominate the product P&L. On-device amortizes to zero marginal cost.

**Objection considered and rejected.** "Use a larger server-side model when
the user has good connectivity, fall back to local otherwise." Rejected
because the SLO is 50 ms *always*, not *when network is good*. A hybrid that
sometimes is slow is worse than a consistent edge experience. Also: users
notice the inconsistency more than the baseline quality improvement.

**Model-size reality check.** An ≈100 M parameter character-or-BPE-level LM is
sufficient for next-word prediction; this is well below the capability
frontier. Capability is **not** the limiting factor here.

### 2.2 Scenario B — Enterprise document summarization service

**Context (given to the learner).** A web app used by legal, finance, and
consulting staff at a Fortune-500 customer. Users paste in 20–200 page
documents and receive a 1-page summary with citations. SLO: 60 s end-to-end
is acceptable; interactive streaming of the summary begins within 5 s. Quality
bar: must handle 128 k+ token contexts, cite specific pages, and produce
structured JSON output.

**Recommendation.** **Server-only.** Centralized multi-GPU fleet serving a
frontier-class long-context model (e.g. 70-B+ class dense model or an MoE
equivalent).

**Dominant factor — model size / capability.** The quality bar — 128 k
context, page-accurate citation, structured JSON output — requires a
frontier-scale model. No current device-class hardware runs a 70-B+ dense
model at acceptable per-token latency. The capability floor rules out edge.

**Next-most — update cadence.** Enterprise document traffic surfaces bugs
(citation hallucinations, structured-output format drift, incorrect jurisdiction
handling) that require model-side fixes rolled out continuously. Server fleet
can ship a fix in a hot-reload window; a device-shipped model cannot.

**Also significant — unit economics.** Enterprise customers pay per-seat and
per-document. Server costs are chargeable; edge-amortization would be
irrelevant because the users are a bounded, high-value pool.

**Objection considered and rejected.** "Add on-device redaction before
ingestion so PII never leaves the device." The customer does already require
this, but *redaction is a preprocessing step*, not inference. It happens
locally; the summarization itself still runs on the server over the redacted
content. The decision for the inference step remains server-side; the data-
handling policy is an orthogonal layer.

**Data residency reality check.** Many enterprise customers require that data
stay within their jurisdiction. Server-only is still defensible here *if* the
fleet is multi-region and contractually bounded (EU-only tenants pinned to EU
regions, etc.). This is a deployment-topology decision, not a push to edge.

### 2.3 Scenario C — Smart-camera "person-in-frame" detection

**Context (given to the learner).** A consumer smart-camera device (video
doorbell / baby monitor / small-business security camera). User-visible
feature: pings the user's phone when a person enters frame. SLO: notification
within 2 s of a person appearing. Quality: tolerable false-positive rate (say,
≤ 1 per day). Device base: a camera with a small embedded NPU (e.g. on the
order of 1–10 TOPS), 2–5 W power budget, occasional Wi-Fi outages.

**Recommendation.** **Hybrid, edge-primary.** A small vision model on the
camera runs continuously; a server-side larger model runs occasionally to
backfill labels, train the edge model's confidence calibration, and handle
false-positive investigation.

**Dominant factor — connectivity.** Wi-Fi outages cannot drop the feature.
"Alerted me *if my Wi-Fi was up*" is unacceptable for a security product. The
primary path must be on-device.

**Next-most — unit economics.** A 24×7 streaming camera generates hours of
video per day. Streaming all of it to a server for inference is a per-device
cost that destroys the product's economics at consumer price points. The
primary path must avoid continuous server round-trips.

**Also significant — data residency / privacy.** Users do not want continuous
indoor video streamed to a vendor. On-device detection with server round-trips
*only on event* is a materially stronger privacy posture than always-uploading
and it is also what regulators in several jurisdictions are converging on.

**Why hybrid, not edge-only.** The edge model has to be small enough to fit
and run in a 2–5 W envelope, which bounds its accuracy. A server-side pass is
valuable for (a) upgrading confidence on ambiguous detections before alerting
the user, (b) collecting labelled false positives to retrain the edge model
quarterly, (c) heavier reasoning ("is this the delivery driver we've seen
before?" with a larger model).

**Objection considered and rejected.** "Push all detection to the server and
make the device dumb." Rejected on connectivity and economics. A dumb
upload-only camera cannot alert during a Wi-Fi outage and blows the P&L.

**Update cadence reality check.** The edge model can be updated quarterly via
OTA; the server-side arbitrator can be updated continuously. The hybrid gives
the product both knobs.

### 2.4 Meta-commentary

Across the three scenarios, **no single factor dominates**. Scenario A was
driven by latency and privacy; scenario B by capability and update cadence;
scenario C by connectivity and economics. The lesson is that server-vs-edge is
not a stance (nor a team culture) — it is a per-feature decision that reduces
to which of the six factors is binding for *that* feature. "Always cloud" is
indefensible for keyboards; "always edge" is indefensible for long-document
summarization; "always pure" (server-only / edge-only) is often indefensible
when the right answer is hybrid-with-a-primary.

Two failure modes recur in real product decisions:

- **Latency arguments being used as cover for capability arguments.** Teams
  advocating cloud-only for a feature where latency allows edge often *really*
  mean "the edge model would be too weak." That should be argued as a
  capability constraint, not a latency one, because the two have different
  mitigations.
- **Privacy arguments being used as cover for cost-avoidance arguments.**
  Teams advocating edge-only sometimes *really* mean "we don't want to pay for
  the server infra." That should be argued as a unit-economics constraint,
  because the appropriate reviewer (CFO) is different from the privacy reviewer
  (legal / DPO).

The decision memo pattern above forces the author to name the dominant factor
and separate it from the next two, which makes those confusions visible.

## 3. Validation steps

A reviewer validating a learner submission should check, per memo:

1. **The recommendation is a specific deployment shape**, not a vague "it
   depends." Server-only, edge-only, or hybrid with a declared primary.
2. **The dominant factor is named** and is one of the six in §1.
3. **At least two next-most factors are named** and discussed.
4. **At least one real objection is considered and rejected**, not a straw-man
   objection that the author set up to knock down.
5. **A reality check** on a factor that *would* have pointed the other way is
   explicitly present. In scenario A, the model-size check (which pushes edge)
   is actually the easier-wins check; in scenario B, the data-residency
   check (which pushes edge) is refuted by multi-region server deployment, not
   ignored.
6. **The meta-commentary** identifies cross-scenario patterns, not just a
   summary.
7. **No hand-waved numbers.** If a memo cites a latency or a cost figure, it
   either cites a public source or labels it as an estimate for the exercise.

## 4. Rubric / review checklist

Score each item 0 (missing), 1 (partial), 2 (fully correct). Passing: ≥ 10 / 14.

| # | Item | Points |
|---|---|---|
| 1 | Decision framework explicitly names the six (or equivalent) factors and acknowledges tradeoffs | 2 |
| 2 | Each scenario memo recommends a specific deployment shape (server / edge / hybrid-primary) | 2 |
| 3 | Each memo names the dominant factor and ≥ 2 next-most factors | 2 |
| 4 | Each memo considers and rejects ≥ 1 real objection (not a straw man) | 2 |
| 5 | Each memo includes a reality check on a factor that would otherwise push the other way | 2 |
| 6 | Meta-commentary identifies cross-scenario patterns and names the recurring confusion modes | 2 |
| 7 | Any numerical claim (latency, cost) is either cited or labelled as an estimate | 2 |

## 5. Common mistakes

- **Arriving with a stance.** "I always prefer edge" or "cloud is where models
  live now" skip the decision framework. The exercise grades the framework,
  not the conclusion.
- **Treating latency and model-size as the same factor.** They are different:
  a model-size argument has mitigations like distillation or routing to
  smaller models; a latency argument has mitigations like edge-caching or
  pre-fetching. Confusing them hides the mitigation path.
- **Ignoring network RTT when citing server-side latency.** "The server takes
  30 ms to decode" is not the same as "the user sees 30 ms". Mobile and
  home-internet RTT distributions are substantial; the budget must include
  round-trip time.
- **Pretending edge models have zero cost.** Edge inference costs device
  battery, thermal headroom, and app binary size. All three are real limits in
  real reviews; none is zero.
- **Pretending server fleet update cadence is instant.** Hot-reloading a
  frontier model across a fleet is not seconds; it is minutes to hours, and
  rollout discipline makes it longer still. "Just update the server" is
  faster than updating the edge but is not free.
- **Conflating data residency with privacy.** Residency is a geographic /
  jurisdictional constraint; privacy is a disclosure constraint. Edge
  satisfies both; server with contractual boundaries can satisfy residency
  without satisfying "no vendor sees this data".
- **Missing hybrid as a real option.** Many real-world designs are
  hybrid-with-a-primary (scenario C). "Server or edge?" is a false binary;
  recognize when the answer is "edge-primary with a server backstop" or
  "server-primary with on-device redaction".
- **Over-fitting to today's hardware.** Mobile NPUs are getting materially
  more capable year-over-year; decisions that are edge-infeasible in 2024 may
  be edge-feasible in 2027. A good memo notes the time horizon it's reasoning
  within.
- **Skipping the governance factor.** Who reviews the model? Who signs off on
  a new version? Edge deployments typically require signed-package review and
  a coordinated OTA rollout; server deployments can be governed by the same
  CI/CD process as any backend. This is the quietest factor and the one most
  often forgotten.

## 6. References

- Lin, T. et al. *On-Device Machine Learning: An Algorithms and Learning
  Theory Perspective.* arXiv:1911.00623 — survey of edge-ML constraints
  (memory, energy, latency) referenced in §1.
- Chen, J., Ran, X. *Deep Learning With Edge Computing: A Review.*
  Proceedings of the IEEE, 2019 — practitioner reference for hybrid
  deployment shapes.
- Apple. *Deploy Core ML models.* https://developer.apple.com/documentation/coreml
  — practitioner reference for on-device inference lifecycle on one of the two
  dominant mobile platforms.
- Google. *LiteRT (formerly TensorFlow Lite) overview.*
  https://ai.google.dev/edge/litert — practitioner reference for on-device
  inference on Android and cross-platform edge targets.
- WebGPU specification. https://www.w3.org/TR/webgpu/ — browser-edge
  reference target (increasingly viable for small LMs client-side).
- ONNX Runtime documentation. https://onnxruntime.ai/ — practitioner reference
  for cross-platform edge deployment.
- OpenAI, Anthropic, Google product docs — not cited inline; server-side
  frontier-model deployments are the status-quo reference.
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/)
  — edge deployments move the attack surface onto the user's device (model
  extraction, local tampering); server deployments concentrate it in the
  fleet. Both are governed, but by different controls.
- [mitre-atlas](https://atlas.mitre.org/) — relevant tactics differ by
  deployment shape: ML model theft is heavier on edge, data exfiltration via
  the API is heavier on server.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework) —
  governance baseline that applies regardless of deployment shape; some
  controls (incident response, model-update records) are easier to satisfy
  server-side.
