# Exercise 05 — Deprecated vs. Maintained Engine Landscape Review

> Paired learning exercise: `lessons/mod-102-inference-engine-internals/exercises/exercise-05-deprecated-vs-maintained-engine-landscape-review.md`.

## 1. Solution overview

The learner must produce a one-page landscape note that classifies each
major open-weights-capable inference engine as **maintained**, **in
transition**, or **deprecated / dormant**, cites the evidence they used,
and ends with a hiring-manager-style recommendation of which engines to
build on and which to retire from the team's stack. The exercise is
deliberately short on prescription: the signal under evaluation is whether
the learner looks at **real** evidence (release cadence, open PR rate, last
tagged release, named maintainers, upstream statements) rather than
vibes from social media.

This solution gives the reference methodology, the evidence rubric the
learner's submission must follow, and a recommendation template.

<!-- verify-source: this landscape changes quickly. Any specific status
claim below is annotated `verify-source` because the only acceptable
answer is "whatever the repository says at the time the learner runs the
review." Do not grade on absolute status, grade on methodology. -->

## 2. Implementation — Worked review methodology

### 2.1 Evidence sources the learner must consult (per engine)

For each engine the learner reviews, they must pull at least:

1. **Last tagged release** — date, version, release notes.
2. **Commit cadence** — commits in the last 30 / 90 days to `main`.
3. **Open PR / issue stats** — count, oldest unanswered, median response
   time.
4. **Named maintainers** — `MAINTAINERS`, `CODEOWNERS`, or equivalent;
   organisations backing the project.
5. **Upstream statements** — explicit deprecation, archived status, or
   maintenance-mode announcements in the README, blog, or release notes.
6. **Dependency drift** — does it still build against current PyTorch /
   CUDA versions supported by the GPUs in scope?
7. **Security posture** — public CVEs, time-to-patch on the last one.

An engine can look "alive" on one axis and be effectively abandoned on
another; the learner's review must triangulate.

### 2.2 Engine shortlist and the classification axes

The review must cover at minimum:

- **vLLM** (vllm-project/vllm)
- **SGLang** (sgl-project/sglang)
- **TensorRT-LLM** (NVIDIA/TensorRT-LLM) + Triton + `tensorrtllm_backend`
- **Hugging Face TGI** (huggingface/text-generation-inference)
- **llama.cpp** (ggml-org/llama.cpp)
- **DeepSpeed-MII / FastGen** (microsoft/DeepSpeed-MII) <!-- verify-source. -->
- **Ray Serve LLM** (ray-project/ray) <!-- verify-source. -->
- **LMDeploy** (InternLM/lmdeploy) <!-- verify-source. -->
- **FasterTransformer** (NVIDIA/FasterTransformer) — the canonical
  "superseded" example

The learner should group these across a 2x3 matrix:

|  | GPU server-side (open weights) | On-device / edge |
|---|---|---|
| **Maintained** | vLLM, SGLang, TRT-LLM + Triton <!-- verify-source --> | llama.cpp <!-- verify-source --> |
| **In transition** | TGI, LMDeploy, DeepSpeed-MII <!-- verify-source --> | — |
| **Deprecated / dormant** | FasterTransformer <!-- verify-source --> | — |

The actual cell contents are a function of the date the review is run;
learners must justify each placement with the evidence from §2.1.

### 2.3 Recommendation template (what the one-pager must contain)

```
# Inference Engine Landscape — <date>
Scope: <models / hardware / workload profile>.

## Status snapshot
<2x3 matrix with the evidence pointers under each cell>

## Recommend to build on
- <Engine A>: <why — concrete evidence, concrete fit to our workload>
- <Engine B>: <why — if we need a second for a different workload>

## Keep as backup
- <Engine C>: <conditions under which we'd fall back to this>

## Retire from the stack
- <Engine D>: <why — e.g. last release > 12 months, maintainer letter,
  upstream points at successor project>

## Open questions / to re-verify
- <what we could not verify and plan to re-check next quarter>
```

### 2.4 Decision rationale — what a defensible recommendation looks like

The reference answer prefers, for a general open-weights server-side
inference role in **2026**, one of the Python-native engines (vLLM or
SGLang) as the primary engine and TRT-LLM + Triton as a specialised path
for NVIDIA-only deployments that need the extra throughput and are willing
to pay the build-then-serve cost. llama.cpp is kept for CPU / edge /
quantised-only use cases. FasterTransformer is retired because NVIDIA has
redirected effort to TensorRT-LLM. <!-- verify-source: confirm this
redirection is still stated in the FasterTransformer README /
TensorRT-LLM docs at the time of the review. -->

The learner's exact ranking is less important than that the rationale is
tied to (a) the workload profile in scope and (b) the evidence they
pulled, not to brand or hype.

## 3. Validation steps

1. **Evidence check.** For each engine, the reviewer verifies that the
   last-release date, commit cadence, maintainer list, and open-issue
   stats cited by the learner are reproducible by clicking the linked
   sources.
2. **Date stamp.** The one-pager must carry a review date and a declared
   expiry (e.g. "re-review in 90 days").
3. **Peer sanity check.** Have a peer (ideally oncall) look at the
   matrix cold and flag anything they'd dispute; disagreements should
   trigger a quick evidence refresh, not a status change without
   evidence.
4. **Workload fit.** Pick one engine from the "build on" column and one
   concrete workload the team runs; walk through why that engine beats
   its alternatives for that workload.

## 4. Rubric / review checklist

| # | Criterion | Weight | Pass bar |
|---|---|---|---|
| 1 | Engine shortlist coverage | 15% | At least the eight engines named in §2.2 covered |
| 2 | Evidence per engine | 25% | All seven axes in §2.1 recorded with links |
| 3 | Classification justified | 20% | Each cell placement tied to a specific piece of evidence |
| 4 | Workload-fit framing | 15% | Recommendation is conditioned on a stated workload, not generic |
| 5 | Retire column is populated | 5% | Learner is willing to say "stop using X" and gives evidence |
| 6 | `verify-source` markers used | 10% | Any claim the learner could not directly verify is marked |
| 7 | Review date + expiry stated | 10% | Explicit "valid as of" + "next review by" |

Passing threshold: 70%.

## 5. Common mistakes

- **Scoring on commit volume alone.** A high commit rate on a project
  whose core maintainers all left last month is not maintenance.
- **Confusing popularity with maintenance.** Stars are not a maintenance
  signal; recent tagged releases + responded-to issues are.
- **Treating the status as permanent.** A six-month-old landscape note is
  worse than no note — it confidently lies. The expiry date is not
  optional.
- **Letting hype rank engines.** Social-media attention on a new project
  is not evidence it is maintained a year from now; conversely, an older
  project with steady releases often wins on reliability.
- **Not covering "retire."** If the learner is unwilling to name an
  engine to stop using, the review is incomplete — part of the value of
  the exercise is the willingness to retire.
- **Inventing numbers.** If the learner cannot find a maintainer list or
  a last-release date, they mark it `verify-source` — they do not
  guess.

## 6. References

- vLLM — https://github.com/vllm-project/vllm (releases, maintainers,
  commit log).
- SGLang — https://github.com/sgl-project/sglang.
- TensorRT-LLM — https://github.com/NVIDIA/TensorRT-LLM.
- Triton Inference Server — https://github.com/triton-inference-server/server.
- Hugging Face TGI — https://github.com/huggingface/text-generation-inference.
- llama.cpp — https://github.com/ggml-org/llama.cpp.
- DeepSpeed-MII — https://github.com/microsoft/DeepSpeed-MII.
- Ray — https://github.com/ray-project/ray (Serve LLM).
- LMDeploy — https://github.com/InternLM/lmdeploy.
- FasterTransformer — https://github.com/NVIDIA/FasterTransformer (read
  the README banner for the current maintenance statement). <!-- verify-source. -->
- OWASP Machine Learning Security Top 10 —
  https://owasp.org/www-project-machine-learning-security-top-10/ (ML06:
  AI Supply Chain Attacks — directly motivates not standing up on a
  deprecated engine).
- NIST AI Risk Management Framework —
  https://www.nist.gov/itl/ai-risk-management-framework ("Govern" and
  "Manage" functions: lifecycle governance of third-party components).
- MITRE ATLAS — https://atlas.mitre.org/ (useful when the review has to
  weigh the attack surface of an unmaintained engine vs. a maintained
  one).
