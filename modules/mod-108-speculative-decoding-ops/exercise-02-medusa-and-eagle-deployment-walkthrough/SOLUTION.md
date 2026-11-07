# Solution — Exercise 02: Medusa and EAGLE Deployment Walkthrough

> Paired learning file: [`lessons/mod-108-speculative-decoding-ops/exercises/exercise-02-medusa-and-eagle-deployment-walkthrough.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-108-speculative-decoding-ops/exercises/exercise-02-medusa-and-eagle-deployment-walkthrough.md)
> Module: `mod-108-speculative-decoding-ops`
> Type: implementation / deployment walkthrough

## 1. Solution overview

Where [[draft-target-speculative-deployment-on-vllm]] (exercise 01) taught
speculative decoding using a **separate small draft model**, this exercise walks
through two **self-speculative** families that live in the target model itself:

- **Medusa** — adds multiple extra "head" layers on top of the target model's
  final hidden state; each head predicts a token at a future position. Tree-based
  verification chooses among top-k candidates from each head.
  ([Cai et al., ICML 2024](https://arxiv.org/abs/2401.10774).)
- **EAGLE** — a small autoregressive draft that runs on the **penultimate feature
  layer** of the target (not on tokens), achieving substantially higher acceptance
  than a token-level draft of similar size.
  ([Li et al., ICML 2024](https://arxiv.org/abs/2401.15077); EAGLE-2 and EAGLE-3
  extend the idea.)

The learner deploys at least one of each family on vLLM, confirms the
speculation path is actually firing, and records acceptance and latency for the
same prompt mix used by exercise 01's baseline. The *point* is not to declare a
winner; different methods excel in different pockets (chat, code, long context,
batch size), and this exercise builds the mental model of which method solves
which failure mode of plain draft-target.

The four artefacts:

| Artefact | Form | Pass bar |
|---|---|---|
| A — Medusa launch command + head checkpoint provenance | Shell script + model-card reference | Starts cleanly; log confirms Medusa is enabled; checkpoint source documented |
| B — EAGLE launch command + head checkpoint provenance | Shell script + model-card reference | Starts cleanly; log confirms EAGLE is enabled; checkpoint source documented |
| C — Acceptance + latency comparison | Three-row table + method note | Baseline (from ex-01), Medusa, EAGLE at identical prompts/concurrency |
| D — Decision note | Short Markdown | States which method the learner would pick for which workload and why — not which is "best" |

### Where each method fits on the map

| Method | Uses a separate model? | Grows target model memory? | Typical α band in published practitioner reports | Key operational lever |
|---|---|---|---|---|
| Draft-target (ex-01) | yes, small LM | no (draft is separate) | ~0.5–0.8 for same-family chat | `num_speculative_tokens`, draft family |
| Medusa | no — adds heads to target | yes, heads add parameters | ~0.5–0.7 (per Medusa paper; verify) | head count, tree width/depth |
| EAGLE | partial — small draft runs on target features | small added decoder | ~0.7–0.85 (per EAGLE paper; verify) | tree width, head depth |

Numbers are illustrative of published practitioner reports; the measured α on the
learner's hardware and workload is the number that matters. The right question is
**which method has the acceptance pattern your workload needs**, not which paper
reports the highest number.

### Picking between Medusa and EAGLE — the short version

- **Medusa** is attractive when you can retrain (or obtain pre-trained heads for)
  the exact target you ship. It removes the draft-model process boundary and
  reduces moving parts.
- **EAGLE** is attractive when acceptance rate is the bottleneck. Running the
  draft on the target's feature space usually buys higher α for the same
  parameter budget, at the cost of a slightly more complex training pipeline for
  the head.
- **Draft-target** remains the right first cut when you need to ship fast and
  already have a same-family small model (e.g., the 1 B / 3 B variants of the 70 B
  you serve). No retraining required.

## 2. Implementation — worked walkthrough

The commands below assume:

- A target model with publicly released Medusa heads and public EAGLE heads at
  the time of the lab. Examples used in the published practitioner literature
  include Vicuna-7B and Llama-based targets; confirm the exact checkpoint from
  the checkpoint's model card before use.
- vLLM version pinned. The flag form for Medusa / EAGLE speculative decoding has
  shifted between vLLM minor releases; verify every flag against
  `vllm serve --help`.
- One node of ≥ 2× 80 GiB accelerators (or 4× / 8× depending on target size).
- Correctness, load-generator, and measurement tooling from exercise 01 reused
  here. The point of this exercise is the *configuration* and the *comparison*,
  not re-building tooling.

Where a flag's exact name is uncertain against a given vLLM release, pin the
release and run `vllm serve --help | grep -iE 'speculat|medusa|eagle'` before
copying commands below.

### 2.1 Medusa deployment

vLLM exposes Medusa via the same structured `--speculative-config` entry point as
draft-target speculation, with `"method": "medusa"` and a Medusa-trained heads
checkpoint. Head provenance must be recorded.

```bash
# config_medusa.sh
export HF_HOME=/scratch/hf
export VLLM_LOGGING_LEVEL=INFO

vllm serve <target-model-with-medusa-heads> \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --disable-log-requests \
  --speculative-config '{
      "method": "medusa",
      "model": "<medusa-heads-checkpoint>",
      "num_speculative_tokens": 4
    }' \
  --host 0.0.0.0 --port 8003
```

Verify from the engine log that Medusa actually initialized:

```
INFO ... Initializing speculative decoding...
INFO ... Scoring method: medusa
INFO ... Loading Medusa heads from <checkpoint> ...
INFO ... num_speculative_tokens=4
```

If the log shows "draft-model" rather than "medusa" as the method, the heads did
not load and the engine is silently falling back; stop and fix.

**Head provenance.** The learner must record:

- The Medusa heads checkpoint URL or model-card ID.
- The target model it was trained for (Medusa heads are **target-specific**;
  heads trained against Target A will not serve Target B).
- The license of the heads and any dataset-provenance notes from the model card.

Medusa heads are a trained artifact and must be governed. See
[[draft-model-lifecycle-contract]] (exercise 04) for the lifecycle rules that
apply.

### 2.2 EAGLE deployment

EAGLE exposes a very similar surface. The EAGLE head's weight format differs from
Medusa's, and the checkpoint is again target-specific (an EAGLE head trained
against Llama-3-8B will not transfer to Llama-3-70B).

```bash
# config_eagle.sh
vllm serve <target-model> \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --disable-log-requests \
  --speculative-config '{
      "method": "eagle",
      "model": "<eagle-head-checkpoint>",
      "num_speculative_tokens": 5
    }' \
  --host 0.0.0.0 --port 8004
```

Expected engine log:

```
INFO ... Initializing speculative decoding...
INFO ... Scoring method: eagle
INFO ... Loading EAGLE draft from <checkpoint> ...
INFO ... num_speculative_tokens=5
```

### 2.3 Correctness check (same discipline as exercise 01)

Both Medusa and EAGLE are **lossless** with respect to the target's distribution,
exactly like draft-target speculation. Under greedy decoding with the same seed,
Medusa and EAGLE outputs must match the baseline from exercise 01 token-for-token
(within fp tolerance).

Re-use `correctness_check.py` from exercise 01, pointed at the Medusa and EAGLE
ports. Any divergence under greedy sampling indicates a bug (wrong tokenizer,
head loaded against the wrong target, verification routine broken). Do not
proceed to measurement until this passes.

### 2.4 Load generation and acceptance read-out

Re-use the `load_gen_spec.py` from exercise 01 against the Medusa and EAGLE
ports. The `vllm:spec_decode_*` metric names are shared across methods, so the
same scraper works.

```bash
for port in 8001 8003 8004; do
  python load_gen_spec.py \
    --url http://localhost:$port/v1/completions \
    --metrics-url http://localhost:$port/metrics \
    --model <target-model> \
    --concurrency 16 --requests 128 \
    --prompt-tokens 512 --max-tokens 128
done
```

### 2.5 Worked comparison table (illustrative)

The numbers below are **placeholders**; the learner reports measured values.
Reviewers judge the *method*, not the figures. The ordering column calls out the
qualitative sign expected from the published literature.

| Config | Method | α (measured) | req/s | p50 lat (ms) | p95 lat (ms) | Notes |
|---|---|---|---|---|---|---|
| Baseline (ex-01) | none | — | *measured* | *measured* | *measured* | Control |
| Draft-target (ex-01) | small LM | *measured* | *measured* | *measured* | *measured* | Same-family draft |
| Medusa | 4 heads, tree verif. | *measured* | *measured* | *measured* | *measured* | Heads sourced from `<url>` |
| EAGLE | feature-level draft | *measured* | *measured* | *measured* | *measured* | Head sourced from `<url>` |

Expected qualitative ordering (per the Medusa and EAGLE papers; verify):

- **α(EAGLE) ≥ α(draft-target) ≥ α(Medusa)** in many chat benchmarks.
  Medusa compensates with a wider tree (more candidates per step) to recover
  effective throughput.
- At low concurrency, latency wins follow `E[accepted]` per round, so EAGLE
  tends to show the biggest latency win.
- At high concurrency, the engine's total GPU budget caps things; all three
  methods converge toward the baseline as the decode batch already saturates
  compute. A sweep at `concurrency ∈ {1, 8, 32}` makes this clear.

### 2.6 Decision note (artefact D)

A short Markdown note that answers three questions:

1. **For a chat-only, latency-sensitive workload where the team already has a
   same-family small model shipping in production, which method would I deploy
   first?** Expected answer: draft-target, because no new trained artefact is
   required.
2. **If acceptance-rate monitoring (exercise 03) shows the draft-target α
   stabilizing around 0.5 on a specific sub-workload and the team has a Medusa
   or EAGLE head available for the target, which would I switch to and why?**
   Expected answer: EAGLE if the head exists and is governed; Medusa if the team
   wants to collapse the two-process topology into one and is willing to accept a
   slightly lower α with wider tree verification.
3. **What would make me abandon speculation entirely for a given workload?**
   Expected answer: measured α below the point where speculation pays for itself
   (the Leviathan §3 inequality), plus tail-latency regressions that cannot be
   recovered by lowering `k` or narrowing the tree.

## 3. Validation steps

A reviewer validating a learner submission should perform the following in order.

1. **Launch commands pinned.** All three engines (baseline, Medusa, EAGLE) pin
   vLLM version, target model, heads checkpoint, and `num_speculative_tokens`.
2. **Method actually initialized.** Engine log line explicitly names the scoring
   method as `medusa` or `eagle`. A log that says `draft-model` when the config
   asked for `medusa` is a silent fallback and fails validation.
3. **Head provenance recorded.** Each self-speculative deployment cites the
   heads checkpoint URL / model-card ID, the target model the heads were trained
   for, and the license. A missing heads-lineage entry fails validation.
4. **Correctness.** Greedy outputs match baseline token-for-token (within fp
   tolerance). If they do not, the submission must name the cause before
   proceeding.
5. **Acceptance recorded.** `vllm:spec_decode_draft_acceptance_rate` (or the
   method-equivalent counter) is non-zero and recorded for every spec engine
   after a non-trivial load run.
6. **Comparison table.** Baseline, draft-target (reused from ex-01 if available),
   Medusa, and EAGLE are compared at identical prompt and output lengths and at
   identical concurrency.
7. **Method note.** Reports prompt length, output length, concurrency, warm-up,
   and an explicit "smoke measurement, not a benchmark" acknowledgement, same as
   exercise 01.
8. **Decision note present.** The short note in §2.6 is included and the three
   questions are answered concretely, not generically.

## 4. Rubric / review checklist

Score each item 0 (missing), 1 (partial), 2 (fully correct). Passing: ≥ 10 / 14.

| # | Item | Points |
|---|---|---|
| 1 | Medusa launch runs clean on a target with published heads; log confirms method = `medusa` | 2 |
| 2 | EAGLE launch runs clean on a target with published head; log confirms method = `eagle` | 2 |
| 3 | Heads checkpoint provenance (URL / model-card ID / license / target) recorded for each | 2 |
| 4 | Correctness check confirms greedy outputs match baseline | 2 |
| 5 | Acceptance-rate counters recorded for Medusa and EAGLE | 2 |
| 6 | Comparison table filled with method note | 2 |
| 7 | Decision note answers the three §2.6 questions with workload-specific reasoning | 2 |

## 5. Common mistakes

- **Treating Medusa heads as portable.** A head trained against Target A is
  numerically and semantically bound to Target A's final hidden state. Reusing it
  against Target B silently produces garbage (acceptance near zero) or fails
  verification. Same rule for EAGLE heads.
- **Treating the self-speculative head as just a config knob.** The head is a
  trained artefact with a license, a dataset lineage, and a retrain cadence. See
  [[draft-model-lifecycle-contract]] (exercise 04).
- **Loading a head without the matching target revision.** Target weights change
  when a new fine-tune lands; heads trained against the previous revision may
  retain usable α for a while and then collapse. Pin head-version ↔ target-version
  explicitly in the deployment manifest.
- **Running Medusa or EAGLE with sampling and expecting matched outputs.** Both
  methods are lossless **in distribution**, not in **samples**. For correctness
  compare greedy outputs; for distribution tests use statistics, not
  byte-for-byte compare.
- **Reporting only aggregate throughput.** Both methods are latency wins first.
  Report both p50/p95 latency and throughput.
- **Skipping the baseline.** Without the exercise-01 baseline in the table, there
  is nothing for Medusa/EAGLE to beat; the comparison is unanchored.
- **Over-wide tree verification in Medusa at high concurrency.** Medusa's tree
  spends target FLOPs on candidates that will not be accepted. At high
  concurrency, verification cost can exceed the decode-step cost it was meant to
  amortize. If α drops and p95 rises together at high concurrency, narrow the
  tree.
- **Using an EAGLE head from one EAGLE release against an engine from another.**
  EAGLE-1, EAGLE-2, and EAGLE-3 have different head architectures. Match head
  version to engine support.

## 6. References

- Cai, T. et al. *Medusa: Simple LLM Inference Acceleration Framework with
  Multiple Decoding Heads.* ICML 2024. https://arxiv.org/abs/2401.10774 —
  original Medusa paper; §2 covers head architecture, §3 covers tree
  verification.
- Li, Y. et al. *EAGLE: Speculative Sampling Requires Rethinking Feature
  Uncertainty.* ICML 2024. https://arxiv.org/abs/2401.15077 — original EAGLE
  paper.
- Li, Y. et al. *EAGLE-2: Faster Inference of Language Models with Dynamic Draft
  Trees.* NeurIPS 2024. https://arxiv.org/abs/2406.16858 — EAGLE-2 extension;
  dynamic tree verification.
- Leviathan, Y., Kalman, M., Matias, Y. *Fast Inference from Transformers via
  Speculative Decoding.* ICML 2023. https://arxiv.org/abs/2211.17192 —
  foundation for the rejection-sampling verification both Medusa and EAGLE
  build on.
- vLLM documentation. *Speculative Decoding.* https://docs.vllm.ai/ —
  practitioner implementation reference for Medusa and EAGLE in vLLM.
- SGLang documentation. https://docs.sglang.ai/ — practitioner implementation
  reference for self-speculative decoding in SGLang, useful cross-check.
- NVIDIA. *TensorRT-LLM.* https://docs.nvidia.com/deeplearning/tensorrt-llm/ —
  alternative practitioner implementation of Medusa and EAGLE.
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/)
  — the Medusa / EAGLE heads are trained artefacts; supply-chain controls on
  them are covered in [[draft-model-lifecycle-contract]] (exercise 04).
- [mitre-atlas](https://atlas.mitre.org/) — same scope; tactics around tampered
  trained components apply to speculative heads.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework) —
  governance baseline that drives the head-lifecycle controls in exercise 04.
