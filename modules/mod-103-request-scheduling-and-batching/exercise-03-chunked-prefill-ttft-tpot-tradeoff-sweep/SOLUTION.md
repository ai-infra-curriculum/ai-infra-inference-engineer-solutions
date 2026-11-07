# Solution — Exercise 03: Chunked Prefill TTFT / TPOT Trade-off Sweep

> Paired learning file: [`lessons/mod-103-request-scheduling-and-batching/exercises/exercise-03-chunked-prefill-ttft-tpot-tradeoff-sweep.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-103-request-scheduling-and-batching/exercises/exercise-03-chunked-prefill-ttft-tpot-tradeoff-sweep.md)

The chunked-prefill knob is where the TTFT / TPOT trade-off is most directly expressed — smaller chunks protect decoders' TPOT at the cost of long-prompt TTFT, larger chunks do the reverse. This exercise finds the Pareto point for the learner's workload, pins the value, and documents the risks that come with it.

Everything numeric below is **illustrative**. The learner's sweep produces their numbers; the reviewer checks the method.

---

## 1. Solution overview

Four artefacts:

| Artefact | Form | Pass bar |
|---|---|---|
| A — Long-prompt workload variant | Load-generator config + distribution writeup | ≥ 15% long-prompt bucket at ≥ 4000 tok P95 |
| B — Chunk-size sweep | Table split by prompt-length bucket | ≥ 5 chunk-size points (off + 4 on) with bucket-wise metrics |
| C — Pareto plot and defended pick | 2 (or 3) plots + one-paragraph pick justification | TTFT P95 for long bucket vs. chunk size shows the trade-off; pick clears both SLOs |
| D — Risk register | ≥ 5 entries with symptom / detection / mitigation | Each risk ties to a metric on the dashboard from [exercise 05](../exercise-05-scheduler-metrics-into-prometheus/SOLUTION.md) |

The method: hold `max-num-seqs` and `max-num-batched-tokens` at the values from [exercise 02](../exercise-02-scheduler-tuning-for-stated-slo-and-traffic-shape/SOLUTION.md); vary only the chunk-size knob (and its engine-specific proxy). The rule for the pick is:

> **Lowest TTFT P95 on the long-prompt bucket subject to TPOT P95 ≤ SLO** (chapter 03).

An "off" baseline is required. If the baseline hits SLO comfortably, chunked prefill may not be worth enabling for this workload.

### Engine-specific knob mapping

Chunked prefill's control surface varies across engines; **verify flag spellings against your installed release's docs** before running any sweep.
| Engine | Enable | Chunk-size control | Notes |
|---|---|---|---|
| vLLM | `--enable-chunked-prefill` | Chunk size is governed indirectly by `--max-num-batched-tokens`; some releases add `--long-prefill-token-threshold` and `--max-num-partial-prefills` | Sweep `--max-num-batched-tokens` ∈ {2048, 4096, 8192, 16384} with chunked prefill on, holding `--max-num-seqs` fixed. |
| SGLang | server-args chunked-prefill knob (`--chunked-prefill-size`-shape) | The chunked-prefill knob itself is the chunk size | Verify flag spelling against the SGLang release. |
| TRT-LLM via Triton | `enable_chunked_context` in the model `config.pbtxt` | Chunk-size parameter at the `tensorrtllm_backend` config | Backend-level knob; not an engine CLI flag. |

---

## 2. Implementation

### 2.1 Artefact A — long-prompt workload variant

Starting from the [exercise 02](../exercise-02-scheduler-tuning-for-stated-slo-and-traffic-shape/SOLUTION.md) `load_gen.py`, extend the prompt-length sampler to split prompts into two buckets with declared ratios:

```python
def sample_prompt_len(rng):
    bucket = rng.choices(["short", "long"], weights=[0.80, 0.20])[0]
    if bucket == "short":
        return int(rng.gauss(500, 150))        # P50 ~500
    # Long-prompt bucket: P95 ~4000-8000
    return int(rng.gauss(5000, 1500))
```

Tag each request row in the output trace with its bucket so the sweep can split metrics by bucket:

```python
@dataclass
class TraceRow:
    ...
    bucket: str  # "short" | "long"
```

Declared ratio for the worked example: **80/20 short/long**, with long-prompt P50 ≈ 5000 tok and P95 ≈ 8000 tok. The chapter requires at least 15% long-prompt share at ≥ 4000 tok P95; 20% at 5000/8000 clears that comfortably and mirrors a RAG-heavy pool. Document the ratio in the writeup: *the sweep's pick is only valid for workloads with this long-prompt share or less.*

Keep concurrent decode traffic: the long prompts' impact is only visible against ongoing decoders. Do not drain the running set between long prompts.

### 2.2 Artefact B — worked sweep (shape and example values)

**Protocol:**

1. Launch engine with the exercise-02 pinned config.
2. Override only the chunked-prefill knobs per sweep point.
3. 5-minute warmup, 10-minute measured window, Artefact A workload.
4. From the trace CSV compute per bucket: TTFT P50/P95/P99 and TPOT P50/P95/P99.
5. From `/metrics` compute: preemption rate (overall), iteration-time distribution if `vllm:iteration_tokens_total` is exposed.
**Example output table** (illustrative placeholders, values are not benchmark claims):

| Config | Long TTFT P95 (ms) | Long TTFT P99 (ms) | Short TTFT P95 (ms) | TPOT P95 (ms) | Preempt rate |
|---|---|---|---|---|---|
| chunked prefill OFF | 5400 | 7100 | 220 | 36 | 0.1% |
| chunk ~ 256 | 3100 | 4500 | 230 | 61 | 0.2% |
| chunk ~ 512 | 2200 | 3200 | 225 | 48 | 0.2% |
| chunk ~ 1024 | 1500 | 2300 | 225 | 42 | 0.3% |
| chunk ~ 2048 | 1050 | 1800 | 225 | 47 | 0.5% |
| chunk ~ 4096 | 850 | 1600 | 220 | 56 | 0.6% |

(vLLM learners: read the `chunk ~ N` row as "`max-num-batched-tokens = N` with chunked prefill on"; the effective chunk size tracks this budget when `max-num-seqs` is fixed.)

Secondary sweep (recommended): hold chunk size at the preferred pick and vary `max-num-batched-tokens` ∈ {4096, 8192, 16384, 32768}; confirm the co-tuning behaviour from chapter 03 — larger token budgets amortise the fixed per-iteration overhead of chunked prefill.

### 2.3 Artefact C — worked Pareto plots and pick

**Plot 1 — long-prompt TTFT P95 vs. chunk size:** monotonically falling curve (smaller chunk → more iterations to complete the prefill → higher TTFT), overlaid with the TTFT SLO line (e.g. 1500 ms for the `chat-prod-us` worked example).

**Plot 2 — TPOT P95 (all prompts) vs. chunk size:** monotonically rising curve (larger chunk → each iteration that carries a chunk delays the decoders riding on the same batch → higher TPOT), overlaid with the TPOT SLO line (e.g. 50 ms).

**Plot 3 (optional) — short-prompt TTFT P95 vs. chunk size:** should be close to flat; short prompts fit in a single chunk regardless, so chunk-size changes do not materially affect them. A non-flat line here is a diagnostic signal that something else is going on (scheduler delay, admission backpressure, prefix-cache cold).

**Worked pick** (against the example values above):

> The chunked-prefill-off baseline misses the long-TTFT SLO by 3.6× (5400 ms vs. 1500 ms); chunked prefill is required. Chunk ~ 256 protects TPOT but overshoots TTFT by 2.1×. Chunk ~ 1024 is the first row that clears TTFT (1500 ms, right at the SLO line — no headroom), and TPOT at 42 ms has 16% headroom. Chunk ~ 2048 clears both axes with margin (TTFT 1050 ms at 30% headroom; TPOT 47 ms at 6% headroom), so by the chapter 03 rule — *lowest long-prompt TTFT P95 subject to TPOT ≤ SLO* — the pick is **chunk ~ 2048** (i.e. `--max-num-batched-tokens=2048` with chunked prefill on). Chunk ~ 4096 would still clear TTFT but regresses TPOT past the SLO; reject.

The pick is comparison-literate: it names the baseline, names the two feasible rows, and applies the chapter's rule. Vague picks ("1024 feels right") fail the rubric.

### 2.4 Artefact D — worked risk register (template)

Each entry: symptom (what you'd see), detection (which metric), mitigation (what to do).

1. **Workload shift — long-prompt ratio growth.**
   - *Symptom:* Long-prompt TTFT P95 regresses while TPOT holds; preemption rate climbs because long prompts take more KV blocks.
   - *Detection:* Prompt-length histogram shifts (tracked by the load generator's bucket-tagged trace, or by a request-length metric at the API gateway); long-prompt P95 TTFT climbs on dashboard 5 ([exercise 05](../exercise-05-scheduler-metrics-into-prometheus/SOLUTION.md)).
   - *Mitigation:* Drop chunk size one step (next Pareto point down — trades long-TTFT for TPOT protection); or route long-prompt traffic to a dedicated pool when disaggregated prefill-decode is available (module 107).

2. **Hardware upgrade (H100 → H200 → Blackwell).**
   - *Symptom:* Iteration-time distribution shifts; the Pareto curve moves, and today's pick may no longer be the knee.
   - *Detection:* Compare iteration-time histogram before and after upgrade; TTFT and TPOT absolute values change even at the same config.
   - *Mitigation:* Re-run this exercise against the upgraded hardware before pinning the config. Chunk-size shape is often robust across generations; setpoint is not.

3. **Engine upgrade changes chunked-prefill scheduling.**
   - *Symptom:* Sweep results don't match the pinned pick; iteration-tokens-per-step histogram shape changes.
   - *Detection:* Release notes diff for chunked-prefill-related changes; sanity re-run of 1–2 sweep points after any engine upgrade across a major scheduler change.
   - *Mitigation:* Treat the sweep as stale on every engine minor version bump; add a CI job that re-runs the pinned-pick row and compares.

4. **Pool turns short-prompt-only.**
   - *Symptom:* Nobody notices for a week, then a dashboard review shows that the long-prompt bucket is < 1% of traffic.
   - *Detection:* Long-prompt bucket share dashboard panel (added to dashboard 2 in [exercise 05](../exercise-05-scheduler-metrics-into-prometheus/SOLUTION.md)).
   - *Mitigation:* If long-prompt share < 5% for a sustained window, chunked prefill becomes pure overhead and should be re-evaluated; a disable-and-recheck experiment is cheap.

5. **Interaction with preemption policy.**
   - *Symptom:* Chunked-prefill-mid-chunk sequences preempted in recompute mode lose chunk progress; preemption rate × chunk size dominates goodput.
   - *Detection:* `preemptions_total` counter rising faster than request rate; swap-count metric flat (confirming recompute mode).
   - *Mitigation:* Prefer swap mode when chunked prefill is on *and* preemption rate > 1%; see [exercise 04](../exercise-04-preemption-policy-swap-vs-recompute-drill/SOLUTION.md).

6. **Short-prompt TTFT regression** (bonus, caught by Plot 3).
   - *Symptom:* Plot 3 is not flat — short-prompt TTFT rises with chunk size.
   - *Detection:* Bucket-split TTFT panel on dashboard 5.
   - *Mitigation:* Typically a scheduler-delay or admission-pressure interaction; verify scheduler delay is off during the sweep and that `max-num-seqs` is not forcing short prompts to queue behind long ones.

Each entry maps to a dashboard panel in [exercise 05](../exercise-05-scheduler-metrics-into-prometheus/SOLUTION.md); the risk register is wired to the alerting stack, not left as a dead document.

---

## 3. Validation steps

A reviewer checks:

1. **Artefact A ratio.** Compute long-prompt share from the trace CSV. If < 15% or long P95 < 4000 tok, reject.
2. **Artefact B buckets split correctly.** Confirm the per-bucket P95 numbers in the table come from a filter on the `bucket` column, not from the aggregate. A single-aggregate P95 table defeats the purpose.
3. **Artefact B knob honesty.** For vLLM, confirm that each sweep row changed only the sweep knob; `max-num-seqs`, scheduler-delay, prefix-cache, and swap-space are held constant. Grep the launch commands.
4. **Artefact C plot shape.** Plot 1 (long TTFT vs. chunk size) should trend downward; Plot 2 (TPOT vs. chunk size) upward; Plot 3 (short TTFT vs. chunk size) roughly flat. If the shapes don't match, the writeup must explain why (e.g. workload too short-prompt, backend not actually implementing mixed batching).
5. **Artefact C pick.** The chosen point clears both SLOs with a named headroom percentage. The pick justification compares to at least two adjacent sweep rows and the off baseline.
6. **Artefact D depth.** ≥ 5 entries; each has all three sub-fields (symptom, detection, mitigation); each detection cites a metric that actually exists in the learner's engine `/metrics` output.
7. **Pinning.** The chosen chunk size is appended to the exercise-02 config file with a reasoning comment that cites the sweep trace location.

---

## 4. Rubric or review checklist

Pass threshold: ≥ 12 / 18.

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| Long-prompt bucket | < 15% share or P95 < 4000 tok | 15–20% share, P95 ≥ 4000 tok | ≥ 20% share, P95 ≥ 4000 tok, bucket tagged in trace |
| Sweep point count | < 5 points, no "off" baseline | 5 points but missing "off" | ≥ 5 points incl. "off" baseline |
| Bucket-split metrics | TTFT reported aggregate only | TTFT split by bucket, TPOT aggregate | Both TTFT and TPOT split by bucket where meaningful |
| Variable isolation | Multiple knobs changed per row | One knob changed, but scheduler-delay / prefix-cache state not noted | Only chunk-size changes; all other knobs frozen and verified |
| Plot shape and SLO overlay | No plots | Plots without SLO lines | Both plots with SLO overlays and visible trend |
| Pick defence | Pick named but not compared | Pick named, compared to 1 neighbour | Pick named, compared to ≥ 2 neighbours + "off" baseline, headroom % cited |
| Risk register count | < 5 entries | 5 entries missing one of the three sub-fields | ≥ 5 entries, each complete |
| Risk register metric grounding | Risks named without detection | Detection vague ("look at the dashboard") | Each detection names a specific metric or panel |
| Version-control pinning | Not committed | Pinned to config but no reasoning comment | Pinned with reasoning comment citing sweep trace |

---

## 5. Common mistakes

- **No long prompts in the workload.** Chunked prefill is a long-prompt mitigation. If the workload is uniformly short, the sweep is flat; document it and move on rather than fake a signal.
- **Mixing variables.** Changing chunk size *and* `max-num-batched-tokens` *and* `max-num-seqs` simultaneously is three experiments at once; hold the other two constant.
- **Global TPOT as the only axis.** Pure-decode windows will have great TPOT regardless of chunk size. The signal lives in mixed prefill+decode windows; split the metric by bucket.
- **Picking the extreme.** Chunk size 256 is pure TPOT protection; chunk size 4096 is pure TTFT protection. The pick is almost always in the middle.
- **Flag-spelling drift.** vLLM has reshaped chunked-prefill flags across releases (`--enable-chunked-prefill`, `--long-prefill-token-threshold`, `--max-num-partial-prefills` have all appeared / moved). Check the installed version's `--help` before writing the sweep script.- **Not verifying the knob is live.** If an engine version has the flag but no backend implementation, the sweep returns flat results. Watch the batched-tokens-per-iteration distribution; it should bimodally split into prefill-heavy and decode-heavy iterations.
- **Scheduler-delay left on.** If scheduler delay is on, chunk-size sweep results are noisier. Prefer scheduler-delay off during this sweep; note the setting.
- **The default might be the right answer.** Modern vLLM defaults chunked prefill on with a reasonable `max-num-batched-tokens`. A sweep that confirms the default is best is a *valid* outcome, not a failure — document it.

---

## 6. References

Primary research for chunked prefill:

- Agrawal et al., *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills* — <https://arxiv.org/abs/2308.16369> — the core insight the knob implements.
- Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, OSDI 2024 — <https://www.usenix.org/conference/osdi24/presentation/agrawal> — the production-ready treatment of the TTFT/TPOT trade-off this exercise sweeps.

Supporting research:

- Yu et al., *Orca*, OSDI 2022 — <https://www.usenix.org/conference/osdi22/presentation/yu>.
- Kwon et al., *PagedAttention* (vLLM), SOSP 2023 — <https://arxiv.org/abs/2309.06180>.
- Dao, *FlashAttention-2* — <https://arxiv.org/abs/2307.08691>; Shah et al., *FlashAttention-3* — <https://arxiv.org/abs/2407.08608>; FlashInfer — <https://github.com/flashinfer-ai/flashinfer>, paper <https://arxiv.org/abs/2501.01005> — attention backends that implement the mixed prefill+decode kernel path.

Engine documentation (verify flag spellings against pinned release):

- vLLM chunked-prefill docs — <https://docs.vllm.ai/> (search for "chunked prefill").
- SGLang server args — <https://docs.sglang.ai/>.
- TensorRT-LLM in-flight batching / `enable_chunked_context` — <https://nvidia.github.io/TensorRT-LLM/>.

Paired learning content:

- Module 103 chapter 03 (*Chunked Prefill and the TTFT / TPOT Trade-off*) in [`ai-infra-inference-engineer-learning`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning) — the source of the pick rule.
- Exercise 02 solution — the config baseline this sweep refines.
- Exercise 05 solution — the dashboards the risk register wires to.
