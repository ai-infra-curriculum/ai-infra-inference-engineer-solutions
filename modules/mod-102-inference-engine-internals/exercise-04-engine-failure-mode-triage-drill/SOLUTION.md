# Exercise 04 — Engine Failure-Mode Triage Drill

> Paired learning exercise: `lessons/mod-102-inference-engine-internals/exercises/exercise-04-engine-failure-mode-triage-drill.md`.

## 1. Solution overview

This is a tabletop drill: the learner is given a set of symptom cards
(latency spike, OOM, hanging request, silent accuracy drop, process crash
with no traceback, etc.) and must produce, for each, a triage runbook —
first-5-minutes checks, hypotheses to confirm / disconfirm, and the fix
owners. The output is not code; it is a decision table and three sample
runbooks worked end-to-end.

This solution provides the reference decision table the learner's
submission must cover and three worked runbooks the learner's submission
should resemble in structure.

## 2. Implementation — Worked triage artefacts

### 2.1 Engine failure-mode decision table (reference set)

The learner's table should cover at minimum these ten modes. Each row
should name: (a) the symptom as seen from a dashboard / client, (b) the
first-5-minutes checks, (c) likely root causes, (d) the fix path.

| # | Symptom | First-5-min checks | Likely causes | Fix path |
|---|---|---|---|---|
| 1 | P99 TTFT spike, throughput flat | Scheduler queue depth, running-batch size, chunked-prefill off? | Bursty traffic, chunked-prefill disabled, prefix-cache cold | Enable chunked prefill, warm cache, admission control |
| 2 | P99 ITL spike, QPS flat | KV free blocks, preemption counter | KV pressure → preemption/recompute | Lower `max_num_seqs`, raise GPU mem fraction, enable KV reuse |
| 3 | OOM at startup | Model size vs. memory, dtype, TP, KV reservation | Precision/TP misconfig, KV fraction too high | Lower KV fraction, enable quantization, raise TP |
| 4 | OOM at steady state | KV pool stats, allocator high-water mark | Context overflow, KV leak on cancelled requests | Hard-cap `max_seq_len`, upgrade engine, verify cancellation path |
| 5 | Hang: one request never completes | Streaming connection state, GPU kernel active? | Lost cancellation, deadlock in a worker, kernel stuck (driver) | Timeout + retry policy, `nvidia-smi` for stuck kernels, driver restart last |
| 6 | Silent accuracy drop | Model/tokenizer/engine version hashes, sampling defaults | Mismatched tokenizer, wrong precision, dtype cast regression | Pin + audit versions, re-run golden set, rollback |
| 7 | Process crash, no Python traceback | `dmesg`, container exit code, CUDA error from driver | CUDA OOM, illegal memory access (kernel bug), ECC error | Check `dmesg`/XID, pin engine version, file upstream bug |
| 8 | Throughput collapse under TP | NCCL comm time, fabric utilisation, GPU idle | Network bottleneck, bad topology, NCCL env misconfig | Confirm NVLink/InfiniBand, set `NCCL_*` env, reseat if hardware |
| 9 | Latency regression after upgrade | Diff of engine version + flags, CUDA graph capture status | Default flag changed, graph capture disabled, new codepath | Bisect flags, revert if needed, open upstream issue |
|10 | Rate of 429 / rejected requests | Admission log, max-running cap, token budget | Overload / caps too tight / client retry storm | Raise caps *if* GPU has headroom; otherwise shed load at LB |

A correct submission does not need to match these exactly, but must cover
the same functional ground: latency failures, memory failures, hangs,
correctness failures, crash failures, distributed failures, and overload /
admission failures.

### 2.2 Three worked runbooks

**Runbook A — "P99 ITL doubled at 10:04 UTC; throughput unchanged."**

1. *Pull the four signals.* Running batch size, KV-pool free blocks,
   preemption count, scheduler iteration time. *Hypothesis:* KV pressure is
   causing preemption (symptom #2).
2. *Confirm.* If free-block percentage has dropped and preemption count is
   climbing in step with the ITL spike, confirmed.
3. *Immediate mitigation.* Lower `max_num_seqs` or the token budget to
   reduce the number of concurrent sequences holding KV. If prefix caching
   is off and traffic has a shared prefix, enable it.
4. *Root-cause follow-up.* Correlate with request mix (longer contexts?
   more concurrent streams?). If it is a traffic shift, update the capacity
   model; if it is a slow leak, grab a heap/pool snapshot and compare
   against last week's baseline.
5. *Exit criteria.* Preemption count back near zero; P99 ITL back within
   SLO for 15 minutes.

**Runbook B — "One request has been in-flight for 300s; others are fine."**

1. *Confirm it's really hung, not just long.* Is the client still connected
   and streaming? What is the token count vs. `max_tokens`?
2. *Check GPU side.* `nvidia-smi` — is a kernel active on the engine's
   GPU? Scheduler logs — is the request still in the running set?
3. *Branch:*
   - If the request is in the running set and tokens are advancing, it's
     just a legitimately long generation; enforce a server-side
     `max_tokens` or wall-clock timeout.
   - If the request is in the running set but tokens are *not* advancing,
     cancellation was lost; forcibly cancel via the engine's abort API,
     file a bug against the engine.
   - If the request is *not* in the running set but the client never got
     a terminal event, the streaming path dropped a message; the fix is
     at the API layer, not the engine.
4. *Exit criteria.* Request terminal event delivered to the client or
   explicit error; follow-up ticket filed for whichever path was broken.

**Runbook C — "Eval score dropped 4 points after today's deploy; no code
changes to the model itself."**

1. *Diff the release.* Engine version, backend version, tokenizer version,
   sampling defaults, precision, quantization. In vLLM this includes
   `--dtype`, `--quantization`, `--kv-cache-dtype`. In TRT-LLM this
   includes the engine build hash and the backend `config.pbtxt`. In SGLang
   this includes `--dtype`, `--quantization`, radix cache / chunked
   prefill defaults.
2. *Rerun the golden set under the previous release.* If the regression
   reproduces on old binaries, it is not the engine; look at data or
   prompt changes. If it reproduces only on the new release, it is the
   engine.
3. *Bisect.* First the big-ticket flags (precision, quantization,
   kv-cache dtype). Then the smaller ones. One flag flip per rerun.
4. *Fix.* Pin the offending flag back to the previous value in the
   deployment config. File an upstream issue with a minimal repro.
5. *Exit criteria.* Golden set back within tolerance; root-cause flag
   documented in the change log.

### 2.3 On-call survival kit (what the triage artefact must include)

- A dashboard pointer per engine (vLLM Prometheus / SGLang server-info /
  Triton metrics endpoint) with the four signals above pre-queried.
- A documented abort / cancel API per engine.
- A documented version-pinning file per deployment.
- A documented rollback procedure that is at most one command.
- A list of the top three bugs currently filed against the pinned engine
  version, with workaround notes. <!-- verify-source: live ticket queue;
  learners should include their own team's. -->

## 3. Validation steps

1. **Peer review.** Have another on-call engineer walk through the
   decision table with a symptom they bring; the table should route them
   to the correct runbook in under a minute.
2. **Game-day.** Inject one of the failure modes in staging (e.g. drive
   KV pressure by firing long-context requests with `max_num_seqs`
   pinned high) and execute the matching runbook end-to-end.
3. **Artefact completeness.** Every runbook must have: hypothesis,
   confirmation step, immediate mitigation, root-cause follow-up, exit
   criteria. A missing one is a fail.
4. **Rollback rehearsal.** Perform the documented rollback at least once
   on a non-production tenant.

## 4. Rubric / review checklist

| # | Criterion | Weight | Pass bar |
|---|---|---|---|
| 1 | Decision table coverage | 20% | All seven failure families (latency, memory, hang, correctness, crash, distributed, overload) represented |
| 2 | Each row actionable | 20% | Every row ends in a concrete fix path, not "investigate" |
| 3 | Three worked runbooks | 25% | Hypothesis → confirm → mitigate → root-cause → exit criteria, all present in all three |
| 4 | Engine-specific signals named | 10% | At least one real metric name from each of vLLM / SGLang / TRT-LLM + Triton |
| 5 | Rollback procedure | 10% | Single-command rollback documented |
| 6 | Game-day evidence | 10% | Screenshot or log from one injected failure + its runbook execution |
| 7 | Version pinning policy | 5% | Where engine version is pinned and how it is bumped |

Passing threshold: 70%.

## 5. Common mistakes

- **Treating "latency up" as a single failure mode.** TTFT and ITL have
  different causes; not separating them means half your runbooks are wrong.
- **Blaming the engine for client bugs.** Lost cancellations and stuck
  streams are often at the API / proxy layer, not the engine.
- **Skipping the version-diff step on accuracy regressions.** Engines ship
  silent precision / default changes; ignoring the version diff sends you
  down the wrong investigation path.
- **Rolling forward through an outage.** The runbook must have a rollback;
  "upgrade past it" is not a rollback.
- **Mixing generic SRE advice with engine-specific triage.** On-call needs
  real metric names and real API calls, not "check the logs."
- **Testing the runbooks only in the drill.** Game-day is the whole point;
  undertested runbooks fail on the one night they matter.

## 6. References

- vLLM observability docs — https://docs.vllm.ai/ (metrics endpoint, debug
  logging). <!-- verify-source: metric names change; pin the version. -->
- SGLang server info / metrics —
  https://docs.sglang.ai/ (endpoint and field names). <!-- verify-source. -->
- NVIDIA Triton metrics —
  https://docs.nvidia.com/deeplearning/triton-inference-server/ (`metrics`
  endpoint, Prometheus format).
- TensorRT-LLM backend observability notes —
  https://github.com/triton-inference-server/tensorrtllm_backend
  (inflight-batcher stats).
- NVIDIA XID error reference —
  https://docs.nvidia.com/deploy/xid-errors/ (driver / GPU hardware error
  codes; needed for crash runbooks).
- OWASP Machine Learning Security Top 10 —
  https://owasp.org/www-project-machine-learning-security-top-10/ (ML05:
  Model Theft and ML09: Output Integrity Attack — both intersect with
  "silent accuracy drop" triage).
- MITRE ATLAS — https://atlas.mitre.org/ (adversary techniques against ML
  systems; useful when triaging unexplained output changes to rule out
  abuse before concluding it is a model regression).
- NIST AI Risk Management Framework —
  https://www.nist.gov/itl/ai-risk-management-framework ("Manage" function:
  incident response and rollback procedures map directly to AI RMF manage-
  phase practice).
