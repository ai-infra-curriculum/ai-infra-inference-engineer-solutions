# Solution — Exercise 05: Agentic Workload Shaped Benchmark

> Paired learning file: [`lessons/mod-112-multimodal-and-emerging-serving/exercises/exercise-05-agentic-workload-shaped-benchmark.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-112-multimodal-and-emerging-serving/exercises/exercise-05-agentic-workload-shaped-benchmark.md)
> Module: `mod-112-multimodal-and-emerging-serving`
> Type: design + implementation

## 1. Solution overview

The learner designs and implements a **shaped benchmark** for an agentic LLM
workload — a benchmark that imitates what an agent loop actually does to an
inference engine rather than running the usual "fixed-prompt, fixed-output"
sweep. The deliverables are a workload specification, a generator that
produces that workload, and a results report with the two metrics that matter
for agent user experience: **time-to-first-tool-call** (TTFT in agent
units) and **inter-step latency**.

Three artefacts:

| Artefact | Form | Pass bar |
|---|---|---|
| A — Workload specification | 1-page spec | Names the prompt-length and output-length distributions, tool-call density per turn, number of turns per session, concurrency shape (bursty vs. steady), and the "action distribution" (text vs. tool-call vs. terminate) |
| B — Shaped load generator | Script | Reproduces the spec against an OpenAI-compatible endpoint; honours each session as an actual multi-turn exchange, not independent requests |
| C — Results report | Table + method note | Reports p50 and p95 for both whole-session wall-clock and per-step latency; names the saturation point (concurrency at which inter-step p95 breaches an agent UX floor, e.g. 2 s) |

### Why agent workloads need their own benchmark

Non-agent benchmarks (fixed 512-in / 128-out at target concurrency) are useful
for engine-level comparisons and have been the reasonable default for serving
work this curriculum has leaned on so far (see module 103, module 107). They
measure the engine's ability to turn fixed shapes into tokens.

Agent workloads break that assumption in five ways at once:

1. **Multi-turn sessions.** A single user request generates N turns of
   LLM-planner → tool-exec → LLM-observe → ... loops. The benchmark unit is
   the **session**, not the request.
2. **Highly asymmetric per-turn shapes.** Planner turns are often short in,
   short out (a JSON tool-call). Observer turns are long in (tool output
   dumped into context), short out. Final-answer turns are short in, long out.
   Averaging over turns hides the shape that stresses the engine.
3. **Growing prefix.** Each turn's prompt is the full conversation so far,
   tool call history, and tool outputs. Prefix length grows monotonically
   through the session. Prefix-caching behaviour determines whether this is
   cheap or expensive, and [[kv-cache-management]] is the mechanism.
4. **Short output budget but tight latency SLO.** The planner must respond
   in a user-perceived "typing" time; users notice step-to-step gaps. A fleet
   that holds 95th-percentile total throughput at a cost of 5 s per-step p95
   is useless for agents.
5. **Burstiness.** Agent fleets don't arrive uniformly; a user clicks "run"
   and N parallel tool calls dispatch simultaneously. The benchmark must
   include burst arrivals, not Poisson-smooth them away.

A "shaped" benchmark is one whose stochastic profile matches those five
properties. It is not an attempt to measure one number ("tokens/sec"); it is
an attempt to answer two questions: what concurrency breaks agent UX, and
which of the five properties above was the cause.

## 2. Implementation — worked design + generator

### 2.1 Workload specification

The specification below is deliberately explicit. Pick concrete numbers from
your fleet's observed telemetry if you have them; otherwise, the values here
are a reasonable starting point for a code-assistant / planner-agent profile
and should be labelled as such in any report.

```yaml
# agentic-shape.yaml
workload: agentic-planner
notes: Shape estimated from a code-assistant style profile; verify against
       own telemetry before citing numbers downstream.

session:
  turns_distribution: {dist: lognormal, mu: 1.8, sigma: 0.7}   # ~6-turn median
  max_turns: 30

turn:
  # Fraction of turns of each kind, drawn independently within a session.
  action_mix:
    tool_call: 0.55         # planner emits JSON tool invocation
    tool_result: 0.30       # client replays tool output into context
    text_reply: 0.10        # planner replies to the user in text
    terminate:  0.05        # session ends

  # Per-turn shape in tokens.
  prompt_tokens_new:       # new tokens added to the prefix this turn
    tool_call:   {dist: lognormal, mu: 3.5, sigma: 0.5}
    tool_result: {dist: lognormal, mu: 6.5, sigma: 1.1}   # long tool output
    text_reply:  {dist: lognormal, mu: 4.0, sigma: 0.6}
  output_tokens:
    tool_call:   {dist: lognormal, mu: 4.0, sigma: 0.5}   # short JSON
    tool_result: {dist: constant,  value: 0}              # client-driven
    text_reply:  {dist: lognormal, mu: 5.0, sigma: 0.7}

arrival:
  base_rate_rps: 2.0              # steady session-start rate
  burst:
    probability: 0.15             # fraction of session starts that are bursts
    size: {dist: uniform, low: 4, high: 12}   # concurrent sessions per burst

session_concurrency_cap: 64        # engine-level ceiling on live sessions
tool_latency_ms: {dist: lognormal, mu: 5.3, sigma: 0.8}   # emulated tool exec
```

The spec is auditable: every distribution has a name, a parameterisation, and
a note on where it came from. Reviewers can question individual choices
without having to reverse-engineer the generator.

### 2.2 Shaped generator

```python
# agentic_bench.py
import argparse, asyncio, math, random, time
import httpx
import numpy as np

random.seed(1337)
rng = random.Random(1337)

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
URL = "http://localhost:8050/v1/chat/completions"

def draw_lognorm(mu, sigma):
    return max(1, int(round(math.exp(rng.gauss(mu, sigma)))))

def pick(action_mix):
    r = rng.random(); acc = 0
    for k, p in action_mix.items():
        acc += p
        if r <= acc: return k
    return list(action_mix.keys())[-1]

ACTION_MIX = {"tool_call": 0.55, "tool_result": 0.30, "text_reply": 0.10, "terminate": 0.05}
PROMPT_SHAPE = {"tool_call": (3.5, 0.5), "tool_result": (6.5, 1.1), "text_reply": (4.0, 0.6)}
OUTPUT_SHAPE = {"tool_call": (4.0, 0.5), "tool_result": (0, 0),     "text_reply": (5.0, 0.7)}

async def one_turn(client, session_id, history_tokens, turn_kind):
    """Return (step_latency_ms, new_output_tokens)."""
    if turn_kind == "tool_result":
        # Client replays a long tool-output chunk — this is a free turn for
        # the engine's decode side but it grows the prefix.
        payload_tokens = draw_lognorm(*PROMPT_SHAPE["tool_result"])
        history_tokens += payload_tokens
        await asyncio.sleep(draw_lognorm(5.3, 0.8) / 1000.0)
        return 0, 0

    prompt_add = draw_lognorm(*PROMPT_SHAPE[turn_kind])
    out = draw_lognorm(*OUTPUT_SHAPE[turn_kind])
    # Build a fake conversation whose token count approximates (history + add).
    prompt_text = "token " * (history_tokens + prompt_add)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": out,
        "temperature": 0.0,
        "user": f"sess-{session_id}",
    }
    t0 = time.perf_counter()
    r = await client.post(URL, json=body, timeout=180)
    r.raise_for_status()
    dt_ms = (time.perf_counter() - t0) * 1000
    return dt_ms, out

async def one_session(client, session_id):
    turns = draw_lognorm(1.8, 0.7)
    history = draw_lognorm(4.5, 0.4)   # initial system + user prompt
    step_latencies = []
    t0 = time.perf_counter()

    for t in range(turns):
        kind = pick(ACTION_MIX)
        if kind == "terminate":
            break
        dt, out = await one_turn(client, session_id, history, kind)
        if dt > 0:
            step_latencies.append(dt)
            history += out
    wall = (time.perf_counter() - t0) * 1000
    return {"session": session_id, "wall_ms": wall, "step_latencies": step_latencies}

async def arrivals(concurrency_cap, duration_s, base_rps, burst_prob, burst_lo, burst_hi):
    """Yield session IDs over `duration_s`, respecting a hard concurrency cap."""
    end = time.perf_counter() + duration_s
    sid = 0
    while time.perf_counter() < end:
        if rng.random() < burst_prob:
            batch = rng.randint(burst_lo, burst_hi)
            for _ in range(batch):
                sid += 1
                yield sid
        else:
            sid += 1
            yield sid
        await asyncio.sleep(rng.expovariate(base_rps))

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration-s", type=int, default=120)
    p.add_argument("--concurrency-cap", type=int, default=64)
    p.add_argument("--base-rps", type=float, default=2.0)
    args = p.parse_args()

    sem = asyncio.Semaphore(args.concurrency_cap)
    results = []

    async with httpx.AsyncClient() as client:
        # Warm-up
        await one_session(client, 0)
        results.clear()

        async def run(sid):
            async with sem:
                results.append(await one_session(client, sid))

        tasks = []
        async for sid in arrivals(args.concurrency_cap, args.duration_s,
                                   args.base_rps, 0.15, 4, 12):
            tasks.append(asyncio.create_task(run(sid)))
        await asyncio.gather(*tasks)

    walls = np.array([r["wall_ms"] for r in results])
    steps = np.array([x for r in results for x in r["step_latencies"]])
    print(f"sessions={len(results)}  total_steps={len(steps)}")
    print(f"session_wall p50={np.percentile(walls,50):.0f} ms  p95={np.percentile(walls,95):.0f} ms")
    print(f"step_latency p50={np.percentile(steps,50):.0f} ms  p95={np.percentile(steps,95):.0f} ms")

asyncio.run(main())
```

The generator intentionally approximates token counts with a filler word; for
higher-fidelity reporting, route a tokenizer-aware prompt builder here. The
distribution shapes are the point.

### 2.3 The two metrics that matter

**Per-step latency (p50 / p95).** The time from the planner issuing a request
to the planner receiving its response. This drives perceived responsiveness.
Humans and downstream tools both notice when it crosses a UX floor. A
reasonable starting floor for a coding-assistant agent is `step p95 ≤ 2 s`;
your product sets its own.

**Session wall-clock (p50 / p95).** End-to-end time for a full agent session.
This matters for cost reporting and for user-flow retention.

Both are needed because they fail independently. An engine can hit a low
median session wall by batching aggressively and still have a horrible p95
step latency that makes the agent feel broken. Conversely, an engine can hit
a low step p95 but spend most of its session wall-clock in tool-exec waits
that have nothing to do with the LLM.

### 2.4 Running the sweep

Hold model, hardware, and engine version fixed. Sweep `session_concurrency_cap`
across a representative range (e.g. 4, 16, 32, 64, 128). Record both metrics
at each cell. Identify the **saturation point** — the concurrency at which
step p95 first breaches the UX floor. That is the actionable output: it tells
capacity planning how many concurrent agent sessions the current engine
tolerates before user experience falls over.

| Concurrency cap | step p50 | step p95 | session wall p50 | session wall p95 | sessions completed |
|---|---|---|---|---|---|
| 4 | *measured* | *measured* | *measured* | *measured* | *measured* |
| 16 | *measured* | *measured* | *measured* | *measured* | *measured* |
| 32 | *measured* | *measured* | *measured* | *measured* | *measured* |
| 64 | *measured* | *measured* | *measured* | *measured* | *measured* |
| 128 | *measured* | *measured* | *measured* | *measured* | *measured* |

Expected qualitative pattern:

- Step p50 is near-flat below saturation and rises sharply beyond it.
- Step p95 is the leading indicator; it rises before p50 does.
- Session wall-clock p50 rises more smoothly: as some turns queue, others are
  still being decoded; the session-level average is less bursty than the
  per-step view.
- Prefix-caching off vs. on changes the saturation point materially for
  long-session workloads. Running the benchmark with and without prefix-cache
  enabled is a useful diagnostic sub-sweep.

## 3. Validation steps

1. **Spec is explicit.** `agentic-shape.yaml` (or equivalent) is checked in,
   lists every distribution with its parameterisation, and names the profile
   source.
2. **Generator reproduces the spec.** The generator's code paths map 1:1 to
   the spec; a reviewer can cross-check the action-mix fractions and the
   per-kind token distributions against the YAML.
3. **Sessions are actual multi-turn flows.** Within one session, the prompt
   grows monotonically turn-over-turn; two sessions do not share state. This
   is directly observable in the engine's prefix-cache metrics, which should
   show high hit rate inside a session.
4. **Both metrics reported.** Step latency and session wall-clock are both
   computed, at p50 and p95.
5. **Burst arrivals present.** The generator's arrival pattern includes
   bursts, not only Poisson base-rate. A 10-session burst is observable as a
   visible step in the submitted engine's concurrent-session gauge.
6. **Saturation point named.** The results report identifies the concurrency
   at which step p95 breaches the stated UX floor, not just a table.
7. **Method note.** Workload YAML version, model, engine version, hardware
   class, and warm-up policy are stated.

## 4. Rubric / review checklist

Score each item 0 (missing), 1 (partial), 2 (fully correct). Passing: ≥ 10 / 14.

| # | Item | Points |
|---|---|---|
| 1 | Shaped workload spec is explicit: distributions, action-mix, burst arrivals, tool-latency | 2 |
| 2 | Generator implements the spec end-to-end; prompts grow monotonically within a session | 2 |
| 3 | Concurrency sweep run; sessions driven, not independent requests | 2 |
| 4 | Step-latency and session-wall-clock metrics both reported at p50 and p95 | 2 |
| 5 | Saturation point named against a stated UX floor | 2 |
| 6 | Prefix-cache behaviour acknowledged (either measured or named as a next step) | 2 |
| 7 | Reviewer can reproduce the sweep from the spec + generator + launch commands | 2 |

## 5. Common mistakes

- **Running independent requests instead of sessions.** The whole point is
  that the prompt grows within a session. Independent-request benchmarks look
  "agent-like" (JSON outputs, short prompts) but miss the prefix-growth and
  prefix-cache dynamics that distinguish agents from chat.
- **Averaging over turn kinds.** Mixing tool-call turns (short-out) with
  text-reply turns (long-out) and reporting one mean output length produces a
  shape no real agent ever sends. Report per-kind, not aggregated.
- **Only reporting one metric.** Either step p95 or session wall-clock alone
  is misleading. Both failure modes exist in production; both must be
  measured.
- **Smoothing the arrival process.** A Poisson-only arrival hides the burst
  failure mode. Real agent fleets fire in bursts whenever a user action
  triggers parallel tool calls.
- **Ignoring the client-side tool wait.** Emulating tool execution with a
  realistic latency distribution is part of the shape. A zero-tool-wait
  benchmark lets the engine run hot continuously; real sessions have idle
  gaps that let the engine recover.
- **Forgetting prefix-cache state across runs.** The engine's prefix cache
  may be warm or cold between sweeps. Either reset it at each sweep cell or
  run multiple repeats and discard the first.
- **Using temperature > 0 with a correctness metric in the loop.** This is a
  latency benchmark; correctness is explicitly out of scope. Temperature = 0
  (or fixed seed) keeps output length distributions tight and repeatable.
- **Reporting token/s.** Agentic workloads are session-latency bound, not
  throughput bound. Token/s is a derived number here, not the KPI.
- **Running at only one concurrency.** The saturation point is the actionable
  output. A single-concurrency number is a point measurement without a curve,
  which is uninterpretable.

## 6. References

- vLLM documentation. *Benchmarks and serving metrics.*
  https://docs.vllm.ai/ — practitioner metrics inventory; pin the
  prefix-cache and concurrent-session series to the deployed release.
- MLPerf Inference. *LLM serving benchmark rules.*
  https://mlcommons.org/en/inference-datacenter/ — reference for latency
  / throughput definitions and warm-up discipline. Not directly reusable
  for agents but useful as a baseline for discipline.
- OpenAI. *Function calling and the Responses API.*
  https://platform.openai.com/docs/guides/function-calling — reference
  wire shape for tool-call turns the benchmark imitates.
- Anthropic. *Tool use.*
  https://docs.anthropic.com/en/docs/build-with-claude/tool-use —
  alternative tool-use wire shape; useful cross-reference.
- LangChain / LangGraph documentation. *Agent execution loops.*
  https://docs.langchain.com/ — practitioner implementation of the planner /
  tool / observer loop the benchmark imitates.
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/)
  — tool-calling exposes prompt-injection pathways not covered by this
  latency benchmark; covered adjacently in a safety module.
- [mitre-atlas](https://atlas.mitre.org/) — tool-use chains map onto
  ATLAS's exfiltration / execution tactics; relevant for abuse testing on
  top of this benchmark.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework) —
  governance baseline for agent workloads whose shape this exercise
  measures.
