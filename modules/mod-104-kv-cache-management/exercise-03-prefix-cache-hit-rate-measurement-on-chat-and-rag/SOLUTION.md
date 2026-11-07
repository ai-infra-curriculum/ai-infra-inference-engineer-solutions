# Exercise 03 — Prefix-Cache Hit-Rate Measurement on Chat and RAG

> Module: `mod-104-kv-cache-management`
> Exercise: `exercise-03-prefix-cache-hit-rate-measurement-on-chat-and-rag`
> Type: implementation / measurement

## 1. Solution overview

The exercise asks the learner to **measure prefix-cache effectiveness on two
contrasting workloads** against a vLLM or SGLang server:

- **Chat.** Multi-turn conversations whose growing transcript is a monotonically
  extending prefix per session. Expected to be *very* cache-friendly — each new user
  turn should hit on essentially all prior turns.
- **RAG.** Each request packs retrieved passages ahead of a short user question.
  Depending on how retrieval is deduplicated and how the template places shared text,
  hit rate ranges from "near zero" (unique passages, system prompt first) to "high"
  (system prompt long, chunks reused).

The deliverable is a short report with (a) hit-rate numbers, (b) TTFT and throughput
deltas with prefix caching on vs off, and (c) an interpretation that ties the
measurements back to how the template was constructed.

The reference solution below uses **vLLM** because its OpenAI-compatible server is
the simplest to drive. A parallel SGLang recipe is included because SGLang's
RadixAttention is the system the lesson pairs with. Both servers expose Prometheus
metrics that the harness scrapes.

## 2. Implementation

The directory contains the following artifacts (create them alongside this
`SOLUTION.md` when running the exercise yourself):

```
exercise-03-prefix-cache-hit-rate-measurement-on-chat-and-rag/
├── SOLUTION.md                 (this file)
├── workloads/
│   ├── chat_workload.py        # synthesizes multi-turn chat sessions
│   └── rag_workload.py         # synthesizes RAG prompts over a document corpus
├── harness/
│   ├── run_vllm.sh             # launch script with/without --enable-prefix-caching
│   ├── run_sglang.sh           # launch script toggling --disable-radix-cache
│   ├── fire.py                 # concurrent OpenAI-API client, records latencies
│   └── scrape_metrics.py       # pulls Prometheus counters at start/end
└── report/
    └── results_template.md     # table skeleton to fill in
```

### 2.1 Launch scripts

`harness/run_vllm.sh` (snippet):

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
PORT="${PORT:-8000}"
MODE="${1:-prefix-on}"    # prefix-on | prefix-off

args=(
  --model "$MODEL"
  --port "$PORT"
  --max-model-len 8192
  --gpu-memory-utilization 0.9
  --disable-log-requests
)

case "$MODE" in
  prefix-on)   args+=(--enable-prefix-caching) ;;
  prefix-off)  args+=(--no-enable-prefix-caching) ;;     # V1 enables it by default; disable explicitly
  *) echo "usage: $0 {prefix-on|prefix-off}" >&2; exit 2 ;;
esac

exec python -m vllm.entrypoints.openai.api_server "${args[@]}"
```

`harness/run_sglang.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
PORT="${PORT:-30000}"
MODE="${1:-prefix-on}"

args=( --model-path "$MODEL" --port "$PORT" )
[[ "$MODE" == "prefix-off" ]] && args+=(--disable-radix-cache)

exec python -m sglang.launch_server "${args[@]}"
```

### 2.2 Chat workload

`workloads/chat_workload.py` (essentials):

```python
"""Synthesize N_SESSIONS multi-turn chats; each session grows monotonically."""
import json, random, pathlib

TEMPLATE_SYS = (
    "You are a careful technical assistant. Answer concisely and cite sources "
    "when the user asks for them."
)

def session(rng, turns):
    history = [{"role": "system", "content": TEMPLATE_SYS}]
    for _ in range(turns):
        history.append({"role": "user", "content": rng.choice(USER_PROMPTS)})
        history.append({"role": "assistant", "content": rng.choice(ASSISTANT_REPLIES)})
    return history

USER_PROMPTS = [
    "Summarize PagedAttention in two sentences.",
    "Explain KV cache fragmentation.",
    "What does chunked prefill solve?",
    # ... load from a static list; keeping prompts fixed is what gives sharing
]
ASSISTANT_REPLIES = [
    "PagedAttention stores KV ...",
    "Fragmentation arises when ...",
    # ... fixed short strings
]

def emit(path, n_sessions=64, max_turns=8, seed=0):
    rng = random.Random(seed)
    with open(path, "w") as f:
        for sid in range(n_sessions):
            turns = rng.randint(2, max_turns)
            full = session(rng, turns)
            # emit incremental requests: at turn k, the prompt is history[:2k+1]
            for k in range(1, turns + 1):
                prompt = full[: 1 + 2 * k - 1]   # system + 2k-1 messages
                f.write(json.dumps({
                    "session_id": sid,
                    "turn": k,
                    "messages": prompt,
                    "max_tokens": 64,
                }) + "\n")

if __name__ == "__main__":
    emit(pathlib.Path(__file__).with_name("chat.jsonl"))
```

Properties this produces:

- **Growing prefix.** Request `(sid, k)` is a strict prefix of `(sid, k+1)` for the
  same session — the exact structure RadixAttention and vLLM's prefix cache target.
- **Fixed system prompt** across all sessions — guaranteed cross-session reuse of the
  first N tokens.

### 2.3 RAG workload

`workloads/rag_workload.py`:

```python
"""Synthesize N_REQS RAG prompts: system + K passages + user question."""
import json, random, pathlib, hashlib

CORPUS = [open(p).read() for p in pathlib.Path("corpus").glob("*.txt")]

SYSTEM = (
    "Use the provided passages to answer the user. "
    "If the passages do not contain an answer, say so."
)
TEMPLATE = (
    "{system}\n\n"
    "[Passages]\n{passages}\n[/Passages]\n\n"
    "Question: {question}\n"
    "Answer:"
)

def emit(path, n=256, k_passages=4, template_variant="system_first", seed=0):
    rng = random.Random(seed)
    with open(path, "w") as f:
        for i in range(n):
            picks = rng.sample(CORPUS, k_passages)
            question = rng.choice(QUESTIONS)
            passages_text = "\n---\n".join(picks)
            if template_variant == "system_first":
                prompt = TEMPLATE.format(system=SYSTEM, passages=passages_text,
                                         question=question)
            elif template_variant == "passages_first":
                prompt = (
                    f"[Passages]\n{passages_text}\n[/Passages]\n\n"
                    f"{SYSTEM}\n\nQuestion: {question}\nAnswer:"
                )
            f.write(json.dumps({"id": i, "prompt": prompt, "max_tokens": 128}) + "\n")

QUESTIONS = ["What is X?", "Summarize Y.", "..."]
```

Template variants are the point of the comparison:

- `system_first`: the long shared system text sits before the variable passages, so
  prefix matching captures `|SYSTEM|` tokens on every request — *modest* hit rate on
  typical templates.
- `passages_first`: the shared text is **after** the variable passages. Prefix
  matching starts at the very first token and diverges immediately; hit rate collapses
  to ~0.

This is the central teaching point: *prefix* caches only help on text that appears at
the **start** of the prompt. Putting a shared system prompt after a variable chunk of
retrieved context destroys the hit rate.

### 2.4 Firing clients

`harness/fire.py` (essentials):

```python
import argparse, asyncio, aiohttp, json, time, statistics
from collections import defaultdict

async def one(session, url, payload):
    t0 = time.perf_counter()
    async with session.post(f"{url}/v1/chat/completions", json=payload) as r:
        first_byte = None
        async for chunk in r.content.iter_any():
            if first_byte is None:
                first_byte = time.perf_counter() - t0
        total = time.perf_counter() - t0
    return first_byte, total

def build_payload(r, model):
    payload = {"model": model, "stream": True,
               "max_tokens": r.get("max_tokens", 128)}
    if "messages" in r:
        payload["messages"] = r["messages"]
    else:
        payload["messages"] = [{"role": "user", "content": r["prompt"]}]
    return payload

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    args = ap.parse_args()

    reqs = [json.loads(l) for l in open(args.jsonl)]
    # Group by session so later turns wait on their predecessors; preserve the
    # file's turn order within each session. Non-chat rows (no session_id) each
    # get their own group so they run with full concurrency.
    groups = defaultdict(list)
    for i, r in enumerate(reqs):
        key = r.get("session_id", f"_solo_{i}")
        groups[key].append(r)

    sem = asyncio.Semaphore(args.concurrency)
    results = [None] * len(reqs)
    # Map each request back to its index in the input order so results align.
    index_of = {id(r): i for i, r in enumerate(reqs)}

    async with aiohttp.ClientSession() as session:
        async def run_session(rs):
            # Sequential within a session: turn k+1 only starts after turn k
            # returns, so prefix lookups see the prior turn's cached KV.
            for r in rs:
                async with sem:
                    results[index_of[id(r)]] = await one(
                        session, args.url, build_payload(r, args.model))

        await asyncio.gather(*[run_session(rs) for rs in groups.values()])

    ttfts = sorted(x[0] for x in results if x[0])
    print(f"n={len(results)}  ttft p50={ttfts[len(ttfts)//2]:.3f}s "
          f"p95={ttfts[int(len(ttfts)*0.95)]:.3f}s "
          f"p99={ttfts[int(len(ttfts)*0.99)]:.3f}s")

asyncio.run(main())
```

### 2.5 Reading hit rate from metrics

**vLLM** exposes prefix-cache counters via its Prometheus `/metrics` endpoint. The
relevant series are (names stable as of vLLM v0.6+; verify against your version with
`curl -s $URL/metrics | grep prefix`):

- `vllm:prefix_cache_queries` — tokens looked up.
- `vllm:prefix_cache_hits` — tokens served from cache.

Hit rate = `prefix_cache_hits / prefix_cache_queries`.

**SGLang** exposes per-request cache hit tokens as well as a server-wide counter at
`/metrics`; the field names in use at the time of writing are
`cache_hit_rate` and `prefix_cache_hit_tokens` (verify with
`curl -s $URL/metrics` on your version).

`harness/scrape_metrics.py`:

```python
import argparse, pathlib, re, urllib.request

def grab(source):
    # Accept either an http(s) URL or a local file path to a saved snapshot.
    if source.startswith(("http://", "https://")):
        text = urllib.request.urlopen(source).read().decode()
    else:
        text = pathlib.Path(source).read_text()
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or " " not in line: continue
        name, _, value = line.partition(" ")
        out[name] = float(value)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    args = ap.parse_args()
    b, a = grab(args.before), grab(args.after)
    for k in sorted(a):
        if any(tag in k for tag in ("prefix", "cache", "tokens")):
            delta = a[k] - b.get(k, 0.0)
            if delta: print(f"{k}: +{delta:.0f}")
```

Usage during a run:

```bash
curl -s "$URL/metrics" > before.prom
python harness/fire.py --url "$URL" --jsonl workloads/chat.jsonl --concurrency 16
curl -s "$URL/metrics" > after.prom
python harness/scrape_metrics.py --before before.prom --after after.prom
```

### 2.6 Expected qualitative results

Using the configuration in §2 and the workloads in §2.2–§2.3, submissions should
observe an ordering consistent with the following predictions. Treat the numeric
ranges as *reasoning checks*, not committed benchmarks — actual values depend on
hardware, model, batch size, template length, and vLLM/SGLang version.

| Workload | Prefix cache | Hit-rate order of magnitude | TTFT change |
|---|---|---|---|
| Chat, within-session | on | Very high (dominates once past turn 1) | Large reduction on turns ≥ 2 |
| Chat, across-session (shared system only) | on | Moderate — proportional to `len(system) / len(full)` | Small reduction |
| RAG, `system_first`, unique passages | on | Moderate — bounded by `len(system) / len(full)` | Small reduction |
| RAG, `passages_first`, unique passages | on | Near zero | None |
| Any | off | 0 | Baseline |

If the measured ordering does **not** match this table, the submission needs to
investigate before claiming results — the common culprits are listed in §5.

## 3. Validation steps

1. **Launch both configurations.** Bring up the server with `prefix-on`, run both
   workloads, scrape metrics. Shut down, relaunch with `prefix-off`, repeat.
2. **Record a results table** with: workload, prefix flag, request count, hit-token
   ratio, TTFT p50/p95, decode throughput tokens/s, GPU KV-blocks free (from
   `vllm:num_free_gpu_blocks` or SGLang equivalent).
3. **Perturbation checks.**
   - *Change one char in the shared system prompt.* Hit rate for across-session chat
     should fall to the within-session component only. Confirms the match is
     token-exact.
   - *Flip the RAG template from `system_first` to `passages_first`.* Hit rate for the
     RAG workload should collapse. Confirms "prefix" means the first tokens, not
     "anywhere in the prompt."
   - *Warm the server first.* Hit rate should be noticeably higher on the second run
     than on the first with a cold pool.
4. **Equivalent-quality check.** The two runs (cache on vs off) should produce
   *identical* token outputs for identical prompts, because the KV values are
   deterministic. If outputs differ, something more than caching (temperature,
   quantization, concurrency-dependent batching) has drifted.
5. **GPU memory check.** Confirm with `nvidia-smi` and the server's own KV-pool
   metrics that enabling prefix caching did not force the pool below the batch size
   the workload needs — otherwise TTFT wins can be masked by queueing.

## 4. Rubric / review checklist

Score 0–2; passing ≥ 14 / 20.

| # | Item | Pts |
|---|---|---|
| 1 | Reproducible launch scripts for cache-on and cache-off configs | 2 |
| 2 | Chat workload produces monotonically extending per-session prefixes | 2 |
| 3 | RAG workload includes at least one template-order perturbation | 2 |
| 4 | Metrics scraped before/after each run; hit-rate computed from counters | 2 |
| 5 | TTFT percentiles reported (not just mean) | 2 |
| 6 | Results table matches expected ordering in §2.6 | 2 |
| 7 | Perturbation test (altered system prompt) run and reported | 2 |
| 8 | Perturbation test (template reorder) run and reported | 2 |
| 9 | Interpretation ties hit rate to template construction, not magic | 2 |
| 10 | Caveats: outputs byte-equal between on and off; KV-pool budget preserved | 2 |

## 5. Common mistakes

- **Running cache-on with a cold pool and reporting a low hit rate.** The first
  request in any workload cannot hit. Report a steady-state window, not the whole
  run.
- **Putting the system prompt at the end of the RAG template.** Hit rate collapses;
  learners sometimes blame the engine when the template is the cause.
- **Comparing TTFT at saturation.** If the server is queueing, TTFT is dominated by
  queue time, not prefill. Measure at a throughput below the knee of the latency
  curve, or report queue depth alongside TTFT.
- **Using a different `max_model_len` or `gpu_memory_utilization` between runs.**
  Changing these alters the KV pool size and biases the comparison.
- **Confusing `cache_hits` (block hits) with `prefix_cache_hits` (token hits).** The
  metric names differ across vLLM and SGLang versions. Verify names from `/metrics`
  text rather than hard-coding.
- **Claiming decode speedup from prefix caching.** Prefix caching only cuts prefill.
  If TPOT drops, a *different* optimization is in play (likely admission pressure or
  batch-composition change).
- **Ignoring concurrency effects.** A high concurrency can flood the KV pool and
  evict prefixes before reuse. Report the concurrency used.

## 6. References

- Kwon, W. et al. *Efficient Memory Management for Large Language Model Serving with
  PagedAttention.* SOSP 2023. arXiv:2309.06180.
- Zheng, L. et al. *SGLang: Efficient Execution of Structured Language Model
  Programs.* NeurIPS 2024. arXiv:2312.07104.
- vLLM documentation — "Automatic Prefix Caching", "Production Metrics" —
  https://docs.vllm.ai/ (practitioner implementation).
- SGLang documentation — "RadixAttention", "Server Metrics" —
  https://docs.sglang.ai/ (practitioner implementation).
- Prometheus text exposition format — https://prometheus.io/docs/instrumenting/exposition_formats/.
