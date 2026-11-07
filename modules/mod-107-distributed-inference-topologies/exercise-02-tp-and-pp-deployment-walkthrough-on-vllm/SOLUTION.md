# Solution — Exercise 02: TP and PP Deployment Walkthrough on vLLM

> Paired learning file: [`lessons/mod-107-distributed-inference-topologies/exercises/exercise-02-tp-and-pp-deployment-walkthrough-on-vllm.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-107-distributed-inference-topologies/exercises/exercise-02-tp-and-pp-deployment-walkthrough-on-vllm.md)
> Module: `mod-107-distributed-inference-topologies`
> Type: implementation / deployment walkthrough

## 1. Solution overview

The learner stands up a **distributed vLLM** deployment with explicit tensor-parallel
(TP) and pipeline-parallel (PP) sharding, first on a single node and then across two
nodes, and then verifies that each configuration is healthy before taking a
measurement. This is a *deployment* exercise, not a benchmark exercise — the output
is a working configuration, a per-GPU memory map, a sanity trace through the engine
logs, and a small latency/throughput table taken with a reproducible load generator.

The three artefacts:

| Artefact | Form | Pass bar |
|---|---|---|
| A — Pinned launch commands | Shell script or Makefile | Runs the three target configs (TP = N; TP = N × PP = M; multi-node TP × PP) |
| B — Healthy-engine checklist | Markdown + log excerpts | Each config: `/health` 200, per-GPU `nvidia-smi` memory, engine log confirms TP/PP sizes |
| C — Smoke measurement | Three-row table + method note | Same prompt mix against each config; wall-clock latency and throughput, with method note describing what is / is not being measured |

Pin the vLLM version in the launch commands. Flag names and defaults have moved
between minor releases; verify every flag against `vllm serve --help` in the pinned
release before copying.

### The two flags that do the work

vLLM exposes the topology via two integer flags. The engine's distributed runtime
is Ray-based when multi-node, and torch-distributed with NCCL when single-node.
Both flags are documented in the [vLLM Distributed Serving
docs](https://docs.vllm.ai/).

| Flag | Meaning |
|---|---|
| `--tensor-parallel-size N` (short: `-tp N`) | Shard each transformer layer across N GPUs. All-reduce after every attention and MLP block. |
| `--pipeline-parallel-size M` (short: `-pp M`) | Partition the layer stack into M stages. Activations flow stage-to-stage via send/recv. |

Total GPUs required = `TP * PP`. Both default to 1. vLLM may require Ray when
`PP > 1` or when running multi-node; single-node TP = N uses torch-distributed
directly.

### The rule that governs the picks

From Exercise 01: **TP should sit inside one NVLink/NVSwitch domain. PP spans
nodes.** For an 8× H100/H200/A100 node, TP = 8 is the safe maximum per node. To use
16 GPUs across two nodes, use **TP = 8 × PP = 2**, not TP = 16.

## 2. Implementation — worked walkthrough

The commands below assume:

- A chat-sized dense model (e.g., `meta-llama/Meta-Llama-3-70B-Instruct` or an
  equivalent 70B-class open-weights checkpoint accessible to the learner).
- One or two nodes of 8× 80 GiB-class accelerators.
- NCCL-visible GPUs (`nvidia-smi` sees them; `NCCL_P2P_LEVEL` left at default).
- A vLLM version pinned in `requirements.txt` or a container image tag.

Substitute model and hardware for whatever the learner's lab provides.

### 2.1 Single-node TP = 8 (worked)

The simplest non-trivial config: one node, all 8 GPUs, no PP.

```bash
# config_a_tp8.sh
export HF_HOME=/scratch/hf                        # cache on fast local disk
export VLLM_LOGGING_LEVEL=INFO                    # include dist init in logs

vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --host 0.0.0.0 --port 8001
```

Expected engine-log evidence (names will differ by release; verify):

```
INFO ... distributed_init ... tensor_parallel_size=8 pipeline_parallel_size=1
INFO ... Loading model weights ...
INFO ... Model loaded. ... num_gpu_blocks=<non-zero> num_cpu_blocks=<non-zero>
INFO ... Engine ready at http://0.0.0.0:8001
```

`nvidia-smi` on the node should show ≈ equal memory use across the 8 GPUs (weight
shard + KV pool + CUDA graphs per rank). A large imbalance means the shard did not
go through — check that `CUDA_VISIBLE_DEVICES` is not masking GPUs, that NCCL
initialized, and that no previous process is holding memory.

### 2.2 Single-node TP = 4 × PP = 2 (worked)

Same node, but demonstrates PP. Useful for learners to see PP semantics without
needing a second machine. Expect lower throughput than TP = 8 at this scale — PP
earns its keep across nodes, not within one node.

```bash
# config_b_tp4_pp2.sh
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --host 0.0.0.0 --port 8002
```

Evidence of a PP = 2 split, from logs: two pipeline stages initialize, each holding
half the layer count (within rounding). Memory per GPU will be lower than config A
because each GPU holds only half the layers' worth of weights (TP = 4 shards within
each half).

This config is included to teach the mechanics; do not read it as a performance
recommendation for a single node.

### 2.3 Two-node TP = 8 × PP = 2 (worked)

The multi-node case. This is where Ray does real work: it wires the ranks across
nodes, and vLLM uses NCCL for the actual collectives.

**Prereqs on every node:**
- Same vLLM version, same CUDA, same PyTorch, same model weights path (shared
  filesystem or pre-populated on each host).
- RDMA / fast interconnect configured (IB or RoCE); NCCL will autodetect but
  verify with `NCCL_DEBUG=INFO` on first run.
- Open the Ray GCS port (default 6379) between nodes.
- `vllm` reachable on the Python path on every node.

**Head node (node-0):**

```bash
# Start Ray head
ray start --head --port=6379 --dashboard-host=0.0.0.0

# Launch vLLM — same model, same max_len, same TP/PP
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --distributed-executor-backend ray \
  --host 0.0.0.0 --port 8003
```

**Worker node (node-1), run before the vLLM launch on node-0:**

```bash
ray start --address='<node-0-ip>:6379'
```

**Confirmation:**
- `ray status` on either node reports 16 GPUs available.
- Engine log on node-0 reports `tensor_parallel_size=8 pipeline_parallel_size=2`
  and places 8 ranks on each node.
- `/v1/models` on `http://node-0:8003` returns the served model.

`--distributed-executor-backend ray` is the explicit, future-proof form. In single-
node runs vLLM selects a backend automatically; for multi-node PP, naming Ray
explicitly avoids ambiguity.

### 2.4 One-shot smoke client

Save and reuse this to verify each config returns reasonable tokens before
measurement.

```python
# smoke_client.py — point URL at each config, in turn.
import time
import httpx

URL = "http://localhost:8001/v1/completions"
PROMPT = "Explain pipeline parallelism in three sentences."

payload = {
    "model": "meta-llama/Meta-Llama-3-70B-Instruct",
    "prompt": PROMPT,
    "max_tokens": 64,
    "temperature": 0.0,
}

t0 = time.perf_counter()
r = httpx.post(URL, json=payload, timeout=120)
r.raise_for_status()
dt = time.perf_counter() - t0
print(f"status={r.status_code} dt={dt*1000:.0f} ms")
print(r.json()["choices"][0]["text"][:200])
```

The three configs should produce *materially similar* output text for a greedy
prompt, because topology does not change the arithmetic (within fp tolerance).
Divergent text usually means two different model revisions were loaded on the two
nodes; re-sync the weights path.

### 2.5 Reproducible load generator

Keep the measurement surface tiny. One prompt-length bucket, one output-length
bucket, one concurrency level, repeated three times. The point is not to publish a
benchmark; the point is to prove the three configs serve traffic.

```python
# load_gen.py — minimal concurrency-N load generator with p50/p95 latency.
import argparse, asyncio, statistics, time
import httpx

async def one_request(client, url, body, label):
    t0 = time.perf_counter()
    r = await client.post(url, json=body, timeout=180)
    r.raise_for_status()
    return (label, (time.perf_counter() - t0) * 1000.0)

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--max-tokens", type=int, default=128)
    args = p.parse_args()

    prompt = "token " * args.prompt_tokens
    body = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
    }
    sem = asyncio.Semaphore(args.concurrency)
    async def guarded(client, i):
        async with sem:
            return await one_request(client, args.url, body, i)

    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *[guarded(client, i) for i in range(args.requests)]
        )
        wall = time.perf_counter() - t0

    latencies = sorted(x[1] for x in results)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    thr = args.requests / wall
    print(f"wall={wall:.1f}s  req/s={thr:.2f}  p50={p50:.0f}ms  p95={p95:.0f}ms")

asyncio.run(main())
```

Run against each config:

```bash
python load_gen.py --url http://localhost:8001/v1/completions \
  --model meta-llama/Meta-Llama-3-70B-Instruct \
  --concurrency 16 --requests 128 \
  --prompt-tokens 512 --max-tokens 128
```

### 2.6 Worked measurement table (illustrative)

The numbers below are **placeholders demonstrating the expected relative ordering
on 8×/16× H100/H200-class hardware with a 70B-class model and the parameters in
§2.5**. The learner reports their own numbers. Reviewers are judging the *method*,
not these numbers.

| Config | GPUs | Topology | req/s | p50 lat (ms) | p95 lat (ms) | per-GPU HBM used |
|---|---|---|---|---|---|---|
| A — single-node TP = 8 | 8 | TP=8 PP=1 | *measured* | *measured* | *measured* | roughly equal, ≈ `(weights/8 + KV_pool + overhead)` |
| B — single-node TP = 4 × PP = 2 | 8 | TP=4 PP=2 | *measured* | *measured* | *measured* | ≈ half the layers' worth per stage |
| C — two-node TP = 8 × PP = 2 | 16 | TP=8 PP=2 | *measured* | *measured* | *measured* | ≈ config A per rank |

Expected qualitative ordering, from topology alone:
- A vs. B on the same hardware: TP = 8 typically wins on latency because PP's bubble
  hurts at low concurrency; PP = 2 does not add GPUs, only slices the layers.
- C vs. A: throughput roughly scales with PP for a fixed max concurrency, with
  diminishing returns from the bubble; see Megatron-LM §3 for the bubble formula.

If the measurements contradict the ordering, the common culprits are (i) running
A under memory pressure that forces KV eviction, (ii) running B/C with a too-small
`max-num-batched-tokens` so the pipeline is never full, or (iii) misconfigured
NCCL that routes through host memory.

## 3. Validation steps

A reviewer validating a learner submission should perform the following in order.

1. **Launch commands pinned.** Commands reference an explicit vLLM version
   (requirements or image tag), explicit model path, explicit TP/PP flags. No
   "default" values on which the topology silently depends.
2. **TP × PP = GPU count.** Confirm each launch's `TP * PP` equals the GPUs
   allocated. Off-by-one here usually means the engine won't start.
3. **Engine health.** `GET /health` returns 200 for each config; `/v1/models` lists
   the loaded model.
4. **Engine log evidence.** Startup log contains the TP and PP sizes it was actually
   configured with, and reports non-zero `num_gpu_blocks`.
5. **Per-GPU memory map.** `nvidia-smi` per node shows the expected shape:
   - TP = 8 → 8 ranks with comparable memory on one node.
   - TP = 4 × PP = 2 → still 8 ranks per node, but the memory per rank is lower
     than TP = 8 at the same `max_model_len` because each rank holds half the
     layers' weights.
   - TP = 8 × PP = 2 on 2 nodes → 8 ranks on each node with comparable memory.
6. **Smoke output.** The §2.4 smoke prompt returns coherent text from each config,
   and greedy outputs are materially similar across configs.
7. **Load measurement.** The §2.6 table is filled with the learner's numbers and
   their method note states (a) the prompt length used, (b) the output length used,
   (c) the concurrency, (d) warm-up policy, and (e) explicit acknowledgement that
   this is a smoke measurement, not a benchmark. (See [module 103 exercise 02's
   benchmark-vs-smoke
   caveat](../../mod-103-request-scheduling-and-batching/exercise-02-scheduler-tuning-for-stated-slo-and-traffic-shape/SOLUTION.md).)
8. **Qualitative ordering plausible.** If the TP = 8 config is slower than
   TP = 4 × PP = 2 on the same node, the learner has investigated and named the
   cause before submitting (NCCL misconfig, memory pressure, Ray overhead, etc.).
9. **Teardown.** The script cleanly stops the engine on each node and
   `ray stop` is run; the next config's launch is not fighting a zombie process.

## 4. Rubric / review checklist

Score each item 0 (missing), 1 (partial), 2 (fully correct). Passing: ≥ 10 / 14.

| # | Item | Points |
|---|---|---|
| 1 | Three launch configs run clean (single-node TP, single-node TP×PP, multi-node TP×PP) | 2 |
| 2 | Each launch command pins vLLM version, model, TP, PP, and `--gpu-memory-utilization` | 2 |
| 3 | Engine logs cited that show TP and PP actually took effect | 2 |
| 4 | Per-GPU memory map captured for each config matches expectation (even TP; halved by PP) | 2 |
| 5 | Smoke client invocation + output excerpt for each config | 2 |
| 6 | Load measurement table filled with method note explaining what is / is not measured | 2 |
| 7 | Multi-node config uses `--distributed-executor-backend ray` and verified `ray status` | 2 |

## 5. Common mistakes

- **TP across nodes.** Running TP = 16 on 2 nodes exports TP's all-reduce to the
  inter-node fabric, where it will dominate every decode step. Fix: TP = 8 × PP = 2.
  (Megatron-LM §3.)
- **Forgetting `ray start` on the worker.** The head-node `vllm serve` will sit in
  rank-acquisition waiting for the worker to join. If the head times out, check
  `ray status` first, then `NCCL_DEBUG=INFO` for fabric issues.
- **Mismatched weights across nodes.** Loading different revisions on the two nodes
  produces divergent output tokens on greedy decoding. Pin a single checkpoint path
  on a shared filesystem, or verify sha256 of each file on both nodes.
- **`--gpu-memory-utilization` too low on PP configs.** PP halves the weight
  footprint per rank; leaving the default (often 0.90) is fine, but setting it low
  (e.g., 0.5) wastes HBM that KV blocks could have used and shrinks `num_gpu_blocks`
  needlessly.
- **Running the measurement without a warm-up.** The first request pays CUDA graph
  capture and model-weight touch costs. Discard the first N requests or send a
  warm-up burst before recording.
- **Shipping a benchmark number without a method note.** A single table of numbers
  without the prompt length, output length, concurrency, hardware, and warm-up
  is not a reproducible result; it is a screenshot.
- **Assuming NVLink is active.** If NCCL is using host memory (bad P2P discovery),
  TP = 8 can be dramatically slower than TP = 4 × PP = 2 on the same node for the
  wrong reason. Verify with `NCCL_DEBUG=INFO` on first launch; look for
  `NCCL INFO Channel .. NVLS` or `NCCL INFO Channel .. via P2P`.
- **Changing more than one thing between A, B, and C.** Keep `max_model_len`,
  `gpu-memory-utilization`, concurrency, prompt length, and output length constant
  across the three configs; only TP and PP differ. Otherwise the comparison is
  confounded.

## 6. References

- vLLM documentation. *Distributed Serving.* https://docs.vllm.ai/ — canonical
  reference for `--tensor-parallel-size`, `--pipeline-parallel-size`, and the Ray
  multi-node setup. (Practitioner implementation; verify flag names against the
  pinned release.)
- Ray documentation. *Ray Cluster — manual cluster setup.*
  https://docs.ray.io/en/latest/cluster/getting-started.html — head/worker join
  procedure.
- Shoeybi, M. et al. *Megatron-LM: Training Multi-Billion Parameter Language Models
  Using Model Parallelism.* arXiv:1909.08053 — TP + PP semantics, the "TP belongs
  on NVLink" argument.
- Narayanan, D. et al. *Efficient Large-Scale Language Model Training on GPU
  Clusters Using Megatron-LM.* SC 2021 — pipeline-bubble formula; useful for
  predicting PP's cost at low concurrency.
- Huang, Y. et al. *GPipe: Efficient Training of Giant Neural Networks using
  Pipeline Parallelism.* NeurIPS 2019 — micro-batching rationale that underpins
  PP.
- NVIDIA. *NCCL User Guide.*
  https://docs.nvidia.com/deeplearning/nccl/user-guide/ — `NCCL_DEBUG`, topology
  detection, NVLink/NVSwitch awareness.
- NVIDIA. *NCCL Troubleshooting.* same portal — common P2P and fabric failure
  modes.
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/)
  — not invoked for this deployment exercise.
- [mitre-atlas](https://atlas.mitre.org/) — not invoked for this deployment
  exercise.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework) — governance
  baseline; not invoked for this deployment exercise.
