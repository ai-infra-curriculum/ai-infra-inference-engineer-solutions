# Exercise 04 — Tiered KV Store with LMCache or Mooncake

> Module: `mod-104-kv-cache-management`
> Exercise: `exercise-04-tiered-kv-store-with-lmcache-or-mooncake`
> Type: implementation / integration

## 1. Solution overview

The exercise asks the learner to **extend a single-node vLLM/SGLang deployment with a
tiered KV cache**, so that KV blocks evicted from GPU HBM are preserved in CPU memory
(and optionally on local NVMe or a remote store) and can be reloaded on a later
request that matches the same prefix. The two canonical open-source choices are:

- **LMCache** — a KV-cache storage and transport layer that plugs into vLLM and
  SGLang. It supports CPU, disk, and remote tiers and exposes prefix-cache reuse
  across requests and across engine processes.
- **Mooncake** — the open-sourced KV-cache store described in Qin, R. et al.,
  *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving*
  (arXiv:2407.00079), from Moonshot AI's Kimi service. It targets large-scale
  disaggregated prefill/decode deployments but can be run in a single-node
  configuration as a KV cache backend.

A complete submission picks one of the two, wires it into vLLM (the easier
integration), runs the same workload with four configurations
(GPU-only, GPU + CPU, GPU + CPU + disk, caching disabled), and reports hit rate,
TTFT, and effective throughput.

Below the solution uses **LMCache + vLLM** as the reference because the install path
is simplest. A Mooncake variant is sketched at the end of §2 for learners who want
to target that path instead.

## 2. Implementation

Directory layout the learner should produce alongside this `SOLUTION.md`:

```
exercise-04-tiered-kv-store-with-lmcache-or-mooncake/
├── SOLUTION.md                  (this file)
├── configs/
│   ├── lmcache_cpu.yaml         # GPU + CPU
│   ├── lmcache_cpu_disk.yaml    # GPU + CPU + disk
│   └── mooncake_node.yaml       # Mooncake single-node (optional)
├── launch/
│   ├── vllm_gpu_only.sh
│   ├── vllm_with_lmcache.sh
│   └── mooncake_vllm.sh         (optional)
├── workload/
│   ├── repeat_prefix.jsonl      # worst-case reuse workload
│   └── reuse_rate_variable.py   # tunable reuse-rate generator
└── results/
    └── results_template.md
```

### 2.1 Baselines

Start with vLLM's **internal** prefix cache only so the paper-level baseline is clear.

`launch/vllm_gpu_only.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}" \
  --max-model-len 8192 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --port 8000
```

### 2.2 Adding LMCache

Install (version-pin against the LMCache release the course targets — pinning is
important because the integration API evolves release to release):

```bash
pip install lmcache
# verify:
python -c "import lmcache; print(lmcache.__version__)"
```

`configs/lmcache_cpu.yaml`:

```yaml
# LMCache config: GPU + CPU tier only.
chunk_size: 256            # tokens per logical chunk stored in the tier
local_cpu: true
max_local_cpu_size: 20     # GiB available for CPU-side KV
remote_url: null           # no remote tier
remote_serde: null
save_decode_cache: false
```

`configs/lmcache_cpu_disk.yaml`:

```yaml
# LMCache config: GPU + CPU + local-disk tier.
chunk_size: 256
local_cpu: true
max_local_cpu_size: 20
local_disk: "file:///var/cache/lmcache/kv"
max_local_disk_size: 200   # GiB
remote_url: null
save_decode_cache: false
```

`launch/vllm_with_lmcache.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
export LMCACHE_CONFIG_FILE="${1:?pass config path}"
exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}" \
  --max-model-len 8192 \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  --gpu-memory-utilization 0.9 \
  --port 8000
```

The `--kv-transfer-config` flag is vLLM's integration point for external KV connectors.
The exact JSON shape and connector class name follow the LMCache + vLLM integration
guide for your pinned version; if the field is renamed in a future release, re-check
against the LMCache docs before merging the exercise.

### 2.3 Workload

Build a workload whose **reuse rate is a tunable knob** so the results sweep over the
dimension that matters for tiered caches: how often does a later request share a long
prefix with an earlier one?

`workload/reuse_rate_variable.py` (essentials):

```python
"""Synthesize a request stream where fraction `p_reuse` of requests reuse one of the
last K unique prefixes with a fresh short suffix, and (1 - p_reuse) are cold."""
import json, random, pathlib, string, argparse

def rand_tokens(rng, n, vocab):
    return " ".join(rng.choice(vocab) for _ in range(n))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--p-reuse", type=float, default=0.5)
    ap.add_argument("--prefix-len", type=int, default=2048)   # tokens in shared pfx
    ap.add_argument("--suffix-len", type=int, default=64)
    ap.add_argument("--k-unique", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    vocab = list(string.ascii_lowercase)
    unique_prefixes = [rand_tokens(rng, args.prefix_len, vocab)
                       for _ in range(args.k_unique)]

    with open(args.out, "w") as f:
        for i in range(args.n):
            if rng.random() < args.p_reuse:
                pfx = rng.choice(unique_prefixes)
            else:
                pfx = rand_tokens(rng, args.prefix_len, vocab)
            suffix = rand_tokens(rng, args.suffix_len, vocab)
            f.write(json.dumps({
                "id": i,
                "prompt": pfx + " " + suffix,
                "max_tokens": 64,
            }) + "\n")

if __name__ == "__main__":
    main()
```

Run with multiple `--p-reuse` values (e.g., 0.0, 0.25, 0.5, 0.75, 1.0).

### 2.4 Runbook

```bash
# 1. GPU-only baseline
launch/vllm_gpu_only.sh &  SERVER=$!
for p in 0.0 0.25 0.5 0.75 1.0; do
  python workload/reuse_rate_variable.py --p-reuse $p --out /tmp/w_$p.jsonl
  python harness/fire.py --url http://localhost:8000 --jsonl /tmp/w_$p.jsonl \
                        --tag "gpu_only_p${p}"
done
kill $SERVER

# 2. GPU + CPU tier
launch/vllm_with_lmcache.sh configs/lmcache_cpu.yaml &  SERVER=$!
for p in 0.0 0.25 0.5 0.75 1.0; do
  python harness/fire.py --url http://localhost:8000 --jsonl /tmp/w_$p.jsonl \
                        --tag "gpu_cpu_p${p}"
done
kill $SERVER

# 3. GPU + CPU + disk
launch/vllm_with_lmcache.sh configs/lmcache_cpu_disk.yaml &  SERVER=$!
for p in 0.0 0.25 0.5 0.75 1.0; do
  python harness/fire.py --url http://localhost:8000 --jsonl /tmp/w_$p.jsonl \
                        --tag "gpu_cpu_disk_p${p}"
done
kill $SERVER
```

Capture Prometheus counters at start and end of each sub-run and compute the tier-wise
hit rate:

- `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries` for the GPU tier.
- LMCache exposes per-tier counters (names depend on version; verify against
  `curl $URL/metrics | grep lmcache`). Typical series cover cpu-hit, disk-hit,
  remote-hit, miss.

### 2.5 Expected shape of results

The table below is the *reasoning check* the report should match, not committed
numbers. Hit rate here means the fraction of prompt tokens served from any cache
tier; latency is TTFT.

| `p_reuse` | GPU-only hit | GPU+CPU hit | GPU+CPU+disk hit | TTFT ordering |
|---|---|---|---|---|
| 0.0 | 0 | 0 | 0 | identical across configs (prefill-bound) |
| 0.25 | low (bounded by HBM) | higher | highest | tiered < GPU-only |
| 0.5  | higher | higher still | highest | tiered clearly below GPU-only |
| 0.75 | moderate — HBM thrashes at a working set > HBM | higher | highest | tiered wins grow |
| 1.0  | depends on working-set vs HBM | near full | near full | tiered wins are large when working set > HBM |

The key observation is that tiered caching **only pays off when the working set of
reusable prefixes exceeds the HBM KV pool**. Below that, the cost of CPU→GPU
transfers may actually make the tiered run *slower* than GPU-only. The exercise
exists to teach learners to recognize that break-even point.

### 2.6 Mooncake variant (sketch)

For learners who choose Mooncake instead of LMCache:

- Build Mooncake from source per its repository (ensure RDMA headers if targeting
  multi-node; a single-node loopback setup works without RDMA).
- Launch a Mooncake store process bound to local CPU memory and NVMe.
- Launch vLLM with the Mooncake KV connector configured through the same
  `--kv-transfer-config` flag used by LMCache. The connector class name differs;
  confirm from the Mooncake docs for your pinned version.
- Use the same workload harness. Metrics names change but the hit/miss semantics are
  parallel.

Mooncake's fundamental advantage over a single-host LMCache deployment is that it is
designed around disaggregated prefill/decode and shared KV across nodes — in a
single-node setup those advantages do not surface. Pick Mooncake if the course flow
includes a multi-node extension; pick LMCache for a cleaner single-node first
exposure.

## 3. Validation steps

1. **Confirm the connector is actually engaged.** Issue two cold-but-identical
   prompts; the second should report a non-zero CPU-tier or disk-tier hit. If not,
   the KV connector is misconfigured — the most common cause is the
   `--kv-transfer-config` JSON not being valid or the connector class name not
   matching your installed LMCache/Mooncake version.
2. **Working-set sweep.** Run the workload for `k_unique ∈ {4, 16, 64}` with
   `p_reuse = 1.0` and verify that GPU-only hit rate falls as `k_unique` grows past
   HBM capacity, while GPU+CPU+disk stays high. This is the central phenomenon the
   exercise should demonstrate.
3. **Byte-equal outputs.** For deterministic sampling (temperature 0, fixed seed),
   outputs must be identical across configs. KV reuse must not change what the model
   emits.
4. **Capacity check.** Confirm CPU resident size tracks `max_local_cpu_size` from the
   config (`ps`, `smem`, or Prometheus). Disk usage should track
   `max_local_disk_size`.
5. **Failure drill.** Kill the LMCache cache location (e.g., `rm -rf` the disk tier
   between sub-runs) and confirm the server degrades to the next tier without
   crashing. The exercise is incomplete without demonstrating that losing a tier is
   non-fatal.

## 4. Rubric / review checklist

Score 0–2; passing ≥ 14 / 20.

| # | Item | Pts |
|---|---|---|
| 1 | Correct vLLM launch with a working external KV connector | 2 |
| 2 | Three configs: GPU-only, GPU+CPU, GPU+CPU+disk, documented and reproducible | 2 |
| 3 | Tunable-reuse workload with multiple `p_reuse` points sampled | 2 |
| 4 | Per-tier hit rate reported (not just aggregate) | 2 |
| 5 | TTFT percentiles reported per config | 2 |
| 6 | Working-set sweep (varying `k_unique`) shows the expected crossover | 2 |
| 7 | Byte-equal output check passed with temperature 0 | 2 |
| 8 | Disk tier failure drill performed and documented | 2 |
| 9 | Discussion of when tiered caching does *not* pay off (small working set) | 2 |
| 10 | Pinned dependency versions recorded for repeatability | 2 |

## 5. Common mistakes

- **Taking credit for a config that silently fell back.** If the LMCache connector
  fails to initialize, vLLM may continue with GPU-only caching and print only a
  warning. Always check that CPU/disk hit counters are advancing.
- **Loading CPU→GPU in the critical path of every token.** Movement happens at the
  *block* level, in bulk, at the start of a request's prefill — not per decode
  token. Submissions that describe per-token transfers are modeling the system wrong.
- **Reporting TTFT improvement on a working set smaller than HBM.** With a small
  working set, the GPU-only prefix cache already holds everything; adding a CPU tier
  cannot improve hit rate and may add overhead. This is a feature, not a bug.
- **Mixing engine versions across runs.** The connector config format is tied to
  specific LMCache and vLLM versions. A `pip install -U` between runs can invalidate
  the comparison.
- **Disk tier on a slow SSD.** A SATA SSD can be slower than refetching prefill on an
  H100, especially for small blocks. The exercise is useful for teaching that the
  disk tier is a *capacity* lever, not a latency lever.
- **Forgetting per-tenant keying.** Cross-tenant KV sharing is a separate concern
  (Exercise 06) — if the workload mixes tenants, either segregate the cache namespace
  or restrict the measurement to a single tenant.

## 6. References

- Qin, R. et al. *Mooncake: A KVCache-centric Disaggregated Architecture for LLM
  Serving.* arXiv:2407.00079.
- LMCache project documentation and vLLM integration guide —
  https://docs.lmcache.ai/ (practitioner implementation).
- Mooncake open-source repository and user guide (practitioner implementation).
- Kwon, W. et al. *Efficient Memory Management for Large Language Model Serving with
  PagedAttention.* SOSP 2023. arXiv:2309.06180 — baseline KV manager.
- vLLM documentation — "KV Transfer Config", "Production Metrics" —
  https://docs.vllm.ai/ (practitioner implementation).
