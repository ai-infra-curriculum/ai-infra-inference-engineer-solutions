# Exercise 01 — PagedAttention Block-Table Walkthrough

> Module: `mod-104-kv-cache-management`
> Exercise: `exercise-01-paged-attention-block-table-walkthrough`
> Type: design / trace walkthrough

## 1. Solution overview

The learner traces how an inference engine built on **PagedAttention** (vLLM) stores a
request's KV cache in fixed-size *blocks*, how a per-sequence *block table* translates
logical token positions to physical GPU-memory blocks, and how that indirection enables
non-contiguous allocation, prefix sharing, and copy-on-write for parallel sampling.

Paged memory for KV cache was introduced by Kwon et al., "Efficient Memory Management
for Large Language Model Serving with PagedAttention" (SOSP 2023). The scheme borrows
directly from OS virtual memory: the physical pool is partitioned into blocks of `B`
tokens, a sequence is a *logical* array of block IDs, and the attention kernel reads
through the block table to gather K/V for all tokens of the sequence.

The core properties a strong answer must recover:

1. KV tensors are chunked into blocks of exactly `B` tokens per layer per head; a block
   is the unit of allocation, sharing, and eviction.
2. Each sequence owns a **block table**: `logical_block_index → physical_block_id`.
3. Only the *last* block of each sequence is partially filled; growth happens by
   appending a new logical entry and allocating a free physical block.
4. Sibling sequences (beam search, `n > 1` sampling) share leading blocks by reference
   count; **copy-on-write** forks the first block that diverges.
5. Fragmentation is bounded to at most one partial block per sequence (internal) and
   zero external fragmentation, which is the central win over per-sequence contiguous
   reservations.

## 2. Implementation — worked walkthrough

### 2.1 Setup

| Symbol | Meaning | Example value |
|---|---|---|
| `B` | Block size (tokens per block) | `16` |
| `L` | Number of transformer layers | `32` |
| `H_kv` | KV heads | `8` (GQA ratio 4) |
| `d_h` | Head dim | `128` |
| `dtype` | KV dtype | `fp16` (2 B) |
| `N_phys` | Physical blocks in pool | `4096` |

Bytes per KV cache block (K **and** V, all layers, all KV heads):

```
bytes_per_block = 2 * L * H_kv * d_h * B * sizeof(dtype)
                = 2 * 32 * 8 * 128 * 16 * 2
                = 2,097,152 B  = 2 MiB
```

Pool size: `N_phys * bytes_per_block = 4096 * 2 MiB = 8 GiB` of HBM reserved for KV.

### 2.2 Three-sequence trace

Follow three sequences through four decode steps. Free list starts at
`[P0, P1, P2, …, P4095]`.

**t = 0 — prompts admitted (prefill).**

| Seq | Prompt len | Logical blocks needed | Physical blocks assigned |
|---|---|---|---|
| S1  | 30 tokens | `ceil(30/16) = 2`     | `[P0, P1]` (P1 has 14 tokens, 2 free slots) |
| S2  | 48 tokens | `ceil(48/16) = 3`     | `[P2, P3, P4]` (P4 full) |
| S3  | 17 tokens | `ceil(17/16) = 2`     | `[P5, P6]` (P6 has 1 token, 15 free slots) |

Block tables after prefill:

```
BT[S1] = [P0, P1]      fill = [16, 14]
BT[S2] = [P2, P3, P4]  fill = [16, 16, 16]
BT[S3] = [P5, P6]      fill = [16, 1]
```

Allocated: 7 physical blocks. Free: 4089.

**t = 1 — first decode step.** Every sequence emits one token.

- S1: last block has 2 free slots → write token 30 into `P1[14]`. No new allocation.
  New `fill = [16, 15]`.
- S2: last block `P4` is full → allocate `P7`, append to BT. New
  `BT[S2] = [P2, P3, P4, P7]`, `fill = [16, 16, 16, 1]`.
- S3: last block `P6` has 15 free slots → write into `P6[1]`.
  `fill = [16, 2]`.

Free list after t=1: 4088 (one allocation for S2).

**t = 2 — beam-split S1 into two beams.**

To support `n=2` sampling without copying all 30 tokens of KV, the engine creates a
sibling sequence `S1b` that **shares the same block table** as `S1`:

```
BT[S1a] = [P0, P1]    refcount: P0→2, P1→2
BT[S1b] = [P0, P1]
```

Both beams then generate a different next token at position 31. Writing into the shared
tail block (`P1`, which has one free slot after t=1) would corrupt the other beam.

**Copy-on-write fires on the first write that would mutate a shared block:**

- Engine allocates `P8`, memcopies `P1[0..15]` into `P8`, decrements
  refcount of `P1` to 1, and rewrites `BT[S1b][1] = P8` before writing beam-b's token.
- Beam-a writes its own token into `P1[15]` as normal (refcount is 1, no CoW needed).

Block table state:

```
BT[S1a] = [P0, P1]     (P0 shared, P1 now beam-a exclusive)
BT[S1b] = [P0, P8]     (P0 shared, P8 is beam-b fork)
```

Free list after t=2: 4087.

**t = 3 — S3 completes and is freed.**

Freeing S3 decrements refcounts of `P5` and `P6`. Both reach zero → both return to the
free list. Free list after t=3: 4089.

### 2.3 Attention kernel view

The paged attention kernel is called once per layer. For each sequence it receives:
- the block table (an array of `num_logical_blocks` physical IDs),
- the number of valid tokens in the sequence (so it can mask padding in the last block),
- query tensor of shape `[batch, H_q, d_h]` for this decode step.

Inside the kernel, position `t` in a sequence maps to `(block_idx, slot_idx)` by
integer division and modulo:

```
block_idx = t // B
slot_idx  = t  % B
phys_block = BT[seq][block_idx]
k_tile = K_pool[phys_block, layer, slot_idx, :, :]
```

K and V are gathered through the indirection; no contiguous reservation per sequence is
needed.

## 3. Validation steps

A graded submission should produce a diagram or table demonstrating the trace above
using the parameters supplied in the exercise prompt. Validate each submission with the
following concrete checks.

1. **Block math.** Compute `bytes_per_block` for the given `L`, `H_kv`, `d_h`, `B`,
   dtype. Verify against `2 * L * H_kv * d_h * B * sizeof(dtype)`. An answer that
   forgets the leading `2` (because it counts K but not V) is the most common error.
2. **Prefill allocations.** For each sequence of length `n`, expected block count is
   `ceil(n / B)`. The last block's fill level is `((n - 1) % B) + 1` (or `B` when
   `n % B == 0`).
3. **Append rule.** During decode a new physical block is allocated **only** when the
   last block's fill reaches `B`.
4. **Beam/parallel-sampling fork.** Sibling sequences must share leading blocks by
   reference count. The first write into a shared tail block must trigger a
   byte-for-byte copy of that single block to a fresh physical block; prior blocks are
   not copied.
5. **Freeing.** When a sequence finishes, every block in its table has its refcount
   decremented. Blocks that reach zero return to the free list in the same step.
6. **Capacity check.** Maximum concurrent sequences of length `S` with sharing-free
   assumptions is `floor(N_phys / ceil(S / B))`. Checking this against the engine's
   admission policy is a good follow-up.

### Optional dynamic validation

If the learner has vLLM installed, they can cross-check the walkthrough against
live engine state:

```bash
# After each step, dump block-manager state through vLLM's internal API.
# (Exact attribute paths change across releases; this is illustrative.)
python - <<'PY'
from vllm import LLM, SamplingParams
llm = LLM(model="facebook/opt-125m", block_size=16, gpu_memory_utilization=0.5)
sp = SamplingParams(max_tokens=4, n=2)  # triggers beam-like fork
out = llm.generate(["Hello world " * 10], sp)
bm = llm.llm_engine.scheduler[0].block_manager
print("num_free_gpu_blocks:", bm.get_num_free_gpu_blocks())
PY
```

The expected result is that `num_free_gpu_blocks` drops by the number of physical
blocks the trace predicts, modulo any pre-reservation vLLM does for CUDA graphs.

## 4. Rubric / review checklist

Score each item 0 (missing), 1 (partially correct), or 2 (fully correct).
Passing: ≥ 10 / 14.

| # | Item | Points |
|---|---|---|
| 1 | Correct per-block byte formula including factor of 2 for K and V | 2 |
| 2 | Block tables for every sequence after prefill with correct fill levels | 2 |
| 3 | Correct append-on-full-block rule during decode | 2 |
| 4 | Explicit reference counting on blocks across sibling sequences | 2 |
| 5 | Copy-on-write triggered only on write to a refcount>1 tail block | 2 |
| 6 | Freeing returns zero-refcount blocks to the free list immediately | 2 |
| 7 | Clear statement that fragmentation is bounded to ≤1 partial block per seq | 2 |

## 5. Common mistakes

- **Reserving a contiguous arena per sequence.** This is the pre-PagedAttention
  baseline and defeats the whole point; it produces external fragmentation and
  wastes HBM proportional to `max_seq_len − actual_len`.
- **Forgetting the `2x` for K and V.** Every PagedAttention block holds both K and V
  interleaved (or in twin pools); memory accounting needs both.
- **CoW on every shared block.** CoW is lazy — only the first *write* on a shared
  block copies, and only that block. Preceding shared blocks remain shared.
- **Allocating a new block every decode step.** New allocation happens only when the
  tail block is full. Learners sometimes draw one block per token.
- **Treating the block table as a hash map.** It is a dense array indexed by logical
  block index; no hashing is involved. (Hashing appears in prefix caching, which is
  Exercise 02.)
- **Mixing block size and chunk size.** `block_size` governs KV allocation.
  `chunked_prefill` governs how many query tokens per iteration. Both are
  independent knobs.

## 6. References

- Kwon, W. et al. *Efficient Memory Management for Large Language Model Serving with
  PagedAttention.* SOSP 2023. arXiv:2309.06180.
- vLLM documentation, "Paged Attention" and "Block Manager" sections —
  https://docs.vllm.ai/ (practitioner implementation).
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/) —
  relevant for Exercise 06; not invoked here.
- [mitre-atlas](https://atlas.mitre.org/) — relevant for Exercise 06; not invoked here.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework) —
  governance baseline, not invoked for this trace exercise.
