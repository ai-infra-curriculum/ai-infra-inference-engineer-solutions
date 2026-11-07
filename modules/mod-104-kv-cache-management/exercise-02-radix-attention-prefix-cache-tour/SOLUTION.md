# Exercise 02 — RadixAttention Prefix-Cache Tour

> Module: `mod-104-kv-cache-management`
> Exercise: `exercise-02-radix-attention-prefix-cache-tour`
> Type: design / trace walkthrough

## 1. Solution overview

**RadixAttention** is the automatic prefix-caching mechanism in SGLang. It stores KV
cache entries keyed by *token-sequence paths* in a **radix (compressed prefix) tree**.
On admission, a request's prompt is matched against the tree; any already-computed
prefix is reused and only the remaining suffix is prefilled. Reference counting plus
LRU over the leaves keep the tree within the GPU KV pool budget.

RadixAttention was introduced in Zheng, L. et al., *SGLang: Efficient Execution of
Structured Language Model Programs* (NeurIPS 2024, arXiv:2312.07104). The design
generalizes vLLM's prefix cache: instead of hashing fixed-size blocks, it maintains a
variable-depth tree of token sequences so that arbitrary shared prefixes (system
prompts, few-shot examples, long user turns in chat) are cached at their true length.

Core properties a strong answer must recover:

1. Nodes in the radix tree hold **token subsequences** (not fixed-size blocks) and
   point at the KV-cache blocks that store their computed state.
2. **Matching** walks the tree from the root along the longest matching token prefix of
   the incoming request; the match ends at a node or partway through an edge label.
3. On a partial edge match the matching node is **split** so the shared portion becomes
   a parent and the divergent tails become two children.
4. Every in-flight request holds **references** to the path from root to its current
   generation head. Nodes with `refcount > 0` are pinned and cannot be evicted.
5. Eviction is **LRU over zero-refcount leaves** and proceeds upward (the whole
   never-referenced subtree becomes evictable once the leaf goes).
6. Hits save the *prefill* cost of the matched tokens; decode cost is unchanged.

## 2. Implementation — worked tour

### 2.1 Example workload

Imagine a tool-augmented chat application that always opens a conversation with the
same system prompt. For readability, use one letter per token.

```
SYS  = "SYSTEM_PROMPT_TOKENS"   (shortened to "SYS")
```

Requests arrive in this order:

| # | Full prompt tokens (ids as letters) |
|---|---|
| R1 | `SYS A B C`                    |
| R2 | `SYS A B D`                    |
| R3 | `SYS A B D E`  (continuation of R2's turn) |
| R4 | `SYS X`                        |
| R5 | `SYS A B C` (new request, same prefix as R1 but R1 has freed) |

Block size `B = 16` tokens for the underlying KV pool; `SYS` is 32 tokens so it spans
two blocks. The letters A, B, C, D, E each represent one additional user/assistant
token for compactness.

### 2.2 Tree state after each request

Dashed boxes mark **evictable** leaves (refcount = 0). Solid boxes mark **pinned**
nodes (refcount ≥ 1, held by an in-flight or recent request). Edge labels show the
token substring; each node stores a pointer to the physical KV blocks covering that
edge.

**After R1 prefill, before R1 finishes:**

```
                [root]
                   |
                 "SYS"            <-- 32-tok edge, 2 phys blocks
                   |
                 "AB"             <-- 2-tok edge, 1 phys block
                   |
                 "C"              refcount=1 (R1 decoding)
```

**After R2 admitted while R1 still decoding:** matching walks root → "SYS" → "AB"
→ hits "C" edge; R2's token at that position is "D", not "C". So the matcher stops
*before* consuming "C" and **splits**:

```
                [root]
                   |
                 "SYS"
                   |
                 "AB"
                 /   \
              "C"     "D"
            (R1,1)   (R2,1)
```

R1 and R2 each share the `SYS→AB` prefix; the first divergent token forced the fork.
Only the single divergent token's KV must be freshly prefilled for R2, not all of its
prompt.

**After R3 admitted:** R3 is `SYS A B D E`. It fully matches `SYS→AB→D` (R2's path)
and extends with one more token `E`.

```
                [root]
                   |
                 "SYS"
                   |
                 "AB"
                 /   \
              "C"     "D"
            (R1,1)   (ref=2)   <-- R2 still has refcount here; R3 adds another
                        |
                       "E"
                     (R3,1)
```

If R2 has already finished, the `"D"` node's refcount from R2 drops to zero and R3 is
the sole holder.

**After R4 admitted:** R4 is `SYS X`. It matches `SYS` only; "AB" vs "X" diverges on
the first token.

```
                [root]
                   |
                 "SYS"  (ref grows to cover R4 too)
                 /    \
              "AB"    "X"
              /  \    (R4,1)
           "C"   "D"
```

**After R1 and R4 finish, and before R5 arrives:** `"C"` and `"X"` become evictable
leaves.

```
                [root]
                   |
                 "SYS"
                 /    \
              "AB"    [X]       <-- evictable
              /  \
           [C]   "D"
                   |
                  "E" (R3,1)
```

**R5 arrives with `SYS A B C`.** The matcher returns a hit covering 100% of its
tokens — all of `SYS→AB→C` still exists in the tree because eviction hasn't fired
yet. R5 re-pins `C` (refcount back to 1), does **zero** prefill, and begins decoding
immediately.

### 2.3 Eviction

Suppose the KV pool fills and the scheduler needs 1 block to admit a new request.
The policy walks the LRU list of zero-refcount leaves:

- `[X]` (last touched when R4 freed) — oldest. Evict `X`; `SYS→X` edge is removed; the
  single block returns to the free list. If `[root → SYS]` now has only one remaining
  child, SGLang does **not** flatten the tree eagerly; flattening would add complexity
  without changing eviction correctness.
- If still short, evict the next oldest zero-refcount leaf, and so on.

Pinned nodes (R3 is still holding `SYS→AB→D→E`) are never candidates, even if
they are the least recently accessed, because evicting them would strand an in-flight
request.

### 2.4 Request lifecycle summary

Admission flow per request:

```
1. tokenize(prompt) → tokens
2. node, offset = radix_tree.match_prefix(tokens)       # longest prefix in tree
3. matched = node.path_length_up_to(offset)
4. if offset < len(node.edge):
       split node at offset
5. miss_tokens = tokens[matched:]
6. new_leaf = tree.insert(edge=miss_tokens, parent=matched_node)
7. new_leaf.refcount += 1
8. schedule_prefill(miss_tokens)                        # only the miss
9. decode loop; each generated token extends new_leaf.edge
10. on finish: new_leaf.refcount -= 1
```

Hit rate is `sum(matched_tokens) / sum(prompt_tokens)` across admitted requests.

## 3. Validation steps

1. **Draw the tree** after each request in §2.2 and compare against the figure. The
   student answer should preserve the split structure and refcounts.
2. **Dry-run match math.** For the sequences given, compute `matched_tokens` for each
   request. Expected totals: R1 = 0 (cold), R2 = `|SYS|+2`, R3 = `|SYS|+3`, R4 =
   `|SYS|`, R5 = `|SYS|+3` (full re-hit).
3. **Live cross-check with SGLang** (optional but strongly recommended). SGLang exposes
   prefix-cache statistics on its server:

   ```bash
   # Launch SGLang server with radix attention (default on)
   python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct \
       --port 30000

   # Fire the five requests
   python replay_trace.py --url http://localhost:30000

   # Dump metrics
   curl -s http://localhost:30000/metrics | grep -E 'cache|radix|token'
   ```

   Expected: non-zero `cache_hit_rate` and `prefix_cache_hit_tokens` after R2+.
4. **Perturbation test.** Change the system prompt by one token in R5. Match length
   drops to `|SYS|-1`. This confirms the match is token-exact, not semantic.

## 4. Rubric / review checklist

Score 0–2; passing ≥ 12 / 18.

| # | Item | Pts |
|---|---|---|
| 1 | Tree is radix-compressed (edges carry token substrings, not single tokens) | 2 |
| 2 | Tree is drawn and updated correctly for R1 through R5 | 2 |
| 3 | Edge-splitting is applied on partial matches (R2 correctly splits "C" vs "D") | 2 |
| 4 | Refcounts noted for in-flight requests; released on completion | 2 |
| 5 | Cache-hit token counts match Section 3.2 for every request | 2 |
| 6 | Eviction described as LRU over zero-refcount leaves | 2 |
| 7 | Clear statement that only the miss suffix is prefilled | 2 |
| 8 | Distinction from vLLM's hash-of-fixed-blocks prefix cache called out | 2 |
| 9 | Correct observation that decode cost does *not* shrink with a hit | 2 |

## 5. Common mistakes

- **Describing a flat hashmap keyed by prompt.** Full-prompt keying only hits on
  identical prompts; radix matching hits on *any* shared prefix depth.
- **Treating nodes as fixed-size blocks.** PagedAttention uses fixed blocks, and
  vLLM's early prefix cache was a hash over block-aligned prefixes. RadixAttention
  separates logical token paths from the physical block layer.
- **Evicting pinned nodes.** Zero-refcount is a precondition. In-flight requests pin
  the whole path to root.
- **Claiming decode speedup from a prefix hit.** Prefix caching cuts prefill (TTFT) by
  skipping forward passes for matched tokens. Per-token decode cost is unchanged; TPOT
  does not improve from a prefix hit alone.
- **Ignoring sampling/randomness as a correctness issue.** When two requests share a
  prefix, their KV caches for that prefix are *identical* — this is safe for
  deterministic prefill but sampling randomness happens only in decode, so no leakage
  occurs from sharing. (The cross-tenant angle is Exercise 06.)
- **Assuming prefix cache always helps.** Cold, high-entropy workloads (unique
  retrieval contexts, fresh user inputs) rarely share meaningful prefixes and pay the
  tree-maintenance overhead for almost no hit.

## 6. References

- Zheng, L. et al. *SGLang: Efficient Execution of Structured Language Model
  Programs.* NeurIPS 2024. arXiv:2312.07104.
- SGLang project documentation and server metrics guide —
  https://docs.sglang.ai/ (practitioner implementation).
- Kwon, W. et al. *Efficient Memory Management for Large Language Model Serving with
  PagedAttention.* SOSP 2023. arXiv:2309.06180 — baseline for the block layer the
  radix tree sits above.
- vLLM documentation, "Automatic Prefix Caching" section — https://docs.vllm.ai/ —
  the hash-of-blocks variant for contrast.
