---
title: "block-sparse offset masks for paged attention"
type: concept
tags: [pallas, paged-attention, sliding-window, sparse-mask, axlearn, stub]
created: 2026-04-23
updated: 2026-04-23
---

Precompute an `(n_kv_blocks, n_kv_blocks)` table of offsets into **unmasked KV blocks only** and load from the offset table in the kernel. Turns sliding-window / block-sparse attention into a dense-load-over-sparse-offsets pattern. First-party reference: AxLearn `tpu_paged_attention_kernel.py`. *Stub — expand when more sources are available.*

## Definition

Sliding-window or block-sparse attention masks mark most `(q_block, kv_block)` pairs as zero. Naively, the kernel still loads those KV blocks and multiplies by zero — wasted DMA + FLOPs. The **offset-mask** approach is to precompute (on host or in a small pre-pass) a table mapping each q_block to the list of kv_blocks that are *not* masked out. The attention kernel then iterates over offsets into that table, loading only unmasked KV blocks.

## Why it matters for TPU perf

On a long-context sliding-window workload (e.g., Gemma local-attention layers with 4K window on 128K sequences), the fraction of unmasked blocks is ~3%. Skipping the other 97% converts attention from a compute-bound dense operation into a memory-bound sparse one, with proportional wall-clock savings. Per AxLearn `tpu_decoding.py`: "speed up roughly equal to ... number of masked kv blocks / total kv blocks".

## Mechanism

1. Precompute `(n_kv_blocks, n_kv_blocks)` offset table — entry `(i, j)` = index of the j-th unmasked kv_block for q_block i.
2. In the kernel, iterate over `j` from 0 to n_unmasked_per_q_block; load `kv_page[offset_table[i, j]]` for each.
3. All loaded blocks are dense-unmasked; no multiply-by-zero waste.

## When it applies / when it doesn't

- **Applies** to sliding-window, chunked-causal, block-sparse, and any paged attention whose mask is statically known per q-block.
- **Does not apply** to dense causal (no mask sparsity) or data-dependent masks (can't precompute the offset table).

## Known results

| Reference | Repo | Notes |
|---|---|---|
| `tpu_paged_attention_kernel.py` + `tpu_decoding.py` | [axlearn](../codebases/axlearn.md) | Canonical block-sparse offset-mask Pallas TPU impl; sliding-window-aware |

## Connections

- [flash-attention](flash-attention.md)
- [splash-attention](splash-attention.md)
- [kv-cache](kv-cache.md)
- [continuous-batching](continuous-batching.md)
- [attention-block-sizes](attention-block-sizes.md)

## Sources

- [axlearn codebase](../codebases/axlearn.md) "Performance-relevant surfaces §3".
- [Pallas kernel directory §4.1](../analyses/pallas-kernel-directory/04-research-labs.md#41-appleaxlearn).
