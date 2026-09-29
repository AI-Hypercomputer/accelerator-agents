---
title: "DMA-overhead-bytes heuristic for TPU Pallas block-size autotuning"
type: concept
tags: [pallas, dma, vmem, autotuning, block-size, simply, stub]
created: 2026-04-23
updated: 2026-04-23
---

Pattern: when autotuning TPU Pallas block sizes, model DMA setup as a **fixed ~0.5 MiB of virtual bytes** and balance that against padding overhead when choosing block shape. Introduced in DeepMind's `simply` repo; portable to any paged TPU kernel. *Stub — expand when more sources are available.*

## Definition

TPU DMA transfers have a fixed setup cost that doesn't scale with transfer size. Smaller blocks mean more DMAs → more setup cost. Larger blocks mean more padding waste when sequence / token axes aren't clean multiples of block size. The **DMA-overhead-bytes** heuristic models the setup cost as an equivalent number of "virtual bytes" (~0.5 MiB per DMA) and the autotuner picks a block size that minimizes (actual bytes + DMA-virtual-bytes + padding bytes).

## Why it matters for TPU perf

Paged attention, ragged matmul, and any kernel with dynamic block counts all benefit — naive block-size choice either dispatches thousands of tiny DMAs (setup-dominated) or wastes VMEM on padding (transfer-dominated). The heuristic gives a closed-form trade-off that matches the measured sweet-spot.

## Mechanism

Given candidate `block_size`:
- `actual_bytes = total_elements × itemsize`
- `num_dmas = ceil(total_elements / block_size) × num_tensors_per_tile`
- `virtual_bytes = num_dmas × DMA_OVERHEAD_BYTES` (≈ 0.5 MiB)
- `padding_bytes = (ceil(total_elements / block_size) × block_size − total_elements) × itemsize`
- Choose block_size minimizing `actual_bytes + virtual_bytes + padding_bytes`.

The DMA-overhead constant is **assumed roughly the same across TPU v4, v5e, v5p, v6e, v7** per the simply reference.

## When it applies / when it doesn't

- **Applies** to paged attention, ragged matmul, and any Pallas kernel whose grid has dynamic per-batch shape.
- **Does not apply** to static-shape kernels where block size is fixed at compile time — the DMA count is constant.

## Known results

| Reference | Repo | Perf claim |
|---|---|---|
| `autotune_block_sizes` in `ragged_paged_attention.py` | [google-deepmind/simply](https://github.com/google-deepmind/simply) (not ingested as its own codebase — wrapper repo) | Code comment: `dma_overhead_equivalent_bytes = 0.5 MiB`; "Increasing [block size] would shift the attention module from memory bandwidth bound to compute bound, but in the meanwhile, it would cause more padding overhead... 32 is a good empirical trade-off so far" |

## Connections

- [autotuning](autotuning.md)
- [vmem](vmem.md)
- [vmem-budget](vmem-budget.md)
- [attention-block-sizes](attention-block-sizes.md)
- [memory-bound](memory-bound.md)

## Sources

- [Pallas kernel directory §4.3](../analyses/pallas-kernel-directory/04-research-labs.md#43-google-deepmindsimply).
