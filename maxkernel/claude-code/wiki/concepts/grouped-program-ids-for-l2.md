---
title: "grouped program-IDs for L2 cache reuse"
type: concept
tags: [pallas, triton, gpu, l2-cache, program-id, alphafold3, stub]
created: 2026-04-23
updated: 2026-04-23
---

Reorder the (block_m, block_n) program-id sequence so tiles that share loaded inputs run back-to-back, letting L2 serve them instead of HBM. GPU-side technique (for Triton/Mosaic-GPU Pallas); analogue on TPU is VMEM reuse via BlockSpec reuse. *Stub — expand when more sources are available.*

## Definition

Default grid iteration on GPU traverses `(m, n)` row-major or column-major — under wide Ns and tall Ms, consecutive program instances share few operands and L2 turns over. **Grouped PIDs** pack together instances that share operands (e.g., same M-row across a group of N-columns) so the M-tile stays warm in L2 across the group.

## Why it matters for TPU perf

Directly relevant on GPU Pallas (Triton lowering). **Not a TPU concept** — TPU Pallas uses BlockSpec + VMEM, and the VMEM model gives explicit tile residency rather than cache-warmth hopes. Included in this wiki because the pattern is referenced in the alphafold3 fused GLU, which is the reference design for any future TPU fused-GLU port.

## Mechanism

Given a grid of `(m_blocks, n_blocks)`:
1. Pick a group size `G` (rows per group).
2. Emit program instances so within each group, all N-columns of G rows run before the next group.
3. In Pallas, this is a `program_id` remap inside the kernel — or a driver-side grid reshape.

AlphaFold3's `_get_best_pids` / `_get_group_cache_usage` solves a simple assignment problem picking `G` that maximizes L2 reuse given known tile sizes.

## When it applies / when it doesn't

- **Applies** to any GPU Pallas GEMM / fused-GEMM kernel where intermediate-weight or input tiles fit in L2 for a group of instances but not for the full sweep.
- **Does not apply on TPU** — Mosaic TPU uses BlockSpec + explicit VMEM. The equivalent concern is VMEM budget (see [vmem-budget](vmem-budget.md)).

## Known results

| Reference | Repo | Notes |
|---|---|---|
| `_get_best_pids` / `_get_group_cache_usage` in `gated_linear_unit/matmul_ext.py` | [alphafold3 @ v3.0.1](../codebases/alphafold3.md) | First-party reference design |

## Connections

- [pallas-kernel](pallas-kernel.md)
- [xla-fusion](xla-fusion.md)
- [vmem](vmem.md) — TPU analogue of cache-warmth reasoning.

## Sources

- [alphafold3 codebase](../codebases/alphafold3.md) "Performance-relevant surfaces §3".
- [Pallas kernel directory §4.5](../analyses/pallas-kernel-directory/04-research-labs.md#45-google-deepmindalphafold3-pinned-to-tag-v301).
