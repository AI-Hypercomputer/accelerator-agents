---
title: "NVIDIA weight-tile bytes limit (101,376 bytes)"
type: concept
tags: [nvidia, gpu, pallas, triton, h100, gb10, a100, shared-memory, stub]
created: 2026-04-23
updated: 2026-04-23
---

Per-SM shared memory budget available for the weight tile in an NVIDIA Pallas/Triton kernel: **101,376 bytes** (H100 has 232,448 per-SM shared, minus ~131 KB overhead for input tiles / accumulators / Triton metadata). Same limit applies across H100 / A100 / GB10. First-party reference: marin/levanter fused CE loss GPU kernel. *Stub — expand when more sources are available.*

## Definition

In NVIDIA Pallas-on-Triton kernels using per-SM shared memory for tile residency, the **weight tile** — the largest tile typically — cannot exceed **101,376 bytes** on current NVIDIA hardware. The constant comes from H100's 232,448-byte per-SM shared minus the ~131 KB consumed by input tiles, accumulators, and Triton's own metadata buffers.

## Why it matters for NVIDIA Pallas perf

This is the primary block-size constraint for NVIDIA-side fused CE loss, fused GLU, and any kernel where the weight tile dominates SMEM usage. Block-size search must respect it, or the kernel either spills (bad) or fails to lower.

## Hardware coverage

| Device | Per-SM shared (bytes) | Weight-tile budget |
|---|---:|---:|
| A100 | 164 KB | **101,376 bytes** (overhead-deducted) |
| H100 | 228 KB | **101,376 bytes** (overhead-deducted) |
| GB10 | larger | **101,376 bytes** (same limit — consistent across generations per marin) |

## Why it matters for this wiki

**TPU wiki, GPU concept**: included because some kernels ingested here ship both TPU and GPU paths (marin CE loss, tokamax ragged_dot, alphafold3 GLU). Cross-reference point when reading their GPU-side code.

## Additional device-specific caps (from marin)

- **GB10**: `_GB10_MAX_H_TILES = 512`; `_GB10_FULL_MATMUL_MAX_OUTPUT_ELEMENTS = 67_108_864`.
- **H100**: V-block caps per device.

## Connections

- [vmem-budget](vmem-budget.md) — TPU analogue of this budget.
- [memory-hierarchy](memory-hierarchy.md)
- [pallas-kernel](pallas-kernel.md)

## Sources

- [marin codebase](../codebases/marin.md) "Performance-relevant surfaces §3".
- `_NVIDIA_WEIGHT_TILE_BYTES_LIMIT = 101_376` in `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/pallas_gpu.py`.
