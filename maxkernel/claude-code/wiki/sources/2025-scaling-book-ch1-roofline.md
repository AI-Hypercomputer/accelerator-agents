---
title: "How to Scale Your Model — Ch 1: All About Rooflines"
type: source
tags: [docs, book, scaling-book, roofline, arithmetic-intensity, compute-bound, memory-bound]
author: DeepMind / JAX ML team
book: "How to Scale Your Model"
book_date: "2025-02-04"
chapter: 1
upstream: https://jax-ml.github.io/scaling-book/roofline
created: 2026-04-23
updated: 2026-04-23
---

Chapter 1 of the scaling-book ([codebase page](../codebases/scaling-book.md)). Establishes the **roofline model** — three fundamental constraints (compute FLOPs/s, bandwidth B/s, capacity B) that bound every algorithm's wall time. Defines **arithmetic intensity** (FLOPs/byte) as the single number predicting whether an op is compute-bound or bandwidth-bound on given hardware, and derives critical batch-size thresholds for every later chapter.

## Key claims

1. Three rooflines bound any algorithm: peak compute FLOPs/s, peak bandwidth (HBM / ICI / DCN / PCIe) B/s, and total capacity B. Execution time `T_lower = max(T_math, T_comms)`; `T_upper = T_math + T_comms` (no-overlap). Practice usually approaches `T_lower`.
2. **Arithmetic intensity** (FLOPs/byte) determines regime: `intensity ≥ peak_FLOPs / peak_BW` ⇒ compute-bound; below ⇒ bandwidth-bound.
3. **Matmul is the canonical "scalable" op**: O(N³) FLOPs per O(N²) bytes ⇒ intensity grows with N, makes bigger matmuls more compute-bound.
4. Per-replica batch size > **~240 tokens** in bf16 makes matmul compute-bound on TPU v5e (critical intensity = `1.97e14 / 8.2e11 ≈ 240`); ~**164** on v5p (`4.59e14 / 2.8e12 ≈ 164`); **~298** on H100.
5. Sharded matmul becomes compute-bound over an ICI axis when contracting-dim > `2 × peak_FLOPs / ICI_BW`.
6. Communication happens at **four hierarchy levels** — within-chip (HBM↔VMEM), inter-chip (ICI), inter-slice (DCN), host-accelerator (PCIe) — each with its own roofline.

## Key data points

### Critical arithmetic intensities (bf16, per-chip)

| Hardware | Peak FLOPs (bf16) | HBM BW | Critical intensity | Note |
|---|---:|---:|---:|---|
| TPU v5e | 1.97×10¹⁴ | 8.2×10¹¹ | **~240** | inference-chip |
| TPU v5p | 4.59×10¹⁴ | 2.8×10¹² | **~164** | training-chip |
| NVIDIA H100 | ~9.9×10¹⁴ | 3.35×10¹² | **~296** | SXM5 |

Formula: `achieved_FLOPs/s = min(peak_FLOPs/s, bandwidth × arithmetic_intensity)`.

### Compute-bound thresholds for common ops

- Dot product (2 FLOPs per 2 bytes) → intensity 1 → always bandwidth-bound.
- Matmul (M×K × K×N → 2·M·K·N FLOPs / 2·(M·K + K·N + M·N) bytes) → intensity ≈ 1/(1/M + 1/N + 1/K) ≈ min(M,N,K) for balanced shapes.

## Techniques referenced

- Block matrix multiplication (tile-and-reuse to raise effective intensity).
- Compute/communication overlap (rings of ICI steps interleaved with MXU matmuls).
- Quantization modifying effective peak FLOPs (e.g., int8 halving bytes, doubling achievable tokens/s at fixed BW).

## Gaps & caveats

- Assumes "perfect overlap" upper bound; real compilers often under-overlap by ~20–50%.
- Ignores cache effects and tiling overhead for small problems.
- Derived for large matrices; small-matmul tile sizes require explicit block-size tuning (see [attention-block-sizes](../concepts/attention-block-sizes.md), [vmem-budget](../concepts/vmem-budget.md)).
- Book is dated **2025-02-04**; **v6e (Trillium)** and **v7 (Ironwood)** not analyzed here but their critical-intensity numbers matter — v7's VMEM halving changes the picture materially.
- Structured sparsity and quantization modify effective peak FLOPs in ways the single-number intensity doesn't capture.

## Connections

Informs every other chapter + most of this wiki:
- [concepts/roofline-model](../concepts/roofline-model.md)
- [concepts/arithmetic-intensity](../concepts/arithmetic-intensity.md)
- [concepts/compute-bound](../concepts/compute-bound.md) / [concepts/memory-bound](../concepts/memory-bound.md)
- [concepts/ridge-point](../concepts/ridge-point.md)
- [concepts/ici-roofline](../concepts/ici-roofline.md)
- [codebases/xprof](../codebases/xprof.md) — xprof's roofline viewer implements exactly this model.

## See also

- [Ch 2 — How to Think About TPUs](2025-scaling-book-ch2-tpus.md)
- [Ch 3 — Sharded Matrices](2025-scaling-book-ch3-sharding.md)
- [codebases/scaling-book](../codebases/scaling-book.md)

## Sources

- `raw/code/scaling-book/roofline.md`
- Upstream: <https://jax-ml.github.io/scaling-book/roofline>
