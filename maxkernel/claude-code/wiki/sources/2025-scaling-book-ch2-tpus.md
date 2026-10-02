---
title: "How to Scale Your Model — Ch 2: How to Think About TPUs"
type: source
tags: [docs, book, scaling-book, tpu, hardware, mxu, vpu, vmem, hbm, ici, dcn, pcie, systolic-array]
author: DeepMind / JAX ML team
book: "How to Scale Your Model"
book_date: "2025-02-04"
chapter: 2
upstream: https://jax-ml.github.io/scaling-book/tpus
created: 2026-04-23
updated: 2026-04-23
---

Chapter 2 of the scaling-book. **The canonical introduction to TPU hardware** — cores, MXU (systolic array), VPU, VMEM, HBM, ICI topology, DCN, PCIe. Every latency/bandwidth constant this wiki uses ultimately traces here or to xprof/pallas-forge derivatives. Book is dated **2025-02-04** — covers **v3–v5p** in depth; **v6e (Trillium)** touched; **v7 (Ironwood) absent**.

## Key claims

1. TPU = MXU (systolic matrix unit) + VPU (vector) + VMEM (fast on-chip) + scalar control unit, backed by HBM; performance is dominated by moving data across the hierarchy.
2. **Systolic array** size: **128×128 on v4/v5, 256×256 on v6e** (Trillium). Executes one `[8, 128] × [128, 128]` multiplication per 8 cycles at ~1.5 GHz ⇒ ~2×10¹⁴ bf16 FLOPs/s per v5e (2 MXUs/core × 2 cores on megacore).
3. **VMEM** (~128 MiB on v5e) has ~22× the bandwidth of HBM; kernels entirely in VMEM only need intensity 10–20 FLOPs/byte to be compute-bound (vs. 240 for HBM).
4. **ICI** (Inter-Chip Interconnect) bandwidth: **90 GB/s per axis on v5p, 45 GB/s on v5e**, bidirectional. 2D/3D **toroidal** connectivity with optical-switch wraparound on full cubes; sub-cube slices (e.g., 4×4×4 on a larger v5p pod) **lose wraparound → 2× comm penalty**.
5. **Pod sizes**: v5e up to 16×16 (256 chips), v5p up to 16×20×28 (superpod, 8960 chips).
6. **DCN** (inter-pod): 6.25 GB/s egress/v5p chip, 3.125 GB/s /v5e chip — 100–1000× slower than ICI.
7. **PCIe** (host↔accelerator): ~1.6 GB/s /v5e chip — slower still; host-offload patterns must be rare.
8. **Tensor shapes must be ≥128 (≥256 on v6e)** to fully populate the systolic array; smaller axes pad with zeros (wasted FLOPs).

## Key data points

### Per-chip specs (book's canonical table)

| Metric | v5e | v5p |
|---|---:|---:|
| HBM capacity | 16 GB | 96 GB |
| HBM bandwidth | 8.1×10¹¹ B/s | 2.8×10¹² B/s |
| ICI bandwidth (1-way) | 4.5×10¹⁰ B/s | 9×10¹⁰ B/s |
| bf16 FLOPs/s | 1.97×10¹⁴ | 4.59×10¹⁴ |
| Pod | 16×16 (256 chips) | 16×20×28 (8960 chips) |
| Systolic array | 128×128 | 128×128 |
| Cores per chip | 1 | 2 (megacore) |

### Other hierarchy numbers

- VMEM: ~128 MiB/core on v5e (book number, pre-Trillium); **v6e 96 MiB / v7 48 MiB** per [tpu-inference](../codebases/tpu-inference.md) — see [vmem-budget](../concepts/vmem-budget.md).
- DCN: 6.25 GB/s per v5p chip egress.
- PCIe: 1.6 GB/s per v5e chip.

## Techniques referenced

- Systolic array (pipelined MAC).
- VPU 8×128 SIMD (softmax, elementwise).
- XLU cross-lane reductions.
- VMEM prefetch (feeds MXU without HBM round-trip).
- Megacore architecture (v4+: two cores sharing HBM).

## Gaps & caveats

- **v6e** (Trillium, 256×256 MXU, ~2× v5p FLOPs) mentioned briefly — this wiki's gemma4 program runs on v6e-4, so the book is **already out-of-date for primary workloads**.
- **v7** (Ironwood) absent — this wiki's future work will need external references.
- Sub-cube topology penalties (no wraparound on partial-pod slices) are analytically covered but not empirically validated.
- VPU throughput is ~10× lower than MXU; book mentions but does not deeply analyze (matters for softmax / GLU / norm FLOPs).
- Doesn't cover **SparseCore** (v5p/v7x) in detail — see [concepts/sparsecore](../concepts/sparsecore.md) + [maxtext](../codebases/maxtext.md) `sc_gather_reduce`.
- Assumes well-known default configurations; custom/non-default topologies (e.g., `2×4` mesh on a bigger slice) require manual analysis.

## Connections

- [concepts/mxu](../concepts/mxu.md), [concepts/vpu](../concepts/vpu.md), [concepts/vmem](../concepts/vmem.md), [concepts/hbm](../concepts/hbm.md)
- [concepts/ici](../concepts/ici.md), [concepts/dcn](../concepts/dcn.md), [concepts/megascale](../concepts/megascale.md)
- [concepts/sparsecore](../concepts/sparsecore.md)
- [concepts/dimension-alignment](../concepts/dimension-alignment.md) — the 128 / 256 rule.
- [concepts/vmem-budget](../concepts/vmem-budget.md) — this wiki's per-generation update.
- [codebases/pallas-forge](../codebases/pallas-forge.md) — `TPU_SPECS` table copies this chapter's numbers.

## See also

- [Ch 1 — Rooflines](2025-scaling-book-ch1-roofline.md)
- [Ch 3 — Sharded Matrices](2025-scaling-book-ch3-sharding.md)
- [Ch 11 — GPUs](2025-scaling-book-ch11-gpus.md)
- [codebases/scaling-book](../codebases/scaling-book.md)

## Sources

- `raw/code/scaling-book/tpus.md`
- Upstream: <https://jax-ml.github.io/scaling-book/tpus>
