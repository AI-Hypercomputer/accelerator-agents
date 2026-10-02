---
title: "How to Scale Your Model — Ch 11: How to Think About GPUs"
type: source
tags: [docs, book, scaling-book, gpu, nvidia, h100, b200, nvlink, nvswitch, infiniband, sparsity, fp8]
author: DeepMind / JAX ML team
book: "How to Scale Your Model"
book_date: "2025-02-04"
chapter: 11
upstream: https://jax-ml.github.io/scaling-book/gpus
created: 2026-04-23
updated: 2026-04-23
---

Chapter 11 of the scaling-book. **GPU-vs-TPU comparison framing**: Tensor Cores, NVLink / NVSwitch / InfiniBand hierarchy, structured sparsity, FP8. For this wiki — a TPU-focused autoresearch — included as GPU↔TPU translation reference rather than primary content.

## Key claims

1. **H100** peak bf16 FLOPs **9.9×10¹⁴/s**; HBM BW **3.35 TB/s** → critical intensity **~298 FLOPs/byte** (vs TPU v5e 240, v5p 164).
2. Tensor Core tile sizes (FP32/TF32): **140×140** — similar to TPU 128×128, similar batch-size thresholds apply.
3. **Structured sparsity** (50% / 2:4 patterns): NVIDIA claims 2× peak FLOPs; real-world gains 1–1.3× due to accuracy trade-offs.
4. **NVLink** (H100): 25 GB/s per link, 8-way per GPU, per direction; optimized for AllReduce within a node.
5. **NVSwitch** (H100 NVL72 pod): all-to-all 25 GB/s between any pair — higher bisection than TPU ICI but more expensive.
6. **Hierarchical networking**: NVLink (within node) >> InfiniBand (cross-node, ~12.5 GB/s /dir) >> rack-to-rack. Contrast with TPU 2D/3D toroidal.
7. **Cost** (Feb 2025 GCP): H100 $10.8/hr, v5p $4.2/hr, v5e $1.2/hr.
8. **FLOPs/$** ranking for inference: v5e (5.8×10¹⁷) > v5p (3.9×10¹⁷) > H100 (3.3×10¹⁷).
9. GPUs excel at **FP8** (TensorFloat32 internal); TPUs similar.

## Key data points

| Metric | H100 | B200 | TPU v5p | TPU v5e |
|---|---:|---:|---:|---:|
| Peak bf16 FLOPs | 9.9×10¹⁴ | ~2×10¹⁵ | 4.59×10¹⁴ | 1.97×10¹⁴ |
| HBM BW | 3.35 TB/s | ~5 TB/s | 2.8 TB/s | 0.82 TB/s |
| Critical intensity (bf16) | ~298 | ~400 | ~164 | ~240 |
| NVLink BW/GPU (1-way) | 25 × 8 GB/s | ~40 × 12 GB/s | — | — |
| Cost ($/hr on GCP) | $10.8 | N/A | $4.2 | $1.2 |
| FLOPs/$ (bf16) | 3.3×10¹⁷ | — | 3.9×10¹⁷ | 5.8×10¹⁷ |

### Critical batch size (H100)

`B_crit = 9.9×10¹⁴ / 3.35×10¹² ≈ 296 tokens` — close to v5e's 240.

## Techniques referenced

- NVLink topology (8 per GPU, 25 GB/s/direction).
- NVSwitch pod-level all-to-all.
- InfiniBand fat-tree (cross-node).
- MIG (multi-instance GPU).
- NCCL collective library.
- Structured sparsity (2:4).
- cuTENSOR tensor contractions.

## Gaps & caveats

- **B200 numbers partial / aspirational** (book Feb 2025).
- Sparse / MoE on GPUs not deeply analyzed.
- Mixed-precision (FP8) quantified for GPUs but not directly compared to TPU int8.
- AMD MI300X and other accelerators absent.
- Multi-GPU AllReduce topology (ring / tree / NVSwitch) not analytically treated.
- Cost snapshot is **Feb 2025 GCP pricing** — current numbers vary.

## Connections

- [Ch 1 — Rooflines](2025-scaling-book-ch1-roofline.md) — same formulas, different hardware constants.
- [Ch 2 — TPUs](2025-scaling-book-ch2-tpus.md) — companion hardware chapter.
- [concepts/mxu](../concepts/mxu.md) — TPU-side MXU analogue of Tensor Cores.
- [codebases/tokamax](../codebases/tokamax.md) — has GPU + TPU paths for direct comparison.
- [codebases/alphafold3](../codebases/alphafold3.md) — GPU Pallas reference.
- [sources/2025-ultrascale-playbook.md](2025-ultrascale-playbook.md) — complementary GPU-cluster training playbook.

## See also

- [Ch 2 — TPUs](2025-scaling-book-ch2-tpus.md)

## Sources

- `raw/code/scaling-book/gpus.md`
- Upstream: <https://jax-ml.github.io/scaling-book/gpus>
