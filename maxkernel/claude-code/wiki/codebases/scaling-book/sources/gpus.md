---
title: "Source: gpus.md"
type: source-page
doc: gpus.md
updated: 2026-07-03
---
# gpus.md — What Is a GPU?

Part 12. Describes the GPU architecture (SMs with a Tensor Core, Warp Scheduler/CUDA cores, SMEM)
as the modular analog of a TPU's TensorCore, contrasting a GPU's ~100+ independent SMs against a
TPU's 1-2 large TensorCores
([what-is-a-gpu](src:gpus.md#what-is-a-gpu)). Tabulates HBM/L2/FLOPs specs across V100 through B200
([summary-of-gpu-specs](src:gpus.md#summary-of-gpu-specs)) and gives a 1:1 GPU-to-TPU component
mapping (SM↔TensorCore, Warp Scheduler↔VPU, SMEM↔VMEM)
([gpus-vs-tpus-at-the-chip-level](src:gpus.md#gpus-vs-tpus-at-the-chip-level)). Covers GPU networking
— NVLink fat-tree intra-node topology vs. TPU's nearest-neighbor ICI torus — and derives collective
costs on GPU (AllGather/ReduceScatter ≈ $B/W_\text{egress}$ intra-node; AllReduce 2x that unless
NVIDIA SHARP in-network reduction is enabled)
([networking](src:gpus.md#networking),
[how-do-collectives-work-on-gpus](src:gpus.md#how-do-collectives-work-on-gpus)).

## Feeds
- [GPU hardware architecture and GPU-vs-TPU](../topics/gpu-hardware-and-gpu-vs-tpu.md) — the core
  topic this chapter defines.
- [Sharding notation and collective communication](../topics/sharding-notation-and-collectives.md) —
  the TPU-side collective-cost comparison point.
