---
title: "Source: tpus.md"
type: source-page
doc: tpus.md
updated: 2026-07-03
---
# tpus.md — How to Think About TPUs

Part 2. Describes the TPU chip: a TensorCore (MXU + VPU + VMEM) attached to HBM
([what-is-a-tpu](src:tpus.md#what-is-a-tpu)). Covers the megacore configuration, PCIe host
attachment, and the ICI network topology (2D/3D torus, wraparounds only on full optical-switch
cubes or full-16 axes) that connects chips within a slice, plus DCN for inter-slice/inter-pod
communication ([tpu-networking](src:tpus.md#tpu-networking)). Gives per-generation HBM
capacity/bandwidth, FLOPs/s, and ICI bandwidth tables for TPU v3 through TPU7x
([tpu-specs](src:tpus.md#tpu-specs)). The appendix details the VPU's `(8,128)` SIMD lane/sublane
structure and VREG count, and explains the systolic array mechanism underlying the MXU
([appendix-a-more-on-tpu-internals](src:tpus.md#appendix-a-more-on-tpu-internals),
[appendix-b-how-does-a-systolic-array-work](src:tpus.md#appendix-b-how-does-a-systolic-array-work)).

## Feeds
- [TPU hardware architecture](../topics/tpu-hardware-architecture.md) — the primary topic this
  chapter defines.
- [GPU hardware architecture and GPU-vs-TPU](../topics/gpu-hardware-and-gpu-vs-tpu.md) — the
  networking-model comparison this chapter sets up against GPUs.
