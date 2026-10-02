---
title: "Context Parallelism"
type: concept
tags: [stub, parallelism, attention]
created: 2026-04-22
updated: 2026-04-22
sources: 1
---

Parallelism axis that splits the **sequence** dimension across devices specifically for attention (where tokens interact), complementing tensor-parallel splits on the hidden dimension. Implemented via Ring Attention or all-to-all variants. Unlocks sequence lengths that exceed any single device's HBM. On TPU, the compute kernel is splash attention (non-ring) or the experimental `ring_attention_kernel` in tokamax.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [Ring Attention](ring-attention.md)
- [Sequence Parallelism](sequence-parallelism.md)
- [Splash Attention](splash-attention.md)
- [Tensor Parallelism](tensor-parallelism.md)
- [Sharding (GSPMD)](sharding.md)

## Sources

- [Ultra-Scale Playbook](../sources/2025-ultrascale-playbook.md) — Section 5
