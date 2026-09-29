---
title: "Memory-Efficient Cross-Entropy"
type: concept
tags: [stub, kernel, loss]
created: 2026-04-22
updated: 2026-04-22
sources: 1
---

Fused linear + log-softmax + NLL that avoids materializing the full `[B, V]` logits tensor. TPU Mosaic kernel in tokamax.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [Mosaic Kernel](mosaic-kernel.md)
- [Pallas Kernel](pallas-kernel.md)
- [Gated Linear Unit](gated-linear-unit.md)
- [Layer Norm / RMS Norm](layer-norm.md)
- [HBM (High-Bandwidth Memory)](hbm.md)

## Sources

- [tokamax supported ops](../sources/2026-tokamax-supported-ops.md) — `raw/code/tokamax/docs/supported_ops.md`
