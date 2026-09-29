---
title: "Ridge Point"
type: concept
tags: [stub, performance, analysis]
created: 2026-04-22
updated: 2026-04-22
sources: 2
---

Arithmetic intensity (FLOPs/byte) at which the slanted memory-bandwidth ceiling meets the horizontal peak-FLOPs ceiling on the roofline. Kernels with intensity below the ridge point are memory-bound; above it, compute-bound. On TPU the ridge point is set by `peak_flops / hbm_bandwidth` and shifts with dtype (bf16 vs fp8 vs int8) and generation.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [Roofline Model](roofline-model.md)
- [Arithmetic Intensity](arithmetic-intensity.md)
- [Memory-Bound](memory-bound.md)
- [Compute-Bound](compute-bound.md)
- [Peak FLOPs](peak-flops.md)

## Sources

- [xprof Roofline Model](../sources/2026-xprof-roofline-model.md) — `raw/code/xprof/docs/roofline_model.md`
- [xprof-mcp TPU optimization guide](../sources/2026-xprof-mcp-tpu-optimization.md) — `raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md`
