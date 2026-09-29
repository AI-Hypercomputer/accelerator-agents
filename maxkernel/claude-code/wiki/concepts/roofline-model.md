---
title: "Roofline Model"
type: concept
tags: [stub, performance, analysis]
created: 2026-04-22
updated: 2026-04-22
sources: 4
---

Performance model stating achievable FLOPs/s = min(arithmetic_intensity × peak_bandwidth, peak_flops); compute-bound above the ridge point, memory-bound below.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [Arithmetic Intensity](arithmetic-intensity.md)
- [Memory-Bound](memory-bound.md)
- [Compute-Bound](compute-bound.md)
- [Peak FLOPs](peak-flops.md)
- [ICI Roofline](ici-roofline.md)

## Sources

- [xprof Roofline Model](../sources/2026-xprof-roofline-model.md) — `raw/code/xprof/docs/roofline_model.md`
- [xprof-mcp TPU optimization guide](../sources/2026-xprof-mcp-tpu-optimization.md) — `raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md`
- [xprof HLO Op Profile](../sources/2026-xprof-hlo-op-profile.md) — `raw/code/xprof/docs/hlo_op_profile.md`
- [xprof HLO Op Stats](../sources/2026-xprof-hlo-op-stats.md) — `raw/code/xprof/docs/hlo_op_stats.md`
