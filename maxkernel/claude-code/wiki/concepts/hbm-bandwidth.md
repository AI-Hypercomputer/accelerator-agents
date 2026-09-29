---
title: "HBM Bandwidth"
type: concept
tags: [stub, metric, memory, hardware]
created: 2026-04-22
updated: 2026-04-22
sources: 4
---

Peak aggregate throughput of on-package HBM per TPU chip (GB/s), set by the chip generation (v4/v5e/v5p/v6e). Distinct from [HBM](hbm.md), which names the memory tier itself — HBM Bandwidth is the scalar that sets the roofline's memory slope and therefore the ridge point beyond which kernels become compute-bound.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [HBM (High-Bandwidth Memory)](hbm.md)
- [Roofline Model](roofline-model.md)
- [Memory-Bandwidth Utilization](memory-bandwidth-utilization.md)
- [Memory-Bound](memory-bound.md)

## Sources

- [xprof Roofline Model](../sources/2026-xprof-roofline-model.md) — `raw/code/xprof/docs/roofline_model.md`
- [xprof-mcp TPU optimization guide](../sources/2026-xprof-mcp-tpu-optimization.md) — `raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md`
- [xprof HLO Op Stats](../sources/2026-xprof-hlo-op-stats.md) — `raw/code/xprof/docs/hlo_op_stats.md`
- [xprof Memory Profile](../sources/2026-xprof-memory-profile.md) — `raw/code/xprof/docs/memory_profile.md`
