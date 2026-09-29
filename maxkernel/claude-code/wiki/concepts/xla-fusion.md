---
title: "XLA Fusion"
type: concept
tags: [stub, compiler, optimization]
created: 2026-04-22
updated: 2026-04-22
sources: 4
---

Compiler pass that merges multiple HLO ops into one kernel; materialized broadcasts (e.g. argmax over add-broadcast) can break fusion and land intermediate tensors in HBM.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [HLO (High Level Optimizer IR)](hlo.md)
- [HLO Op](hlo-op.md)
- [XLA Flags](xla-flags.md)
- [HBM (High-Bandwidth Memory)](hbm.md)
- [Memory-Bound](memory-bound.md)

## Sources

- [xprof-mcp TPU optimization guide](../sources/2026-xprof-mcp-tpu-optimization.md) — `raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md`
- [xprof HLO Op Stats](../sources/2026-xprof-hlo-op-stats.md) — `raw/code/xprof/docs/hlo_op_stats.md`
- [xprof HLO Op Profile](../sources/2026-xprof-hlo-op-profile.md) — `raw/code/xprof/docs/hlo_op_profile.md`
- [xprof Graph Viewer](../sources/2026-xprof-graph-viewer.md) — `raw/code/xprof/docs/graph_viewer.md`
