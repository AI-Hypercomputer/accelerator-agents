---
title: "XProf analysis tools"
type: topic
provenance: docs
updated: 2026-07-03
status: fresh
---
# XProf analysis tools

Once a profile is captured, XProf exposes a suite of viewer/stats tools. The Overview page is the entry point; the others drill into the trace, framework/HLO ops, and the compute graph.

## Tool catalog
- **Overview page** — top-level dashboard (TPU training/inference, GPU) ([overview-page-components](src:docs/overview_page.md#overview-page-components)); see [Overview page](../sources/overview_page.md).
- **Trace viewer** — the host+device event timeline ([Trace viewer](../sources/trace_viewer.md)).
- **Framework Op Stats** — per-framework-op aggregates ([framework-op-stats-components](src:docs/framework_op_stats.md#framework-op-stats-components)); see [Framework op stats](../sources/framework_op_stats.md).
- **HLO Op Stats / HLO Op Profile** — compiler-level op statistics ([hlo-op-stats-tool-components](src:docs/hlo_op_stats.md#hlo-op-stats-tool-components)) and cost ranking ([using-the-hlo-op-profile-tool](src:docs/hlo_op_profile.md#using-the-hlo-op-profile-tool)); see [HLO op stats](../sources/hlo_op_stats.md), [HLO op profile](../sources/hlo_op_profile.md).
- **Graph viewer** — the XLA HLO computation graph ([using-graph-viewer](src:docs/graph_viewer.md#using-graph-viewer)); see [Graph viewer](../sources/graph_viewer.md).
- **GPU Kernel Stats** — per-CUDA-kernel aggregates ([interface-components](src:docs/gpu_kernel_stats.md#interface-components)); see [GPU kernel stats](../sources/gpu_kernel_stats.md).

## See also
- [Memory analysis](memory-analysis.md), [Roofline & performance counters](roofline-and-counters.md), [Terminology](terminology.md)

