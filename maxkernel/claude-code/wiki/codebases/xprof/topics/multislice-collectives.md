---
title: "Multi-slice collectives (Megascale)"
type: topic
provenance: docs
updated: 2026-07-03
status: fresh
---
# Multi-slice collectives (Megascale)

For multi-slice / multi-host TPU jobs, XProf's Megascale tools analyze collective-communication (DCN) performance — the action graph, per-collective stats, and PerfettoSQL queries over the trace.

## Tools
- **Megascale Stats** — collective-communication summaries and collective terminology ([analyzing-megascale-stats-tool-data](src:docs/megascale_stats.md#analyzing-megascale-stats-tool-data), [terminology-related-to-collectives](src:docs/megascale_stats.md#terminology-related-to-collectives)); see [Megascale stats](../sources/megascale_stats.md).
- **Megascale Viewer** — the action graph connecting TPU ops to collective actions, for latency diagnosis ([connecting-tpu-ops-to-megascale-action-graph](src:docs/megascale_viewer.md#connecting-tpu-ops-to-megascale-action-graph)); see [Megascale viewer](../sources/megascale_viewer.md).
- **PerfettoSQL queries** — extract recv-done ops/stats from the trace ([example-perfettosql-queries](src:docs/megascale_viewer_sql.md#example-perfettosql-queries-for-megascale-viewer)); see [Megascale viewer SQL](../sources/megascale_viewer_sql.md).

## See also
- [DCN/PCIe profiling](roofline-and-counters.md), [Deploying XProf](deployment.md) (distributed)

