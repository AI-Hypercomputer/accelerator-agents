---
title: "XProf Graph Viewer (docs)"
type: source
tags: [docs, profiler, xprof, graph-viewer, hlo, xla, tpu, gpu]
created: 2026-04-22
updated: 2026-04-22
---

Graph Viewer renders the XLA **optimized-HLO** graph with overlaid runtime statistics — the go-to tool for inspecting the shape of the computation the compiler actually produced and the measured cost of individual ops.

## Overview

Graph Viewer takes a selected XLA module and a primary HLO op, and renders the op's neighborhood (controlled by `Graph Width`). Static graph data comes from XLA without execution; on top of that, hover overlays show measured execution count, average time, and rate-based utilization/bandwidth derived from XLA's static FLOPS/bytes cost analysis. The view is directly linkable from Op Profile and Trace Viewer.

## Key claims

- Graph Viewer operates on **optimized HLO** — the form close to backend lowering — **not** StableHLO or a framework-level graph. All XProf tools share this level.
- The graph structure is **static** (no run needed) but runtime stats are overlaid: execution count, average execution time, and computed utilization/bandwidth rates.
- Utilization and bandwidth numbers on hover are derived by combining **static cost analysis (FLOPS, bytes)** from XLA with **measured execution time** — not hardware perf counters.
- `Merge Fusion` toggles whether fused-op components are collapsed or expanded, which changes both display size and per-op attribution granularity.
- `Show Metadata` toggles compiler metadata overlay; the graph must be re-searched after toggling.
- The user-code source line can appear on hover for an op, **only if** framework/compiler layers propagated metadata through the transformation chain.
- Entry point is often another tool: Op Profile (to find the hottest op) or Trace Viewer (to find the cause of a pipeline bubble) — both deep-link into Graph Viewer at the selected op.
- Export: HLO module can be downloaded as `.pb`, `.pbtxt`, short text, and long text; the graph itself as SVG, HTML, or DOT.

## Key data points

### Controls

| Control | Purpose |
|---|---|
| XLA Modules dropdown | Pick HLO module to visualize |
| Graph Type dropdown | Pick graph rendering type |
| XLA Op Name box | Primary node to center the view on |
| Graph Width | Max distance (hops) from primary node |
| Show Metadata | Toggle compiler metadata overlay (requires re-search) |
| Merge Fusion | Collapse/expand fused-op internals (requires re-search) |
| Zoom in/out/reset | UI + mouse |
| Search | Locate specific ops within large graphs |
| Click op | Freeze runtime data panel on that op |

### Exports

| Target | Formats |
|---|---|
| HLO module | `.pb`, `.pbtxt`, short text, long text |
| Graph image | SVG, HTML, DOT |
| HLO full text | Short or long representation (via top button) |

### Runtime overlay (on hover)

- Execution count during profile.
- Average execution time.
- Utilization % (derived: static FLOPS / measured time).
- Bandwidth (derived: static bytes / measured time).
- User-code line (if metadata propagated).

## Techniques referenced

- Optimized-HLO-level inspection (pre-backend-lowering view).
- Static graph + runtime-overlay hybrid visualization.
- Rate-based utilization via static cost analysis × measured time (contrast with hardware counters used in Utilization Viewer).
- Fusion collapse/expand as an analysis knob.
- Cross-tool linking: Op Profile → Graph Viewer, Trace Viewer → Graph Viewer.
- Graph export to DOT/SVG for external processing or documentation.

## Gaps & caveats

- Graph Viewer does **not** show framework-level graphs — only optimized HLO. Users expecting PyTorch/JAX-level structure must bridge via metadata.
- Utilization and bandwidth are **derived estimates**, not measured — they do not reflect spills, cache misses, or dynamic scheduling overhead. Utilization Viewer uses hardware counters for the real number.
- Graph size can be huge; `Graph Width` limits neighborhood, not total complexity.
- Metadata (source line, framework op) is optional — absence is common in heavily-optimized builds.
- Static FLOPS/bytes values may over- or under-count relative to what actually ran (skipped fusions, rematerialization).

## Connections

Concept slugs this source informs:

- `hlo` — the representation Graph Viewer renders.
- `optimized-hlo` — specifically the post-optimization form used by XProf.
- `xla-compiler` — source of the static graph and cost analysis.
- `xla-fusion` — Merge Fusion control exposes fused-op structure.
- `xla-op` — the node type in the graph.
- `hlo-op-profile-link` — entry from Op Profile.
- `static-cost-analysis` — FLOPS/bytes source for rate derivation.
- `graph-export-dot` — external visualization workflow.

## See also

- [xprof](../codebases/xprof.md)
- [xprof trace viewer](2026-xprof-trace-viewer.md)
- [xprof memory viewer](2026-xprof-memory-viewer.md)
- [xprof utilization viewer](2026-xprof-utilization-viewer.md)
- [xprof overview page](2026-xprof-overview-page.md)
- [xprof terminology](2026-xprof-terminology.md)

## Sources

- `raw/code/xprof/docs/graph_viewer.md`
