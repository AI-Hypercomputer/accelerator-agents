---
title: "XProf — HLO Op Profile Tool"
type: source
tags: [docs, profiler, hlo, xprof, op-profile, utilization, roofline]
created: 2026-04-22
updated: 2026-04-22
---

XProf documentation for the **HLO Op Profile** tool: a hierarchical, utilization-first view of HLO ops grouped by module → category → op, with overall FLOPs and HBM-bandwidth utilization at the top. Where HLO Op Stats ranks by time, HLO Op Profile is designed to find where **hardware is under-utilized** — especially ops that are both slow and wasteful of FLOPs.

## Overview

The top of the page shows the profiling session's overall compute (FLOPs) utilization and HBM bandwidth utilization. Below it, a drillable table breaks the work down by module, then by category within a module, then by individual op. Fusions can be further expanded to show the element-wise, non-fusion ops they contain.

Two sort modes matter:

1. **By fraction of total time** (default): classic time ranking.
2. **By under-utilization weighted by runtime** (the doc's "wasted time"): surfaces ops with low FLOPs utilization *and* high time consumption — the best targets for optimization.

Hovering an op shows a detail card (pinnable by clicking) with a Graph Viewer link, average execution time, absolute rates (TFLOP/s, HBM GB/s, on-chip read/write GB/s), full XLA op details (shapes, layouts), framework provenance, occurrences, and aggregate time.

## Key claims

- Utilization numbers (percentages) = absolute resource consumption ÷ per-accelerator peak.
- Absolute rates (TFLOP/s, GB/s) = compiler-static FLOPs or bytes ÷ measured profile duration.
- Fusion categorization is mostly compiler-defined; XProf adds heuristics (e.g., it parses the HLO graph to identify "convolution fusions").
- Expanding a fusion reveals the non-fusion element-wise ops inside it — useful when a fusion's aggregate looks fine but one inner op dominates.
- The "wasted time" sort (under-utilization × runtime) is the tool's primary optimization-targeting affordance.
- Top-of-page overall FLOPs utilization and HBM BW utilization are the first two numbers to read from any profile.

## Key data points

### Hierarchy

| Level | What it groups | When to drill |
|---|---|---|
| Module | One HLO module (program) | Multi-module runs; find which module dominates |
| Category | Compiler categories + XProf heuristics (convolution fusion, etc.) | Identify where time bucketizes |
| Op | Individual HLO op | Candidate for targeted optimization |
| Fusion expansion | Non-fusion element-wise ops inside a fusion | Confirm whether a fusion is well-structured |

### Op detail card fields

| Field | Meaning |
|---|---|
| Graph Viewer link | Jump to op in HLO graph |
| Average execution time | Avg per-occurrence time |
| TFLOP/s (absolute) | Raw compute rate, not utilization |
| HBM GB/s (absolute) | Raw HBM rate |
| On-chip read/write GB/s | On-chip (e.g., VMEM/CMEM) rates |
| Full XLA op details | Shapes, layouts |
| Framework provenance | JAX/TF origin |
| Occurrences + aggregate time | Sanity check time share |

### Absolute rates vs utilization

| Concept | Formula | Shown where |
|---|---|---|
| Absolute rate (TFLOP/s or GB/s) | static numerator ÷ measured duration | Op detail card |
| Utilization (%) | absolute rate ÷ peak device capability | Table cells, top-of-page overview |

## Techniques referenced

- Roofline-style thinking: low FLOPs utilization + high runtime ⇒ optimization candidate.
- Fusion analysis via drill-down to non-fusion element-wise components.
- Graph Viewer cross-linking (detail cards link into the HLO graph).
- Convolution-fusion heuristic identification in HLO.

## Gaps & caveats

- All utilization and rate numbers depend on the XLA compiler's static cost analysis — custom ops with no cost model will report zero/garbage rates even if they consume real time.
- "Wasted time" sort uses FLOPs utilization specifically; a memory-bound op with high HBM utilization but low FLOPs utilization can appear "wasteful" when it isn't actually addressable by more compute. Cross-check with the compute/memory-bound signal from HLO Op Stats.
- The top-of-page overall utilization is aggregate over the profiling window — a single long low-utilization op can drag it down even when most ops are fine.
- Fusion drill-down exposes element-wise ops only; non-element-wise children inside a fusion are not expanded the same way.

## Connections

- `hlo-op` — the unit displayed.
- `mxu-utilization` / `flops-utilization` — the primary top-of-page number.
- `hbm-bandwidth` — the other top-of-page number.
- `fusion` — category and drill-down behavior.
- `convolution-fusion` — XProf heuristic category.
- `roofline` — under-utilization × time logic.
- `graph-viewer` — cross-linked tool.
- `wasted-time-sort` — the tool's targeting heuristic.

## See also

- [xprof](../codebases/xprof.md)
- [XProf HLO Op Stats](2026-xprof-hlo-op-stats.md)
- [XProf Framework Op Stats](2026-xprof-framework-op-stats.md)
- [XProf Perf Counters](2026-xprof-perf-counters.md)
- [XProf Custom Call Profiling](2026-xprof-custom-call-profiling.md)

## Sources

- `raw/code/xprof/docs/hlo_op_profile.md`
