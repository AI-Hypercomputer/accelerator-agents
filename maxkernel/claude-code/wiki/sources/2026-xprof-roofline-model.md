---
title: "XProf Roofline Model Tool"
type: source
tags: [docs, profiler, roofline, arithmetic-intensity, memory-bound, compute-bound, hbm, vmem, cmem]
created: 2026-04-22
updated: 2026-04-22
---

The Roofline Model tool in xprof is a visual performance model that classifies a program (or individual ops) as memory-bound vs. compute-bound by plotting achieved FLOPS/s against operational intensity (FLOPS per byte accessed). It draws the hardware's theoretical "roof" — the minimum of peak FLOPS and (operational intensity × peak memory bandwidth) — and shows how far the workload sits from that ceiling. Supported on TPU (GA) and GPU (beta).

## Overview

The roofline chart makes the classic bound analysis visual. The x-axis is operational intensity (FLOPS/byte); the y-axis is achieved FLOPS/s. The roof is:

$$
Roofline = \min(\text{Operational Intensity} \times \text{Peak Memory Bandwidth},\ \text{Peak FLOPS})
$$

Two regimes:

- Slanted segment: performance = operational_intensity × peak_bandwidth. A point here is **memory-bound** — raising arithmetic intensity (more compute per byte moved) is the main lever.
- Flat segment: performance = peak FLOPS. A point here is **compute-bound** — raising intensity further yields no gain; need to improve FLOPS utilization (better kernels, precision, fusion).

The **ridge point** is the minimum operational intensity required to reach peak FLOPS; it is a hardware property per memory tier.

The tool draws one roof line per supported memory. For TPUs: **HBM**, **VMEM**, and **CMEM** (TPU v4 only). For GPUs: **HBM** and **L1/SharedMem**. A given memory's roof line only appears if some op in the profile was bound by that memory — if every op is HBM- or compute-bound, the VMEM/CMEM lines are omitted.

The external reference for the model is Williams, Waterman, and Patterson's "Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures" (CACM 2008).

## Key claims

- **Memory-bound vs. compute-bound is readable off the chart.** Points on the slanted part are memory-bound; points on the flat part are compute-bound. The distance below the roof is the available headroom.
- **Two roofline sections per profile.** Program-level (whole profile, per-step averages, per-step points) and operation-level (top-1000 most time-consuming ops). Both sections have companion statistics tables with peak FLOP rate %, max memory utilization %, etc.
- **Hardware counters vs. XLA cost model.** The program-level chart shows the profile both with FLOPS/s from XLA's static cost model and, separately, from hardware performance counters. These can disagree — the counter value is ground truth for FLOP rate; the cost model is what the compiler "thinks" it is doing.
- **Per-memory roofs diagnose which memory is the bottleneck.** On TPU v4 the chart can separate ops bound by HBM vs. VMEM vs. CMEM. If a point sits on the VMEM roof but has slack to the HBM roof, the bottleneck is on-chip bandwidth, not HBM.
- **Infeed/outfeed can be toggled.** Both charts have a dropdown to include or exclude infeed/outfeed ops — host-input stalls otherwise dominate and mask compute bottlenecks.
- **Operation-level filtering.** The op chart can be filtered by op category, by resource-bound class, and by op name — this is how you isolate attention, all-reduces, layernorms, etc.

## Key data points

### Roof construction

| Input | Source | Role |
|---|---|---|
| Peak FLOPS | Device spec | Horizontal part of the roof |
| Peak memory bandwidth (per memory tier) | Device spec | Slope of the slanted part |
| Operational intensity | FLOPs / bytes accessed (per op or aggregate) | x-coordinate of each data point |
| Achieved FLOPS/s | XLA cost model or HW counter | y-coordinate of each data point |

### Memory tiers plotted (per platform)

| Platform | Memories with roof lines |
|---|---|
| TPU (v4) | HBM, VMEM, CMEM |
| TPU (non-v4) | HBM, VMEM |
| GPU (beta) | HBM, L1/SharedMem |

### Program-level data points (Section 1)

| Data point | Meaning |
|---|---|
| Total profile duration (cost model) | Whole-profile FLOPS/s from XLA static cost model |
| Total profile duration (HW counters) | Whole-profile FLOPS/s from hardware performance counters |
| Step average | Mean across complete training steps in the profile |
| Per-step point | One point per complete step executed during the profile |

### Interpretation rules

| Position | Diagnosis | Next move |
|---|---|---|
| On slanted part | Memory-bound | Raise operational intensity (fuse, tile, reuse, re-materialize) |
| On flat part | Compute-bound | Raise peak-FLOPS utilization (precision, better kernel, fewer pipeline bubbles) |
| Well below roof | Both memory and compute are under-utilized | Investigate — likely scheduling, sync, or host/infeed stalls |
| On VMEM roof, below HBM roof (v4) | On-chip bandwidth bound | Change layout / tiling to reduce VMEM traffic |

### Op-chart filtering dimensions

| Filter | What it does |
|---|---|
| Include/exclude infeed and outfeed | Removes host-input-bound data points that dwarf everything else |
| Op category | Focus on matmul / conv / reduction / collective / etc. |
| Resource-bound class | Show only ops bound by a specific memory or by compute |
| Op name substring | Isolate a specific HLO op family |

### Hover tooltips and per-point stats (from tables)

| Field | Example use |
|---|---|
| Bandwidth numbers per memory | Which memory tier was the binder |
| Total time spent | Rank ops by cost — spotlight the top contributors |
| Max memory utilization % | Closeness to a given memory's peak |
| Peak FLOP rate % | Closeness to the compute roof (MFU-like signal) |

## Techniques referenced

- **Roofline model** (Williams/Waterman/Patterson 2008) — the underlying visualization.
- **Operational intensity (arithmetic intensity)** — FLOPs per byte accessed; the x-axis of the chart.
- **HLO cost model vs. hardware performance counters** — two distinct FLOPS/s sources for the same workload.
- **Per-memory rooflines (HBM / VMEM / CMEM)** — multi-roof variant that distinguishes which memory is the binder.
- **Infeed/outfeed filtering** — excluding host-input stalls to see the compute story.
- **Op categorization and resource-bound filtering** — slicing the top-1000 ops by kind or by binder.

## Gaps & caveats

- **XLA cost model is not ground truth.** The program-level chart intentionally shows both cost-model and HW-counter FLOPS/s because they can disagree. Any hypothesis driven off the cost-model point should be cross-checked against the counter point before acting.
- **Only top-1000 ops appear on the op chart.** Long-tail small ops are invisible there; they may still matter in aggregate and must be read from profile summary or op profile.
- **CMEM roof is TPU v4 only.** v5e/v5p/v6e do not expose CMEM as a separate memory tier on the chart; this doc does not describe how newer generations are handled.
- **Doc does not give numeric peak values.** Peak FLOPS and peak bandwidths per generation are not listed here — the tool pulls them from Device Information and draws the roofs automatically.
- **Proximity to roof is qualitative.** "Distance from the roof indicates potential improvement" — but the doc does not give a threshold below which a point is considered optimized.
- **Collectives are not called out.** The doc does not address how ICI/DCN collectives are placed on the chart (whether they count as compute, are excluded, or have their own tier).
- **Operational intensity definition.** The doc does not pin down whether "byte accessed" is from HBM only, or includes VMEM/CMEM traffic. The multi-roof design implies per-memory intensities exist, but the mapping is not spelled out.

## Connections

- `roofline-model`
- `arithmetic-intensity`
- `operational-intensity`
- `memory-bound`
- `compute-bound`
- `ridge-point`
- `hbm-bandwidth`
- `vmem`
- `cmem`
- `peak-flops`
- `xla-cost-model`
- `hardware-performance-counters`
- `mfu`
- `infeed-outfeed`

## See also

- [xprof](../codebases/xprof.md)
- [XProf Megascale Stats Tool](2026-xprof-megascale-stats.md)
- [XProf Megascale Viewer](2026-xprof-megascale-viewer.md)
- [XProf Megascale Viewer SQL](2026-xprof-megascale-viewer-sql.md)

## Sources

- `raw/code/xprof/docs/roofline_model.md`
