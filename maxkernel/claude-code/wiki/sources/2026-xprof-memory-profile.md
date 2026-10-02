---
title: "XProf Memory Profile (docs)"
type: source
tags: [docs, profiler, xprof, memory-profile, hbm, tpu, gpu]
created: 2026-04-22
updated: 2026-04-22
---

Memory Profile is the dynamic, time-series view of device memory use over the profiling interval. It is the sibling of Memory Viewer (static, per-module), and its main jobs are OOM debugging and fragmentation diagnosis.

## Overview

Memory Profile monitors device memory throughout the profiling interval and reports allocations, deallocations, peak usage, fragmentation percentage, and the framework ops that own memory at peak. Allocations are managed by XLA's runtime allocator which owns the entire HBM space. On TPU the allocation curve is often a **flat line** because XLA performs a large upfront allocation — Memory Profile is therefore typically more informative on GPU than TPU.

## Key claims

- Memory Profile is **dynamic** (over profiling-interval time), whereas Memory Viewer is **static** (over program order of a single XLA module).
- Memory Profile gives a **global view** across all XLA modules on the device; Memory Viewer is per-module only. Multi-module workloads need Memory Profile to see the total picture.
- XLA's runtime allocator owns all of HBM; allocations and frees visible in Memory Profile are its bookkeeping, not framework-level allocator activity.
- A high `Fragmentation` percentage means free memory exists but is non-contiguous — large allocations may OOM even with apparent headroom.
- Peak-usage analysis reports the **lifetime** peak since the model started (possibly before profiling began) separately from the in-window peak.
- On TPU the allocation chart tends to be flat because XLA front-loads allocations; the tool is most valuable on GPU workloads and for TPU fragmentation debugging.
- The `Memory Breakdown Table` shows per-framework-op contributions at the peak-usage instant within the window, with shape and dtype when the compiler exposes them.

## Key data points

### UI components

| Component | Role |
|---|---|
| Host ID selector | Choose which host to profile |
| Memory ID selector | Choose HBM attached to a specific accelerator (or host memory) |
| Memory Profile Summary | Totals, capacity, lifetime peak, in-window peak, fragmentation |
| Memory Timeline Graph | Usage vs. time + fragmentation %, broken into stack/heap/free |
| Memory Breakdown Table | Per-op contributions at peak |

### Memory Profile Summary — fields

- Total allocations during interval.
- Total deallocations during interval.
- Total memory capacity of selected memory system.
- Lifetime peak heap usage (possibly outside profiling interval).
- In-window peak usage.
- Fragmentation percentage.

### Memory Timeline Graph — axes

| Axis | What it shows |
|---|---|
| X | Time within profiling interval |
| Left Y | Total memory usage, stacked as stack (red) / heap (orange) / free (green) |
| Right Y | Fragmentation percentage |
| Hover | Info card of allocation/deallocation events at that timestamp |

### Memory Breakdown Table — columns (when available)

- Framework op name.
- Allocation size.
- Shape.
- Data type.
- Additional compiler-supplied metadata.

### Memory Profile vs. Memory Viewer

| Axis | Memory Profile | Memory Viewer |
|---|---|---|
| Time axis | Wall-clock within profiling interval | Program order of an HLO module |
| Scope | Global across XLA modules on device | Per XLA module |
| Data source | Runtime allocator events | Static XLA compile-time info |
| Best for | OOM debug, fragmentation, dynamic allocation patterns | Understanding program-order peak, buffer lifetimes, padding overhead |

## Techniques referenced

- Stack/heap/free stacked-area visualization over time.
- Fragmentation as a percentage alongside absolute usage.
- Framework-op-level attribution at the peak instant (OOM forensic pattern).
- Dual-peak reporting (lifetime vs. in-window) to avoid missing pre-profile allocations.
- Distinct host-memory vs. per-accelerator-HBM views.

## Gaps & caveats

- On TPU the chart is often flat and uninformative because XLA allocates upfront — don't expect dynamic signal unless your workload truly allocates/frees at runtime.
- Lifetime peak may lie outside the profiling window — you may see the "peak" but not the events that caused it.
- Breakdown table only reflects peak-instant ownership; transient peaks between allocator events are not attributed.
- Metadata (shape, dtype, framework op) depends on the compiler surfacing it — missing metadata means only sizes are visible.
- Fragmentation %% semantics (definition, numerator/denominator) are not spelled out in the doc.
- Host memory view is available "in certain cases" but conditions are not enumerated.

## Connections

Concept slugs this source informs:

- `hbm` — the device memory being monitored.
- `memory-fragmentation` — the core failure mode the tool surfaces.
- `oom-debugging` — primary use case of the tool.
- `xla-runtime-allocator` — owner of the HBM space.
- `memory-timeline-graph` — the stack/heap/free-over-time visualization.
- `peak-memory-usage` — in-window vs. lifetime distinction.
- `memory-breakdown-table` — per-op attribution pattern.
- `host-memory` — optional view within the tool.

## See also

- [xprof](../codebases/xprof.md)
- [xprof memory viewer](2026-xprof-memory-viewer.md)
- [xprof overview page](2026-xprof-overview-page.md)
- [xprof trace viewer](2026-xprof-trace-viewer.md)
- [xprof terminology](2026-xprof-terminology.md)

## Sources

- `raw/code/xprof/docs/memory_profile.md`
