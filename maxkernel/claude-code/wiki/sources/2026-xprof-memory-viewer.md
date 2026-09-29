---
title: "XProf Memory Viewer (docs)"
type: source
tags: [docs, profiler, xprof, memory-viewer, hbm, vmem, cmem, tpu, gpu]
created: 2026-04-22
updated: 2026-04-22
---

Memory Viewer is the **static** memory-analysis tool in XProf — built entirely from XLA compiler output, it displays peak-usage attribution, buffer lifetimes, and padding overhead for a single HLO module.

## Overview

Memory Viewer plots memory allocation vs. program order (HLO sequence, not wall clock) for a chosen XLA module and memory type. At the peak-usage point it breaks down all live buffers three ways — by program order, by size, and by padding overhead — and offers per-buffer detail cards linking back to source code and Graph Viewer. Because it is static, it works without running the program; dynamic runtime data is in Memory Profile.

## Key claims

- All Memory Viewer data is **static**, pulled from the XLA compiler — no runtime profile needed.
- The x-axis of the allocation chart is **program order (HLO Sequence), not time**. This distinguishes it from Memory Profile.
- Memory Viewer is **per XLA module**; it does not know about other modules co-resident on the chip. The compiler annotates a baseline offset for each module, but later allocations by other modules are invisible.
- TPUs expose richer memory-type choices than GPU: `HBM`, `VMEM`, `SMEM`, `CMEM`, `Sync Flags (SFlag)`, `Sparsecore`, and `Host Memory`. GPU mostly exposes HBM and host memory.
- The tool reports `padding overhead` explicitly — large padding share is a direct optimization signal (shape layout / tiling).
- The three buffer-chart sortings (program order, size, padding overhead) are orthogonal lenses on the same peak-instant buffer set.
- The `timeline` link transforms buffer allocations into a 2-D view where x=program order, y=buffer-size, width=buffer lifetime — a classic buffer-lifetime plot.
- Buffer detail cards expose `Allocation type` classification: `Parameter`, `Output`, `Thread-local`, `Temporary` (e.g., intra-fusion).

## Key data points

### Memory types exposed (TPU)

| Memory | Role |
|---|---|
| HBM | Main device memory |
| VMEM | Vector memory close to TensorCore |
| SMEM | Scalar memory |
| CMEM | Instruction/constant memory |
| SFlag (Sync Flags) | Synchronization primitives |
| Sparsecore | SparseCore-attached memory |
| Host Memory | System RAM visible to TPU |

GPU exposes HBM and Host Memory.

### Textual overview panel — fields

- Peak memory allocation required for the program.
- Split between arguments and temporary variables.
- Padding overhead (fraction of total allocation due to shape restrictions).

### Main line chart

- **X-axis**: HLO Sequence program order (NOT time).
- **Y-axis**: total allocated memory of the chosen module in the chosen memory type.
- **Peak marker**: vertical line at the program point with peak memory utilization for the selected module.
- **Baseline**: compiler-annotated offset representing allocations by prior modules (does not track later modules' dynamics).

### Buffer charts at peak (three views)

| Sort | Purpose |
|---|---|
| Program order | Oldest-first view — reveals long-lived buffers |
| Size | Largest-first — top contributors to peak |
| Padding overhead | Most hardware-inefficient buffers — tiling/layout targets |

### Timeline visualization

- X = program order; Y = buffer size; box width = allocation lifetime.
- Each box = one allocation; hover yields HLO op, shape, etc.

### Buffer detail card — fields

| Field | Content |
|---|---|
| Name | XLA op name (searchable in Graph Viewer / Trace Viewer) |
| Size | Allocation size with and without padding |
| Shape | Rank, dimensions, dtype |
| Framework op name | Associated framework-level op |
| Allocation type | Parameter / Output / Thread-local / Temporary |
| Source | File and line of the op |
| Source Stack | Full call stack leading to the allocation |

## Techniques referenced

- Static compile-time memory analysis (no execution required).
- Peak-instant buffer breakdowns sorted by distinct criteria (size, lifetime, padding).
- Buffer-lifetime 2-D plot (program order × size × lifetime).
- Padding overhead as a first-class optimization target (shape/tiling).
- Allocation-type taxonomy (parameter / output / thread-local / temporary-in-fusion).
- Cross-linking from buffer → XLA op → Graph Viewer / Trace Viewer.

## Gaps & caveats

- Co-residence across modules is **not modeled**: peaks reported are per-module, so multi-module workloads can OOM at a point that looks safe in Memory Viewer. Use Memory Profile for the cross-module picture.
- X-axis is program order, not time — do not visually compare against a wall-clock trace.
- Static-only: nothing about actual allocator behavior, fragmentation, or runtime peaks. Pair with Memory Profile for dynamic truth.
- Source-code / framework-op metadata is only shown when the compiler surfaces it through the transformation chain.
- Padding overhead is reported but the tool does not directly suggest a better layout.
- On-chip-memory categories (VMEM/SMEM/CMEM) are TPU-specific; patterns learned on TPU don't port directly to GPU memory analysis.

## Connections

Concept slugs this source informs:

- `memory-hierarchy` — full TPU memory stack (HBM/VMEM/SMEM/CMEM/SFlag/Sparsecore/host).
- `vmem` — on-chip vector memory.
- `smem` — on-chip scalar memory.
- `cmem` — on-chip instruction/constant memory.
- `hbm` — main device memory.
- `padding-overhead` — shape-restriction-driven waste.
- `buffer-lifetime` — program-order × size × lifetime plot.
- `allocation-type` — parameter / output / thread-local / temporary taxonomy.
- `hlo-sequence` — program-order axis used across XLA tools.
- `peak-memory-usage` — per-module static peak concept.
- `xla-module` — the unit that Memory Viewer scopes to.

## See also

- [xprof](../codebases/xprof.md)
- [xprof memory profile](2026-xprof-memory-profile.md)
- [xprof graph viewer](2026-xprof-graph-viewer.md)
- [xprof trace viewer](2026-xprof-trace-viewer.md)
- [xprof overview page](2026-xprof-overview-page.md)
- [xprof terminology](2026-xprof-terminology.md)

## Sources

- `raw/code/xprof/docs/memory_viewer.md`
