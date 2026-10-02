---
title: "XProf Trace Viewer (docs)"
type: source
tags: [docs, profiler, xprof, trace-viewer, tpu, gpu]
created: 2026-04-22
updated: 2026-04-22
---

The XProf Trace Viewer is a timeline UI for events captured during profiling — host threads, accelerator cores, streams, XLA ops, framework ops, source annotations, and host offload — used to identify bottlenecks, gaps, and launch latencies.

## Overview

Trace Viewer organizes events into `sections` (one per processing element — TPU node, SparseCore node, GPU node, host component) containing `tracks` (per-core/per-stream/per-thread timelines). Events are colored rectangles whose color has no semantic meaning. A details pane appears when events are selected and, for XLA ops, links back to Graph Viewer plus statically-known FLOPS and bytes-accessed data.

## Key claims

- The timeline streams data on demand (like a map UI) to scale to millions of events; very large profiles should use **Trace Viewer v2**, a WebGPU/Canvas renderer enabled via the "Switch to V2" button.
- Only `XLA Ops` for TPU and `stream data` for GPU are directly grounded in the collected profile. All other tracks (Steps, Framework Ops, Name Scope, Source code, TraceMes, etc.) are **derived** from optional sideband info, user annotations, or XProf heuristics — they may not appear in every profile.
- On GPU the `XLA Ops` track is derived from streams and is **not always accurate**, because an XLA op can map N:M to kernels across dynamically scheduled streams/SMs.
- `Flow Events` draw arrows between related events across threads/lines (e.g., host launch → device execution) using CUPTI IDs, TPU runtime info, and heuristics.
- The FLOPS and bytes-accessed fields in the details pane are **compile-time static cost analysis**, not runtime measurements.
- The "Find events" search bar only searches within the currently visible time window, not the full trace.

## Key data points

### TPU section — tracks

| Track | What it shows |
|---|---|
| Steps | Training-step durations, if annotated |
| XLA Modules | XLA program being executed |
| XLA Ops | HLO ops that ran on the TPU core (ground-truth track) |
| XLA TraceMe | User/XLA-inserted annotations for logical units |
| Framework Ops | JAX/TF/PyTorch ops annotated onto the timeline |
| Framework Name Scope | Visualized stack trace per framework op (one device only) |
| Source code | Path to source being executed, if available |
| Scalar unit | Events on the TPU scalar unit |
| TensorCore Sync Flags | Synchronization primitives |
| Host Offload | Async host↔accelerator transfers; start/stop markers on XLA Ops |
| LLO Utilization | HW utilization for XLA Custom Calls when flags enabled |

### GPU section — tracks

- One track per CUDA stream (name includes type: Memcpy, Compute, ...).
- `Launch Stats`: max and average launch-phase time.
- `Steps`, `XLA Modules`, `Framework Ops`, `Framework Name Scope`, `Source code` — similar to TPU.
- `XLA TraceMe` is **not supported** for GPUs.

### SparseCore section

- Present on v5p and Trillium (v6e); contains modules, ops, and TraceMes associated with SparseCore units (distinct from dense MXU).

### Keyboard / tool shortcuts

| Key | Action |
|---|---|
| W/S | Zoom in/out |
| A/D | Pan left/right |
| 1 / ! | Selection tool |
| 2 / @ | Pan tool |
| 3 / # | Zoom tool |
| 4 / $ | Timing tool |
| m | Mark selection and report total duration |
| f | Zoom to selected events |
| ctrl+click | Multi-select for summary |

## Techniques referenced

- Streaming tile-based timeline rendering (load-on-demand, WebGPU v2 path).
- Flow-event correlation (host-launch to device-execute) via CUPTI / TPU runtime IDs.
- Derived tracks from optional compiler sideband — tracks are opt-in per profile.
- Static cost analysis (FLOPS / bytes) vs. measured runtime — the distinction matters for Op Profile and Graph Viewer as well.
- Host-offload annotations to reason about async HBM↔host transfers.
- XLA TraceMe as a user-level annotation mechanism for custom spans.

## Gaps & caveats

- GPU `XLA Ops` track is advisory, not authoritative — do not use it to attribute time on GPU.
- Derived tracks may silently be absent; their absence does not mean the underlying activity didn't happen.
- `Framework Name Scope` is rendered for a single device only (space-saving), so multi-device stack traces aren't directly comparable in one view.
- `Find events` scoped to visible window can mislead users searching a long trace.
- Compile-time FLOPS and bytes don't reflect runtime spills, rematerialization, or skipped fusions.
- Event color carries no information — readers who assume color-coded categories will be wrong.

## Connections

Concept slugs this source informs:

- `trace-event-categories` — catalog of tracks (Steps, XLA Ops, Framework Ops, TraceMe, Host Offload, SparseCore, Launch Stats).
- `xla-traceme` — annotation mechanism for logical spans.
- `host-offload` — async host↔accelerator transfer pattern.
- `sparsecore` — SparseCore sections on v5p/Trillium.
- `flow-events` — host-launch → device-execute correlation.
- `kernel-launch-latency` — surfaced through GPU Launch Stats.
- `pipeline-bubble` — gaps in the timeline reveal bubbles.
- `tpu-scalar-unit` — scalar-unit events on TPU timelines.
- `tensorcore-sync-flags` — TPU sync primitives on the trace.
- `xla-custom-call` — LLO Utilization track surfaces these.
- `trace-viewer-v2` — WebGPU renderer for large traces.

## See also

- [xprof](../codebases/xprof.md)
- [xprof overview page](2026-xprof-overview-page.md)
- [xprof graph viewer](2026-xprof-graph-viewer.md)
- [xprof memory profile](2026-xprof-memory-profile.md)
- [xprof utilization viewer](2026-xprof-utilization-viewer.md)
- [xprof terminology](2026-xprof-terminology.md)

## Sources

- `raw/code/xprof/docs/trace_viewer.md`
