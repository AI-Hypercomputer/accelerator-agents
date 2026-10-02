---
title: "XProf Overview Page (docs)"
type: source
tags: [docs, profiler, xprof, overview, tpu, gpu]
created: 2026-04-22
updated: 2026-04-22
---

The XProf Overview Page is the top-level, aggregated entry point into a profile — it summarizes step time, hardware utilization, and where time is spent, and acts as the launchpad for the more detailed XProf tools (Trace Viewer, Memory Viewer, Graph Viewer, etc.).

## Overview

The overview page differs between TPU and GPU and between training and inference profiles. It renders a `Performance Summary` panel plus a time-oriented chart (`Step-time Graph` for training, `Inference Session Latency Breakdown` for inference). The intent is diagnostic triage: identify whether a job is compute-bound, memory-bound, host-bound, or suffers from idle time / bad precision / eager execution before drilling into other tools.

## Key claims

- The overview page is the recommended **starting point** for XProf analysis; other tools are drill-downs from here.
- TPU and GPU overview pages expose different metrics because the execution models differ (XLA program on TPU vs. streams/kernels on GPU).
- `Program Goodput Efficiency` quantifies how close the run is to ideal performance on the given hardware, providing a single headline ratio for optimization headroom.
- On TPU, step time is decomposed into stacked categories in the `Step-time Graph` (e.g., `TensorCore idle time`, time spent `communicating with the host`), so a glance tells you whether the bottleneck is compute, host, or sync.
- On GPU, the `Step-time Breakdown` uses a finer-grained category set (kernel launch, host compute, device compute, D2D comm, collective comm, input/output/compilation, plus "all other").
- `Device Compute Precisions` tracks the % of device compute time spent in 16-bit vs. 32-bit — low-precision share is a proxy for MXU-friendly execution.
- `Op Time Spent on Eager Executions` flags cases where eager (non-graph) execution is eating time; that is actionable for PyTorch/TF users who can switch to graph mode.

## Key data points

### TPU training — Performance Summary fields

| Field | Meaning |
|---|---|
| Average Step Time | Mean step time across sampled steps |
| FLOPS Utilization | Achieved FLOPs / peak FLOPs on device |
| TPU Duty Cycle | Fraction of wall time the TPU was busy |
| Memory Bandwidth Utilization | Achieved HBM BW / peak HBM BW |
| Program Goodput Efficiency | Observed vs. ideal performance on this HW |
| TF Op Placement | Which ops are on host vs. device |
| Op Time Spent on Eager Executions | Time outside graph execution |
| Device Compute Precisions | % compute in 16-bit vs. 32-bit |

### TPU inference — deltas

- `Average Step Time` is replaced by `Average Session Time`, plus a distribution chart over sessions.
- `Step-time Graph` is replaced by `Inference Session Latency Breakdown`, which at a chosen percentile shows the split between host compute, device compute, and host-device communication.

### GPU — Step-time Breakdown categories

| Category | What it captures |
|---|---|
| All Other Time | Residual incl. Python overhead |
| Compilation Time | Kernel compilation |
| Output Time | Writing output data |
| Input Time | Reading input data |
| Kernel Launch Time | Host-side launch latency |
| Host Compute Time | Host computation |
| Device Collective Communication Time | GPU collectives |
| Device to Device Time | D2D transfers |
| Device Compute Time | On-device compute |

### TPU training Step-time Graph (stacked)

Each step's bar is composed of category-colored segments; TensorCore-idle and host-communication time are explicitly called out as categories, making the stacked plot a first-pass bottleneck detector.

## Techniques referenced

- Program goodput / MFU-style ratios as a single headline metric.
- Step-time decomposition (stacked-bar over steps) for temporal regression detection.
- Precision mix reporting as a proxy for MXU use.
- Eager vs. graph execution share as an actionable refactor signal.

## Gaps & caveats

- The overview's `Performance Summary` numbers are averages over sampled steps; step-to-step variance is only visible in the step-time graph, not in the summary fields.
- `Program Goodput Efficiency` is defined qualitatively ("relative to ideal") — the underlying model for "ideal" is not documented on this page.
- TPU and GPU summaries are not 1:1; cross-platform comparisons from overview alone are apples-to-oranges.
- `Device Compute Precisions` splits 16-bit vs. 32-bit but does not distinguish bf16 from fp16, nor tag mixed-precision patterns beyond those two buckets.
- The overview is aggregated; pinpointing which op or step drives a bad number requires Trace Viewer / Op Profile / Memory Viewer.

## Connections

Concept slugs this source informs:

- `mfu` — FLOPS utilization is essentially MFU surfaced at the overview level.
- `step-time` — definitional, and the top-line training metric for the loop.
- `program-goodput-efficiency` — XProf's headline "how close to ideal" ratio.
- `tpu-duty-cycle` — busy-fraction metric on TPU.
- `memory-bandwidth-utilization` — HBM BW fraction.
- `tensorcore-idle-time` — step-time category specific to TPU.
- `device-compute-precisions` — 16-bit vs. 32-bit time split.
- `eager-vs-graph-execution` — actionable framework-level lever.
- `inference-session-latency` — TPU inference breakdown concept.
- `gpu-step-time-breakdown` — GPU-side step decomposition categories.

## See also

- [xprof](../codebases/xprof.md)
- [xprof trace viewer](2026-xprof-trace-viewer.md)
- [xprof memory profile](2026-xprof-memory-profile.md)
- [xprof memory viewer](2026-xprof-memory-viewer.md)
- [xprof graph viewer](2026-xprof-graph-viewer.md)
- [xprof utilization viewer](2026-xprof-utilization-viewer.md)
- [xprof terminology](2026-xprof-terminology.md)

## Sources

- `raw/code/xprof/docs/overview_page.md`
