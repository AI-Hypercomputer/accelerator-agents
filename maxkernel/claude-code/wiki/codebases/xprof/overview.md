---
title: "xprof — overview"
type: overview
updated: 2026-07-03
---
# xprof — what it is and how it fits together

## In one paragraph
XProf is the OpenXLA / TensorBoard ML performance profiler and visualizer: you capture a **profile** of a running JAX / PyTorch-XLA / TensorFlow workload, then explore it through a suite of analysis tools (overview dashboard, trace viewer, framework- and HLO-op stats, graph/memory/roofline viewers, and multi-slice collective tools). Its reporting is organized around a handful of core terms — profile, session, run, host, device, and **step time**, the primary performance unit ([terminology](../codebases/xprof/topics/terminology.md)). This docs wiki is organized as **topics** (cross-cutting concepts) backed by per-document **source summaries**.

## Main topics
- **[Terminology](topics/terminology.md)** — profile/session/run/host/device/step; read first.
- **[Capturing profiles](topics/capturing-profiles.md)** — programmatic, on-demand/manual, continuous, remote, and cloud capture modes.
- **[Framework-specific profiling](topics/framework-profiling.md)** — JAX (incl. GPU), PyTorch/XLA, TensorFlow instrumentation and options.
- **[XProf analysis tools](topics/analysis-tools.md)** — overview page, trace viewer, framework/HLO op stats, HLO op profile, graph viewer, GPU kernel stats.
- **[Memory analysis](topics/memory-analysis.md)** — memory profile, memory viewer, and leak tracking.
- **[Roofline & performance counters](topics/roofline-and-counters.md)** — roofline model, utilization viewer, perf counters, kernel profiling, custom-call and DCN/PCIe profiling.
- **[Multi-slice collectives (Megascale)](topics/multislice-collectives.md)** — Megascale stats/viewer and PerfettoSQL queries for multi-host collective comms.
- **[Deploying XProf](topics/deployment.md)** — standalone/TensorBoard, Docker, and Kubernetes (distributed).

## Which topic answers which question
- "What does *step time* / *session* mean?" → [Terminology](topics/terminology.md).
- "How do I get a profile out of my JAX/PyTorch/TF job?" → [Capturing profiles](topics/capturing-profiles.md) + [Framework-specific profiling](topics/framework-profiling.md).
- "Where did device time go / which ops are expensive?" → [Analysis tools](topics/analysis-tools.md) (HLO op stats/profile, framework op stats, trace viewer).
- "Am I compute- or memory-bound?" → [Roofline & performance counters](topics/roofline-and-counters.md).
- "Why is my HBM full / where's the leak?" → [Memory analysis](topics/memory-analysis.md).
- "Why are my multi-slice collectives slow?" → [Multi-slice collectives](topics/multislice-collectives.md).
- "How do I run XProf at scale?" → [Deploying XProf](topics/deployment.md).

## Map of the wiki
- Per-document summaries live under `sources/`; cross-cutting synthesis under `topics/`.
- Full concept table: `index.md`.

## Sources
- raw/code/xprof/docs (commit 145d0d1f14)
- raw/code/xprof/README.md
