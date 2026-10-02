---
title: "XProf terminology"
type: source
provenance: docs
source_doc: docs/terminology.md
updated: 2026-07-03
status: fresh
---
# XProf terminology

Defines the core vocabulary XProf uses to talk about a captured profile: profile, session, run, host, device, and step. Step time is XProf's fundamental unit of reported performance.

## Key points
- A **profile** is the captured performance data (memory, op durations, transfer sizes); a **session** is one capture instance (a subdirectory under `plugins/profile/`); a **run** is one training job/experiment ([xprof-terminology](src:docs/terminology.md#xprof-terminology))
- **Host** = the system CPU (host memory = RAM); **Device** = the GPU/TPU accelerator (device memory = HBM); a **step** is one training-loop iteration and **step time** is the unit XProf reports ([xprof-terminology](src:docs/terminology.md#xprof-terminology))

## Feeds topics
- [XProf terminology](../topics/terminology.md)
- [XProf analysis tools](../topics/analysis-tools.md)

## Sources
- raw/code/xprof/docs/terminology.md
