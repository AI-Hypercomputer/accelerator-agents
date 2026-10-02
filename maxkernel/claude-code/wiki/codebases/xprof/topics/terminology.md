---
title: "XProf terminology"
type: topic
provenance: docs
updated: 2026-07-03
status: fresh
---
# XProf terminology

The vocabulary XProf uses. Read this first — every other tool and guide assumes these terms.

## Core terms
- **Profile** — the captured performance data for a program run (memory, op durations, transfer sizes) ([xprof-terminology](src:docs/terminology.md#xprof-terminology)).
- **Session** — one capture instance; each subdirectory under `plugins/profile/` is a session ([xprof-terminology](src:docs/terminology.md#xprof-terminology)). The [README](../sources/readme.md) describes this [log-directory structure](src:README.md#log-directory-structure).
- **Run** — one training job/experiment ([xprof-terminology](src:docs/terminology.md#xprof-terminology)).
- **Host** vs **Device** — the CPU (host memory = RAM) vs the GPU/TPU accelerator (device memory = HBM) ([xprof-terminology](src:docs/terminology.md#xprof-terminology)).
- **Step** and **step time** — one training-loop iteration; step time is XProf's primary reported unit ([xprof-terminology](src:docs/terminology.md#xprof-terminology)).

## See also
- [Full terminology doc](../sources/terminology.md)
- [XProf analysis tools](analysis-tools.md) — the tools that report these quantities.

