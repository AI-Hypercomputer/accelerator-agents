---
title: "Memory analysis"
type: topic
provenance: docs
updated: 2026-07-03
status: fresh
---
# Memory analysis

XProf's memory tools show device (HBM) usage over time and per buffer — used to find peak memory, fragmentation, and leaks.

## Tools
- **Memory Profile** — the device-memory timeline over a step ([using-the-memory-profile-tool](src:docs/memory_profile.md#using-the-memory-profile-tool)); see [Memory profile](../sources/memory_profile.md).
- **Memory Viewer** — per-buffer allocation at peak, with buffer charts ([how-to-interpret-the-memory-viewer-buffer-charts](src:docs/memory_viewer.md#how-to-interpret-the-memory-viewer-buffer-charts)); see [Memory viewer](../sources/memory_viewer.md).
- **Finding leaks** — track tensor/buffer lifetimes with both tools ([overview](src:docs/memory_leaks.md#overview), [analyze-memory-timeline-with-memory-profile](src:docs/memory_leaks.md#analyze-memory-timeline-with-memory-profile)); see [Memory leaks](../sources/memory_leaks.md).

## See also
- [XProf analysis tools](analysis-tools.md), [Terminology](terminology.md) (host vs device memory)

