---
title: "Capturing profiles"
type: topic
provenance: docs
updated: 2026-07-03
status: fresh
---
# Capturing profiles

How to get a profile out of a running workload. XProf supports several capture modes; the right one depends on whether you can instrument the code and whether the run is short or long-lived.

## Capture modes
- **Programmatic capture** — instrument the workload to profile a code region ([programmatic-capture](src:docs/capturing_profiles.md#programmatic-capture)); per-framework in [JAX](src:docs/jax_profiling.md#programmatic-capture) and [PyTorch/XLA](src:docs/pytorch_xla_profiling.md#capture-a-trace-programmatically).
- **On-demand / manual capture** — trigger from the XProf UI against a running profiler server ([on-demand-capture](src:docs/capturing_profiles.md#on-demand-capture-aka-manual-capture)).
- **Continuous snapshots** — periodic captures during a long run ([continuous-profiling-snapshots](src:docs/capturing_profiles.md#continuous-profiling-snapshots)).
- **Remote / cloud** — profiling a remote machine or Google Cloud ([profiling-on-a-remote-machine](src:docs/capturing_profiles.md#profiling-on-a-remote-machine)); multiple sessions per run ([multiple-sessions-per-run](src:docs/capturing_profiles.md#multiple-sessions-per-run)).

## Framework specifics
See [framework-specific instructions](src:docs/capturing_profiles.md#framework-specific-instructions) and the [Framework-specific profiling](framework-profiling.md) topic.

## See also
- [Capturing profiles doc](../sources/capturing_profiles.md), [Advanced profiler options](../sources/advanced_profiler_options.md)

