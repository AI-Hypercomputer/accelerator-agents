---
title: "Framework-specific profiling"
type: topic
provenance: docs
updated: 2026-07-03
status: fresh
---
# Framework-specific profiling

XProf profiles JAX, PyTorch/XLA, and TensorFlow. Each framework has its own instrumentation API and options, but all feed the same set of XProf analysis tools.

## Per-framework
- **JAX** — programmatic/manual capture, custom trace events, and general/GPU/TPU profiler options ([configuring-profiler-options](src:docs/jax_profiling.md#configuring-profiler-options)); see [JAX profiling](../sources/jax_profiling.md).
- **JAX on GPU** — instrument → containerize → deploy on Kubernetes → capture ([step-1](src:docs/jax_gpu_profiling.md#step-1-instrumenting-the-jax-workload)); see [JAX GPU profiling](../sources/jax_gpu_profiling.md).
- **PyTorch/XLA** — start the profiler server, set trace duration, add custom labels ([1-start-the-profiler-server](src:docs/pytorch_xla_profiling.md#1-start-the-profiler-server)); see [PyTorch/XLA profiling](../sources/pytorch_xla_profiling.md).
- **TensorFlow** — see [TensorFlow profiling](../sources/tensorflow_profiling.md).

## TPU-specific options
Advanced TPU capture options (counter sampling, power/thermal, tracemark) are in [Advanced profiler options](../sources/advanced_profiler_options.md) ([periodic-counter-sampling-options](src:docs/advanced_profiler_options.md#periodic-counter-sampling-options)).

## See also
- [Capturing profiles](capturing-profiles.md)

