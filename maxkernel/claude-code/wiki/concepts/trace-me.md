---
title: "TraceMe"
type: concept
tags: [stub, profiling]
created: 2026-04-22
updated: 2026-04-22
sources: 5
---

Lower-level annotation primitive in the XProf tracing runtime that emits a named timeline event; all framework-specific trace wrappers (`jax.profiler.TraceAnnotation`, `torch_xla.debug.profiler.Trace` / `xp.Trace`, `tf.profiler.experimental.Trace`) ultimately lower to `TraceMe`. Appears as its own track in the Trace Viewer.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [Custom Trace Annotations](custom-trace-annotations.md)
- [Trace Viewer](trace-viewer.md)
- [Trace Event Categories](trace-event-categories.md)

## Sources

- [xprof Trace Viewer](../sources/2026-xprof-trace-viewer.md) — `raw/code/xprof/docs/trace_viewer.md`
- [xprof Capturing Profiles](../sources/2026-xprof-capturing-profiles.md) — `raw/code/xprof/docs/capturing_profiles.md`
- [xprof JAX Profiling](../sources/2026-xprof-jax-profiling.md) — `raw/code/xprof/docs/jax_profiling.md`
- [xprof PyTorch/XLA Profiling](../sources/2026-xprof-pytorch-xla-profiling.md) — `raw/code/xprof/docs/pytorch_xla_profiling.md`
- [xprof TensorFlow Profiling](../sources/2026-xprof-tensorflow-profiling.md) — `raw/code/xprof/docs/tensorflow_profiling.md`
