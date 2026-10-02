---
title: "Mark-Step Sync"
type: concept
tags: [stub, profiling, pytorch, compile-time]
created: 2026-04-22
updated: 2026-04-22
sources: 3
---

PyTorch/XLA lazy-execution boundary (`xm.mark_step`, `torch_xla.step`) that forces graph compilation and execution — required to get meaningful profile intervals.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [PyTorch/XLA Trace](pytorch-xla-trace.md)
- [Profile Capture](profile-capture.md)
- [Custom Trace Annotations](custom-trace-annotations.md)
- [Trace Viewer](trace-viewer.md)
- [Scan Over Layers](scan-over-layers.md)

## Sources

- [xprof PyTorch/XLA Profiling](../sources/2026-xprof-pytorch-xla-profiling.md) — `raw/code/xprof/docs/pytorch_xla_profiling.md`
- [xprof-mcp TPU optimization guide](../sources/2026-xprof-mcp-tpu-optimization.md) — `raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md`
- [JAX HuggingFace Part 1](../sources/2026-jax-huggingface-part-1.md) — torchax/JAX analogue of the PyTorch-XLA lazy boundary: first `jax.jit` call 4.365 s (compile), cached call 13 ms; `jax.block_until_ready` needed for correct timing
