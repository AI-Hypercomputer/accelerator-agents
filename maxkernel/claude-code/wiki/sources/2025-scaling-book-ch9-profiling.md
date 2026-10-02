---
title: "How to Scale Your Model — Ch 9: How to Profile TPU Programs"
type: source
tags: [docs, book, scaling-book, profiling, xprof, hlo, xla, trace-viewer, graph-viewer, memory-profile]
author: DeepMind / JAX ML team
book: "How to Scale Your Model"
book_date: "2025-02-04"
chapter: 9
upstream: https://jax-ml.github.io/scaling-book/profiling
created: 2026-04-23
updated: 2026-04-23
---

Chapter 9 of the scaling-book. **How to read a TPU profile** — the XLA / HLO / LLO lowering chain, HLO-op notation (shape × layout × tiling × memory-space), and the three main xprof viewers. Worked example verifying an FFW matmul matches its roofline within 0.4%.

## Key claims

1. Software stack: JAX (NumPy-style) → **StableHLO** (platform-agnostic) → **HLO** (XLA IR) → **LLO** (TPU-specific scheduling) → machine code.
2. HLO notation encodes four things in one line: `op_name[shape{layout:tiling}@memory_location] = op(args)`.
3. Logical `[a, b, c]` maps to physical memory through `{p1, p0, p2}` permutation + `T(tile1, tile2)...` tiling + padding.
4. Memory-space sigil: **`S(0)` = HBM**, **`S(1)` = VMEM** (fast, small). Retiling (layout mismatches causing copies) shows up as separate HLO ops.
5. **Trace Viewer**: XLA-op timeline on top, named_scopes below; WASD navigation; click an op to see source code + jump to Graph Viewer.
6. **Graph Viewer**: fusions shown as HLO op graphs; helps interpret compound ops like `all-reduce-scatter`.
7. **Memory Profile**: HBM usage over time; diagnoses OOMs and peak-step allocations.
8. **Worked matmul example** (8 × 1024 × 8192 × 32768 bf16 on 8 v2 cores): predicted 95.6 ms, measured 96 ms → **perfect overlap, 0.4% error**. AllReduce 4×2 (128 MB): predicted 1.1 ms, measured 1.13 ms.

## Key data points

### HLO-op example (decoded verbatim)

```
bf16[32, 32, 4096]{2, 1, 0 : T(8, 128)(2, 1) S(1)}
```

- Shape: `32 × 32 × 4096` of bf16.
- Layout `{2, 1, 0}`: physical memory order = axis 2 innermost, axis 0 outermost.
- Tiling `T(8, 128)(2, 1)`: outer tile `(8, 128)`, inner `(2, 1)`.
- Memory: `S(1)` = VMEM.

### Latency verification worked example

- Matmul `8 × 1024 × 8192 × 32768` bf16 on 8 v2 cores: predicted 95.6 ms, measured 96 ms.
- AllReduce 4×2 (128 MB): predicted 1.1 ms, measured 1.13 ms.

## Techniques referenced

- `jax.jit` + `jax.profiler.trace` (see [concepts/jax-trace](../concepts/jax-trace.md)).
- TensorBoard / xprof UI.
- Perfetto embedded trace viewing (see [concepts/perfetto](../concepts/perfetto.md)).
- HLO fusion (`kCustom` kind).
- Layout inference via `jax.numpy.layout` (see [codebases/jax](../codebases/jax.md) `jax.experimental.layout`).
- Manual roofline-vs-measured checks.

## Gaps & caveats

- Assumes access to TensorBoard UI (Colab / local); distributed / cloud profiling limited (see [xprof Kubernetes deployment](2026-xprof-kubernetes-deployment.md)).
- HLO notation drifts with XLA version; backward compat not guaranteed.
- Trace Viewer doesn't show memory-access patterns or cache misses.
- **Pallas / custom-kernel profiling only lightly covered** — see [xprof custom-call-profiling](2026-xprof-custom-call-profiling.md).
- GPU equivalents (nvprof / nsys) mentioned briefly.

## Connections

- [concepts/profile-capture](../concepts/profile-capture.md), [jax-trace](../concepts/jax-trace.md), [trace-viewer](../concepts/trace-viewer.md), [trace-event-categories](../concepts/trace-event-categories.md).
- [concepts/hlo](../concepts/hlo.md), [hlo-op](../concepts/hlo-op.md), [hlo-dumping-and-diffing](../concepts/hlo-dumping-and-diffing.md).
- [codebases/xprof](../codebases/xprof.md), [xprof-mcp](../codebases/xprof-mcp.md).
- All xprof-documentation source pages (`2026-xprof-*.md`) are applied versions of this chapter's material.

## See also

- [Ch 10 — JAX Programming](2025-scaling-book-ch10-jax.md)
- [xprof-mcp TPU optimization guide](2026-xprof-mcp-tpu-optimization.md) — crown-jewel practical companion.

## Sources

- `raw/code/scaling-book/profiling.md`
- Upstream: <https://jax-ml.github.io/scaling-book/profiling>
