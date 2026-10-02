---
title: "Source: profiling.md"
type: source-page
doc: profiling.md
updated: 2026-07-03
---
# profiling.md — Profiling

Part 9. Walks the TPU software stack from JAX code through StableHLO, XLA-compiled HLO, and LLO down
to TPU machine code
([a-thousand-foot-view-of-the-tpu-software-stack](src:profiling.md#a-thousand-foot-view-of-the-tpu-software-stack)).
Introduces `jax.profiler.trace` and the TensorBoard/xprof viewer's three key tabs: Trace Viewer
(chronological per-core timeline of XLA ops), Graph Viewer (HLO dataflow graph), and Memory
Profile/Viewer
([the-jax-profiler-a-multi-purpose-tpu-profiler](src:profiling.md#the-jax-profiler-a-multi-purpose-tpu-profiler)).
Teaches how to read an HLO op's shape/layout/memory-space/argument annotations (e.g. `S(1)` = VMEM,
`T(8,128)(2,1)` = two-level tiling) directly from the trace
([how-to-read-an-xla-op](src:profiling.md#how-to-read-an-xla-op)), and works a real profile example
where a sharded FFW matmul's measured 96ms matches its roofline-predicted 95.6ms almost exactly
([looking-at-a-realish-example-profile](src:profiling.md#looking-at-a-realish-example-profile)).

## Feeds
- [Profiling methodology and reading XLA/HLO](../topics/profiling-methodology-and-hlo.md) — the
  core topic this chapter defines.
