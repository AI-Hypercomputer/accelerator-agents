---
title: "Source: microbenchmarks/README.md"
type: source-page
doc: microbenchmarks/README.md
updated: 2026-07-03
---
# microbenchmarks/README.md

The single-chip microbenchmark hub (v6e-1 TPU VM), covering two scripts:
`benchmark_matmul.py` and `benchmark_hbm.py`
([setup](src:microbenchmarks/README.md#setup)). The matmul benchmark's key methodological point:
`--trace_matcher` controls whether timing is measured via the compiled function's trace duration
(TPU-only time, excluding host↔TPU communication) or via host-side wall-clock timing (which
additionally includes compilation and kernel-launch/data-transfer/sync overhead) — the same
8192×8192×8192 bf16 matmul reports 827 TFLOPS trace-matched vs. 754 TFLOPS host-timed
([run-matmul-benchmark](src:microbenchmarks/README.md#run-matmul-benchmark)). The HBM benchmark
measures bandwidth for a configurable tensor copy size
([hbm-bandwidth-benchmark](src:microbenchmarks/README.md#hbm-bandwidth-benchmark)).

## Feeds
- [TPU microbenchmarks](../topics/tpu-microbenchmarks.md) — the single-chip baseline this topic's
  multi-chip (Ironwood/Trillium) benchmarks build on.
