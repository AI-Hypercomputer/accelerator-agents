---
title: 'Module: benchmarks/bench_matmul.py'
type: catalog
provenance: extracted
module: benchmarks/bench_matmul.py
status: fresh
symbol_base: scip-python python pallas-forge 0.0.0 `benchmarks.bench_matmul`/
symbols:
  main: main().
  xla_baseline: xla_baseline().
  input_fn: input_fn().
  flops_fn: flops_fn().
  bytes_fn: bytes_fn().
  K: K.
  M: M.
  N: N.
  kernel_fn: kernel_fn().
  DTYPE: DTYPE.
  xla_baseline.xla_matmul: xla_baseline().xla_matmul().
---
# Module: [`benchmarks/bench_matmul.py`](../../../../../raw/code/pallas-forge/benchmarks/bench_matmul.py)

## Functions
- `bytes_fn(config)` — [`L39`](../../../../../raw/code/pallas-forge/benchmarks/bench_matmul.py#L39) — Total bytes accessed: read x + w, write output (bfloat16 = 2 bytes).
- `flops_fn(config)` — [`L34`](../../../../../raw/code/pallas-forge/benchmarks/bench_matmul.py#L34) — Total FLOPs for a matmul: 2*M*K*N.
- `input_fn(config)` — [`L20`](../../../../../raw/code/pallas-forge/benchmarks/bench_matmul.py#L20) — Create random inputs for benchmarking.
- `kernel_fn(x, w, *, block_m, block_k, block_n, **_)` — [`L29`](../../../../../raw/code/pallas-forge/benchmarks/bench_matmul.py#L29) — Wrapper matching the tune() calling convention.
- `main()` — [`L63`](../../../../../raw/code/pallas-forge/benchmarks/bench_matmul.py#L63) — documented in [pallas_forge-tune-config](../../concepts/pallas_forge-tune-config.md)
- `xla_baseline()` — [`L45`](../../../../../raw/code/pallas-forge/benchmarks/bench_matmul.py#L45) — Benchmark JAX's default matmul (XLA compiler) for comparison. — documented in [pallas_forge-tune-runner](../../concepts/pallas_forge-tune-runner.md)
- `xla_matmul(x, w)` — [`L49`](../../../../../raw/code/pallas-forge/benchmarks/bench_matmul.py#L49)

## Module values
- `DTYPE` — [`L17`](../../../../../raw/code/pallas-forge/benchmarks/bench_matmul.py#L17)
- `K` — [`L16`](../../../../../raw/code/pallas-forge/benchmarks/bench_matmul.py#L16)
- `M` — [`L16`](../../../../../raw/code/pallas-forge/benchmarks/bench_matmul.py#L16)
- `N` — [`L16`](../../../../../raw/code/pallas-forge/benchmarks/bench_matmul.py#L16)

