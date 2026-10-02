---
title: 'Module: benchmarks/bench_rmsnorm.py'
type: catalog
provenance: extracted
module: benchmarks/bench_rmsnorm.py
status: fresh
symbol_base: scip-python python pallas-forge 0.0.0 `benchmarks.bench_rmsnorm`/
symbols:
  main: main().
  xla_baseline: xla_baseline().
  input_fn: input_fn().
  bytes_fn: bytes_fn().
  DIM: DIM.
  DTYPE: DTYPE.
  BATCH: BATCH.
  SEQ_LEN: SEQ_LEN.
  kernel_fn: kernel_fn().
  xla_baseline.xla_rmsnorm_residual: xla_baseline().xla_rmsnorm_residual().
---
# Module: [`benchmarks/bench_rmsnorm.py`](../../../../../raw/code/pallas-forge/benchmarks/bench_rmsnorm.py)

## Functions
- `bytes_fn(config)` — [`L34`](../../../../../raw/code/pallas-forge/benchmarks/bench_rmsnorm.py#L34) — Bytes accessed: read x + residual + weight, write output + new_residual.
- `input_fn(config)` — [`L20`](../../../../../raw/code/pallas-forge/benchmarks/bench_rmsnorm.py#L20)
- `kernel_fn(x, residual, weight, *, block_size, **_)` — [`L29`](../../../../../raw/code/pallas-forge/benchmarks/bench_rmsnorm.py#L29)
- `main()` — [`L65`](../../../../../raw/code/pallas-forge/benchmarks/bench_rmsnorm.py#L65) — documented in [pallas_forge-tune-config](../../concepts/pallas_forge-tune-config.md)
- `xla_baseline()` — [`L43`](../../../../../raw/code/pallas-forge/benchmarks/bench_rmsnorm.py#L43) — Benchmark unfused JAX ops for comparison. — documented in [pallas_forge-tune-runner](../../concepts/pallas_forge-tune-runner.md)
- `xla_rmsnorm_residual(x, residual, weight)` — [`L47`](../../../../../raw/code/pallas-forge/benchmarks/bench_rmsnorm.py#L47)

## Module values
- `BATCH` — [`L16`](../../../../../raw/code/pallas-forge/benchmarks/bench_rmsnorm.py#L16)
- `DIM` — [`L16`](../../../../../raw/code/pallas-forge/benchmarks/bench_rmsnorm.py#L16)
- `DTYPE` — [`L17`](../../../../../raw/code/pallas-forge/benchmarks/bench_rmsnorm.py#L17)
- `SEQ_LEN` — [`L16`](../../../../../raw/code/pallas-forge/benchmarks/bench_rmsnorm.py#L16)

