---
title: 'Module: benchmarks/bench_swiglu.py'
type: catalog
provenance: extracted
module: benchmarks/bench_swiglu.py
status: fresh
symbol_base: scip-python python pallas-forge 0.0.0 `benchmarks.bench_swiglu`/
symbols:
  main: main().
  xla_baseline: xla_baseline().
  input_fn: input_fn().
  DIM: DIM.
  FFN_DIM: FFN_DIM.
  flops_fn: flops_fn().
  bytes_fn: bytes_fn().
  BATCH_SEQ: BATCH_SEQ.
  DTYPE: DTYPE.
  kernel_fn: kernel_fn().
  xla_baseline.xla_swiglu: xla_baseline().xla_swiglu().
---
# Module: [`benchmarks/bench_swiglu.py`](../../../../../raw/code/pallas-forge/benchmarks/bench_swiglu.py)

## Functions
- `bytes_fn(config)` — [`L43`](../../../../../raw/code/pallas-forge/benchmarks/bench_swiglu.py#L43) — Bytes: read x + w_gate + w_up, write output.
- `flops_fn(config)` — [`L35`](../../../../../raw/code/pallas-forge/benchmarks/bench_swiglu.py#L35) — FLOPs: two matmuls (2*M*K*N each) + activation + elementwise mul.
- `input_fn(config)` — [`L22`](../../../../../raw/code/pallas-forge/benchmarks/bench_swiglu.py#L22)
- `kernel_fn(x, w_gate, w_up, *, block_m, block_n, **_)` — [`L31`](../../../../../raw/code/pallas-forge/benchmarks/bench_swiglu.py#L31)
- `main()` — [`L71`](../../../../../raw/code/pallas-forge/benchmarks/bench_swiglu.py#L71) — documented in [pallas_forge-tune-config](../../concepts/pallas_forge-tune-config.md)
- `xla_baseline()` — [`L51`](../../../../../raw/code/pallas-forge/benchmarks/bench_swiglu.py#L51) — Benchmark unfused JAX ops for comparison. — documented in [pallas_forge-tune-runner](../../concepts/pallas_forge-tune-runner.md)
- `xla_swiglu(x, w_gate, w_up)` — [`L55`](../../../../../raw/code/pallas-forge/benchmarks/bench_swiglu.py#L55)

## Module values
- `BATCH_SEQ` — [`L16`](../../../../../raw/code/pallas-forge/benchmarks/bench_swiglu.py#L16)
- `DIM` — [`L17`](../../../../../raw/code/pallas-forge/benchmarks/bench_swiglu.py#L17)
- `DTYPE` — [`L19`](../../../../../raw/code/pallas-forge/benchmarks/bench_swiglu.py#L19)
- `FFN_DIM` — [`L18`](../../../../../raw/code/pallas-forge/benchmarks/bench_swiglu.py#L18)

