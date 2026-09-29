---
title: 'Module: tokamax/_src/ops/ragged_dot/bench.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/bench.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.bench`/
symbols:
  _IMPLS: _IMPLS.
  _register_benchmarks: _register_benchmarks().
  _register_benchmark: _register_benchmark.
  ARG_SPECS: ARG_SPECS.
  _BENCHMARK_IMPLS: _BENCHMARK_IMPLS.
  _xla_only_group0: _xla_only_group0().
  _xla_even_groups: _xla_even_groups().
  _flops: _flops().
  _transpose_rhs: _transpose_rhs().
---
# Module: [`tokamax/_src/ops/ragged_dot/bench.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/bench.py)

## Functions
- `_flops(lhs, rhs, *, group_sizes)` — [`L52`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/bench.py#L52) — Returns the floating-point operations.
- `_register_benchmarks()` — [`L84`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/bench.py#L84) — Registers benchmarks.
- `_transpose_rhs(x: jax.ShapeDtypeStruct)` — [`L77`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/bench.py#L77)
- `_xla_even_groups(lhs, rhs, *, group_sizes)` — [`L43`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/bench.py#L43) — XLA baseline that ignores group sizes and splits evenly between groups.
- `_xla_only_group0(lhs, rhs, *, group_sizes)` — [`L35`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/bench.py#L35) — XLA baseline that ignores group sizes and only uses group 0 weights.

## Module values
- `ARG_SPECS` — [`L31`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/bench.py#L31)
- `_BENCHMARK_IMPLS` — [`L66`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/bench.py#L66)
- `_IMPLS` — [`L59`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/bench.py#L59)
- `_register_benchmark` — [`L69`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/bench.py#L69)

