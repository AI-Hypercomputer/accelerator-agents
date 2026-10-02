---
title: 'Module: axlearn/common/flash_attention/gpu_attention_benchmark.py'
type: catalog
provenance: extracted
module: axlearn/common/flash_attention/gpu_attention_benchmark.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.flash_attention.gpu_attention_benchmark`/
symbols:
  bench_flash_attention: bench_flash_attention().
  benchmark_sweep: benchmark_sweep().
  measure: measure().
  _sweep: _sweep().
  main: main().
  BenchFn.__call__: BenchFn#__call__().
  BenchFnResult: BenchFnResult.
  bench_flash_attention_fwd_bwd: bench_flash_attention_fwd_bwd().
  benchmark_decode: benchmark_decode().
  SweepFn.__call__: SweepFn#__call__().
  measure.run: measure().run().
  X: X.
  Y: Y.
  BenchFn: BenchFn#
  SweepFn: SweepFn#
  _sweep.check_fn: _sweep().check_fn().
---
# Module: [`axlearn/common/flash_attention/gpu_attention_benchmark.py`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_benchmark.py)

## Classes
### `BenchFn`  ·  implements/extends Protocol
- def: [`axlearn/common/flash_attention/gpu_attention_benchmark.py:155`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_benchmark.py#L155)
- signature: `class BenchFn(Protocol):`
- protocol/private: `__call__`[`L156`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_benchmark.py#L156)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor), [`BenchFnResult`](gpu_attention_benchmark.md#BenchFnResult)
- used by: [`measure`](gpu_attention_benchmark.md#measure)

### `SweepFn`  ·  implements/extends Protocol
- def: [`axlearn/common/flash_attention/gpu_attention_benchmark.py:159`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_benchmark.py#L159)
- signature: `class SweepFn(Protocol):`
- protocol/private: `__call__`[`L160`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_benchmark.py#L160)
- uses (calls/refs, reference-scoped): [`BenchFnResult`](gpu_attention_benchmark.md#BenchFnResult)
- used by: [`_sweep`](gpu_attention_benchmark.md#_sweep)

## Functions
- `_sweep(fn: SweepFn, libraries: list[str], common_kwargs: dict[str, Any], **_sweep_kwargs: list[Any])` — [`L268`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_benchmark.py#L268) — Benchmarks `fn` by sweeping through combinations of parameters.
- `bench_flash_attention(library: str, bs: int, num_heads: int, num_kv_heads: Optional[int], seq_len: int, per_head_dim: int, is_decode: bool, use_bwd: bool, sw_sz: int = -1, dtype=jnp.float16)` — [`L207`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_benchmark.py#L207)
- `bench_flash_attention_fwd_bwd(use_bwd: bool)` — [`L341`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_benchmark.py#L341)
- `benchmark_decode()` — [`L320`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_benchmark.py#L320)
- `benchmark_sweep(libraries: list[str], common_kwargs: dict[str, Any], **sweep_args: list[Any])` — [`L316`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_benchmark.py#L316)
- `check_fn(result, ref_result)` — [`L285`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_benchmark.py#L285)
- `main()` — [`L370`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_benchmark.py#L370) — Main function to run benchmarks.
- `measure(f: BenchFn, *args: Tensor)` — [`L163`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_benchmark.py#L163) — Measures the time it takes to execute the function on the GPU.
- `run(*args)` — [`L190`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_benchmark.py#L190)

## Module values
- `BenchFnResult` — [`L152`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_benchmark.py#L152)
- `X` — [`L149`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_benchmark.py#L149)
- `Y` — [`L150`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/gpu_attention_benchmark.py#L150)

