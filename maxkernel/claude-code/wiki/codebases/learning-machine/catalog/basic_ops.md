---
title: 'Module: basic_ops.py'
type: catalog
provenance: extracted
module: basic_ops.py
status: fresh
symbol_base: scip-python python learning-machine 0.0.0 basic_ops/
symbols:
  allcases: allcases.
  P: P.
  L: L.
  _llama_ffn_shmap: _llama_ffn_shmap().
  _llama_ffn_spmd: _llama_ffn_spmd().
  main: main().
  dim: dim.
  mesh: mesh.
  BenchmarkCase: BenchmarkCase#
  BenchmarkCase.name: BenchmarkCase#name.
  BenchmarkCase.args_shape: BenchmarkCase#args_shape.
  BenchmarkCase.args_sharding: BenchmarkCase#args_sharding.
  _new_args: _new_args().
  hidden_dim: hidden_dim.
  _all_gather: _all_gather().
  _all_reduce: _all_reduce().
  start_key: start_key.
  devices: devices.
  _new_arg: _new_arg().
  _run_case: _run_case().
  _run_call_cases: _run_call_cases().
  _llama_ffn: _llama_ffn().
  BATCH: BATCH.
  BenchmarkCase.profiler_output: BenchmarkCase#profiler_output.
  multiple_of: multiple_of.
---
# Module: [`basic_ops.py`](../../../../raw/code/learning-machine/basic_ops.py)

## Classes
### `BenchmarkCase`
- def: [`basic_ops.py:29`](../../../../raw/code/learning-machine/basic_ops.py#L29) — documented in [basic_ops](../concepts/basic_ops.md)
- doc: BenchmarkCase.
- signature: `class BenchmarkCase:`
- members:
  - `args_shape` — [`L34`](../../../../raw/code/learning-machine/basic_ops.py#L34) — documented in [basic_ops](../concepts/basic_ops.md)
  - `args_sharding` — [`L35`](../../../../raw/code/learning-machine/basic_ops.py#L35) — documented in [basic_ops](../concepts/basic_ops.md)
  - `name` — [`L32`](../../../../raw/code/learning-machine/basic_ops.py#L32) — documented in [basic_ops](../concepts/basic_ops.md)
  - `profiler_output` — [`L36`](../../../../raw/code/learning-machine/basic_ops.py#L36)
- used by: [`allcases`](basic_ops.md#allcases)

## Functions
- `_all_gather(x)` — [`L128`](../../../../raw/code/learning-machine/basic_ops.py#L128) — documented in [basic_ops](../concepts/basic_ops.md)
- `_all_reduce(x)` — [`L136`](../../../../raw/code/learning-machine/basic_ops.py#L136) — documented in [basic_ops](../concepts/basic_ops.md)
- `_llama_ffn(x, w1, w2, w3)` — [`L82`](../../../../raw/code/learning-machine/basic_ops.py#L82) — documented in [basic_ops](../concepts/basic_ops.md)
- `_llama_ffn_shmap(x, w1, w2, w3)` — [`L96`](../../../../raw/code/learning-machine/basic_ops.py#L96) — documented in [basic_ops](../concepts/basic_ops.md)
- `_llama_ffn_spmd(x, w1, w2, w3)` — [`L104`](../../../../raw/code/learning-machine/basic_ops.py#L104) — documented in [basic_ops](../concepts/basic_ops.md)
- `_new_arg(shape, dtype)` — [`L42`](../../../../raw/code/learning-machine/basic_ops.py#L42)
- `_new_args(case, dtype)` — [`L52`](../../../../raw/code/learning-machine/basic_ops.py#L52)
- `_run_call_cases(cases)` — [`L196`](../../../../raw/code/learning-machine/basic_ops.py#L196)
- `_run_case(case, warmup=2, runtimes=5, dtype=jnp.bfloat16.dtype)` — [`L62`](../../../../raw/code/learning-machine/basic_ops.py#L62)
- `main()` — [`L212`](../../../../raw/code/learning-machine/basic_ops.py#L212) — documented in [basic_ops](../concepts/basic_ops.md)

## Module values
- `BATCH` — [`L117`](../../../../raw/code/learning-machine/basic_ops.py#L117) — documented in [basic_ops](../concepts/basic_ops.md)
- `L` — [`L25`](../../../../raw/code/learning-machine/basic_ops.py#L25) — documented in [basic_ops](../concepts/basic_ops.md)
- `P` — [`L19`](../../../../raw/code/learning-machine/basic_ops.py#L19) — documented in [basic_ops](../concepts/basic_ops.md)
- `allcases` — [`L140`](../../../../raw/code/learning-machine/basic_ops.py#L140) — documented in [basic_ops](../concepts/basic_ops.md)
- `devices` — [`L18`](../../../../raw/code/learning-machine/basic_ops.py#L18)
- `dim` — [`L111`](../../../../raw/code/learning-machine/basic_ops.py#L111) — documented in [basic_ops](../concepts/basic_ops.md)
- `hidden_dim` — [`L116`](../../../../raw/code/learning-machine/basic_ops.py#L116) — documented in [basic_ops](../concepts/basic_ops.md)
- `mesh` — [`L22`](../../../../raw/code/learning-machine/basic_ops.py#L22) — documented in [basic_ops](../concepts/basic_ops.md)
- `multiple_of` — [`L112`](../../../../raw/code/learning-machine/basic_ops.py#L112)
- `start_key` — [`L39`](../../../../raw/code/learning-machine/basic_ops.py#L39)

