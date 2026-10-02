---
title: 'Module: jax/experimental/pallas/ops/gpu/reduce_scatter_mgpu.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/gpu/reduce_scatter_mgpu.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.gpu.reduce_scatter_mgpu`/
symbols:
  reduce_scatter: reduce_scatter().
  _run_example: _run_example().
  reduce_scatter.kernel._transfer_loop: reduce_scatter().kernel()._transfer_loop().
  reduce_scatter.kernel: reduce_scatter().kernel().
  _run_example.ref_fn: _run_example().ref_fn().
  _run_example.kernel_fn: _run_example().kernel_fn().
---
# Module: [`jax/experimental/pallas/ops/gpu/reduce_scatter_mgpu.py`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/reduce_scatter_mgpu.py)

## Functions
- `_run_example()` — [`L180`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/reduce_scatter_mgpu.py#L180)
- `_transfer_loop(loop_info: plgpu.NDLoopInfo)` — [`L148`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/reduce_scatter_mgpu.py#L148)
- `kernel(x_ref, y_ref, done_barrier)` — [`L135`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/reduce_scatter_mgpu.py#L135)
- `kernel_fn(x)` — [`L224`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/reduce_scatter_mgpu.py#L224)
- `reduce_scatter(x: jax.Array, *, axis_name, scatter_dimension: int | None = 0, reduction: Literal["add", "min", "max", "and", "or", "xor"] = "add", num_blocks: int | None = None, tile_size: int | None = None, vec_size: int | None = None)` — [`L32`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/reduce_scatter_mgpu.py#L32) — Performs a reduce-scatter or all-reduce operation across devices using multimem instructions.
- `ref_fn(x)` — [`L201`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/reduce_scatter_mgpu.py#L201)

