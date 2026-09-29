---
title: 'Module: jax/experimental/pallas/ops/gpu/all_gather_mgpu.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/gpu/all_gather_mgpu.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.gpu.all_gather_mgpu`/
symbols:
  all_gather: all_gather().
  _run_example: _run_example().
  all_gather.kernel._transfer_loop: all_gather().kernel()._transfer_loop().
  all_gather.kernel: all_gather().kernel().
  _run_example.ref_fn: _run_example().ref_fn().
  _run_example.kernel_fn: _run_example().kernel_fn().
  all_gather.kernel._never: all_gather().kernel()._never().
---
# Module: [`jax/experimental/pallas/ops/gpu/all_gather_mgpu.py`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/all_gather_mgpu.py)

## Functions
- `_never()` — [`L151`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/all_gather_mgpu.py#L151)
- `_run_example()` — [`L167`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/all_gather_mgpu.py#L167)
- `_transfer_loop(loop_info: plgpu.NDLoopInfo)` — [`L131`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/all_gather_mgpu.py#L131)
- `all_gather(x: jax.Array, *, axis_name: Hashable, gather_dimension: int = 0, num_blocks: int | None = None, tile_size: int | None = None, vec_size: int | None = None)` — [`L32`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/all_gather_mgpu.py#L32) — Performs an all-gather operation using multimem instructions.
- `kernel(x_ref, y_ref, done_barrier)` — [`L120`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/all_gather_mgpu.py#L120)
- `kernel_fn(x)` — [`L208`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/all_gather_mgpu.py#L208)
- `ref_fn(x)` — [`L186`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/all_gather_mgpu.py#L186)

