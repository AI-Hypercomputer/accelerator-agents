---
title: 'Module: jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.gpu.blackwell_matmul_mgpu`/
symbols:
  matmul_kernel: matmul_kernel().
  matmul_kernel.kernel.mn_loop: matmul_kernel().kernel().mn_loop().
  main: main().
  matmul_kernel.kernel.mn_loop._._per_warp._memory._loop_body: matmul_kernel().kernel().mn_loop()._()._per_warp()._memory()._loop_body().
  matmul_kernel.kernel.mn_loop._._per_warp._compute._loop_body: matmul_kernel().kernel().mn_loop()._()._per_warp()._compute()._loop_body().
  matmul_kernel.kernel.mn_loop._._per_warp: matmul_kernel().kernel().mn_loop()._()._per_warp().
  matmul_kernel.kernel.mn_loop._._per_warp._compute: matmul_kernel().kernel().mn_loop()._()._per_warp()._compute().
  matmul_kernel.kernel.mn_loop._._per_warp._memory: matmul_kernel().kernel().mn_loop()._()._per_warp()._memory().
  matmul_kernel.kernel.mn_loop._._per_warp._wait_store: matmul_kernel().kernel().mn_loop()._()._per_warp()._wait_store().
  TuningConfig.grid_minor_dim: TuningConfig#grid_minor_dim.
  MatmulDimension: MatmulDimension#
  matmul_kernel.kernel: matmul_kernel().kernel().
  TuningConfig: TuningConfig#
  TuningConfig.tile_m: TuningConfig#tile_m.
  TuningConfig.tile_n: TuningConfig#tile_n.
  TuningConfig.tile_k: TuningConfig#tile_k.
  TuningConfig.max_concurrent_steps: TuningConfig#max_concurrent_steps.
  TuningConfig.collective: TuningConfig#collective.
  TuningConfig.epilogue_tile_n: TuningConfig#epilogue_tile_n.
  TuningConfig.grid_tile_width: TuningConfig#grid_tile_width.
  MatmulDimension.N: MatmulDimension#N.
  MatmulDimension.M: MatmulDimension#M.
---
# Module: [`jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py)

## Classes
### `MatmulDimension`  ·  implements/extends IntEnum
- def: [`jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py:30`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py#L30)
- signature: `class MatmulDimension(enum.IntEnum):`
- members:
  - `M` — [`L31`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py#L31)
  - `N` — [`L32`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py#L32)
- used by: [`main`](blackwell_matmul_mgpu.md#main), [`main`](blackwell_ragged_dot_mgpu.md#main), [`grid_minor_dim`](blackwell_matmul_mgpu.md#TuningConfig.grid_minor_dim), [`grid_minor_dim`](blackwell_ragged_dot_mgpu.md#TuningConfig.grid_minor_dim)

### `TuningConfig`
- def: [`jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py:36`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py#L36)
- signature: `class TuningConfig:`
- members:
  - `collective` — [`L41`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py#L41)
  - `epilogue_tile_n` — [`L42`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py#L42)
  - `grid_minor_dim` — [`L43`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py#L43)
  - `grid_tile_width` — [`L44`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py#L44)
  - `max_concurrent_steps` — [`L40`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py#L40)
  - `tile_k` — [`L39`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py#L39)
  - `tile_m` — [`L37`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py#L37)
  - `tile_n` — [`L38`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py#L38)
- uses (calls/refs, reference-scoped): [`MatmulDimension`](blackwell_matmul_mgpu.md#MatmulDimension), [`N`](blackwell_matmul_mgpu.md#MatmulDimension.N)
- used by: [`matmul_kernel`](blackwell_matmul_mgpu.md#matmul_kernel), [`mn_loop`](blackwell_matmul_mgpu.md#matmul_kernel.kernel.mn_loop), [`main`](blackwell_matmul_mgpu.md#main)

## Functions
- `_compute()` — [`L162`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py#L162)
- `_loop_body(ki, _)` — [`L131`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py#L131)
- `_loop_body(ki, _)` — [`L163`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py#L163)
- `_memory()` — [`L130`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py#L130)
- `_per_warp()` — [`L127`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py#L127)
- `_wait_store()` — [`L159`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py#L159)
- `kernel(a_gmem, b_gmem, out_gmem, a_smem, b_smem, acc_tmem, acc_smem, ab_tma_barrier, store_done_barrier, mma_done_barrier, consumed_barrier)` — [`L99`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py#L99)
- `main(_)` — [`L264`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py#L264)
- `matmul_kernel(a, b, config: TuningConfig)` — [`L47`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py#L47)
- `mn_loop(loop_info: plgpu.NDLoopInfo)` — [`L108`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/blackwell_matmul_mgpu.py#L108)

