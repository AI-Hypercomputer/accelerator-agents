---
title: 'Module: jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.gpu.hopper_mixed_type_matmul_mgpu`/
symbols:
  mixed_matmul_kernel: mixed_matmul_kernel().
  main: main().
  mixed_matmul_kernel.kernel._pipeline_scope._mn_loop: mixed_matmul_kernel().kernel()._pipeline_scope()._mn_loop().
  mixed_matmul_kernel.kernel.get_pipeline: mixed_matmul_kernel().kernel().get_pipeline().
  MatmulDimension: MatmulDimension#
  reference: reference().
  mixed_matmul_kernel.kernel._pipeline_scope._mn_loop.compute_context._acc_scope: mixed_matmul_kernel().kernel()._pipeline_scope()._mn_loop().compute_context()._acc_scope().
  mixed_matmul_kernel.kernel._pipeline_scope._mn_loop.mma_body: mixed_matmul_kernel().kernel()._pipeline_scope()._mn_loop().mma_body().
  TuningConfig.cluster_dimension: TuningConfig#cluster_dimension.
  mixed_matmul_kernel.kernel._pipeline_scope: mixed_matmul_kernel().kernel()._pipeline_scope().
  TuningConfig.wg_dimension: TuningConfig#wg_dimension.
  MatmulDimension.N: MatmulDimension#N.
  TuningConfig.grid_minor_dim: TuningConfig#grid_minor_dim.
  MatmulDimension.M: MatmulDimension#M.
  mixed_matmul_kernel.kernel: mixed_matmul_kernel().kernel().
  TuningConfig: TuningConfig#
  TuningConfig.tile_m: TuningConfig#tile_m.
  TuningConfig.tile_n: TuningConfig#tile_n.
  TuningConfig.tile_k: TuningConfig#tile_k.
  TuningConfig.max_concurrent_steps: TuningConfig#max_concurrent_steps.
  TuningConfig.epi_tile_n: TuningConfig#epi_tile_n.
  TuningConfig.epi_tile_m: TuningConfig#epi_tile_m.
  TuningConfig.grid_tile_width: TuningConfig#grid_tile_width.
  mixed_matmul_kernel.kernel._pipeline_scope._mn_loop.compute_context: mixed_matmul_kernel().kernel()._pipeline_scope()._mn_loop().compute_context().
  MatmulDimension.__str__: MatmulDimension#__str__().
  MatmulDimension.__repr__: MatmulDimension#__repr__().
---
# Module: [`jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py)

## Classes
### `MatmulDimension`  ·  implements/extends IntEnum
- def: [`jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py:32`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py#L32)
- signature: `class MatmulDimension(enum.IntEnum):`
- members:
  - `M` — [`L33`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py#L33)
  - `N` — [`L34`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py#L34)
- protocol/private: `__repr__`[`L39`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py#L39), `__str__`[`L36`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py#L36)
- used by: [`mixed_matmul_kernel`](hopper_mixed_type_matmul_mgpu.md#mixed_matmul_kernel), [`main`](hopper_mixed_type_matmul_mgpu.md#main), [`_mn_loop`](hopper_mixed_type_matmul_mgpu.md#mixed_matmul_kernel.kernel._pipeline_scope._mn_loop), [`get_pipeline`](hopper_mixed_type_matmul_mgpu.md#mixed_matmul_kernel.kernel.get_pipeline), [`cluster_dimension`](hopper_mixed_type_matmul_mgpu.md#TuningConfig.cluster_dimension), [`wg_dimension`](hopper_mixed_type_matmul_mgpu.md#TuningConfig.wg_dimension), [`grid_minor_dim`](hopper_mixed_type_matmul_mgpu.md#TuningConfig.grid_minor_dim)

### `TuningConfig`
- def: [`jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py:44`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py#L44)
- signature: `class TuningConfig:`
- members:
  - `cluster_dimension` — [`L54`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py#L54)
  - `epi_tile_m` — [`L50`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py#L50)
  - `epi_tile_n` — [`L49`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py#L49)
  - `grid_minor_dim` — [`L51`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py#L51)
  - `grid_tile_width` — [`L52`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py#L52)
  - `max_concurrent_steps` — [`L48`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py#L48)
  - `tile_k` — [`L47`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py#L47)
  - `tile_m` — [`L45`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py#L45)
  - `tile_n` — [`L46`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py#L46)
  - `wg_dimension` — [`L53`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py#L53)
- uses (calls/refs, reference-scoped): [`MatmulDimension`](hopper_mixed_type_matmul_mgpu.md#MatmulDimension), [`N`](hopper_mixed_type_matmul_mgpu.md#MatmulDimension.N)
- used by: [`mixed_matmul_kernel`](hopper_mixed_type_matmul_mgpu.md#mixed_matmul_kernel), [`main`](hopper_mixed_type_matmul_mgpu.md#main), [`_mn_loop`](hopper_mixed_type_matmul_mgpu.md#mixed_matmul_kernel.kernel._pipeline_scope._mn_loop), [`get_pipeline`](hopper_mixed_type_matmul_mgpu.md#mixed_matmul_kernel.kernel.get_pipeline)

## Functions
- `_acc_scope(acc_ref)` — [`L203`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py#L203)
- `_mn_loop(loop_info: plgpu.NDLoopInfo)` — [`L176`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py#L176)
- `_pipeline_scope(pipeline_allocs)` — [`L172`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py#L172)
- `compute_context(eval_pipeline)` — [`L199`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py#L199)
- `get_pipeline(pipeline_body, compute_context)` — [`L134`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py#L134)
- `kernel(a_gmem, b_gmem, out_gmem, out_smem)` — [`L132`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py#L132)
- `main(_)` — [`L278`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py#L278)
- `mixed_matmul_kernel(a: jax.Array, b: jax.Array, *, out_dtype: jnp.dtype, config: TuningConfig)` — [`L57`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py#L57) — Mixed-type matrix multiplication kernel for Hopper GPUs.
- `mma_body(_, a_smem, b_smem, acc_ref)` — [`L222`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py#L222)
- `reference(a: jax.Array, b: jax.Array, *, out_dtype: jnp.dtype)` — [`L269`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_mixed_type_matmul_mgpu.py#L269) — Reference implementation of a mixed-type matrix multiplication.

