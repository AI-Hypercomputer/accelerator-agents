---
title: 'Module: jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.gpu.hopper_matmul_mgpu`/
symbols:
  main: main().
  kernel: kernel().
  kernel._pipeline_scope._mn_loop: kernel()._pipeline_scope()._mn_loop().
  matmul: matmul().
  kernel.get_pipeline: kernel().get_pipeline().
  MatmulDimension: MatmulDimension#
  kernel._pipeline_scope: kernel()._pipeline_scope().
  kernel._pipeline_scope._mn_loop.compute_context._acc_scope: kernel()._pipeline_scope()._mn_loop().compute_context()._acc_scope().
  TuningConfig.cluster_dimension: TuningConfig#cluster_dimension.
  TuningConfig.wg_dimension: TuningConfig#wg_dimension.
  MatmulDimension.M: MatmulDimension#M.
  MatmulDimension.N: MatmulDimension#N.
  TuningConfig.grid_minor_dim: TuningConfig#grid_minor_dim.
  get_num_sms: get_num_sms().
  kernel._pipeline_scope._mn_loop.mma_body: kernel()._pipeline_scope()._mn_loop().mma_body().
  TuningConfig: TuningConfig#
  TuningConfig.tile_m: TuningConfig#tile_m.
  TuningConfig.tile_n: TuningConfig#tile_n.
  TuningConfig.tile_k: TuningConfig#tile_k.
  TuningConfig.max_concurrent_steps: TuningConfig#max_concurrent_steps.
  TuningConfig.epi_tile_n: TuningConfig#epi_tile_n.
  TuningConfig.epi_tile_m: TuningConfig#epi_tile_m.
  TuningConfig.grid_tile_width: TuningConfig#grid_tile_width.
  kernel._pipeline_scope._mn_loop.compute_context: kernel()._pipeline_scope()._mn_loop().compute_context().
  MatmulDimension.__str__: MatmulDimension#__str__().
  MatmulDimension.__repr__: MatmulDimension#__repr__().
---
# Module: [`jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py)

## Classes
### `MatmulDimension`  ·  implements/extends IntEnum
- def: [`jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py:37`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py#L37)
- signature: `class MatmulDimension(enum.IntEnum):`
- members:
  - `M` — [`L38`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py#L38)
  - `N` — [`L39`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py#L39)
- protocol/private: `__repr__`[`L44`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py#L44), `__str__`[`L41`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py#L41)
- used by: [`all_gather_lhs_matmul`](collective_matmul_mgpu.md#all_gather_lhs_matmul), [`_run_example`](collective_matmul_mgpu.md#_run_example), [`main`](hopper_matmul_mgpu.md#main), [`kernel`](hopper_matmul_mgpu.md#kernel), [`_mn_loop`](hopper_matmul_mgpu.md#kernel._pipeline_scope._mn_loop), [`get_pipeline`](hopper_matmul_mgpu.md#kernel.get_pipeline), [`cluster_dimension`](hopper_matmul_mgpu.md#TuningConfig.cluster_dimension), [`wg_dimension`](hopper_matmul_mgpu.md#TuningConfig.wg_dimension), [`grid_minor_dim`](hopper_matmul_mgpu.md#TuningConfig.grid_minor_dim), [`MatmulDimension`](collective_matmul_mgpu.md#MatmulDimension)

### `TuningConfig`
- def: [`jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py:49`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py#L49)
- signature: `class TuningConfig:`
- members:
  - `cluster_dimension` — [`L59`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py#L59)
  - `epi_tile_m` — [`L55`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py#L55)
  - `epi_tile_n` — [`L54`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py#L54)
  - `grid_minor_dim` — [`L56`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py#L56)
  - `grid_tile_width` — [`L57`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py#L57)
  - `max_concurrent_steps` — [`L53`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py#L53)
  - `tile_k` — [`L52`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py#L52)
  - `tile_m` — [`L50`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py#L50)
  - `tile_n` — [`L51`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py#L51)
  - `wg_dimension` — [`L58`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py#L58)
- uses (calls/refs, reference-scoped): [`MatmulDimension`](hopper_matmul_mgpu.md#MatmulDimension), [`N`](hopper_matmul_mgpu.md#MatmulDimension.N)
- used by: [`all_gather_lhs_matmul`](collective_matmul_mgpu.md#all_gather_lhs_matmul), [`_run_example`](collective_matmul_mgpu.md#_run_example), [`main`](hopper_matmul_mgpu.md#main), [`kernel`](hopper_matmul_mgpu.md#kernel), [`_mn_loop`](hopper_matmul_mgpu.md#kernel._pipeline_scope._mn_loop), [`matmul`](hopper_matmul_mgpu.md#matmul), [`get_pipeline`](hopper_matmul_mgpu.md#kernel.get_pipeline), [`TuningConfig`](collective_matmul_mgpu.md#TuningConfig)

## Functions
- `_acc_scope(acc_ref)` — [`L183`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py#L183)
- `_mn_loop(loop_info: plgpu.NDLoopInfo)` — [`L156`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py#L156)
- `_pipeline_scope(pipeline_allocs, out_smem, c_barrier)` — [`L152`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py#L152)
- `compute_context(eval_pipeline)` — [`L179`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py#L179)
- `get_num_sms()` — [`L31`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py#L31)
- `get_pipeline(pipeline_body, compute_context)` — [`L106`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py#L106)
- `kernel(a_gmem, b_gmem, c_gmem, out_gmem, config: TuningConfig, pipeline_callback=None, delay_release=0)` — [`L63`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py#L63)
- `main(_)` — [`L275`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py#L275)
- `matmul(a, b, c, config: TuningConfig)` — [`L232`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py#L232)
- `mma_body(idxs, a_smem, b_smem, acc_ref)` — [`L215`](../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/gpu/hopper_matmul_mgpu.py#L215)

