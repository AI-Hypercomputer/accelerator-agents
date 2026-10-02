---
title: 'Module: tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.ragged_dot.pallas_mosaic_gpu`/
symbols:
  PallasMosaicGpuRaggedDot._fwd: PallasMosaicGpuRaggedDot#_fwd().
  PallasMosaicGpuRaggedDot._get_sm90_autotuning_configs: PallasMosaicGpuRaggedDot#_get_sm90_autotuning_configs().
  PallasMosaicGpuRaggedDot._get_heuristics_config: PallasMosaicGpuRaggedDot#_get_heuristics_config().
  PallasMosaicGpuRaggedDot._generate_configs: PallasMosaicGpuRaggedDot#_generate_configs().
  Config: Config.
  PallasMosaicGpuRaggedDot._get_sm100_autotuning_configs: PallasMosaicGpuRaggedDot#_get_sm100_autotuning_configs().
  QArray: QArray.
  PallasMosaicGpuRaggedDot._get_autotuning_cache_key: PallasMosaicGpuRaggedDot#_get_autotuning_cache_key().
  PallasMosaicGpuRaggedDot: PallasMosaicGpuRaggedDot#
  PallasMosaicGpuRaggedDot._get_autotuning_configs: PallasMosaicGpuRaggedDot#_get_autotuning_configs().
  PallasMosaicGpuRaggedDot.__post_init__: PallasMosaicGpuRaggedDot#__post_init__().
  AsQArray: AsQArray.
  GroupSizes: GroupSizes.
  PallasMosaicGpuRaggedDot.config_cls: PallasMosaicGpuRaggedDot#config_cls.
  PallasMosaicGpuRaggedDot.supports_symbolic_shapes: PallasMosaicGpuRaggedDot#supports_symbolic_shapes.
  PallasMosaicGpuRaggedDot.supported_on: PallasMosaicGpuRaggedDot#supported_on().
---
# Module: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu.py)

## Classes
### `PallasMosaicGpuRaggedDot`  ·  implements/extends RaggedDot
- def: [`tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu.py:51`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu.py#L51)
- doc: Pallas-Mosaic-GPU ragged dot implementation.
- signature: `class PallasMosaicGpuRaggedDot(base.RaggedDot[Config, None]):`
- members:
  - `supported_on(self, device: jax.Device)` — [`L453`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu.py#L453)
  - `config_cls` — [`L57`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu.py#L57) — ---
  - `supports_symbolic_shapes` — [`L58`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu.py#L58)
- protocol/private: `__post_init__`[`L60`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu.py#L60), `_fwd`[`L68`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu.py#L68), `_generate_configs`[`L376`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu.py#L376), `_get_autotuning_cache_key`[`L218`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu.py#L218), `_get_autotuning_configs`[`L298`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu.py#L298), `_get_heuristics_config`[`L237`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu.py#L237), `_get_sm100_autotuning_configs`[`L349`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu.py#L349), `_get_sm90_autotuning_configs`[`L303`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu.py#L303)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`ragged_dot_gpu_fp8_quant_blackwell_kernel`](pallas_mosaic_gpu_kernel_sm100_fp8_quant.md#ragged_dot_gpu_fp8_quant_blackwell_kernel), [`ragged_dot_gpu_i8_quant_blackwell_kernel`](pallas_mosaic_gpu_kernel_sm100_i8_quant.md#ragged_dot_gpu_i8_quant_blackwell_kernel), [`ragged_dot_gpu_quant_blackwell_kernel`](pallas_mosaic_gpu_kernel_sm100_quant.md#ragged_dot_gpu_quant_blackwell_kernel), [`ragged_dot_gpu_quant_post_scale_blackwell_kernel`](pallas_mosaic_gpu_kernel_sm100_quant_post_scale.md#ragged_dot_gpu_quant_post_scale_blackwell_kernel), [`ragged_dot_quantized_kernel`](pallas_mosaic_gpu_kernel_sm90_quant.md#ragged_dot_quantized_kernel), [`ragged_dot_gpu_non_quant_blackwell_kernel`](pallas_mosaic_gpu_kernel_sm100.md#ragged_dot_gpu_non_quant_blackwell_kernel), [`ragged_dot_kernel`](pallas_mosaic_gpu_kernel_sm90.md#ragged_dot_kernel), [`RaggedDot`](base.md#RaggedDot), [`args`](../op.md#BoundArguments.args), [`arguments`](../op.md#BoundArguments.arguments), [`block_k`](pallas_mosaic_gpu_common.md#Config.block_k), [`vjp`](../op.md#Op.vjp), [`_get_autotuning_cache_key`](../op.md#Op._get_autotuning_cache_key), [`block_m`](pallas_mosaic_gpu_common.md#Config.block_m), [`Config`](pallas_mosaic_gpu.md#Config), [`kwargs`](../op.md#BoundArguments.kwargs), [`ActivationFunction`](base.md#ActivationFunction), [`block_n`](pallas_mosaic_gpu_common.md#Config.block_n), [`num_stages`](pallas_mosaic_gpu_common.md#Config.num_stages), [`ragged_contracting_dim_dot_kernel`](pallas_mosaic_gpu_kernel_sm90.md#ragged_contracting_dim_dot_kernel), [`grid_minor_dim`](pallas_mosaic_gpu_common.md#Config.grid_minor_dim), [`QArray`](pallas_mosaic_gpu.md#QArray), [`vjp`](base.md#vjp), [`grid_tile_width`](pallas_mosaic_gpu_common.md#Config.grid_tile_width), [`split_k`](pallas_mosaic_gpu_common.md#Config.split_k), [`MatmulDimension`](pallas_mosaic_gpu_common.md#MatmulDimension), [`DEFAULT_RAGGED_DOT_DIM_NUMS`](base.md#DEFAULT_RAGGED_DOT_DIM_NUMS), [`collective`](pallas_mosaic_gpu_common.md#Config.collective), [`persistent`](pallas_mosaic_gpu_common.md#Config.persistent), [`Residuals`](base.md#Residuals), [`CanonicalPrecision`](base.md#CanonicalPrecision), [`M`](pallas_mosaic_gpu_common.md#MatmulDimension.M), [`post_scale`](pallas_mosaic_gpu_common.md#Config.post_scale), [`split_m`](pallas_mosaic_gpu_common.md#Config.split_m), [`AsQArray`](pallas_mosaic_gpu.md#AsQArray), [`GroupSizes`](pallas_mosaic_gpu.md#GroupSizes), [`RAGGED_CONTRACTING_DOT_DIM_NUMS`](base.md#RAGGED_CONTRACTING_DOT_DIM_NUMS), [`N`](pallas_mosaic_gpu_common.md#MatmulDimension.N), [`TRANS_RHS_RAGGED_DOT_DIM_NUMS`](base.md#TRANS_RHS_RAGGED_DOT_DIM_NUMS)
- used by: [`_get_heuristics_config`](../op.md#Op._get_heuristics_config), [`_get_autotuning_configs`](../op.md#Op._get_autotuning_configs), [`supported_on`](../op.md#Op.supported_on), [`RaggedDot`](base.md#RaggedDot), [`_fwd`](base.md#RaggedDot._fwd), [`_get_autotuning_cache_key`](../op.md#Op._get_autotuning_cache_key), [`_DEFAULT_IMPLEMENTATIONS`](api.md#_DEFAULT_IMPLEMENTATIONS), [`_IMPLS`](bench.md#_IMPLS), [`__init__`](pallas_mosaic_gpu_kernel_sm100_fp8_quant_test.md#PallasMosaicGpuKernelSm100FP8QuantTest.__init__), [`__init__`](pallas_mosaic_gpu_kernel_sm100_i8_quant_test.md#PallasMosaicGpuKernelSm100I8QuantTest.__init__), [`__init__`](pallas_mosaic_gpu_test.md#PallasMosaicGpuRaggedDotTest.__init__)

## Module values
- `AsQArray` — [`L40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu.py#L40)
- `Config` — [`L38`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu.py#L38)
- `GroupSizes` — [`L41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu.py#L41)
- `QArray` — [`L39`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu.py#L39)

