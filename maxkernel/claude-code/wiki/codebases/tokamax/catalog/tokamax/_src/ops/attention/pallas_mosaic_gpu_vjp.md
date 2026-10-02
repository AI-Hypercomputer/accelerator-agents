---
title: 'Module: tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.attention.pallas_mosaic_gpu_vjp`/
symbols:
  PallasMosaicGpuFlashAttentionVjp._fwd: PallasMosaicGpuFlashAttentionVjp#_fwd().
  PallasMosaicGpuFlashAttentionVjp._get_heuristics_config: PallasMosaicGpuFlashAttentionVjp#_get_heuristics_config().
  PallasMosaicGpuFlashAttentionVjp._get_autotuning_configs: PallasMosaicGpuFlashAttentionVjp#_get_autotuning_configs().
  PallasMosaicGpuFlashAttentionVjp: PallasMosaicGpuFlashAttentionVjp#
  PallasMosaicGpuFlashAttentionVjp.cast: PallasMosaicGpuFlashAttentionVjp#cast().
  Config: Config.
  PallasMosaicGpuFlashAttentionVjp.dbias_intermediate_dtype: PallasMosaicGpuFlashAttentionVjp#dbias_intermediate_dtype.
  PallasMosaicGpuFlashAttentionVjp.config_cls: PallasMosaicGpuFlashAttentionVjp#config_cls.
  CanonicalPrecision: CanonicalPrecision.
  _broadcast_to_rank: _broadcast_to_rank().
  PagingInfo: PagingInfo.
  Residuals: Residuals.
  Mask: Mask.
  PallasMosaicGpuFlashAttentionVjp.supports_symbolic_shapes: PallasMosaicGpuFlashAttentionVjp#supports_symbolic_shapes.
  PallasMosaicGpuFlashAttentionVjp.supported_on: PallasMosaicGpuFlashAttentionVjp#supported_on().
---
# Module: [`tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp.py)

## Classes
### `PallasMosaicGpuFlashAttentionVjp`  ·  implements/extends DotProductAttentionVjp
- def: [`tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp.py:50`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp.py#L50)
- doc: Pallas-Triton FlashAttention VJP implementation.
- signature: `class PallasMosaicGpuFlashAttentionVjp(base.DotProductAttentionVjp[Config, None]):`
- members:
  - `cast(x, precision)` — [`L120`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp.py#L120)
  - `supported_on(self, device: jax.Device)` — [`L208`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp.py#L208)
  - `config_cls` — [`L55`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp.py#L55)
  - `dbias_intermediate_dtype` — [`L57`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp.py#L57)
  - `supports_symbolic_shapes` — [`L56`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp.py#L56)
- protocol/private: `_fwd`[`L61`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp.py#L61), `_get_autotuning_configs`[`L200`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp.py#L200), `_get_heuristics_config`[`L192`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp.py#L192)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`Mask`](base.md#Mask), [`get_autotuning_configs`](pallas_mosaic_gpu_vjp_kernel_sm100.md#get_autotuning_configs), [`flash_attention_vjp_kernel`](pallas_mosaic_gpu_vjp_kernel_sm100.md#flash_attention_vjp_kernel), [`jaxtyped`](../../jaxtyping.md#jaxtyped), [`flash_attention_vjp_kernel`](pallas_mosaic_gpu_vjp_kernel_sm90.md#flash_attention_vjp_kernel), [`get_heuristics_config`](pallas_mosaic_gpu_vjp_kernel_sm90.md#get_heuristics_config), [`DotProductAttentionVjp`](base.md#DotProductAttentionVjp), [`get_autotuning_configs`](pallas_mosaic_gpu_vjp_kernel_sm90.md#get_autotuning_configs), [`Config`](pallas_mosaic_gpu_vjp_kernel_sm100.md#Config), [`vmap_batch_dims`](base.md#vmap_batch_dims), [`get_heuristics_config`](pallas_mosaic_gpu_vjp_kernel_sm100.md#get_heuristics_config), [`Config`](pallas_mosaic_gpu_vjp_kernel_sm90.md#Config), [`DotProductAttentionGrads`](base.md#DotProductAttentionGrads), [`decompose_mask`](pallas_mosaic_gpu_common.md#decompose_mask), [`Config`](pallas_mosaic_gpu_vjp.md#Config), [`CanonicalPrecision`](pallas_mosaic_gpu_vjp.md#CanonicalPrecision), [`_broadcast_to_rank`](pallas_mosaic_gpu_vjp.md#_broadcast_to_rank), [`k`](base.md#DotProductAttentionGrads.k), [`q`](base.md#DotProductAttentionGrads.q), [`safe_downcast`](pallas_mosaic_gpu_common.md#safe_downcast), [`v`](base.md#DotProductAttentionGrads.v), [`PagingInfo`](pallas_mosaic_gpu_vjp.md#PagingInfo), [`Residuals`](pallas_mosaic_gpu_vjp.md#Residuals), [`bias`](base.md#DotProductAttentionGrads.bias)
- used by: [`_get_heuristics_config`](../op.md#Op._get_heuristics_config), [`_get_autotuning_configs`](../op.md#Op._get_autotuning_configs), [`supported_on`](../op.md#Op.supported_on), [`_fwd`](base.md#DotProductAttentionVjp._fwd), [`DotProductAttentionVjp`](base.md#DotProductAttentionVjp), [`__init__`](pallas_mosaic_gpu_test.md#PallasMosaicGpuFlashAttentionTest.__init__), [`__post_init__`](pallas_mosaic_gpu.md#PallasMosaicGpuFlashAttention.__post_init__)

## Functions
- `_broadcast_to_rank(x, rank)` — [`L45`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp.py#L45)

## Module values
- `CanonicalPrecision` — [`L38`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp.py#L38)
- `Config` — [`L39`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp.py#L39)
- `Mask` — [`L40`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp.py#L40)
- `PagingInfo` — [`L41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp.py#L41)
- `Residuals` — [`L42`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp.py#L42)

