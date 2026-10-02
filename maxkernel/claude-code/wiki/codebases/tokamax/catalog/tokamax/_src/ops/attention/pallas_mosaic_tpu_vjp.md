---
title: 'Module: tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.attention.pallas_mosaic_tpu_vjp`/
symbols:
  PallasMosaicTpuFlashAttentionVjp._fwd: PallasMosaicTpuFlashAttentionVjp#_fwd().
  PallasMosaicTpuFlashAttentionVjp._get_autotuning_configs: PallasMosaicTpuFlashAttentionVjp#_get_autotuning_configs().
  PallasMosaicTpuFlashAttentionVjp._get_heuristics_config: PallasMosaicTpuFlashAttentionVjp#_get_heuristics_config().
  Config: Config#
  PallasMosaicTpuFlashAttentionVjp: PallasMosaicTpuFlashAttentionVjp#
  Config.block_kv_dkv: Config#block_kv_dkv.
  Config.block_kv_dkv_compute: Config#block_kv_dkv_compute.
  Config.__post_init__: Config#__post_init__().
  Config.block_q_dkv: Config#block_q_dkv.
  Residuals: Residuals.
  PagingInfo: PagingInfo.
  PallasMosaicTpuFlashAttentionVjp.bwd_fn: PallasMosaicTpuFlashAttentionVjp#bwd_fn().
  QArray: QArray.
  PallasMosaicTpuFlashAttentionVjp.config_cls: PallasMosaicTpuFlashAttentionVjp#config_cls.
  PallasMosaicTpuFlashAttentionVjp.supports_symbolic_shapes: PallasMosaicTpuFlashAttentionVjp#supports_symbolic_shapes.
  Config.use_base2_exp: Config#use_base2_exp.
  PallasMosaicTpuFlashAttentionVjp.supported_on: PallasMosaicTpuFlashAttentionVjp#supported_on().
  Key.Key: Key.Key.
---
# Module: [`tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py)

## Classes
### `Config`
- def: [`tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py:42`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py#L42)
- signature: `class Config:`
- members:
  - `block_kv_dkv` — [`L44`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py#L44)
  - `block_kv_dkv_compute` — [`L45`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py#L45)
  - `block_q_dkv` — [`L43`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py#L43)
  - `use_base2_exp` — [`L46`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py#L46)
- protocol/private: `__post_init__`[`L48`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py#L48)
- uses (calls/refs, reference-scoped): [`NUM_LANES`](pallas_mosaic_tpu_common.md#NUM_LANES.NUM_LANES)
- used by: [`_fwd`](pallas_mosaic_tpu_vjp.md#PallasMosaicTpuFlashAttentionVjp._fwd), [`_get_autotuning_configs`](pallas_mosaic_tpu_vjp.md#PallasMosaicTpuFlashAttentionVjp._get_autotuning_configs), [`_get_heuristics_config`](pallas_mosaic_tpu_vjp.md#PallasMosaicTpuFlashAttentionVjp._get_heuristics_config), [`PallasMosaicTpuFlashAttentionVjp`](pallas_mosaic_tpu_vjp.md#PallasMosaicTpuFlashAttentionVjp), [`config_cls`](pallas_mosaic_tpu_vjp.md#PallasMosaicTpuFlashAttentionVjp.config_cls)

### `PallasMosaicTpuFlashAttentionVjp`  ·  implements/extends DotProductAttentionVjp
- def: [`tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py:58`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py#L58)
- doc: Pallas-Mosaic FlashAttention VJP implementation.
- signature: `class PallasMosaicTpuFlashAttentionVjp(base.DotProductAttentionVjp[Config, None]):`
- members:
  - `bwd_fn(res, cotangents, splash_mask)` — [`L162`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py#L162)
  - `supported_on(self, device: jax.Device)` — [`L237`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py#L237)
  - `config_cls` — [`L63`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py#L63)
  - `supports_symbolic_shapes` — [`L64`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py#L64)
- protocol/private: `_fwd`[`L68`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py#L68), `_get_autotuning_configs`[`L207`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py#L207), `_get_heuristics_config`[`L198`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py#L198)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`Mask`](base.md#Mask), [`jaxtyped`](../../jaxtyping.md#jaxtyped), [`_splash_attention_bwd`](../experimental/tpu/splash_attention/splash_attention_kernel.md#_splash_attention_bwd), [`arguments`](../op.md#BoundArguments.arguments), [`get_default`](../experimental/tpu/splash_attention/splash_attention_kernel.md#SplashConfig.get_default), [`build_splash_kernel`](pallas_mosaic_tpu_common.md#build_splash_kernel), [`DotProductAttentionVjp`](base.md#DotProductAttentionVjp), [`SplashConfig`](../experimental/tpu/splash_attention/splash_attention_kernel.md#SplashConfig), [`CanonicalPrecision`](base.md#CanonicalPrecision), [`LOG2E`](../experimental/tpu/splash_attention/splash_attention_kernel.md#LOG2E), [`DotProductAttentionGrads`](base.md#DotProductAttentionGrads), [`Config`](pallas_mosaic_tpu_vjp.md#Config), [`block_kv_dkv`](pallas_mosaic_tpu_vjp.md#Config.block_kv_dkv), [`block_kv_dkv_compute`](pallas_mosaic_tpu_vjp.md#Config.block_kv_dkv_compute), [`block_q_dkv`](pallas_mosaic_tpu_vjp.md#Config.block_q_dkv), [`k`](base.md#DotProductAttentionGrads.k), [`q`](base.md#DotProductAttentionGrads.q), [`v`](base.md#DotProductAttentionGrads.v), [`PagingInfo`](pallas_mosaic_tpu_vjp.md#PagingInfo), [`Residuals`](pallas_mosaic_tpu_vjp.md#Residuals), [`check_inputs_support`](pallas_mosaic_tpu_common.md#check_inputs_support), [`use_base2_exp`](pallas_mosaic_tpu_vjp.md#Config.use_base2_exp)
- used by: [`_get_heuristics_config`](../op.md#Op._get_heuristics_config), [`_get_autotuning_configs`](../op.md#Op._get_autotuning_configs), [`supported_on`](../op.md#Op.supported_on), [`_fwd`](base.md#DotProductAttentionVjp._fwd), [`DotProductAttentionVjp`](base.md#DotProductAttentionVjp), [`_test_attention`](pallas_mosaic_tpu_test.md#PallasMosaicTpuFlashAttentionTest._test_attention), [`test_autotune_vjp`](pallas_mosaic_tpu_test.md#PallasMosaicTpuFlashAttentionTest.test_autotune_vjp), [`__post_init__`](pallas_mosaic_tpu.md#PallasMosaicTpuFlashAttention.__post_init__)

## Module values
- `Key` — [`L38`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py#L38)
- `PagingInfo` — [`L37`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py#L37)
- `QArray` — [`L35`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py#L35)
- `Residuals` — [`L36`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu_vjp.py#L36)

