---
title: 'Module: tokamax/_src/ops/attention/pallas_mosaic_tpu.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/attention/pallas_mosaic_tpu.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.attention.pallas_mosaic_tpu`/
symbols:
  PallasMosaicTpuFlashAttention._get_autotuning_configs: PallasMosaicTpuFlashAttention#_get_autotuning_configs().
  PallasMosaicTpuFlashAttention._fwd: PallasMosaicTpuFlashAttention#_fwd().
  PallasMosaicTpuFlashAttention._get_heuristics_config: PallasMosaicTpuFlashAttention#_get_heuristics_config().
  PallasMosaicTpuFlashAttention: PallasMosaicTpuFlashAttention#
  Config: Config#
  Config.block_kv: Config#block_kv.
  Config.block_kv_compute: Config#block_kv_compute.
  QArray: QArray.
  Config.__post_init__: Config#__post_init__().
  PallasMosaicTpuFlashAttention.__post_init__: PallasMosaicTpuFlashAttention#__post_init__().
  Config.block_q: Config#block_q.
  Config.q_layout: Config#q_layout.
  Config.k_layout: Config#k_layout.
  Config.v_layout: Config#v_layout.
  Residuals: Residuals.
  PagingInfo: PagingInfo.
  PallasMosaicTpuFlashAttention.config_cls: PallasMosaicTpuFlashAttention#config_cls.
  Config.use_experimental_scheduler: Config#use_experimental_scheduler.
  Config.use_base2_exp: Config#use_base2_exp.
  PallasMosaicTpuFlashAttention.splash_fn: PallasMosaicTpuFlashAttention#splash_fn().
  Key.Key: Key.Key.
  PallasMosaicTpuFlashAttention.supports_symbolic_shapes: PallasMosaicTpuFlashAttention#supports_symbolic_shapes.
  PallasMosaicTpuFlashAttention.supported_on: PallasMosaicTpuFlashAttention#supported_on().
---
# Module: [`tokamax/_src/ops/attention/pallas_mosaic_tpu.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py)

## Classes
### `Config`
- def: [`tokamax/_src/ops/attention/pallas_mosaic_tpu.py:42`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L42)
- signature: `class Config:`
- members:
  - `block_kv` — [`L44`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L44)
  - `block_kv_compute` — [`L45`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L45)
  - `block_q` — [`L43`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L43)
  - `k_layout` — [`L47`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L47)
  - `q_layout` — [`L46`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L46)
  - `use_base2_exp` — [`L50`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L50)
  - `use_experimental_scheduler` — [`L49`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L49)
  - `v_layout` — [`L48`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L48)
- protocol/private: `__post_init__`[`L52`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L52)
- uses (calls/refs, reference-scoped): [`QKVLayout`](../experimental/tpu/splash_attention/splash_attention_kernel.md#QKVLayout), [`NUM_LANES`](pallas_mosaic_tpu_common.md#NUM_LANES.NUM_LANES)
- used by: [`_get_autotuning_configs`](pallas_mosaic_tpu.md#PallasMosaicTpuFlashAttention._get_autotuning_configs), [`_fwd`](pallas_mosaic_tpu.md#PallasMosaicTpuFlashAttention._fwd), [`_get_heuristics_config`](pallas_mosaic_tpu.md#PallasMosaicTpuFlashAttention._get_heuristics_config), [`PallasMosaicTpuFlashAttention`](pallas_mosaic_tpu.md#PallasMosaicTpuFlashAttention), [`config_cls`](pallas_mosaic_tpu.md#PallasMosaicTpuFlashAttention.config_cls)

### `PallasMosaicTpuFlashAttention`  ·  implements/extends DotProductAttention
- def: [`tokamax/_src/ops/attention/pallas_mosaic_tpu.py:60`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L60)
- doc: Flash attention with Mosaic TPU.
- signature: `class PallasMosaicTpuFlashAttention(base.DotProductAttention[Config, Key]):`
- members:
  - `_fwd(self, q: Float[Array | QArray, "*B T H D"], k: Float[Array | QArray, "*B t h D"], v: Float[Array | QArray, "*B t h d"], *, precision: tuple[base.CanonicalPrecision, base.CanonicalPrecision], logits_dtype: jnp.dtype, logits_scale: float, bias: Float[Array, "*#B #H #T #t"] | None, logits_soft_cap: float | None, mask: base.Mask, dropout_mask: Bool[Array, "*#B #H #T #t"] | None, dropout_rate: float, paging_info: PagingInfo | None, q_indices: Int[Array, "*#B #H T"] | None, k_indices: Int[Array, "*#B #h t"] | None, normalize_output: bool, return_residuals: bool, config: Config)` — [`L76`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L76) — Performs attention, optionally returning softmax residuals.
  - `splash_fn(q, k, v, mask)` — [`L144`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L144)
  - `supported_on(self, device: jax.Device)` — [`L228`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L228)
  - `config_cls` — [`L63`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L63)
  - `supports_symbolic_shapes` — [`L64`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L64)
- protocol/private: `__post_init__`[`L66`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L66), `_get_autotuning_configs`[`L188`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L188), `_get_heuristics_config`[`L174`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L174)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`DotProductAttention`](base.md#DotProductAttention), [`Mask`](base.md#Mask), [`jaxtyped`](../../jaxtyping.md#jaxtyped), [`arguments`](../op.md#BoundArguments.arguments), [`get_default`](../experimental/tpu/splash_attention/splash_attention_kernel.md#SplashConfig.get_default), [`build_splash_kernel`](pallas_mosaic_tpu_common.md#build_splash_kernel), [`QKVLayout`](../experimental/tpu/splash_attention/splash_attention_kernel.md#QKVLayout), [`vjp`](../op.md#Op.vjp), [`SplashConfig`](../experimental/tpu/splash_attention/splash_attention_kernel.md#SplashConfig), [`CanonicalPrecision`](base.md#CanonicalPrecision), [`HEAD_DIM_MINOR`](../experimental/tpu/splash_attention/splash_attention_kernel.md#QKVLayout.HEAD_DIM_MINOR), [`PallasMosaicTpuFlashAttentionVjp`](pallas_mosaic_tpu_vjp.md#PallasMosaicTpuFlashAttentionVjp), [`Config`](pallas_mosaic_tpu.md#Config), [`block_kv`](pallas_mosaic_tpu.md#Config.block_kv), [`block_kv_compute`](pallas_mosaic_tpu.md#Config.block_kv_compute), [`QArray`](pallas_mosaic_tpu.md#QArray), [`block_q`](pallas_mosaic_tpu.md#Config.block_q), [`k_layout`](pallas_mosaic_tpu.md#Config.k_layout), [`q_layout`](pallas_mosaic_tpu.md#Config.q_layout), [`v_layout`](pallas_mosaic_tpu.md#Config.v_layout), [`PagingInfo`](pallas_mosaic_tpu.md#PagingInfo), [`Residuals`](pallas_mosaic_tpu.md#Residuals), [`check_inputs_support`](pallas_mosaic_tpu_common.md#check_inputs_support), [`use_experimental_scheduler`](pallas_mosaic_tpu.md#Config.use_experimental_scheduler), [`Key`](pallas_mosaic_tpu.md#Key.Key), [`SEQ_MINOR`](../experimental/tpu/splash_attention/splash_attention_kernel.md#QKVLayout.SEQ_MINOR), [`use_base2_exp`](pallas_mosaic_tpu.md#Config.use_base2_exp)
- used by: [`DotProductAttention`](base.md#DotProductAttention), [`_get_heuristics_config`](../op.md#Op._get_heuristics_config), [`_get_autotuning_configs`](../op.md#Op._get_autotuning_configs), [`supported_on`](../op.md#Op.supported_on), [`_fwd`](base.md#DotProductAttention._fwd), [`_DEFAULT_IMPLEMENTATIONS`](api.md#_DEFAULT_IMPLEMENTATIONS), [`_test_attention`](pallas_mosaic_tpu_test.md#PallasMosaicTpuFlashAttentionTest._test_attention), [`test_autotune_vjp`](pallas_mosaic_tpu_test.md#PallasMosaicTpuFlashAttentionTest.test_autotune_vjp), [`test_autotune_configs`](pallas_mosaic_tpu_test.md#PallasMosaicTpuFlashAttentionTest.test_autotune_configs)

## Module values
- `Key` — [`L38`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L38)
- `PagingInfo` — [`L37`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L37)
- `QArray` — [`L35`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L35)
- `Residuals` — [`L36`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/attention/pallas_mosaic_tpu.py#L36)

