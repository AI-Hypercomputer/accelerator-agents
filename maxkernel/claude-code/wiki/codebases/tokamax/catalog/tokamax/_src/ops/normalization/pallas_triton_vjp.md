---
title: 'Module: tokamax/_src/ops/normalization/pallas_triton_vjp.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/normalization/pallas_triton_vjp.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.normalization.pallas_triton_vjp`/
symbols:
  PallasTritonNormalizationVjp._get_autotuning_configs: PallasTritonNormalizationVjp#_get_autotuning_configs().
  PallasTritonNormalizationVjp._fwd: PallasTritonNormalizationVjp#_fwd().
  PallasTritonNormalizationVjp._get_heuristics_config: PallasTritonNormalizationVjp#_get_heuristics_config().
  PallasTritonNormalizationVjp._get_autotuning_cache_key: PallasTritonNormalizationVjp#_get_autotuning_cache_key().
  PallasTritonNormalizationVjp: PallasTritonNormalizationVjp#
  Config.Config: Config.Config.
  Key.Key: Key.Key.
  Residuals.Residuals: Residuals.Residuals.
  PallasTritonNormalizationVjp.config_cls: PallasTritonNormalizationVjp#config_cls.
  _NUM_REGISTERS_PER_SM: _NUM_REGISTERS_PER_SM.
  _normalization_vjp_kernel: _normalization_vjp_kernel().
  PallasTritonNormalizationVjp.supported_on: PallasTritonNormalizationVjp#supported_on().
---
# Module: [`tokamax/_src/ops/normalization/pallas_triton_vjp.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_vjp.py)

## Classes
### `PallasTritonNormalizationVjp`  ·  implements/extends NormalizationVjp
- def: [`tokamax/_src/ops/normalization/pallas_triton_vjp.py:82`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_vjp.py#L82)
- doc: Pallas-Triton normalization VJP.
- signature: `class PallasTritonNormalizationVjp(base.NormalizationVjp[Config, Key]):`
- members:
  - `_fwd(self, residuals: Residuals, out: jax.Array, dout: jax.Array, x: jax.Array, scale: jax.Array | None, offset: jax.Array | None, *, axis: int, epsilon: float, scale_offset: float, subtract_mean: bool, return_residuals: bool, config: Config)` — [`L88`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_vjp.py#L88) — Computes normalization VJP `(dx, dscale, doffset)`.
  - `supported_on(self, device: jax.Device)` — [`L205`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_vjp.py#L205)
  - `config_cls` — [`L85`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_vjp.py#L85)
- protocol/private: `_get_autotuning_cache_key`[`L176`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_vjp.py#L176), `_get_autotuning_configs`[`L181`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_vjp.py#L181), `_get_heuristics_config`[`L170`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_vjp.py#L170)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`args`](../op.md#BoundArguments.args), [`kwargs`](../op.md#BoundArguments.kwargs), [`pallas_call`](../../pallas/block.md#pallas_call), [`NormalizationVjp`](base.md#NormalizationVjp), [`get_key`](pallas_triton_vjp_config.md#get_key), [`Config`](pallas_triton_vjp.md#Config.Config), [`get_heuristics_config`](pallas_triton_vjp_config.md#get_heuristics_config), [`vmap_axis_sizes`](../op.md#BoundArguments.vmap_axis_sizes), [`canonicalize_shape`](pallas_triton_config.md#canonicalize_shape), [`Key`](pallas_triton_vjp.md#Key.Key), [`canonicalize_shape_3d`](pallas_triton_config.md#canonicalize_shape_3d), [`Residuals`](pallas_triton_vjp.md#Residuals.Residuals), [`block_n`](pallas_triton_vjp_config.md#Config.block_n), [`_NUM_REGISTERS_PER_SM`](pallas_triton_vjp.md#_NUM_REGISTERS_PER_SM), [`block_m`](pallas_triton_vjp_config.md#Config.block_m), [`num_warps`](pallas_triton_vjp_config.md#Config.num_warps), [`_normalization_vjp_kernel`](pallas_triton_vjp.md#_normalization_vjp_kernel)
- used by: [`_get_heuristics_config`](../op.md#Op._get_heuristics_config), [`_get_autotuning_configs`](../op.md#Op._get_autotuning_configs), [`supported_on`](../op.md#Op.supported_on), [`_get_autotuning_cache_key`](../op.md#Op._get_autotuning_cache_key), [`NormalizationVjp`](base.md#NormalizationVjp), [`_fwd`](base.md#NormalizationVjp._fwd), [`_IMPLEMENTATIONS_VJP`](api_test.md#_IMPLEMENTATIONS_VJP._IMPLEMENTATIONS_VJP), [`__post_init__`](pallas_triton.md#PallasTritonNormalization.__post_init__)

## Functions
- `_normalization_vjp_kernel(dout_ref, x_ref, scale_ref, mean_ref, rstddev_ref, dx_ref, dscale_ref, doffset_ref, *, scale_offset)` — [`L42`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_vjp.py#L42) — Normalization VJP kernel.

## Module values
- `Config` — [`L37`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_vjp.py#L37)
- `Key` — [`L38`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_vjp.py#L38)
- `Residuals` — [`L39`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_vjp.py#L39)
- `_NUM_REGISTERS_PER_SM` — [`L34`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton_vjp.py#L34)

