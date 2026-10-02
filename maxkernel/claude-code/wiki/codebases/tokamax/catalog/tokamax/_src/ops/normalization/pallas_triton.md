---
title: 'Module: tokamax/_src/ops/normalization/pallas_triton.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/normalization/pallas_triton.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.normalization.pallas_triton`/
symbols:
  PallasTritonNormalization._fwd: PallasTritonNormalization#_fwd().
  PallasTritonNormalization._get_autotuning_configs: PallasTritonNormalization#_get_autotuning_configs().
  PallasTritonNormalization._get_heuristics_config: PallasTritonNormalization#_get_heuristics_config().
  PallasTritonNormalization._get_autotuning_cache_key: PallasTritonNormalization#_get_autotuning_cache_key().
  PallasTritonNormalization: PallasTritonNormalization#
  Config.Config: Config.Config.
  PallasTritonNormalization.__post_init__: PallasTritonNormalization#__post_init__().
  Key.Key: Key.Key.
  FusedInputArray: FusedInputArray.
  PallasTritonNormalization.config_cls: PallasTritonNormalization#config_cls.
  _NUM_REGISTERS_PER_SM: _NUM_REGISTERS_PER_SM.
  PallasTritonNormalization.input_output_alias: PallasTritonNormalization#input_output_alias.
  _normalization_kernel: _normalization_kernel().
  PallasTritonNormalization.supports_symbolic_shapes: PallasTritonNormalization#supports_symbolic_shapes.
  PallasTritonNormalization.supported_on: PallasTritonNormalization#supported_on().
---
# Module: [`tokamax/_src/ops/normalization/pallas_triton.py`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton.py)

## Classes
### `PallasTritonNormalization`  ·  implements/extends Normalization
- def: [`tokamax/_src/ops/normalization/pallas_triton.py:84`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton.py#L84)
- doc: Pallas-Triton normalization op.
- signature: `class PallasTritonNormalization(base.Normalization[Config, Key]):`
- members:
  - `supported_on(self, device: jax.Device)` — [`L227`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton.py#L227)
  - `config_cls` — [`L87`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton.py#L87)
  - `input_output_alias` — [`L90`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton.py#L90)
  - `supports_symbolic_shapes` — [`L88`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton.py#L88)
- protocol/private: `__post_init__`[`L92`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton.py#L92), `_fwd`[`L98`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton.py#L98), `_get_autotuning_cache_key`[`L195`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton.py#L195), `_get_autotuning_configs`[`L200`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton.py#L200), `_get_heuristics_config`[`L189`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton.py#L189)
- uses (calls/refs, reference-scoped): [`BoundArguments`](../op.md#BoundArguments), [`args`](../op.md#BoundArguments.args), [`vjp`](../op.md#Op.vjp), [`Normalization`](base.md#Normalization), [`kwargs`](../op.md#BoundArguments.kwargs), [`get_heuristics_config`](pallas_triton_config.md#get_heuristics_config), [`pallas_call`](../../pallas/block.md#pallas_call), [`Config`](pallas_triton.md#Config.Config), [`PallasTritonNormalizationVjp`](pallas_triton_vjp.md#PallasTritonNormalizationVjp), [`get_key`](pallas_triton_config.md#get_key), [`vmap_axis_sizes`](../op.md#BoundArguments.vmap_axis_sizes), [`canonicalize_shape`](pallas_triton_config.md#canonicalize_shape), [`Residuals`](base.md#Residuals.Residuals), [`block_n`](pallas_triton_config.md#Config.block_n), [`Key`](pallas_triton.md#Key.Key), [`block_m`](pallas_triton_config.md#Config.block_m), [`canonicalize_shape_3d`](pallas_triton_config.md#canonicalize_shape_3d), [`num_warps`](pallas_triton_config.md#Config.num_warps), [`FusedInputArray`](pallas_triton.md#FusedInputArray), [`_NUM_REGISTERS_PER_SM`](pallas_triton.md#_NUM_REGISTERS_PER_SM), [`_normalization_kernel`](pallas_triton.md#_normalization_kernel)
- used by: [`_get_heuristics_config`](../op.md#Op._get_heuristics_config), [`_get_autotuning_configs`](../op.md#Op._get_autotuning_configs), [`supported_on`](../op.md#Op.supported_on), [`_get_autotuning_cache_key`](../op.md#Op._get_autotuning_cache_key), [`Normalization`](base.md#Normalization), [`_DEFAULT_IMPLEMENTATIONS`](api.md#_DEFAULT_IMPLEMENTATIONS), [`_IMPLS`](bench.md#_IMPLS), [`_fwd`](base.md#Normalization._fwd), [`__init__`](pallas_triton_test.md#PallasTritonNormalizationTest.__init__)

## Functions
- `_normalization_kernel(x_value_refs, scale_ref, offset_ref, y_ref, mean_ref, rstddev_ref, *, x_fn, epsilon, scale_offset, subtract_mean)` — [`L41`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton.py#L41) — Normalization kernel.

## Module values
- `Config` — [`L35`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton.py#L35)
- `FusedInputArray` — [`L37`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton.py#L37)
- `Key` — [`L36`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton.py#L36)
- `_NUM_REGISTERS_PER_SM` — [`L38`](../../../../../../../../raw/code/tokamax/tokamax/_src/ops/normalization/pallas_triton.py#L38)

