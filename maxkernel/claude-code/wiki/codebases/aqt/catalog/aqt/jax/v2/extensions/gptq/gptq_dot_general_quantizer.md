---
title: 'Module: aqt/jax/v2/extensions/gptq/gptq_dot_general_quantizer.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/extensions/gptq/gptq_dot_general_quantizer.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.extensions.gptq.gptq_dot_general_quantizer`/
symbols:
  _get_quant_mode: _get_quant_mode().
  _get_divisible_blocksize: _get_divisible_blocksize().
  _reshape_kernel_for_gptq: _reshape_kernel_for_gptq().
  _recover_kernel_from_gptq_result: _recover_kernel_from_gptq_result().
  _init_hinv_for_calibration: _init_hinv_for_calibration().
  GptqHinvCollector: GptqHinvCollector#
  GptqHinvCollector.quant_collection: GptqHinvCollector#quant_collection.
  GptqHinvCollector.sharding_axes: GptqHinvCollector#sharding_axes.
  GptqHinvCollector.perc_damp: GptqHinvCollector#perc_damp.
  GptqHinvCollector.act_order: GptqHinvCollector#act_order.
  GptqHinvCollector.__call__: GptqHinvCollector#__call__().
  GptqDotGeneralQuantizer: GptqDotGeneralQuantizer#
  GptqDotGeneralQuantizer.sharding_axes: GptqDotGeneralQuantizer#sharding_axes.
  GptqDotGeneralQuantizer.quant_collection: GptqDotGeneralQuantizer#quant_collection.
  GptqDotGeneralQuantizer.perc_damp: GptqDotGeneralQuantizer#perc_damp.
  GptqDotGeneralQuantizer.act_order: GptqDotGeneralQuantizer#act_order.
  GptqDotGeneralQuantizer.is_rhs_kernel: GptqDotGeneralQuantizer#is_rhs_kernel.
  GptqDotGeneralQuantizer.calibrate: GptqDotGeneralQuantizer#calibrate().
  GptqDotGeneralQuantizer.swap_lhs_and_rhs: GptqDotGeneralQuantizer#swap_lhs_and_rhs().
---
# Module: [`aqt/jax/v2/extensions/gptq/gptq_dot_general_quantizer.py`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/extensions/gptq/gptq_dot_general_quantizer.py)

## Classes
### `GptqDotGeneralQuantizer`
- def: [`aqt/jax/v2/extensions/gptq/gptq_dot_general_quantizer.py:234`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/extensions/gptq/gptq_dot_general_quantizer.py#L234)
- doc: GPTQ dot_general quantizer.
- signature: `class GptqDotGeneralQuantizer(aqt_dot_general.DefaultDotGeneralQuantizer):`
- members:
  - `calibrate(self, lhs: jax.Array, rhs: jax.Array, dimension_numbers: None | jax.lax.DotDimensionNumbers, lhs_mode: None | aqt_dot_general.CalibrationMode, rhs_mode: None | aqt_dot_general.CalibrationMode)` — [`L251`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/extensions/gptq/gptq_dot_general_quantizer.py#L251) — GPTQ calibration.
  - `swap_lhs_and_rhs(self)` — [`L390`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/extensions/gptq/gptq_dot_general_quantizer.py#L390) — Swaps lhs and rhs configuration.
  - `act_order` — [`L246`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/extensions/gptq/gptq_dot_general_quantizer.py#L246)
  - `is_rhs_kernel` — [`L249`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/extensions/gptq/gptq_dot_general_quantizer.py#L249)
  - `perc_damp` — [`L242`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/extensions/gptq/gptq_dot_general_quantizer.py#L242)
  - `quant_collection` — [`L239`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/extensions/gptq/gptq_dot_general_quantizer.py#L239)
  - `sharding_axes` — [`L237`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/extensions/gptq/gptq_dot_general_quantizer.py#L237)

### `GptqHinvCollector`
- def: [`aqt/jax/v2/extensions/gptq/gptq_dot_general_quantizer.py:124`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/extensions/gptq/gptq_dot_general_quantizer.py#L124)
- doc: GPTQ hinv collector module.
- signature: `class GptqHinvCollector(nn.Module):`
- members:
  - `__call__(self, x: jnp.ndarray, ca: Sequence[utils.AxisIdx], quant_mode: utils.QuantMode)` — [`L144`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/extensions/gptq/gptq_dot_general_quantizer.py#L144) — Collects Inverse of the hessian.
  - `act_order` — [`L141`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/extensions/gptq/gptq_dot_general_quantizer.py#L141)
  - `perc_damp` — [`L137`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/extensions/gptq/gptq_dot_general_quantizer.py#L137)
  - `quant_collection` — [`L132`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/extensions/gptq/gptq_dot_general_quantizer.py#L132)
  - `sharding_axes` — [`L134`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/extensions/gptq/gptq_dot_general_quantizer.py#L134)

## Functions
- `_get_divisible_blocksize(dim: int, blocksize_top: int)` — [`L38`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/extensions/gptq/gptq_dot_general_quantizer.py#L38) — Returns the blocksize which could divide the given dimension.
- `_get_quant_mode(context: None | utils.Context)` — [`L34`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/extensions/gptq/gptq_dot_general_quantizer.py#L34)
- `_init_hinv_for_calibration(inputs, perc_damp=0.01)` — [`L116`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/extensions/gptq/gptq_dot_general_quantizer.py#L116) — Initializes hinv with a damping term.
- `_recover_kernel_from_gptq_result(kernel: jnp.ndarray, ca: Sequence[utils.AxisIdx], sharding_axes: None | str, act_order: bool, perm: None | Sequence[utils.AxisIdx], kernel_dtype: jnp.dtype, kernel_feature_grouped_shape: Sequence[int])` — [`L91`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/extensions/gptq/gptq_dot_general_quantizer.py#L91) — Recovers original kernel shape.
- `_reshape_kernel_for_gptq(kernel: jnp.ndarray, ca: Sequence[utils.AxisIdx], sharding_axes: None | str, act_order: bool, perm: None | Sequence[utils.AxisIdx], blocksize: int)` — [`L59`](../../../../../../../../../raw/code/aqt/aqt/jax/v2/extensions/gptq/gptq_dot_general_quantizer.py#L59) — Reshapes kernel to (features / blocksize, blocksize, -1) and potentially reshard.

