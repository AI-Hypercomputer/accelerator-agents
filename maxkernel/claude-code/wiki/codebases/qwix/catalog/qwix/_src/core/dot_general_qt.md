---
title: 'Module: qwix/_src/core/dot_general_qt.py'
type: catalog
provenance: extracted
module: qwix/_src/core/dot_general_qt.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix._src.core.dot_general_qt`/
symbols:
  dot_general_qt_bwd._compute_gradient_for_operand: dot_general_qt_bwd()._compute_gradient_for_operand().
  dot_general_qt: dot_general_qt().
  dot_general_qt_fwd: dot_general_qt_fwd().
  dot_general_qt_bwd: dot_general_qt_bwd().
  DotGeneralQtConfig: DotGeneralQtConfig#
  DotGeneralQtConfig.lhs_qtype: DotGeneralQtConfig#lhs_qtype.
  DotGeneralQtConfig.rhs_qtype: DotGeneralQtConfig#rhs_qtype.
  _get_residual_for_backward: _get_residual_for_backward().
  DotGeneralQtConfig.tile_size: DotGeneralQtConfig#tile_size.
  DotGeneralQtConfig.dlhs_grad_qtype: DotGeneralQtConfig#dlhs_grad_qtype.
  DotGeneralQtConfig.drhs_grad_qtype: DotGeneralQtConfig#drhs_grad_qtype.
  dot_general_qt_fwd_bwd: dot_general_qt_fwd_bwd().
  DotGeneralQtConfig.sparsity_rule: DotGeneralQtConfig#sparsity_rule.
  DotGeneralQtConfig.drhs_tile_size: DotGeneralQtConfig#drhs_tile_size.
  _apply_rhs_scale_to_lhs: _apply_rhs_scale_to_lhs().
  DotGeneralQtConfig.dlhs_stochastic_rounding_noise_fn: DotGeneralQtConfig#dlhs_stochastic_rounding_noise_fn.
  DotGeneralQtConfig.drhs_stochastic_rounding_noise_fn: DotGeneralQtConfig#drhs_stochastic_rounding_noise_fn.
  DotGeneralQtConfig.lhs_calibration_method: DotGeneralQtConfig#lhs_calibration_method.
  DotGeneralQtConfig.rhs_calibration_method: DotGeneralQtConfig#rhs_calibration_method.
  DotGeneralQtConfig.dlhs_tile_size: DotGeneralQtConfig#dlhs_tile_size.
  DotGeneralQtConfig.use_original_residuals: DotGeneralQtConfig#use_original_residuals.
  DotGeneralQtConfig.tree_flatten: DotGeneralQtConfig#tree_flatten().
  _update_dimension_numbers_for_backward: _update_dimension_numbers_for_backward().
  DotGeneralQtConfig.lhs_collect_quant_stat: DotGeneralQtConfig#lhs_collect_quant_stat.
  DotGeneralQtConfig.rhs_collect_quant_stat: DotGeneralQtConfig#rhs_collect_quant_stat.
  DotGeneralQtConfig.lhs_disable_channelwise_axes: DotGeneralQtConfig#lhs_disable_channelwise_axes.
  DotGeneralQtConfig.rhs_disable_channelwise_axes: DotGeneralQtConfig#rhs_disable_channelwise_axes.
  DotGeneralQtConfig.dlhs_grad_calibration_method: DotGeneralQtConfig#dlhs_grad_calibration_method.
  DotGeneralQtConfig.dlhs_grad_disable_channelwise_axes: DotGeneralQtConfig#dlhs_grad_disable_channelwise_axes.
  DotGeneralQtConfig.drhs_grad_calibration_method: DotGeneralQtConfig#drhs_grad_calibration_method.
  DotGeneralQtConfig.drhs_grad_disable_channelwise_axes: DotGeneralQtConfig#drhs_grad_disable_channelwise_axes.
  DotGeneralQtConfig.dlhs_residual_qtype: DotGeneralQtConfig#dlhs_residual_qtype.
  DotGeneralQtConfig.drhs_residual_qtype: DotGeneralQtConfig#drhs_residual_qtype.
  _ranges_like: _ranges_like().
  DotGeneralQtConfig.disable_gradient_clipping: DotGeneralQtConfig#disable_gradient_clipping.
  DotGeneralQtConfig.dlhs_residual_calibration_method: DotGeneralQtConfig#dlhs_residual_calibration_method.
  DotGeneralQtConfig.dlhs_residual_disable_channelwise_axes: DotGeneralQtConfig#dlhs_residual_disable_channelwise_axes.
  DotGeneralQtConfig.drhs_residual_calibration_method: DotGeneralQtConfig#drhs_residual_calibration_method.
  DotGeneralQtConfig.drhs_residual_disable_channelwise_axes: DotGeneralQtConfig#drhs_residual_disable_channelwise_axes.
  DotGeneralQtConfig.tree_unflatten: DotGeneralQtConfig#tree_unflatten().
---
# Module: [`qwix/_src/core/dot_general_qt.py`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py)

## Classes
### `DotGeneralQtConfig`
- def: [`qwix/_src/core/dot_general_qt.py:33`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L33) — documented in [qwix-_src-core-dot_general_qt](../../../../concepts/qwix-_src-core-dot_general_qt.md)
- doc: Configuration for dot_general_qt.
- signature: `class DotGeneralQtConfig:`
- members:
  - `tree_flatten(self)` — [`L87`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L87)
  - `tree_unflatten(cls, aux_data, children)` — [`L98`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L98)
  - `disable_gradient_clipping` — [`L65`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L65)
  - `dlhs_grad_calibration_method` — [`L49`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L49)
  - `dlhs_grad_disable_channelwise_axes` — [`L52`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L52)
  - `dlhs_grad_qtype` — [`L48`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L48) — documented in [qwix-_src-providers-qt](../../../../concepts/qwix-_src-providers-qt.md)
  - `dlhs_residual_calibration_method` — [`L79`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L79)
  - `dlhs_residual_disable_channelwise_axes` — [`L80`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L80)
  - `dlhs_residual_qtype` — [`L78`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L78)
  - `dlhs_stochastic_rounding_noise_fn` — [`L51`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L51)
  - `dlhs_tile_size` — [`L50`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L50)
  - `drhs_grad_calibration_method` — [`L56`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L56)
  - `drhs_grad_disable_channelwise_axes` — [`L59`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L59)
  - `drhs_grad_qtype` — [`L55`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L55) — documented in [qwix-_src-providers-qt](../../../../concepts/qwix-_src-providers-qt.md)
  - `drhs_residual_calibration_method` — [`L82`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L82)
  - `drhs_residual_disable_channelwise_axes` — [`L83`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L83)
  - `drhs_residual_qtype` — [`L81`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L81)
  - `drhs_stochastic_rounding_noise_fn` — [`L58`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L58)
  - `drhs_tile_size` — [`L57`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L57) — documented in [qwix-_src-providers-qt](../../../../concepts/qwix-_src-providers-qt.md)
  - `lhs_calibration_method` — [`L40`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L40)
  - `lhs_collect_quant_stat` — [`L42`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L42)
  - `lhs_disable_channelwise_axes` — [`L44`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L44)
  - `lhs_qtype` — [`L37`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L37) — documented in [qwix-_src-core-dot_general_qt](../../../../concepts/qwix-_src-core-dot_general_qt.md)
  - `rhs_calibration_method` — [`L41`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L41)
  - `rhs_collect_quant_stat` — [`L43`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L43)
  - `rhs_disable_channelwise_axes` — [`L45`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L45)
  - `rhs_qtype` — [`L38`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L38) — documented in [qwix-_src-core-dot_general_qt](../../../../concepts/qwix-_src-core-dot_general_qt.md)
  - `sparsity_rule` — [`L85`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L85) — documented in [qwix-_src-core-sparsity](../../../../concepts/qwix-_src-core-sparsity.md)
  - `tile_size` — [`L39`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L39) — documented in [qwix-_src-providers-qt](../../../../concepts/qwix-_src-providers-qt.md)
  - `use_original_residuals` — [`L73`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L73)
- uses (calls/refs, reference-scoped): [`SparsityRule`](sparsity.md#SparsityRule), [`NoiseFn`](stochastic_rounding.md#NoiseFn)
- used by: [`_create_dot_general_qt_config`](../providers/qt.md#QtProvider._create_dot_general_qt_config), [`_compute_gradient_for_operand`](dot_general_qt.md#dot_general_qt_bwd._compute_gradient_for_operand), [`dot_general_qt`](dot_general_qt.md#dot_general_qt), [`dot_general_qt_fwd`](dot_general_qt.md#dot_general_qt_fwd), [`dot_general_qt_bwd`](dot_general_qt.md#dot_general_qt_bwd), [`_get_residual_for_backward`](dot_general_qt.md#_get_residual_for_backward), [`dot_general_qt_fwd_bwd`](dot_general_qt.md#dot_general_qt_fwd_bwd)  (9 test-only)

## Functions
- `_apply_rhs_scale_to_lhs(lhs, rhs_scale, dnums)` — [`L161`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L161) — Applies the rhs_scale to lhs. — documented in [qwix-_src-core-dot_general_qt](../../../../concepts/qwix-_src-core-dot_general_qt.md)
- `_compute_gradient_for_operand(g: jax.Array, *, for_dlhs: bool)` — [`L255`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L255) — Compute dot_general for gradient and other_fwd_operand. — documented in [qwix-_src-core-dot_general_qt](../../../../concepts/qwix-_src-core-dot_general_qt.md)
- `_get_residual_for_backward(config: DotGeneralQtConfig, operand_in: jax.Array, operand_qt: qarray.MaybeQArray)` — [`L171`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L171) — Returns the residual to be used in the backward pass. — documented in [qwix-_src-core-dot_general_qt](../../../../concepts/qwix-_src-core-dot_general_qt.md)
- `_ranges_like(*xs)` — [`L109`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L109)
- `_update_dimension_numbers_for_backward(fwd_dimension_numbers: jax.lax.DotDimensionNumbers, fwd_ndims: tuple[int, int], *, for_dlhs: bool)` — [`L116`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L116) — Generates a new dimension number for backward pass.
- `dot_general_qt(lhs: jax.Array, rhs: jax.Array, dimension_numbers: jax.lax.DotDimensionNumbers, config: DotGeneralQtConfig)` — [`L359`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L359) — Quantized dot_general with backpropagation support. — documented in [qwix-_src-core-dot_general_qt](../../../../concepts/qwix-_src-core-dot_general_qt.md)
- `dot_general_qt_bwd(fwd_dimension_numbers: jax.lax.DotDimensionNumbers, residuals: tuple[jax.Array, jax.Array, qarray.MaybeQArray, qarray.MaybeQArray, dict[str, jax.Array] | None, dict[str, jax.Array] | None, DotGeneralQtConfig,], g: jax.Array)` — [`L239`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L239) — Backward pass for dot_general_qt custom VJP. — documented in [qwix-_src-core-dot_general_qt](../../../../concepts/qwix-_src-core-dot_general_qt.md)
- `dot_general_qt_fwd(lhs: jax.Array, rhs: jax.Array, lhs_calibration: dict[str, jax.Array] | None, rhs_calibration: dict[str, jax.Array] | None, dimension_numbers: jax.lax.DotDimensionNumbers, config: DotGeneralQtConfig)` — [`L203`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L203) — Forward pass for dot_general_qt custom VJP. — documented in [qwix-_src-core-dot_general_qt](../../../../concepts/qwix-_src-core-dot_general_qt.md)
- `dot_general_qt_fwd_bwd(lhs: jax.Array, rhs: jax.Array, lhs_calibration: dict[str, jax.Array] | None, rhs_calibration: dict[str, jax.Array] | None, dimension_numbers: jax.lax.DotDimensionNumbers, config: DotGeneralQtConfig)` — [`L341`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general_qt.py#L341) — dot_general custom VJP.

