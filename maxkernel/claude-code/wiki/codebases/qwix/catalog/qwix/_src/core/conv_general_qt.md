---
title: 'Module: qwix/_src/core/conv_general_qt.py'
type: catalog
provenance: extracted
module: qwix/_src/core/conv_general_qt.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix._src.core.conv_general_qt`/
symbols:
  conv_general_qt_bwd: conv_general_qt_bwd().
  conv_general_qt_fwd._quantize_operand: conv_general_qt_fwd()._quantize_operand().
  conv_general_qt_fwd: conv_general_qt_fwd().
  conv_general_qt: conv_general_qt().
  ConvGeneralQtConfig: ConvGeneralQtConfig#
  _compute_dilated_shape: _compute_dilated_shape().
  ConvGeneralQtConfig.lhs_qtype: ConvGeneralQtConfig#lhs_qtype.
  ConvGeneralQtConfig.rhs_qtype: ConvGeneralQtConfig#rhs_qtype.
  ConvGeneralQtConfig.dlhs_grad_qtype: ConvGeneralQtConfig#dlhs_grad_qtype.
  ConvGeneralQtConfig.drhs_grad_qtype: ConvGeneralQtConfig#drhs_grad_qtype.
  _conv_spec_transpose: _conv_spec_transpose.
  _conv_general_vjp_lhs_padding: _conv_general_vjp_lhs_padding().
  _conv_general_vjp_rhs_padding: _conv_general_vjp_rhs_padding().
  ConvGeneralQtConfig.lhs_calibration_method: ConvGeneralQtConfig#lhs_calibration_method.
  ConvGeneralQtConfig.rhs_calibration_method: ConvGeneralQtConfig#rhs_calibration_method.
  ConvGeneralQtConfig.lhs_collect_quant_stat: ConvGeneralQtConfig#lhs_collect_quant_stat.
  ConvGeneralQtConfig.rhs_collect_quant_stat: ConvGeneralQtConfig#rhs_collect_quant_stat.
  ConvGeneralQtConfig.lhs_disable_channelwise_axes: ConvGeneralQtConfig#lhs_disable_channelwise_axes.
  ConvGeneralQtConfig.rhs_disable_channelwise_axes: ConvGeneralQtConfig#rhs_disable_channelwise_axes.
  ConvGeneralQtConfig.dlhs_grad_calibration_method: ConvGeneralQtConfig#dlhs_grad_calibration_method.
  ConvGeneralQtConfig.drhs_grad_calibration_method: ConvGeneralQtConfig#drhs_grad_calibration_method.
  ConvGeneralQtConfig.dlhs_grad_disable_channelwise_axes: ConvGeneralQtConfig#dlhs_grad_disable_channelwise_axes.
  ConvGeneralQtConfig.drhs_grad_disable_channelwise_axes: ConvGeneralQtConfig#drhs_grad_disable_channelwise_axes.
  _apply_fwd_scale_to_g: _apply_fwd_scale_to_g().
  _conv_sdims: _conv_sdims.
---
# Module: [`qwix/_src/core/conv_general_qt.py`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general_qt.py)

## Classes
### `ConvGeneralQtConfig`
- def: [`qwix/_src/core/conv_general_qt.py:31`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general_qt.py#L31) — documented in [qwix-_src-core-conv_general_qt](../../../../concepts/qwix-_src-core-conv_general_qt.md)
- doc: Configuration for conv_general_qt.
- signature: `class ConvGeneralQtConfig:`
- members:
  - `dlhs_grad_calibration_method` — [`L47`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general_qt.py#L47)
  - `dlhs_grad_disable_channelwise_axes` — [`L49`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general_qt.py#L49)
  - `dlhs_grad_qtype` — [`L45`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general_qt.py#L45) — documented in [qwix-_src-core-conv_general_qt](../../../../concepts/qwix-_src-core-conv_general_qt.md)
  - `drhs_grad_calibration_method` — [`L48`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general_qt.py#L48)
  - `drhs_grad_disable_channelwise_axes` — [`L50`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general_qt.py#L50)
  - `drhs_grad_qtype` — [`L46`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general_qt.py#L46) — documented in [qwix-_src-core-conv_general_qt](../../../../concepts/qwix-_src-core-conv_general_qt.md)
  - `lhs_calibration_method` — [`L37`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general_qt.py#L37)
  - `lhs_collect_quant_stat` — [`L39`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general_qt.py#L39)
  - `lhs_disable_channelwise_axes` — [`L41`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general_qt.py#L41)
  - `lhs_qtype` — [`L35`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general_qt.py#L35) — documented in [qwix-_src-core-conv_general_qt](../../../../concepts/qwix-_src-core-conv_general_qt.md)
  - `rhs_calibration_method` — [`L38`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general_qt.py#L38)
  - `rhs_collect_quant_stat` — [`L40`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general_qt.py#L40)
  - `rhs_disable_channelwise_axes` — [`L42`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general_qt.py#L42)
  - `rhs_qtype` — [`L36`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general_qt.py#L36) — documented in [qwix-_src-core-conv_general_qt](../../../../concepts/qwix-_src-core-conv_general_qt.md)
- used by: [`_create_conv_general_qt_config`](../providers/qt.md#QtProvider._create_conv_general_qt_config), [`conv_general_qt_bwd`](conv_general_qt.md#conv_general_qt_bwd), [`_quantize_operand`](conv_general_qt.md#conv_general_qt_fwd._quantize_operand), [`conv_general_qt_fwd`](conv_general_qt.md#conv_general_qt_fwd), [`conv_general_qt`](conv_general_qt.md#conv_general_qt)  (2 test-only)

## Functions
- `_apply_fwd_scale_to_g(scale: jax.Array, g: jax.Array, g_axis: int)` — [`L120`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general_qt.py#L120) — Applies the scale from the forward pass to the backward gradient.
- `_compute_dilated_shape(shape: Sequence[int], dilation: Sequence[int])` — [`L62`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general_qt.py#L62) — Computes the shape of an array after dilation.
- `_conv_general_vjp_lhs_padding(in_shape, window_dimensions, window_strides, out_shape, padding, lhs_dilation, rhs_dilation)` — [`L73`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general_qt.py#L73) — Computes padding for the VJP wrt the LHS of a convolution. — documented in [qwix-_src-core-conv_general_qt](../../../../concepts/qwix-_src-core-conv_general_qt.md)
- `_conv_general_vjp_rhs_padding(in_shape, window_dimensions, window_strides, out_shape, padding, lhs_dilation, rhs_dilation)` — [`L96`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general_qt.py#L96) — Computes padding for the VJP wrt the RHS of a convolution. — documented in [qwix-_src-core-conv_general_qt](../../../../concepts/qwix-_src-core-conv_general_qt.md)
- `_quantize_operand(operand: jax.Array, *, for_lhs: bool)` — [`L159`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general_qt.py#L159) — Quantizes a single operand for the forward pass if configured to do so. — documented in [qwix-_src-core-conv_general_qt](../../../../concepts/qwix-_src-core-conv_general_qt.md)
- `conv_general_qt(lhs: jax.Array, rhs: jax.Array, config: ConvGeneralQtConfig, window_strides: Sequence[int], padding: str | Sequence[tuple[int, int]], lhs_dilation: Sequence[int] | None = None, rhs_dilation: Sequence[int] | None = None, dimension_numbers: jax.lax.ConvDimensionNumbers | None = None, feature_group_count: int = 1, batch_group_count: int = 1, out_sharding=None)` — [`L345`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general_qt.py#L345) — Quantized conv_general using a simple, hashable config dataclass. — documented in [qwix-_src-core-conv_general_qt](../../../../concepts/qwix-_src-core-conv_general_qt.md)
- `conv_general_qt_bwd(config: ConvGeneralQtConfig, window_strides: Sequence[int], padding: str | Sequence[tuple[int, int]], lhs_dilation: Sequence[int] | None, rhs_dilation: Sequence[int] | None, dimension_numbers: jax.lax.ConvDimensionNumbers | None, feature_group_count: int, batch_group_count: int, out_sharding, res: tuple[qarray.MaybeQArray, qarray.MaybeQArray], g: jax.Array)` — [`L213`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general_qt.py#L213) — Backward pass for conv_general_qt custom VJP. — documented in [qwix-_src-core-conv_general_qt](../../../../concepts/qwix-_src-core-conv_general_qt.md)
- `conv_general_qt_fwd(lhs: jax.Array, rhs: jax.Array, config: ConvGeneralQtConfig, window_strides: Sequence[int], padding: str | Sequence[tuple[int, int]], lhs_dilation: Sequence[int] | None, rhs_dilation: Sequence[int] | None, dimension_numbers: jax.lax.ConvDimensionNumbers | None, feature_group_count: int, batch_group_count: int, out_sharding=None)` — [`L141`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general_qt.py#L141) — Forward pass for conv_general_qt custom VJP. — documented in [qwix-_src-core-conv_general_qt](../../../../concepts/qwix-_src-core-conv_general_qt.md)

## Module values
- `_conv_sdims` — [`L59`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general_qt.py#L59)
- `_conv_spec_transpose` — [`L55`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general_qt.py#L55)

