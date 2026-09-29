---
title: 'Module: jax/_src/lax/convolution.py'
type: catalog
provenance: extracted
module: jax/_src/lax/convolution.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.lax.convolution`/
symbols:
  conv_general_dilated: conv_general_dilated().
  conv_transpose: conv_transpose().
  conv_general_dilated_p: conv_general_dilated_p.
  _conv_general_dilated_transpose_lhs: _conv_general_dilated_transpose_lhs().
  _conv_general_dilated_shape_rule: _conv_general_dilated_shape_rule().
  _conv_general_dilated_lower: _conv_general_dilated_lower().
  _conv_general_dilated_batch_rule: _conv_general_dilated_batch_rule().
  _conv_general_dilated_transpose_rhs: _conv_general_dilated_transpose_rhs().
  _conv_general_dilated_sharding_rule: _conv_general_dilated_sharding_rule().
  _reshape_axis_into: _reshape_axis_into().
  _reshape_axis_out_of: _reshape_axis_out_of().
  _conv_general_dilated_batch_rule.get_out_sharding: _conv_general_dilated_batch_rule().get_out_sharding().
  ConvDimensionNumbers: ConvDimensionNumbers#
  ConvDimensionNumbers.rhs_spec: ConvDimensionNumbers#rhs_spec.
  _conv_general_dilated_dtype_rule: _conv_general_dilated_dtype_rule().
  _complex_mul: _complex_mul().
  conv: conv().
  conv_with_general_padding: conv_with_general_padding().
  conv_dimension_numbers: conv_dimension_numbers().
  conv_shape_tuple: conv_shape_tuple().
  _conv_general_vjp_lhs_padding: _conv_general_vjp_lhs_padding().
  ConvGeneralDilatedDimensionNumbers: ConvGeneralDilatedDimensionNumbers.
  _conv_general_vjp_rhs_padding: _conv_general_vjp_rhs_padding().
  conv_general_permutations: conv_general_permutations().
  conv_general_shape_tuple: conv_general_shape_tuple().
  conv_transpose_shape_tuple: conv_transpose_shape_tuple().
  _conv_sdims: _conv_sdims.
  _conv_general_dilated_lower.prep_one_pad: _conv_general_dilated_lower().prep_one_pad().
  ConvDimensionNumbers.lhs_spec: ConvDimensionNumbers#lhs_spec.
  _conv_transpose_padding: _conv_transpose_padding().
  _conv_spec_transpose: _conv_spec_transpose.
  _flip_axes: _flip_axes().
  _real_dtype: _real_dtype.
  conv_general_permutations.getperm: conv_general_permutations().getperm().
  ConvDimensionNumbers.out_spec: ConvDimensionNumbers#out_spec.
---
# Module: [`jax/_src/lax/convolution.py`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py)

## Classes
### `ConvDimensionNumbers`  ·  implements/extends NamedTuple
- def: [`jax/_src/lax/convolution.py:38`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L38)
- doc: Describes batch, spatial, and feature dimensions of a convolution.
- signature: `class ConvDimensionNumbers(NamedTuple):`
- members:
  - `lhs_spec` — [`L49`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L49)
  - `out_spec` — [`L51`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L51)
  - `rhs_spec` — [`L50`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L50)
- used by: [`conv_transpose`](convolution.md#conv_transpose), [`_conv_general_dilated_transpose_lhs`](convolution.md#_conv_general_dilated_transpose_lhs), [`_conv_general_dilated_lower`](convolution.md#_conv_general_dilated_lower), [`_conv_general_dilated_shape_rule`](convolution.md#_conv_general_dilated_shape_rule), [`_conv_general_dilated_transpose_rhs`](convolution.md#_conv_general_dilated_transpose_rhs), [`conv_dimension_numbers`](convolution.md#conv_dimension_numbers), [`ConvGeneralDilatedDimensionNumbers`](convolution.md#ConvGeneralDilatedDimensionNumbers)

## Functions
- `_complex_mul(mul, x, y)` — [`L754`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L754)
- `_conv_general_dilated_batch_rule(axis_data, batched_args, batch_dims, *, window_strides, padding, lhs_dilation, rhs_dilation, dimension_numbers, feature_group_count, batch_group_count, precision, preferred_element_type, out_sharding, **unused_kwargs)` — [`L600`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L600)
- `_conv_general_dilated_dtype_rule(lhs, rhs, *, window_strides, padding, lhs_dilation, rhs_dilation, dimension_numbers, preferred_element_type, **unused_kwargs)` — [`L479`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L479)
- `_conv_general_dilated_lower(ctx, lhs, rhs, *, window_strides, padding, lhs_dilation, rhs_dilation, dimension_numbers, feature_group_count, batch_group_count, precision, preferred_element_type, out_sharding, expand_complex_convolutions=False, **unused_kwargs)` — [`L775`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L775)
- `_conv_general_dilated_shape_rule(lhs: core.ShapedArray, rhs: core.ShapedArray, *, window_strides, padding, lhs_dilation, rhs_dilation, dimension_numbers, feature_group_count, batch_group_count, **unused_kwargs)` — [`L391`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L391)
- `_conv_general_dilated_sharding_rule(lhs: core.ShapedArray, rhs: core.ShapedArray, *, window_strides, padding, lhs_dilation, rhs_dilation, dimension_numbers, feature_group_count, batch_group_count, out_sharding, **unused_kwargs)` — [`L457`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L457)
- `_conv_general_dilated_transpose_lhs(g, lhs, rhs, *, window_strides, padding, lhs_dilation, rhs_dilation, dimension_numbers, feature_group_count, batch_group_count, precision, preferred_element_type, out_sharding)` — [`L519`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L519)
- `_conv_general_dilated_transpose_rhs(g, lhs, rhs, *, window_strides, padding, lhs_dilation, rhs_dilation, dimension_numbers: ConvDimensionNumbers, feature_group_count: int, batch_group_count: int, precision, preferred_element_type, out_sharding)` — [`L562`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L562)
- `_conv_general_vjp_lhs_padding(in_shape, window_dimensions, window_strides, out_shape, padding, lhs_dilation, rhs_dilation)` — [`L1020`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L1020)
- `_conv_general_vjp_rhs_padding(in_shape, window_dimensions, window_strides, out_shape, padding, lhs_dilation, rhs_dilation)` — [`L1032`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L1032)
- `_conv_transpose_padding(k, s, padding)` — [`L260`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L260) — Calculate before and after padding for a dim of transposed convolution.
- `_flip_axes(x, axes)` — [`L289`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L289) — Flip ndarray 'x' along each axis specified in axes tuple.
- `_reshape_axis_into(src, dst, x)` — [`L879`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L879)
- `_reshape_axis_out_of(src, size1, x)` — [`L890`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L890)
- `conv(lhs: Array, rhs: Array, window_strides: Sequence[int], padding: str, precision: lax.PrecisionLike = None, preferred_element_type: DTypeLike | None = None)` — [`L194`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L194) — Convenience wrapper around `conv_general_dilated`.
- `conv_dimension_numbers(lhs_shape, rhs_shape, dimension_numbers)` — [`L948`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L948) — Converts convolution `dimension_numbers` to a `ConvDimensionNumbers`.
- `conv_general_dilated(lhs: Array, rhs: Array, window_strides: Sequence[int], padding: str | Sequence[tuple[int, int]], lhs_dilation: Sequence[int] | None = None, rhs_dilation: Sequence[int] | None = None, dimension_numbers: ConvGeneralDilatedDimensionNumbers = None, feature_group_count: int = 1, batch_group_count: int = 1, precision: lax.PrecisionLike = None, preferred_element_type: DTypeLike | None = None, out_sharding: NamedSharding | P | None = None)` — [`L61`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L61) — General n-dimensional convolution operator, with optional dilation.
- `conv_general_permutations(dimension_numbers)` — [`L991`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L991) — Utility for convolution dimension permutations relative to Conv HLO.
- `conv_general_shape_tuple(lhs_shape, rhs_shape, window_strides, padding, dimension_numbers)` — [`L921`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L921)
- `conv_shape_tuple(lhs_shape, rhs_shape, strides, pads, batch_group_count=1)` — [`L898`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L898) — Compute the shape tuple of a conv given input shapes in canonical order.
- `conv_transpose(lhs: Array, rhs: Array, strides: Sequence[int], padding: str | Sequence[tuple[int, int]], rhs_dilation: Sequence[int] | None = None, dimension_numbers: ConvGeneralDilatedDimensionNumbers = None, transpose_kernel: bool = False, precision: lax.PrecisionLike = None, preferred_element_type: DTypeLike | None = None, use_consistent_padding: bool = False)` — [`L296`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L296) — Convenience wrapper for calculating the N-d convolution "transpose".
- `conv_transpose_shape_tuple(lhs_shape, rhs_shape, window_strides, padding, dimension_numbers)` — [`L930`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L930)
- `conv_with_general_padding(lhs: Array, rhs: Array, window_strides: Sequence[int], padding: str | Sequence[tuple[int, int]], lhs_dilation: Sequence[int] | None, rhs_dilation: Sequence[int] | None, precision: lax.PrecisionLike = None, preferred_element_type: DTypeLike | None = None)` — [`L220`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L220) — Convenience wrapper around `conv_general_dilated`.
- `get_out_sharding(axis)` — [`L649`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L649)
- `getperm(spec, charpair)` — [`L1010`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L1010)
- `prep_one_pad(pad_lo_hi: tuple[core.DimSize, core.DimSize])` — [`L842`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L842)

## Module values
- `ConvGeneralDilatedDimensionNumbers` — [`L53`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L53)
- `_conv_sdims` — [`L490`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L490)
- `_conv_spec_transpose` — [`L489`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L489)
- `_real_dtype` — [`L773`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L773)
- `conv_general_dilated_p` — [`L742`](../../../../../../../raw/code/jax/jax/_src/lax/convolution.py#L742)

