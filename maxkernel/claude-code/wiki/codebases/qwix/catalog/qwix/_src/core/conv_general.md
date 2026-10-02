---
title: 'Module: qwix/_src/core/conv_general.py'
type: catalog
provenance: extracted
module: qwix/_src/core/conv_general.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix._src.core.conv_general`/
symbols:
  _fast_conv_general_dilated: _fast_conv_general_dilated().
  get_how_to_quantize: get_how_to_quantize().
  conv_general_dilated: conv_general_dilated().
  _slow_conv_general_dilated: _slow_conv_general_dilated().
  get_transpose: get_transpose().
---
# Module: [`qwix/_src/core/conv_general.py`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general.py)

## Functions
- `_fast_conv_general_dilated(lhs: qarray.MaybeQArray, rhs: qarray.MaybeQArray, window_strides: Sequence[int], padding: str | Sequence[tuple[int, int]], lhs_dilation: Sequence[int] | None = None, rhs_dilation: Sequence[int] | None = None, dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers = None, feature_group_count: int = 1, batch_group_count: int = 1, preferred_element_type: jax.typing.DTypeLike | None = None, out_sharding=None)` — [`L103`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general.py#L103) — Quantized jax.lax.conv_general_dilated. Both sides must be QArrays.
- `_slow_conv_general_dilated(lhs: qarray.MaybeQArray, rhs: qarray.MaybeQArray, window_strides: Sequence[int], padding: str | Sequence[tuple[int, int]], lhs_dilation: Sequence[int] | None = None, rhs_dilation: Sequence[int] | None = None, dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers = None, feature_group_count: int = 1, batch_group_count: int = 1, precision: jax.lax.PrecisionLike = None, preferred_element_type: jax.typing.DTypeLike | None = None, out_sharding=None)` — [`L68`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general.py#L68) — Dequantizes first then computes in floating-point types.
- `conv_general_dilated(lhs: qarray.MaybeQArray, rhs: qarray.MaybeQArray, window_strides: Sequence[int], padding: str | Sequence[tuple[int, int]], lhs_dilation: Sequence[int] | None = None, rhs_dilation: Sequence[int] | None = None, dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers = None, feature_group_count: int = 1, batch_group_count: int = 1, precision: jax.lax.PrecisionLike = None, preferred_element_type: jax.typing.DTypeLike | None = None, out_sharding=None)` — [`L190`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general.py#L190) — Computes a general dilated convolution with support for ``QArray`` inputs.
- `get_how_to_quantize(*, dimension_numbers: jax.lax.ConvDimensionNumbers, for_lhs: bool, **kwargs: Any)` — [`L25`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general.py#L25) — Gets how to quantize from conv's dimension_numbers.
- `get_transpose(dimension_numbers: jax.lax.ConvDimensionNumbers, for_lhs: bool)` — [`L54`](../../../../../../../raw/code/qwix/qwix/_src/core/conv_general.py#L54) — Returns the transpose list for the given dimension_numbers.

