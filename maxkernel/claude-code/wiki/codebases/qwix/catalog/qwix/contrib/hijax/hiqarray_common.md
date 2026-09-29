---
title: 'Module: qwix/contrib/hijax/hiqarray_common.py'
type: catalog
provenance: extracted
module: qwix/contrib/hijax/hiqarray_common.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix.contrib.hijax.hiqarray_common`/QuantizationMetadata#
symbols:
  QuantizationMetadata.__repr__: __repr__().
  QuantizationMetadata.init: init().
  QuantizationMetadata: ''
  QuantizationMetadata.quant_axes: quant_axes.
  QuantizationMetadata.group_sizes: group_sizes.
  QuantizationMetadata.dtype: dtype.
  QuantizationMetadata.qtype: qtype.
  QuantizationMetadata._get_reduction_shape_and_axes: _get_reduction_shape_and_axes().
  QuantizationMetadata._get_quant_shape: _get_quant_shape().
  QuantizationMetadata.data_shape: data_shape.
  QuantizationMetadata.quant_shape: quant_shape.
  QuantizationMetadata.data_compatible_shape: data_compatible_shape.
  QuantizationMetadata.quant_compatible_shape: quant_compatible_shape.
  QuantizationMetadata._tiled_reduction_axes: _tiled_reduction_axes.
  QuantizationMetadata._full_reduction_axes: _full_reduction_axes.
---
# Module: [`qwix/contrib/hijax/hiqarray_common.py`](../../../../../../../raw/code/qwix/qwix/contrib/hijax/hiqarray_common.py)

## Classes
### `QuantizationMetadata`
- def: [`qwix/contrib/hijax/hiqarray_common.py:25`](../../../../../../../raw/code/qwix/qwix/contrib/hijax/hiqarray_common.py#L25)
- doc: This class contains information used to quantize and dequantize an array.
- signature: `class QuantizationMetadata:`
- members:
  - `_get_quant_shape(intermediate_shape: tuple[int, ...], tiled_reduction_axes: tuple[int, ...], full_reduction_axes: tuple[int, ...])` — [`L194`](../../../../../../../raw/code/qwix/qwix/contrib/hijax/hiqarray_common.py#L194) — Returns the shape of the tensor after quantization.
  - `_get_reduction_shape_and_axes(original_shape: tuple[int, ...], quant_axes: tuple[int, ...], group_sizes: tuple[int, ...])` — [`L108`](../../../../../../../raw/code/qwix/qwix/contrib/hijax/hiqarray_common.py#L108) — This function returns the intermediate shape needed for performing reductions as well as the axes along which to reduce.
  - `init(cls, data_shape: tuple[int, ...], quant_info: dict[int, int], original_dtype: jnp.dtype, quantized_dtype: jnp.dtype)` — [`L53`](../../../../../../../raw/code/qwix/qwix/contrib/hijax/hiqarray_common.py#L53) — Initializes the quantization metadata for an array.
  - `data_compatible_shape` — [`L41`](../../../../../../../raw/code/qwix/qwix/contrib/hijax/hiqarray_common.py#L41)
  - `data_shape` — [`L37`](../../../../../../../raw/code/qwix/qwix/contrib/hijax/hiqarray_common.py#L37)
  - `dtype` — [`L49`](../../../../../../../raw/code/qwix/qwix/contrib/hijax/hiqarray_common.py#L49)
  - `group_sizes` — [`L34`](../../../../../../../raw/code/qwix/qwix/contrib/hijax/hiqarray_common.py#L34)
  - `qtype` — [`L50`](../../../../../../../raw/code/qwix/qwix/contrib/hijax/hiqarray_common.py#L50)
  - `quant_axes` — [`L33`](../../../../../../../raw/code/qwix/qwix/contrib/hijax/hiqarray_common.py#L33)
  - `quant_compatible_shape` — [`L42`](../../../../../../../raw/code/qwix/qwix/contrib/hijax/hiqarray_common.py#L42)
  - `quant_shape` — [`L38`](../../../../../../../raw/code/qwix/qwix/contrib/hijax/hiqarray_common.py#L38)
- protocol/private: `__repr__`[`L215`](../../../../../../../raw/code/qwix/qwix/contrib/hijax/hiqarray_common.py#L215), `_full_reduction_axes`[`L46`](../../../../../../../raw/code/qwix/qwix/contrib/hijax/hiqarray_common.py#L46), `_tiled_reduction_axes`[`L45`](../../../../../../../raw/code/qwix/qwix/contrib/hijax/hiqarray_common.py#L45)

