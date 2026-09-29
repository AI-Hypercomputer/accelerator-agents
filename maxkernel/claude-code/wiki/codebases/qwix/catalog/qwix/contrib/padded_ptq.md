---
title: 'Module: qwix/contrib/padded_ptq.py'
type: catalog
provenance: extracted
module: qwix/contrib/padded_ptq.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix.contrib.padded_ptq`/
symbols:
  einsum: einsum().
  quantize: quantize().
  dequantize: dequantize().
  dot_general: dot_general().
  _pad_operand_if_qarray: _pad_operand_if_qarray().
  PaddedQArray: PaddedQArray#
  create_quantized_param: create_quantized_param().
  PaddedPtqProvider: PaddedPtqProvider.
  MaybeQArray.MaybeQArray: MaybeQArray.MaybeQArray.
  pad_to_shape: pad_to_shape().
  quantize_act: quantize_act().
  HowToQuantize: HowToQuantize.
  PaddedQArray.original_shape: PaddedQArray#original_shape.
  PtqProvider: PtqProvider.
  PaddedQArray.padded_shape: PaddedQArray#padded_shape.
  calibrate: calibrate.
  quantize_params: quantize_params().
  QARRAY_KEEP_PADDED_SHAPE: QARRAY_KEEP_PADDED_SHAPE.
  get_padded_shape: get_padded_shape().
  __all__: __all__.
---
# Module: [`qwix/contrib/padded_ptq.py`](../../../../../../raw/code/qwix/qwix/contrib/padded_ptq.py)

## Classes
### `PaddedQArray`  ·  implements/extends QArray
- def: [`qwix/contrib/padded_ptq.py:53`](../../../../../../raw/code/qwix/qwix/contrib/padded_ptq.py#L53) — documented in [qwix-contrib-padded_ptq](../../../concepts/qwix-contrib-padded_ptq.md)
- doc: Quantized array with padding support.
- signature: `class PaddedQArray(qarray.QArray):`
- members:
  - `original_shape` — [`L67`](../../../../../../raw/code/qwix/qwix/contrib/padded_ptq.py#L67) — documented in [qwix-contrib-padded_ptq](../../../concepts/qwix-contrib-padded_ptq.md)
  - `padded_shape` — [`L64`](../../../../../../raw/code/qwix/qwix/contrib/padded_ptq.py#L64)
- uses (calls/refs, reference-scoped): [`QArray`](../_src/core/qarray.md#QArray)
- used by: [`QArray`](../_src/core/qarray.md#QArray), [`einsum`](padded_ptq.md#einsum), [`quantize`](padded_ptq.md#quantize), [`dequantize`](padded_ptq.md#dequantize), [`dot_general`](padded_ptq.md#dot_general), [`_pad_operand_if_qarray`](padded_ptq.md#_pad_operand_if_qarray), [`MaybeQArray`](padded_ptq.md#MaybeQArray.MaybeQArray)

## Functions
- `_pad_operand_if_qarray(x)` — [`L152`](../../../../../../raw/code/qwix/qwix/contrib/padded_ptq.py#L152) — documented in [qwix-contrib-padded_ptq](../../../concepts/qwix-contrib-padded_ptq.md)
- `create_quantized_param(name: str, value: jax.Array, how: HowToQuantize)` — [`L284`](../../../../../../raw/code/qwix/qwix/contrib/padded_ptq.py#L284) — Wrapper that delegates to PTQ.create_quantized_param using this backend.
- `dequantize(array: PaddedQArray)` — [`L137`](../../../../../../raw/code/qwix/qwix/contrib/padded_ptq.py#L137) — Dequantizes an array. The reverse of |quantize|. — documented in [qwix-contrib-padded_ptq](../../../concepts/qwix-contrib-padded_ptq.md)
- `dot_general(lhs: MaybeQArray, rhs: MaybeQArray, dimension_numbers: jax.lax.DotDimensionNumbers, precision: jax.lax.PrecisionLike = None, preferred_element_type: jax.typing.DTypeLike | None = None, **kwargs)` — [`L159`](../../../../../../raw/code/qwix/qwix/contrib/padded_ptq.py#L159) — Pad operands and delegate to core dot_general. — documented in [qwix-_src-core-dot_general](../../../concepts/qwix-_src-core-dot_general.md)
- `einsum(einsum_str: str, lhs: MaybeQArray, rhs: MaybeQArray, *, preferred_element_type: jax.typing.DTypeLike | None = None, **kwargs)` — [`L214`](../../../../../../raw/code/qwix/qwix/contrib/padded_ptq.py#L214) — Pad operands and delegate to core einsum. — documented in [qwix-contrib-padded_ptq](../../../concepts/qwix-contrib-padded_ptq.md)
- `get_padded_shape(original_shape: tuple[int, ...], tiled_axes: Mapping[int, int | float])` — [`L88`](../../../../../../raw/code/qwix/qwix/contrib/padded_ptq.py#L88) — Computes the target padded shape given tiled axes.
- `pad_to_shape(array: jax.Array, target_shape: tuple[int, ...])` — [`L77`](../../../../../../raw/code/qwix/qwix/contrib/padded_ptq.py#L77) — Pads array to target shape.
- `quantize(array: jax.Array, how: HowToQuantize)` — [`L119`](../../../../../../raw/code/qwix/qwix/contrib/padded_ptq.py#L119) — Quantizes an array using a dynamic range with padding support. — documented in [qwix-_src-core-qarray](../../../concepts/qwix-_src-core-qarray.md)
- `quantize_act(array: jax.Array, how: HowToQuantize, rule, act_name: str | None)` — [`L272`](../../../../../../raw/code/qwix/qwix/contrib/padded_ptq.py#L272) — Wrapper to reuse PTQ.quantize_act with this module as qarray backend.
- `quantize_params(params: Any, abstract_quantized_params: Any, quant_stats: Any = flax.core.FrozenDict())` — [`L295`](../../../../../../raw/code/qwix/qwix/contrib/padded_ptq.py#L295) — Wrapper that delegates to PTQ.quantize_params using this backend.

## Module values
- `HowToQuantize` — [`L41`](../../../../../../raw/code/qwix/qwix/contrib/padded_ptq.py#L41)
- `MaybeQArray` — [`L74`](../../../../../../raw/code/qwix/qwix/contrib/padded_ptq.py#L74)
- `PaddedPtqProvider` — [`L313`](../../../../../../raw/code/qwix/qwix/contrib/padded_ptq.py#L313) — documented in [qwix-contrib-padded_ptq](../../../concepts/qwix-contrib-padded_ptq.md)
- `PtqProvider` — [`L39`](../../../../../../raw/code/qwix/qwix/contrib/padded_ptq.py#L39)
- `QARRAY_KEEP_PADDED_SHAPE` — [`L44`](../../../../../../raw/code/qwix/qwix/contrib/padded_ptq.py#L44)
- `__all__` — [`L321`](../../../../../../raw/code/qwix/qwix/contrib/padded_ptq.py#L321)
- `calibrate` — [`L40`](../../../../../../raw/code/qwix/qwix/contrib/padded_ptq.py#L40)

