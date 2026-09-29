---
title: 'Module: qwix/_src/core/dot_general.py'
type: catalog
provenance: extracted
module: qwix/_src/core/dot_general.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix._src.core.dot_general`/
symbols:
  dot_general: dot_general().
  _fast_dot_general: _fast_dot_general().
  loop_dot_general: loop_dot_general().
  get_how_to_quantize: get_how_to_quantize().
  _slow_dot_general: _slow_dot_general().
  loop_dot_general.take_slice: loop_dot_general().take_slice().
  _get_scale_transpose: _get_scale_transpose().
  _apply_tiling: _apply_tiling().
  _broadcast_axes: _broadcast_axes().
  MIN_TILE_SIZE_TO_DEQUANT_ON_OUTPUT: MIN_TILE_SIZE_TO_DEQUANT_ON_OUTPUT.
---
# Module: [`qwix/_src/core/dot_general.py`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general.py)

## Functions
- `_apply_tiling(contracting_axes: Sequence[int], batch_axes: Sequence[int], tiled_axes: Collection[int])` — [`L93`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general.py#L93) — Apply tiling to dimension numbers. — documented in [qwix-_src-core-dot_general](../../../../concepts/qwix-_src-core-dot_general.md)
- `_broadcast_axes(array: jax.Array, shape: tuple[int, ...], axes: Collection[int])` — [`L125`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general.py#L125) — Broadcast the given axes in the array to the given shape.
- `_fast_dot_general(lhs: qarray.MaybeQArray, rhs: qarray.MaybeQArray, dimension_numbers: jax.lax.DotDimensionNumbers, preferred_element_type: jax.typing.DTypeLike | None = None, **kwargs)` — [`L135`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general.py#L135) — Dot general in optimized path by computing in quantized types first then dequantize. — documented in [qwix-_src-core-dot_general](../../../../concepts/qwix-_src-core-dot_general.md)
- `_get_scale_transpose(dimension_numbers: jax.lax.DotDimensionNumbers, ndims: tuple[int, int])` — [`L79`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general.py#L79) — Returns the transpose list for lhs_scale and rhs_scale.
- `_slow_dot_general(lhs: qarray.MaybeQArray, rhs: qarray.MaybeQArray, dimension_numbers: jax.lax.DotDimensionNumbers, **kwargs)` — [`L264`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general.py#L264) — Dot general in slow path by dequantizing first then computing in floating-point types. — documented in [qwix-_src-core-dot_general](../../../../concepts/qwix-_src-core-dot_general.md)
- `dot_general(lhs: qarray.MaybeQArray, rhs: qarray.MaybeQArray, dimension_numbers: jax.lax.DotDimensionNumbers, precision: jax.lax.PrecisionLike = None, preferred_element_type: jax.typing.DTypeLike | None = None, **kwargs)` — [`L407`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general.py#L407) — Computes a general dot product with support for ``QArray`` inputs. — documented in [qwix-_src-core-dot_general](../../../../concepts/qwix-_src-core-dot_general.md)
- `get_how_to_quantize(*, dimension_numbers: jax.lax.DotDimensionNumbers, ndims: tuple[int, int], for_lhs: bool, tile_size: Mapping[int, int | float] | int | float | None, **kwargs: Any)` — [`L28`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general.py#L28) — Get how to quantize from dimension_numbers and remaining_dims. — documented in [qwix-_src-core-dot_general](../../../../concepts/qwix-_src-core-dot_general.md)
- `loop_dot_general(lhs: qarray.MaybeQArray, rhs: qarray.MaybeQArray, dimension_numbers: jax.lax.DotDimensionNumbers, preferred_element_type: jax.typing.DTypeLike | None = None, **kwargs)` — [`L278`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general.py#L278) — Loop-based tiled dot general implementation for the internal accumulation loop. — documented in [qwix-_src-core-dot_general](../../../../concepts/qwix-_src-core-dot_general.md)
- `take_slice(array: jax.Array, ca: Sequence[int], ca_tile_indices: Sequence[int])` — [`L364`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general.py#L364)

## Module values
- `MIN_TILE_SIZE_TO_DEQUANT_ON_OUTPUT` — [`L404`](../../../../../../../raw/code/qwix/qwix/_src/core/dot_general.py#L404)

