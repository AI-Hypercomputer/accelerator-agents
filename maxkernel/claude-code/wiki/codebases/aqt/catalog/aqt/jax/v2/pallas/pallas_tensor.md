---
title: 'Module: aqt/jax/v2/pallas/pallas_tensor.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/pallas/pallas_tensor.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.pallas.pallas_tensor`/
symbols:
  TransposedTensor: TransposedTensor#
  transpose_tensor_for_memory_saving: transpose_tensor_for_memory_saving().
  make_qtensor_blockspec: make_qtensor_blockspec().
  T: T.
  Array: Array.
  QTensor: QTensor.
  transpose: transpose.
  _called_within_pallas_kernel: _called_within_pallas_kernel().
  TransposedTensor.transposed_tensor: TransposedTensor#transposed_tensor.
  TransposedTensor.permute_axes: TransposedTensor#permute_axes.
  TransposedTensor.untransposed: TransposedTensor#untransposed().
  _count_less: _count_less().
---
# Module: [`aqt/jax/v2/pallas/pallas_tensor.py`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_tensor.py)

## Classes
### `TransposedTensor`
- def: [`aqt/jax/v2/pallas/pallas_tensor.py:43`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_tensor.py#L43)
- doc: Transposed tensor.
- signature: `class TransposedTensor(Generic[T]):`
- members:
  - `untransposed(self)` — [`L53`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_tensor.py#L53)
  - `permute_axes` — [`L47`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_tensor.py#L47)
  - `transposed_tensor` — [`L46`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_tensor.py#L46)
- used by: [`TransposedTensor`](pallas_call.md#TransposedTensor)

## Functions
- `_called_within_pallas_kernel(func)` — [`L37`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_tensor.py#L37) — Decorator to indicate that the function is called within Pallas kernel.
- `_count_less(list_, item)` — [`L59`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_tensor.py#L59) — Count the number of elements in list_ that are less than item.
- `make_qtensor_blockspec(qtensor: aqt_tensor.QTensor, block_spec: pl.BlockSpec)` — [`L120`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_tensor.py#L120) — Build a block spec for QTensor.
- `transpose_tensor_for_memory_saving(s: jax.Array, block_spec: pl.BlockSpec)` — [`L64`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_tensor.py#L64) — Transpose given tensor s for memory saving.

## Module values
- `Array` — [`L31`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_tensor.py#L31)
- `QTensor` — [`L32`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_tensor.py#L32)
- `T` — [`L30`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_tensor.py#L30)
- `transpose` — [`L34`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_tensor.py#L34)

