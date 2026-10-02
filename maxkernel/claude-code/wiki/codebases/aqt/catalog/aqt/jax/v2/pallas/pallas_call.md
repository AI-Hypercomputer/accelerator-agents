---
title: 'Module: aqt/jax/v2/pallas/pallas_call.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/pallas/pallas_call.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.pallas.pallas_call`/
symbols:
  pallas_call.wrapped: pallas_call().wrapped().
  tree_util: tree_util.
  _make_qtensor_blockspec: _make_qtensor_blockspec().
  _transpose_tensor_for_memory_saving: _transpose_tensor_for_memory_saving().
  pallas_call: pallas_call().
  pallas_call.wrapped.kernel: pallas_call().wrapped().kernel().
  QTensor: QTensor.
  TransposedTensor: TransposedTensor.
  _is_qtensor: _is_qtensor().
  _is_arg_and_block_spec: _is_arg_and_block_spec().
  _is_transposed_tensor: _is_transposed_tensor().
  ArgAndBlockSpec: ArgAndBlockSpec.
  BlockSpec: BlockSpec.
  no_block_spec: no_block_spec.
---
# Module: [`aqt/jax/v2/pallas/pallas_call.py`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_call.py)

## Functions
- `_is_arg_and_block_spec(x)` — [`L61`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_call.py#L61)
- `_is_qtensor(x)` — [`L53`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_call.py#L53)
- `_is_transposed_tensor(x)` — [`L57`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_call.py#L57)
- `_make_qtensor_blockspec(arg, block_spec)` — [`L36`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_call.py#L36)
- `_transpose_tensor_for_memory_saving(arg: Any, block_spec: BlockSpec)` — [`L43`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_call.py#L43) — Transposes tensor for memory optimization.
- `kernel(*args)` — [`L131`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_call.py#L131)
- `pallas_call(f: Callable[..., None], *pl_call_args, grid_spec=None, in_specs=no_block_spec, **pl_call_kwrags)` — [`L65`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_call.py#L65) — pl.pallas_call wrapper that can pass QTensor as input.
- `wrapped(*args)` — [`L84`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_call.py#L84)

## Module values
- `ArgAndBlockSpec` — [`L33`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_call.py#L33)
- `BlockSpec` — [`L26`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_call.py#L26)
- `QTensor` — [`L30`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_call.py#L30) — documented in [aqt-jax-v2-aqt_tensor](../../../../../concepts/aqt-jax-v2-aqt_tensor.md)
- `TransposedTensor` — [`L31`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_call.py#L31)
- `no_block_spec` — [`L27`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_call.py#L27)
- `tree_util` — [`L28`](../../../../../../../../raw/code/aqt/aqt/jax/v2/pallas/pallas_call.py#L28)

