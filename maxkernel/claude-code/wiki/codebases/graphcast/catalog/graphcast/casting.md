---
title: 'Module: graphcast/casting.py'
type: catalog
provenance: extracted
module: graphcast/casting.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.casting`/
symbols:
  Bfloat16Cast.loss: Bfloat16Cast#loss().
  Bfloat16Cast.loss_and_predictions: Bfloat16Cast#loss_and_predictions().
  Bfloat16Cast.__call__: Bfloat16Cast#__call__().
  bfloat16_variable_view: bfloat16_variable_view().
  tree_map_cast: tree_map_cast().
  infer_floating_dtype: infer_floating_dtype().
  Bfloat16Cast._predictor: Bfloat16Cast#_predictor.
  Bfloat16Cast._enabled: Bfloat16Cast#_enabled.
  _all_inputs_to_bfloat16: _all_inputs_to_bfloat16().
  Bfloat16Cast: Bfloat16Cast#
  Bfloat16Cast.__init__: Bfloat16Cast#__init__().
  PyTree: PyTree.
  tree_map_cast.cast_fn: tree_map_cast().cast_fn().
  _bfloat16_creator: _bfloat16_creator().
  _bfloat16_getter: _bfloat16_getter().
  _bfloat16_setter: _bfloat16_setter().
---
# Module: [`graphcast/casting.py`](../../../../../raw/code/graphcast/graphcast/casting.py)

## Classes
### `Bfloat16Cast`  ·  implements/extends Predictor
- def: [`graphcast/casting.py:31`](../../../../../raw/code/graphcast/graphcast/casting.py#L31)
- doc: Wrapper that casts all inputs to bfloat16 and outputs to targets dtype.
- signature: `class Bfloat16Cast(predictor_base.Predictor):`
- members:
  - `__init__(self, predictor: predictor_base.Predictor, enabled: bool = True)` — [`L34`](../../../../../raw/code/graphcast/graphcast/casting.py#L34) — Inits the wrapper.
  - `loss(self, inputs: xarray.Dataset, targets: xarray.Dataset, forcings: xarray.Dataset, **kwargs)` — [`L67`](../../../../../raw/code/graphcast/graphcast/casting.py#L67) — documented in [graphcast-casting](../../concepts/graphcast-casting.md)
  - `loss_and_predictions(self, inputs: xarray.Dataset, targets: xarray.Dataset, forcings: xarray.Dataset, **kwargs)` — [`L92`](../../../../../raw/code/graphcast/graphcast/casting.py#L92) — documented in [graphcast-casting](../../concepts/graphcast-casting.md)
- protocol/private: `__call__`[`L45`](../../../../../raw/code/graphcast/graphcast/casting.py#L45), `_enabled`[`L42`](../../../../../raw/code/graphcast/graphcast/casting.py#L42), `_predictor`[`L43`](../../../../../raw/code/graphcast/graphcast/casting.py#L43)
- uses (calls/refs, reference-scoped): [`Predictor`](predictor_base.md#Predictor), [`loss`](predictor_base.md#Predictor.loss), [`loss_and_predictions`](predictor_base.md#Predictor.loss_and_predictions), [`LossAndDiagnostics`](predictor_base.md#LossAndDiagnostics), [`bfloat16_variable_view`](casting.md#bfloat16_variable_view), [`tree_map_cast`](casting.md#tree_map_cast), [`infer_floating_dtype`](casting.md#infer_floating_dtype), [`_all_inputs_to_bfloat16`](casting.md#_all_inputs_to_bfloat16)
- used by: [`Predictor`](predictor_base.md#Predictor), [`loss`](predictor_base.md#Predictor.loss), [`loss_and_predictions`](predictor_base.md#Predictor.loss_and_predictions)

## Functions
- `_all_inputs_to_bfloat16(inputs: xarray.Dataset, targets: xarray.Dataset, forcings: xarray.Dataset)` — [`L135`](../../../../../raw/code/graphcast/graphcast/casting.py#L135)
- `_bfloat16_creator(next_creator, shape, dtype, init, context)` — [`L186`](../../../../../raw/code/graphcast/graphcast/casting.py#L186) — Creates float32 variables when bfloat16 is requested.
- `_bfloat16_getter(next_getter, value, context)` — [`L193`](../../../../../raw/code/graphcast/graphcast/casting.py#L193) — Casts float32 to bfloat16 when bfloat16 was originally requested.
- `_bfloat16_setter(next_setter, value, context)` — [`L201`](../../../../../raw/code/graphcast/graphcast/casting.py#L201) — Casts bfloat16 to float32 when bfloat16 was originally set.
- `bfloat16_variable_view(enabled: bool = True)` — [`L156`](../../../../../raw/code/graphcast/graphcast/casting.py#L156) — Context for Haiku modules with float32 params, but bfloat16 activations. — documented in [graphcast-casting](../../concepts/graphcast-casting.md)
- `cast_fn(x)` — [`L149`](../../../../../raw/code/graphcast/graphcast/casting.py#L149)
- `infer_floating_dtype(data_vars: Mapping[str, chex.Array])` — [`L120`](../../../../../raw/code/graphcast/graphcast/casting.py#L120) — Infers a floating dtype from an input mapping of data.
- `tree_map_cast(inputs: PyTree, input_dtype: np.dtype, output_dtype: np.dtype)` — [`L147`](../../../../../raw/code/graphcast/graphcast/casting.py#L147)

## Module values
- `PyTree` — [`L28`](../../../../../raw/code/graphcast/graphcast/casting.py#L28)

