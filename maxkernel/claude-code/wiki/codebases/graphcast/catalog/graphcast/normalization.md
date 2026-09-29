---
title: 'Module: graphcast/normalization.py'
type: catalog
provenance: extracted
module: graphcast/normalization.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.normalization`/
symbols:
  InputsAndResiduals.loss_and_predictions: InputsAndResiduals#loss_and_predictions().
  InputsAndResiduals.loss: InputsAndResiduals#loss().
  InputsAndResiduals.__call__: InputsAndResiduals#__call__().
  InputsAndResiduals._unnormalize_prediction_and_add_input: InputsAndResiduals#_unnormalize_prediction_and_add_input().
  InputsAndResiduals._subtract_input_and_normalize_target: InputsAndResiduals#_subtract_input_and_normalize_target().
  normalize: normalize().
  InputsAndResiduals._scales: InputsAndResiduals#_scales.
  InputsAndResiduals._locations: InputsAndResiduals#_locations.
  unnormalize: unnormalize().
  InputsAndResiduals._predictor: InputsAndResiduals#_predictor.
  InputsAndResiduals: InputsAndResiduals#
  InputsAndResiduals.__init__: InputsAndResiduals#__init__().
  InputsAndResiduals._residual_scales: InputsAndResiduals#_residual_scales.
  InputsAndResiduals._residual_locations: InputsAndResiduals#_residual_locations.
  normalize.normalize_array: normalize().normalize_array().
  unnormalize.unnormalize_array: unnormalize().unnormalize_array().
---
# Module: [`graphcast/normalization.py`](../../../../../raw/code/graphcast/graphcast/normalization.py)

## Classes
### `InputsAndResiduals`  ·  implements/extends Predictor
- def: [`graphcast/normalization.py:73`](../../../../../raw/code/graphcast/graphcast/normalization.py#L73)
- doc: Wraps with a residual connection, normalizing inputs and target residuals.
- signature: `class InputsAndResiduals(predictor_base.Predictor):`
- members:
  - `loss(self, inputs: xarray.Dataset, targets: xarray.Dataset, forcings: xarray.Dataset, **kwargs)` — [`L162`](../../../../../raw/code/graphcast/graphcast/normalization.py#L162) — Returns the loss computed on normalized inputs and targets.
  - `loss_and_predictions(self, inputs: xarray.Dataset, targets: xarray.Dataset, forcings: xarray.Dataset, **kwargs)` — [`L177`](../../../../../raw/code/graphcast/graphcast/normalization.py#L177) — The loss computed on normalized data, with unnormalized predictions.
- protocol/private: `__call__`[`L148`](../../../../../raw/code/graphcast/graphcast/normalization.py#L148), `__init__`[`L101`](../../../../../raw/code/graphcast/graphcast/normalization.py#L101), `_locations`[`L109`](../../../../../raw/code/graphcast/graphcast/normalization.py#L109), `_predictor`[`L107`](../../../../../raw/code/graphcast/graphcast/normalization.py#L107), `_residual_locations`[`L111`](../../../../../raw/code/graphcast/graphcast/normalization.py#L111), `_residual_scales`[`L110`](../../../../../raw/code/graphcast/graphcast/normalization.py#L110), `_scales`[`L108`](../../../../../raw/code/graphcast/graphcast/normalization.py#L108), `_subtract_input_and_normalize_target`[`L134`](../../../../../raw/code/graphcast/graphcast/normalization.py#L134), `_unnormalize_prediction_and_add_input`[`L113`](../../../../../raw/code/graphcast/graphcast/normalization.py#L113)
- uses (calls/refs, reference-scoped): [`Predictor`](predictor_base.md#Predictor), [`loss`](predictor_base.md#Predictor.loss), [`map_structure`](xarray_tree.md#map_structure), [`loss_and_predictions`](predictor_base.md#Predictor.loss_and_predictions), [`LossAndDiagnostics`](predictor_base.md#LossAndDiagnostics), [`normalize`](normalization.md#normalize), [`unnormalize`](normalization.md#unnormalize)
- used by: [`Predictor`](predictor_base.md#Predictor), [`loss`](predictor_base.md#Predictor.loss), [`loss_and_predictions`](predictor_base.md#Predictor.loss_and_predictions)

## Functions
- `normalize(values: xarray.Dataset, scales: xarray.Dataset, locations: Optional[xarray.Dataset])` — [`L29`](../../../../../raw/code/graphcast/graphcast/normalization.py#L29) — Normalize variables using the given scales and (optionally) locations.
- `normalize_array(array)` — [`L34`](../../../../../raw/code/graphcast/graphcast/normalization.py#L34)
- `unnormalize(values: xarray.Dataset, scales: xarray.Dataset, locations: Optional[xarray.Dataset])` — [`L51`](../../../../../raw/code/graphcast/graphcast/normalization.py#L51) — Unnormalize variables using the given scales and (optionally) locations.
- `unnormalize_array(array)` — [`L56`](../../../../../raw/code/graphcast/graphcast/normalization.py#L56)

