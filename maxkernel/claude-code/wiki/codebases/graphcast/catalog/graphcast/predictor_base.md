---
title: 'Module: graphcast/predictor_base.py'
type: catalog
provenance: extracted
module: graphcast/predictor_base.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.predictor_base`/
symbols:
  Predictor: Predictor#
  Predictor.loss: Predictor#loss().
  Predictor.loss_and_predictions: Predictor#loss_and_predictions().
  LossAndDiagnostics: LossAndDiagnostics.
  Predictor.__call__: Predictor#__call__().
---
# Module: [`graphcast/predictor_base.py`](../../../../../raw/code/graphcast/graphcast/predictor_base.py)

## Classes
### `Predictor`  ·  implements/extends ABC
- def: [`graphcast/predictor_base.py:28`](../../../../../raw/code/graphcast/graphcast/predictor_base.py#L28)
- doc: A possibly-trainable predictor of weather, exposing an xarray-based API.
- signature: `class Predictor(abc.ABC):`
- members:
  - `__call__(self, inputs: xarray.Dataset, targets_template: xarray.Dataset, forcings: xarray.Dataset, **optional_kwargs)` — [`L45`](../../../../../raw/code/graphcast/graphcast/predictor_base.py#L45) — Makes predictions.
  - `loss(self, inputs: xarray.Dataset, targets: xarray.Dataset, forcings: xarray.Dataset, **optional_kwargs)` — [`L87`](../../../../../raw/code/graphcast/graphcast/predictor_base.py#L87) — Computes a training loss, for predictors that are trainable. — documented in [graphcast-casting](../../concepts/graphcast-casting.md)
  - `loss_and_predictions(self, inputs: xarray.Dataset, targets: xarray.Dataset, forcings: xarray.Dataset, **optional_kwargs)` — [`L133`](../../../../../raw/code/graphcast/graphcast/predictor_base.py#L133) — Like .loss but also returns corresponding predictions. — documented in [graphcast-casting](../../concepts/graphcast-casting.md)
- uses (calls/refs, reference-scoped): [`DataArray`](xarray_jax.md#DataArray), [`loss`](gencast.md#GenCast.loss), [`loss`](autoregressive.md#Predictor.loss), [`loss_and_predictions`](normalization.md#InputsAndResiduals.loss_and_predictions), [`loss`](casting.md#Bfloat16Cast.loss), [`loss`](normalization.md#InputsAndResiduals.loss), [`loss_and_predictions`](casting.md#Bfloat16Cast.loss_and_predictions), [`loss_and_predictions`](nan_cleaning.md#NaNCleaner.loss_and_predictions), [`LossAndDiagnostics`](predictor_base.md#LossAndDiagnostics), [`loss`](nan_cleaning.md#NaNCleaner.loss), [`loss_and_predictions`](graphcast.md#GraphCast.loss_and_predictions), [`loss`](graphcast.md#GraphCast.loss), [`loss_and_predictions`](gencast.md#GenCast.loss_and_predictions), [`Bfloat16Cast`](casting.md#Bfloat16Cast), [`GenCast`](gencast.md#GenCast), [`GraphCast`](graphcast.md#GraphCast), [`InputsAndResiduals`](normalization.md#InputsAndResiduals), [`NaNCleaner`](nan_cleaning.md#NaNCleaner), [`Predictor`](autoregressive.md#Predictor)
- used by: [`loss`](autoregressive.md#Predictor.loss), [`loss_and_predictions`](normalization.md#InputsAndResiduals.loss_and_predictions), [`loss`](casting.md#Bfloat16Cast.loss), [`loss`](normalization.md#InputsAndResiduals.loss), [`loss_and_predictions`](casting.md#Bfloat16Cast.loss_and_predictions), [`one_step_loss`](autoregressive.md#Predictor.one_step_loss), [`loss_and_predictions`](nan_cleaning.md#NaNCleaner.loss_and_predictions), [`loss`](nan_cleaning.md#NaNCleaner.loss), [`Bfloat16Cast`](casting.md#Bfloat16Cast), [`GenCast`](gencast.md#GenCast), [`GraphCast`](graphcast.md#GraphCast), [`InputsAndResiduals`](normalization.md#InputsAndResiduals), [`NaNCleaner`](nan_cleaning.md#NaNCleaner), [`Predictor`](autoregressive.md#Predictor), [`__init__`](autoregressive.md#Predictor.__init__), [`__init__`](casting.md#Bfloat16Cast.__init__), [`__init__`](nan_cleaning.md#NaNCleaner.__init__), [`__init__`](normalization.md#InputsAndResiduals.__init__)

## Module values
- `LossAndDiagnostics` — [`L25`](../../../../../raw/code/graphcast/graphcast/predictor_base.py#L25) — documented in [graphcast-casting](../../concepts/graphcast-casting.md)

