---
title: 'Module: graphcast/nan_cleaning.py'
type: catalog
provenance: extracted
module: graphcast/nan_cleaning.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.nan_cleaning`/NaNCleaner#
symbols:
  NaNCleaner.loss_and_predictions: loss_and_predictions().
  NaNCleaner._var_to_clean: _var_to_clean.
  NaNCleaner._clean: _clean().
  NaNCleaner.__call__: __call__().
  NaNCleaner.loss: loss().
  NaNCleaner._maybe_reintroduce_nans: _maybe_reintroduce_nans().
  NaNCleaner._reintroduce_nans: _reintroduce_nans.
  NaNCleaner._predictor: _predictor.
  NaNCleaner: ''
  NaNCleaner.__init__: __init__().
  NaNCleaner._fill_value: _fill_value.
---
# Module: [`graphcast/nan_cleaning.py`](../../../../../raw/code/graphcast/graphcast/nan_cleaning.py)

## Classes
### `NaNCleaner`  ·  implements/extends Predictor
- def: [`graphcast/nan_cleaning.py:27`](../../../../../raw/code/graphcast/graphcast/nan_cleaning.py#L27)
- doc: A predictor wrapper than removes NaNs from ingested data.
- signature: `class NaNCleaner(base.Predictor):`
- members:
  - `__init__(self, predictor: base.Predictor, var_to_clean: str, fill_value: xarray.Dataset, reintroduce_nans: bool = False)` — [`L33`](../../../../../raw/code/graphcast/graphcast/nan_cleaning.py#L33) — Initializes the NaNCleaner.
  - `_clean(self, dataset: xarray.Dataset)` — [`L46`](../../../../../raw/code/graphcast/graphcast/nan_cleaning.py#L46) — Cleans the dataset of NaNs.
  - `loss(self, inputs: xarray.Dataset, targets: xarray.Dataset, forcings: Optional[xarray.Dataset] = None, **kwargs)` — [`L86`](../../../../../raw/code/graphcast/graphcast/nan_cleaning.py#L86)
  - `loss_and_predictions(self, inputs: xarray.Dataset, targets: xarray.Dataset, forcings: Optional[xarray.Dataset] = None, **kwargs)` — [`L103`](../../../../../raw/code/graphcast/graphcast/nan_cleaning.py#L103)
- protocol/private: `__call__`[`L65`](../../../../../raw/code/graphcast/graphcast/nan_cleaning.py#L65), `_fill_value`[`L42`](../../../../../raw/code/graphcast/graphcast/nan_cleaning.py#L42), `_maybe_reintroduce_nans`[`L54`](../../../../../raw/code/graphcast/graphcast/nan_cleaning.py#L54), `_predictor`[`L41`](../../../../../raw/code/graphcast/graphcast/nan_cleaning.py#L41), `_reintroduce_nans`[`L44`](../../../../../raw/code/graphcast/graphcast/nan_cleaning.py#L44), `_var_to_clean`[`L43`](../../../../../raw/code/graphcast/graphcast/nan_cleaning.py#L43)
- uses (calls/refs, reference-scoped): [`Predictor`](predictor_base.md#Predictor), [`loss`](predictor_base.md#Predictor.loss), [`loss_and_predictions`](predictor_base.md#Predictor.loss_and_predictions), [`LossAndDiagnostics`](predictor_base.md#LossAndDiagnostics)
- used by: [`Predictor`](predictor_base.md#Predictor), [`loss`](predictor_base.md#Predictor.loss), [`loss_and_predictions`](predictor_base.md#Predictor.loss_and_predictions)

