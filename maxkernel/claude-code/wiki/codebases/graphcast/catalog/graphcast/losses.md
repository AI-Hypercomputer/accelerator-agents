---
title: 'Module: graphcast/losses.py'
type: catalog
provenance: extracted
module: graphcast/losses.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.losses`/
symbols:
  weighted_mse_per_level: weighted_mse_per_level().
  weighted_mse_per_level.loss: weighted_mse_per_level().loss().
  normalized_latitude_weights: normalized_latitude_weights().
  LossAndDiagnostics: LossAndDiagnostics.
  sum_per_variable_losses: sum_per_variable_losses().
  _weight_for_latitude_vector_without_poles: _weight_for_latitude_vector_without_poles().
  _weight_for_latitude_vector_with_poles: _weight_for_latitude_vector_with_poles().
  LossFunction.__call__: LossFunction#__call__().
  _check_uniform_spacing_and_get_delta: _check_uniform_spacing_and_get_delta().
  _mean_preserving_batch: _mean_preserving_batch().
  normalized_level_weights: normalized_level_weights().
  LossFunction: LossFunction#
---
# Module: [`graphcast/losses.py`](../../../../../raw/code/graphcast/graphcast/losses.py)

## Classes
### `LossFunction`  ·  implements/extends Protocol
- def: [`graphcast/losses.py:27`](../../../../../raw/code/graphcast/graphcast/losses.py#L27)
- doc: A loss function.
- signature: `class LossFunction(Protocol):`
- members:
  - `__call__(self, predictions: xarray.Dataset, targets: xarray.Dataset, **optional_kwargs)` — [`L34`](../../../../../raw/code/graphcast/graphcast/losses.py#L34) — Computes a loss function.
- uses (calls/refs, reference-scoped): [`LossAndDiagnostics`](losses.md#LossAndDiagnostics)

## Functions
- `_check_uniform_spacing_and_get_delta(vector)` — [`L175`](../../../../../raw/code/graphcast/graphcast/losses.py#L175)
- `_mean_preserving_batch(x: xarray.DataArray)` — [`L73`](../../../../../raw/code/graphcast/graphcast/losses.py#L73)
- `_weight_for_latitude_vector_with_poles(latitude)` — [`L161`](../../../../../raw/code/graphcast/graphcast/losses.py#L161) — Weights for uniform latitudes of the form [+- 90, ..., -+90].
- `_weight_for_latitude_vector_without_poles(latitude)` — [`L150`](../../../../../raw/code/graphcast/graphcast/losses.py#L150) — Weights for uniform latitudes of the form [+-90-+d/2, ..., -+90+-d/2].
- `loss(prediction, target)` — [`L62`](../../../../../raw/code/graphcast/graphcast/losses.py#L62)
- `normalized_latitude_weights(data: xarray.DataArray)` — [`L103`](../../../../../raw/code/graphcast/graphcast/losses.py#L103) — Weights based on latitude, roughly proportional to grid cell area.
- `normalized_level_weights(data: xarray.DataArray)` — [`L97`](../../../../../raw/code/graphcast/graphcast/losses.py#L97) — Weights proportional to pressure at each level.
- `sum_per_variable_losses(per_variable_losses: Mapping[str, xarray.DataArray], weights: Mapping[str, float])` — [`L77`](../../../../../raw/code/graphcast/graphcast/losses.py#L77) — Weighted sum of per-variable losses.
- `weighted_mse_per_level(predictions: xarray.Dataset, targets: xarray.Dataset, per_variable_weights: Mapping[str, float])` — [`L56`](../../../../../raw/code/graphcast/graphcast/losses.py#L56) — Latitude- and pressure-level-weighted MSE loss.

## Module values
- `LossAndDiagnostics` — [`L24`](../../../../../raw/code/graphcast/graphcast/losses.py#L24)

