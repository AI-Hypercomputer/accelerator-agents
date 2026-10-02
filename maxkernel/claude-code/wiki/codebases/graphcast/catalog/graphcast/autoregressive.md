---
title: 'Module: graphcast/autoregressive.py'
type: catalog
provenance: extracted
module: graphcast/autoregressive.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.autoregressive`/
symbols:
  Predictor.loss: Predictor#loss().
  Predictor.one_step_loss: Predictor#one_step_loss().
  Predictor.__call__: Predictor#__call__().
  Predictor.one_step_prediction: Predictor#one_step_prediction().
  Predictor.add_noise: Predictor#add_noise().
  _unflatten_and_expand_time: _unflatten_and_expand_time().
  _get_flat_arrays_and_single_timestep_treedef: _get_flat_arrays_and_single_timestep_treedef().
  Predictor._predictor: Predictor#_predictor.
  Predictor: Predictor#
  Predictor.__init__: Predictor#__init__().
  Predictor._noise_level: Predictor#_noise_level.
  Predictor._gradient_checkpointing: Predictor#_gradient_checkpointing.
  Predictor._get_and_validate_constant_inputs: Predictor#_get_and_validate_constant_inputs().
  Predictor._validate_targets_and_forcings: Predictor#_validate_targets_and_forcings().
  Predictor._update_inputs: Predictor#_update_inputs().
---
# Module: [`graphcast/autoregressive.py`](../../../../../raw/code/graphcast/graphcast/autoregressive.py)

## Classes
### `Predictor`  ·  implements/extends Predictor
- def: [`graphcast/autoregressive.py:39`](../../../../../raw/code/graphcast/graphcast/autoregressive.py#L39)
- doc: Wraps a one-step Predictor to make multi-step predictions autoregressively.
- signature: `class Predictor(predictor_base.Predictor):`
- members:
  - `__call__(self, inputs: xarray.Dataset, targets_template: xarray.Dataset, forcings: xarray.Dataset, **kwargs)` — [`L127`](../../../../../raw/code/graphcast/graphcast/autoregressive.py#L127) — Calls the Predictor.
  - `__init__(self, predictor: predictor_base.Predictor, noise_level: Optional[float] = None, gradient_checkpointing: bool = False)` — [`L63`](../../../../../raw/code/graphcast/graphcast/autoregressive.py#L63) — Initializes an autoregressive predictor wrapper.
  - `add_noise(x)` — [`L245`](../../../../../raw/code/graphcast/graphcast/autoregressive.py#L245)
  - `loss(self, inputs: xarray.Dataset, targets: xarray.Dataset, forcings: xarray.Dataset, **kwargs)` — [`L224`](../../../../../raw/code/graphcast/graphcast/autoregressive.py#L224) — The mean of the per-timestep losses of the underlying predictor. — documented in [graphcast-casting](../../concepts/graphcast-casting.md)
  - `one_step_loss(inputs, scan_variables)` — [`L262`](../../../../../raw/code/graphcast/graphcast/autoregressive.py#L262)
  - `one_step_prediction(inputs, scan_variables)` — [`L175`](../../../../../raw/code/graphcast/graphcast/autoregressive.py#L175)
- protocol/private: `_get_and_validate_constant_inputs`[`L88`](../../../../../raw/code/graphcast/graphcast/autoregressive.py#L88), `_gradient_checkpointing`[`L86`](../../../../../raw/code/graphcast/graphcast/autoregressive.py#L86), `_noise_level`[`L85`](../../../../../raw/code/graphcast/graphcast/autoregressive.py#L85), `_predictor`[`L84`](../../../../../raw/code/graphcast/graphcast/autoregressive.py#L84), `_update_inputs`[`L114`](../../../../../raw/code/graphcast/graphcast/autoregressive.py#L114), `_validate_targets_and_forcings`[`L100`](../../../../../raw/code/graphcast/graphcast/autoregressive.py#L100)
- uses (calls/refs, reference-scoped): [`DataArray`](xarray_jax.md#DataArray), [`Predictor`](predictor_base.md#Predictor), [`loss`](predictor_base.md#Predictor.loss), [`map_structure`](xarray_tree.md#map_structure), [`loss_and_predictions`](predictor_base.md#Predictor.loss_and_predictions), [`LossAndDiagnostics`](predictor_base.md#LossAndDiagnostics), [`unwrap_data`](xarray_jax.md#unwrap_data), [`_get_flat_arrays_and_single_timestep_treedef`](autoregressive.md#_get_flat_arrays_and_single_timestep_treedef), [`_unflatten_and_expand_time`](autoregressive.md#_unflatten_and_expand_time)
- used by: [`Predictor`](predictor_base.md#Predictor), [`loss`](predictor_base.md#Predictor.loss)

## Functions
- `_get_flat_arrays_and_single_timestep_treedef(variables)` — [`L33`](../../../../../raw/code/graphcast/graphcast/autoregressive.py#L33)
- `_unflatten_and_expand_time(flat_variables, tree_def, time_coords)` — [`L28`](../../../../../raw/code/graphcast/graphcast/autoregressive.py#L28)

