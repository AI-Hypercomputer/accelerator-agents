---
title: 'Module: graphcast/rollout.py'
type: catalog
provenance: extracted
module: graphcast/rollout.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.rollout`/
symbols:
  chunked_prediction_generator_multiple_runs: chunked_prediction_generator_multiple_runs().
  chunked_prediction_generator: chunked_prediction_generator().
  _replicate_dataset.replicate_variable: _replicate_dataset().replicate_variable().
  _replicate_dataset: _replicate_dataset().
  chunked_prediction: chunked_prediction().
  extend_targets_template: extend_targets_template().
  _replicate_dataset.replicate_dataset: _replicate_dataset().replicate_dataset().
  chunked_prediction_generator_multiple_runs.predictor_fn_pmap_named_args: chunked_prediction_generator_multiple_runs().predictor_fn_pmap_named_args().
  PredictorFn: PredictorFn#
  chunked_prediction_generator.split_rng_fn: chunked_prediction_generator().split_rng_fn().
  _device_put_sharded: _device_put_sharded().
  _get_next_inputs: _get_next_inputs().
  extend_targets_template.extend_time: extend_targets_template().extend_time().
  PredictorFn.__call__: PredictorFn#__call__().
---
# Module: [`graphcast/rollout.py`](../../../../../raw/code/graphcast/graphcast/rollout.py)

## Classes
### `PredictorFn`  ·  implements/extends Protocol
- def: [`graphcast/rollout.py:55`](../../../../../raw/code/graphcast/graphcast/rollout.py#L55)
- doc: Functional version of base.Predictor.__call__ with explicit rng.
- signature: `class PredictorFn(typing_extensions.Protocol):`
- protocol/private: `__call__`[`L58`](../../../../../raw/code/graphcast/graphcast/rollout.py#L58)
- used by: [`chunked_prediction_generator_multiple_runs`](rollout.md#chunked_prediction_generator_multiple_runs), [`chunked_prediction_generator`](rollout.md#chunked_prediction_generator), [`chunked_prediction`](rollout.md#chunked_prediction)

## Functions
- `_device_put_sharded(data_list, devices, axis_name)` — [`L30`](../../../../../raw/code/graphcast/graphcast/rollout.py#L30) — Stack data and put on devices with consistent sharding.
- `_get_next_inputs(prev_inputs: xarray.Dataset, next_frame: xarray.Dataset)` — [`L416`](../../../../../raw/code/graphcast/graphcast/rollout.py#L416) — Computes next inputs, from previous inputs and predictions.
- `_replicate_dataset(data: xarray.Dataset, replica_dim: str, replicate_to_device: bool, devices: Sequence[jax.Device])` — [`L67`](../../../../../raw/code/graphcast/graphcast/rollout.py#L67) — Used to prepare for xarray_jax.pmap.
- `chunked_prediction(predictor_fn: PredictorFn, rng: chex.PRNGKey, inputs: xarray.Dataset, targets_template: xarray.Dataset, forcings: xarray.Dataset, num_steps_per_chunk: int = 1, verbose: bool = False)` — [`L231`](../../../../../raw/code/graphcast/graphcast/rollout.py#L231) — Outputs a long trajectory by iteratively concatenating chunked predictions.
- `chunked_prediction_generator(predictor_fn: PredictorFn, rng: chex.PRNGKey, inputs: xarray.Dataset, targets_template: xarray.Dataset, forcings: xarray.Dataset, num_steps_per_chunk: int = 1, verbose: bool = False, pmap_devices: Sequence[jax.Device] | None = None, replica_axis: str | None = None)` — [`L272`](../../../../../raw/code/graphcast/graphcast/rollout.py#L272) — Outputs a long trajectory by yielding chunked predictions.
- `chunked_prediction_generator_multiple_runs(predictor_fn: PredictorFn, rngs: chex.PRNGKey, inputs: xarray.Dataset, targets_template: xarray.Dataset, forcings: Optional[xarray.Dataset], num_samples: Optional[int], pmap_devices: Optional[Sequence[jax.Device]] = None, **chunked_prediction_kwargs)` — [`L102`](../../../../../raw/code/graphcast/graphcast/rollout.py#L102) — Outputs a trajectory of multiple samples by yielding chunked predictions.
- `extend_targets_template(targets_template: xarray.Dataset, required_num_steps: int)` — [`L441`](../../../../../raw/code/graphcast/graphcast/rollout.py#L441) — Extends `targets_template` to `required_num_steps` with lazy arrays.
- `extend_time(data_array: xarray.DataArray)` — [`L478`](../../../../../raw/code/graphcast/graphcast/rollout.py#L478)
- `predictor_fn_pmap_named_args(rng, inputs, targets_template, forcings)` — [`L138`](../../../../../raw/code/graphcast/graphcast/rollout.py#L138)
- `replicate_dataset(dataset: xarray.Dataset)` — [`L89`](../../../../../raw/code/graphcast/graphcast/rollout.py#L89)
- `replicate_variable(variable: xarray.Variable)` — [`L74`](../../../../../raw/code/graphcast/graphcast/rollout.py#L74)
- `split_rng_fn(rng)` — [`L344`](../../../../../raw/code/graphcast/graphcast/rollout.py#L344)

