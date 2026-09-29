---
title: 'Module: aqt/jax/v2/flax/delayed_scaling_calibration.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/flax/delayed_scaling_calibration.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.flax.delayed_scaling_calibration`/
symbols:
  CALIBRATION_STATS: CALIBRATION_STATS.
  DelayedScalingCalibration: DelayedScalingCalibration#
  DelayedScalingCalibration.amax_history_length: DelayedScalingCalibration#amax_history_length.
  DelayedScalingCalibration.setup: DelayedScalingCalibration#setup().
  DelayedScalingCalibration.get_bound: DelayedScalingCalibration#get_bound().
  DelayedScalingCalibration.get_scale_and_bias_and_sparsity: DelayedScalingCalibration#get_scale_and_bias_and_sparsity().
  DelayedScalingCalibration.compute_bound: DelayedScalingCalibration#compute_bound().
  DelayedScalingCalibration.compute_history: DelayedScalingCalibration#compute_history().
  DelayedScalingCalibration.init_calibration: DelayedScalingCalibration#init_calibration().
---
# Module: [`aqt/jax/v2/flax/delayed_scaling_calibration.py`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/delayed_scaling_calibration.py)

## Classes
### `DelayedScalingCalibration`
- def: [`aqt/jax/v2/flax/delayed_scaling_calibration.py:28`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/delayed_scaling_calibration.py#L28)
- doc: Calibration module with logic from Transformer Engine, utilizing Delayed Scaling.
- signature: `class DelayedScalingCalibration(calibration.Calibration, nn.Module):`
- members:
  - `compute_bound(self, amax, prev_bound)` — [`L107`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/delayed_scaling_calibration.py#L107)
  - `compute_history(self, x, amax_history)` — [`L113`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/delayed_scaling_calibration.py#L113)
  - `get_bound(self, x: jnp.ndarray, shared_axes: None | Sequence[utils.AxisIdx], context: None | utils.Context=None)` — [`L57`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/delayed_scaling_calibration.py#L57)
  - `get_scale_and_bias_and_sparsity(self, x: jnp.ndarray, shared_axes: None | Sequence[utils.AxisIdx], numerics_: numerics.AqtNumerics, context: None | utils.Context=None)` — [`L94`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/delayed_scaling_calibration.py#L94)
  - `init_calibration(self)` — [`L120`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/delayed_scaling_calibration.py#L120)
  - `setup(self)` — [`L33`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/delayed_scaling_calibration.py#L33)
  - `amax_history_length` — [`L31`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/delayed_scaling_calibration.py#L31)

## Module values
- `CALIBRATION_STATS` — [`L24`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/delayed_scaling_calibration.py#L24)

