---
title: 'Module: aqt/jax/v2/flax/aqt_flax_calibration.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/flax/aqt_flax_calibration.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.flax.aqt_flax_calibration`/
symbols:
  _SUM_OF_ONES: _SUM_OF_ONES.
  _SUM_OF_VALS: _SUM_OF_VALS.
  _MAX_OF_ABS_VALS: _MAX_OF_ABS_VALS.
  _SUM_OF_L1_VALS: _SUM_OF_L1_VALS.
  _SUM_OF_LP_VALS: _SUM_OF_LP_VALS.
  MeanOfAbsMaxCalibration: MeanOfAbsMaxCalibration#
  MeanOfAbsMaxCalibration.quant_collection: MeanOfAbsMaxCalibration#quant_collection.
  MeanOfAbsMaxCalibration.get_bound: MeanOfAbsMaxCalibration#get_bound().
  MeanOfAbsMaxCalibration.get_scale_and_bias_and_sparsity: MeanOfAbsMaxCalibration#get_scale_and_bias_and_sparsity().
  WeightedStatsCalibration: WeightedStatsCalibration#
  WeightedStatsCalibration.quant_collection: WeightedStatsCalibration#quant_collection.
  WeightedStatsCalibration.l1_dev_coeff: WeightedStatsCalibration#l1_dev_coeff.
  WeightedStatsCalibration.lp_dev_coeff: WeightedStatsCalibration#lp_dev_coeff.
  WeightedStatsCalibration.max_dev_coeff: WeightedStatsCalibration#max_dev_coeff.
  WeightedStatsCalibration.const_bound_coeff: WeightedStatsCalibration#const_bound_coeff.
  WeightedStatsCalibration.update_count_prior: WeightedStatsCalibration#update_count_prior.
  WeightedStatsCalibration.max_dev_prior: WeightedStatsCalibration#max_dev_prior.
  WeightedStatsCalibration.mean_prior: WeightedStatsCalibration#mean_prior.
  WeightedStatsCalibration.l1_dev_prior: WeightedStatsCalibration#l1_dev_prior.
  WeightedStatsCalibration.lp_dev_prior: WeightedStatsCalibration#lp_dev_prior.
  WeightedStatsCalibration.lp_order: WeightedStatsCalibration#lp_order.
  WeightedStatsCalibration.safe_divide: WeightedStatsCalibration#safe_divide.
  WeightedStatsCalibration.filter_zeros: WeightedStatsCalibration#filter_zeros.
  WeightedStatsCalibration.tpu_cross_replica_sum: WeightedStatsCalibration#tpu_cross_replica_sum.
  WeightedStatsCalibration.ema_update_count: WeightedStatsCalibration#ema_update_count.
  WeightedStatsCalibration._get_value: WeightedStatsCalibration#_get_value().
  WeightedStatsCalibration._mean: WeightedStatsCalibration#_mean().
  WeightedStatsCalibration._max_dev: WeightedStatsCalibration#_max_dev().
  WeightedStatsCalibration._l1_dev: WeightedStatsCalibration#_l1_dev().
  WeightedStatsCalibration._lp_dev: WeightedStatsCalibration#_lp_dev().
  WeightedStatsCalibration._update_var: WeightedStatsCalibration#_update_var().
  WeightedStatsCalibration._divide: WeightedStatsCalibration#_divide().
  WeightedStatsCalibration.get_bound: WeightedStatsCalibration#get_bound().
  WeightedStatsCalibration.get_scale_and_bias_and_sparsity: WeightedStatsCalibration#get_scale_and_bias_and_sparsity().
---
# Module: [`aqt/jax/v2/flax/aqt_flax_calibration.py`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py)

## Classes
### `MeanOfAbsMaxCalibration`
- def: [`aqt/jax/v2/flax/aqt_flax_calibration.py:33`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L33)
- doc: State for static range AQT/PTQ/QAT.
- signature: `class MeanOfAbsMaxCalibration(calibration.Calibration, nn.Module):`
- members:
  - `get_bound(self, x: jnp.ndarray, shared_axes: None | Sequence[utils.AxisIdx], context: None | utils.Context=None)` — [`L39`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L39)
  - `get_scale_and_bias_and_sparsity(self, x: jnp.ndarray, shared_axes: None | Sequence[utils.AxisIdx], numerics_: numerics.AqtNumerics, context: None | utils.Context=None)` — [`L77`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L77)
  - `quant_collection` — [`L36`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L36)

### `WeightedStatsCalibration`
- def: [`aqt/jax/v2/flax/aqt_flax_calibration.py:92`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L92)
- doc: Migration of AQTv1 calibration to AQTv2.
- signature: `class WeightedStatsCalibration(calibration.Calibration, nn.Module):`
- members:
  - `_update_var(self, var: nn.Variable, s: jnp.ndarray, shared_axes: None | Sequence[utils.AxisIdx], weight: None | jnp.ndarray=None, reduce_max: bool=False)` — [`L143`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L143) — Updates the given Flax variable.
  - `get_bound(self, x: jnp.ndarray, shared_axes: None | Sequence[utils.AxisIdx], context: None | utils.Context=None)` — [`L177`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L177)
  - `get_scale_and_bias_and_sparsity(self, x: jnp.ndarray, shared_axes: None | Sequence[utils.AxisIdx], numerics_: numerics.AqtNumerics, context: None | utils.Context=None)` — [`L240`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L240)
  - `const_bound_coeff` — [`L101`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L101)
  - `ema_update_count` — [`L114`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L114)
  - `filter_zeros` — [`L112`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L112)
  - `l1_dev_coeff` — [`L98`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L98)
  - `l1_dev_prior` — [`L107`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L107)
  - `lp_dev_coeff` — [`L99`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L99)
  - `lp_dev_prior` — [`L108`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L108)
  - `lp_order` — [`L110`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L110)
  - `max_dev_coeff` — [`L100`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L100)
  - `max_dev_prior` — [`L105`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L105)
  - `mean_prior` — [`L106`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L106)
  - `quant_collection` — [`L95`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L95)
  - `safe_divide` — [`L111`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L111)
  - `tpu_cross_replica_sum` — [`L113`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L113)
  - `update_count_prior` — [`L104`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L104)
- protocol/private: `_divide`[`L168`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L168), `_get_value`[`L116`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L116), `_l1_dev`[`L127`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L127), `_lp_dev`[`L132`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L132), `_max_dev`[`L124`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L124), `_mean`[`L119`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L119)

## Module values
- `_MAX_OF_ABS_VALS` — [`L27`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L27)
- `_SUM_OF_L1_VALS` — [`L28`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L28)
- `_SUM_OF_LP_VALS` — [`L29`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L29)
- `_SUM_OF_ONES` — [`L25`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L25)
- `_SUM_OF_VALS` — [`L26`](../../../../../../../../raw/code/aqt/aqt/jax/v2/flax/aqt_flax_calibration.py#L26)

