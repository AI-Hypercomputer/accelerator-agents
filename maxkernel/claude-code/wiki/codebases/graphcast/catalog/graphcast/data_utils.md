---
title: 'Module: graphcast/data_utils.py'
type: catalog
provenance: extracted
module: graphcast/data_utils.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.data_utils`/
symbols:
  add_derived_vars: add_derived_vars().
  extract_inputs_targets_forcings: extract_inputs_targets_forcings().
  SEC_PER_DAY: SEC_PER_DAY.
  AVG_SEC_PER_YEAR: AVG_SEC_PER_YEAR.
  add_tisr_var: add_tisr_var().
  extract_input_target_times: extract_input_target_times().
  DAY_PROGRESS: DAY_PROGRESS.
  YEAR_PROGRESS: YEAR_PROGRESS.
  get_year_progress: get_year_progress().
  get_day_progress: get_day_progress().
  TISR: TISR.
  _DERIVED_VARS: _DERIVED_VARS.
  _process_target_lead_times_and_get_duration: _process_target_lead_times_and_get_duration().
  TargetLeadTimes: TargetLeadTimes.
  TimedeltaLike: TimedeltaLike.
  featurize_progress: featurize_progress().
  _AVG_DAY_PER_YEAR: _AVG_DAY_PER_YEAR.
  _SEC_PER_HOUR: _SEC_PER_HOUR.
  _HOUR_PER_DAY: _HOUR_PER_DAY.
  get_seconds_since_epoch: get_seconds_since_epoch().
  TimedeltaStr: TimedeltaStr.
---
# Module: [`graphcast/data_utils.py`](../../../../../raw/code/graphcast/graphcast/data_utils.py)

## Functions
- `_process_target_lead_times_and_get_duration(target_lead_times: TargetLeadTimes)` — [`L296`](../../../../../raw/code/graphcast/graphcast/data_utils.py#L296) — Returns the minimum duration for the target lead times.
- `add_derived_vars(data: xarray.Dataset)` — [`L142`](../../../../../raw/code/graphcast/graphcast/data_utils.py#L142) — Adds year and day progress features to `data` in place if missing.
- `add_tisr_var(data: xarray.Dataset)` — [`L184`](../../../../../raw/code/graphcast/graphcast/data_utils.py#L184) — Adds TISR feature to `data` in place if missing.
- `extract_input_target_times(dataset: xarray.Dataset, input_duration: TimedeltaLike, target_lead_times: TargetLeadTimes)` — [`L215`](../../../../../raw/code/graphcast/graphcast/data_utils.py#L215) — Extracts inputs and targets for prediction, from a Dataset with a time dim.
- `extract_inputs_targets_forcings(dataset: xarray.Dataset, *, input_variables: Tuple[str, ...], target_variables: Tuple[str, ...], forcing_variables: Tuple[str, ...], pressure_levels: Tuple[int, ...], input_duration: TimedeltaLike, target_lead_times: TargetLeadTimes)` — [`L322`](../../../../../raw/code/graphcast/graphcast/data_utils.py#L322) — Extracts inputs, targets and forcings according to requirements.
- `featurize_progress(name: str, dims: Sequence[str], progress: np.ndarray)` — [`L103`](../../../../../raw/code/graphcast/graphcast/data_utils.py#L103) — Derives features used by ML models from the `progress` variable.
- `get_day_progress(seconds_since_epoch: np.ndarray, longitude: np.ndarray)` — [`L74`](../../../../../raw/code/graphcast/graphcast/data_utils.py#L74) — Computes day progress for times in seconds at each longitude.
- `get_seconds_since_epoch(datetime_sequence: xarray.DataArray)` — [`L135`](../../../../../raw/code/graphcast/graphcast/data_utils.py#L135) — Computes seconds since epoch from `data` in place if missing.
- `get_year_progress(seconds_since_epoch: np.ndarray)` — [`L51`](../../../../../raw/code/graphcast/graphcast/data_utils.py#L51) — Computes year progress for times in seconds.

## Module values
- `AVG_SEC_PER_YEAR` — [`L36`](../../../../../raw/code/graphcast/graphcast/data_utils.py#L36)
- `DAY_PROGRESS` — [`L38`](../../../../../raw/code/graphcast/graphcast/data_utils.py#L38)
- `SEC_PER_DAY` — [`L34`](../../../../../raw/code/graphcast/graphcast/data_utils.py#L34)
- `TISR` — [`L48`](../../../../../raw/code/graphcast/graphcast/data_utils.py#L48)
- `TargetLeadTimes` — [`L26`](../../../../../raw/code/graphcast/graphcast/data_utils.py#L26)
- `TimedeltaLike` — [`L23`](../../../../../raw/code/graphcast/graphcast/data_utils.py#L23)
- `TimedeltaStr` — [`L24`](../../../../../raw/code/graphcast/graphcast/data_utils.py#L24)
- `YEAR_PROGRESS` — [`L39`](../../../../../raw/code/graphcast/graphcast/data_utils.py#L39)
- `_AVG_DAY_PER_YEAR` — [`L35`](../../../../../raw/code/graphcast/graphcast/data_utils.py#L35)
- `_DERIVED_VARS` — [`L40`](../../../../../raw/code/graphcast/graphcast/data_utils.py#L40)
- `_HOUR_PER_DAY` — [`L33`](../../../../../raw/code/graphcast/graphcast/data_utils.py#L33)
- `_SEC_PER_HOUR` — [`L32`](../../../../../raw/code/graphcast/graphcast/data_utils.py#L32)

