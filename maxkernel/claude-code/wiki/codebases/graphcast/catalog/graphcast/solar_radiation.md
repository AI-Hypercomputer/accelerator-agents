---
title: 'Module: graphcast/solar_radiation.py'
type: catalog
provenance: extracted
module: graphcast/solar_radiation.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.solar_radiation`/
symbols:
  get_toa_incident_solar_radiation: get_toa_incident_solar_radiation().
  _get_orbital_parameters: _get_orbital_parameters().
  _get_solar_sin_altitude: _get_solar_sin_altitude().
  get_toa_incident_solar_radiation_for_xarray: get_toa_incident_solar_radiation_for_xarray().
  _get_radiation_flux: _get_radiation_flux().
  _DEFAULT_TSI_DATA_LOADER._DEFAULT_TSI_DATA_LOADER: _DEFAULT_TSI_DATA_LOADER._DEFAULT_TSI_DATA_LOADER.
  get_tsi: get_tsi().
  _get_integrated_radiation: _get_integrated_radiation().
  reference_tsi_data: reference_tsi_data().
  _get_j2000_days: _get_j2000_days().
  _get_integrated_radiation_jitted: _get_integrated_radiation_jitted.
  _OrbitalParameters: _OrbitalParameters#
  _DEFAULT_INTEGRATION_PERIOD: _DEFAULT_INTEGRATION_PERIOD.
  _DEFAULT_NUM_INTEGRATION_BINS: _DEFAULT_NUM_INTEGRATION_BINS.
  _TimestampLike: _TimestampLike.
  _TimedeltaLike: _TimedeltaLike.
  TsiDataLoader: TsiDataLoader.
  era5_tsi_data: era5_tsi_data().
  _OrbitalParameters.rotational_phase: _OrbitalParameters#rotational_phase.
  _OrbitalParameters.sin_declination: _OrbitalParameters#sin_declination.
  _OrbitalParameters.cos_declination: _OrbitalParameters#cos_declination.
  _OrbitalParameters.eq_of_time_seconds: _OrbitalParameters#eq_of_time_seconds.
  _OrbitalParameters.solar_distance_au: _OrbitalParameters#solar_distance_au.
  _JULIAN_YEAR_LENGTH_IN_DAYS: _JULIAN_YEAR_LENGTH_IN_DAYS.
  _J2000_EPOCH: _J2000_EPOCH.
  _SECONDS_PER_DAY: _SECONDS_PER_DAY.
  _REFERENCE_TSI: _REFERENCE_TSI.
  _OrbitalParameters.theta: _OrbitalParameters#theta.
---
# Module: [`graphcast/solar_radiation.py`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py)

## Classes
### `_OrbitalParameters`
- def: [`graphcast/solar_radiation.py:157`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L157)
- doc: Parameters characterising Earth's position relative to the Sun.
- signature: `class _OrbitalParameters:`
- members:
  - `cos_declination` — [`L180`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L180)
  - `eq_of_time_seconds` — [`L181`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L181)
  - `rotational_phase` — [`L178`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L178)
  - `sin_declination` — [`L179`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L179)
  - `solar_distance_au` — [`L182`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L182)
  - `theta` — [`L177`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L177)
- used by: [`_get_orbital_parameters`](solar_radiation.md#_get_orbital_parameters), [`_get_solar_sin_altitude`](solar_radiation.md#_get_solar_sin_altitude), [`_get_radiation_flux`](solar_radiation.md#_get_radiation_flux)

## Functions
- `_get_integrated_radiation(j2000_days: chex.Array, sin_latitude: chex.Array, cos_latitude: chex.Array, longitude: chex.Array, tsi: chex.Array, integration_period: pd.Timedelta, num_integration_bins: int)` — [`L368`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L368) — Returns the TOA solar radiation flux integrated over a time period.
- `_get_j2000_days(timestamp: pd.Timestamp)` — [`L185`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L185) — Returns the number of days since the J2000 epoch.
- `_get_orbital_parameters(j2000_days: chex.Array)` — [`L197`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L197) — Computes the orbital parameters for the given J2000 days.
- `_get_radiation_flux(j2000_days: chex.Array, sin_latitude: chex.Array, cos_latitude: chex.Array, longitude: chex.Array, tsi: chex.Array)` — [`L328`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L328) — Computes the instantaneous TOA incident solar radiation flux.
- `_get_solar_sin_altitude(op: _OrbitalParameters, sin_latitude: chex.Array, cos_latitude: chex.Array, longitude: chex.Array)` — [`L293`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L293) — Returns the sine of the solar altitude angle.
- `era5_tsi_data()` — [`L83`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L83) — A TsiDataProvider that returns ERA5 compatible TSI data.
- `get_toa_incident_solar_radiation(timestamps: Sequence[_TimestampLike], latitude: chex.Array, longitude: chex.Array, tsi_data: xa.DataArray | None = None, integration_period: _TimedeltaLike = _DEFAULT_INTEGRATION_PERIOD, num_integration_bins: int = _DEFAULT_NUM_INTEGRATION_BINS, use_jit: bool = False)` — [`L443`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L443) — Computes the solar radiation incident at the top of the atmosphere.
- `get_toa_incident_solar_radiation_for_xarray(data_array_like: xa.DataArray | xa.Dataset, tsi_data: xa.DataArray | None = None, integration_period: _TimedeltaLike = _DEFAULT_INTEGRATION_PERIOD, num_integration_bins: int = _DEFAULT_NUM_INTEGRATION_BINS, use_jit: bool = False)` — [`L523`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L523) — Computes the solar radiation incident at the top of the atmosphere.
- `get_tsi(timestamps: Sequence[_TimestampLike], tsi_data: xa.DataArray)` — [`L131`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L131) — Returns TSI values for the given timestamps.
- `reference_tsi_data()` — [`L74`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L74) — A TsiDataProvider that returns a single reference TSI value.

## Module values
- `TsiDataLoader` — [`L63`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L63)
- `_DEFAULT_INTEGRATION_PERIOD` — [`L37`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L37)
- `_DEFAULT_NUM_INTEGRATION_BINS` — [`L41`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L41)
- `_DEFAULT_TSI_DATA_LOADER` — [`L128`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L128)
- `_J2000_EPOCH` — [`L49`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L49)
- `_JULIAN_YEAR_LENGTH_IN_DAYS` — [`L45`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L45)
- `_REFERENCE_TSI` — [`L71`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L71)
- `_SECONDS_PER_DAY` — [`L52`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L52)
- `_TimedeltaLike` — [`L56`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L56)
- `_TimestampLike` — [`L55`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L55)
- `_get_integrated_radiation_jitted` — [`L437`](../../../../../raw/code/graphcast/graphcast/solar_radiation.py#L437)

