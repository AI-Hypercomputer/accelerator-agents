---
title: 'Module: graphcast/solar_radiation_test.py'
type: catalog
provenance: extracted
module: graphcast/solar_radiation_test.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.solar_radiation_test`/
symbols:
  GetTsiTest.test_mid_2020_lookup: GetTsiTest#test_mid_2020_lookup().
  SolarRadiationTest.test_full_spatial_resolution: SolarRadiationTest#test_full_spatial_resolution().
  SolarRadiationTest.benchmark: SolarRadiationTest#benchmark().
  SolarRadiationTest.test_missing_dim_raises_value_error: SolarRadiationTest#test_missing_dim_raises_value_error().
  SolarRadiationTest.test_missing_coordinate_raises_value_error: SolarRadiationTest#test_missing_coordinate_raises_value_error().
  SolarRadiationTest.test_shape_multiple_timestamps: SolarRadiationTest#test_shape_multiple_timestamps().
  SolarRadiationTest.test_shape_single_timestamp: SolarRadiationTest#test_shape_single_timestamp().
  GetTsiTest.test_interpolation: GetTsiTest#test_interpolation().
  _get_grid_lat_lon_coords: _get_grid_lat_lon_coords().
  SolarRadiationTest: SolarRadiationTest#
  SolarRadiationTest.setUp: SolarRadiationTest#setUp().
  GetTsiTest: GetTsiTest#
---
# Module: [`graphcast/solar_radiation_test.py`](../../../../../raw/code/graphcast/graphcast/solar_radiation_test.py)

## Classes
### `GetTsiTest`  ·  implements/extends TestCase
- def: [`graphcast/solar_radiation_test.py:163`](../../../../../raw/code/graphcast/graphcast/solar_radiation_test.py#L163)
- signature: `class GetTsiTest(parameterized.TestCase):`
- members:
  - `test_interpolation(self, timestamps: Sequence[np.datetime64], expected_tsi: np.ndarray)` — [`L225`](../../../../../raw/code/graphcast/graphcast/solar_radiation_test.py#L225)
  - `test_mid_2020_lookup(self, loader: solar_radiation.TsiDataLoader, expected_tsi: np.ndarray)` — [`L177`](../../../../../raw/code/graphcast/graphcast/solar_radiation_test.py#L177)
- uses (calls/refs, reference-scoped): [`get_tsi`](solar_radiation.md#get_tsi), [`reference_tsi_data`](solar_radiation.md#reference_tsi_data), [`TsiDataLoader`](solar_radiation.md#TsiDataLoader), [`era5_tsi_data`](solar_radiation.md#era5_tsi_data)

### `SolarRadiationTest`  ·  implements/extends TestCase
- def: [`graphcast/solar_radiation_test.py:44`](../../../../../raw/code/graphcast/graphcast/solar_radiation_test.py#L44)
- signature: `class SolarRadiationTest(parameterized.TestCase):`
- members:
  - `benchmark()` — [`L142`](../../../../../raw/code/graphcast/graphcast/solar_radiation_test.py#L142)
  - `setUp(self)` — [`L46`](../../../../../raw/code/graphcast/graphcast/solar_radiation_test.py#L46)
  - `test_full_spatial_resolution(self, periods: int, repeats: int, use_jit: bool)` — [`L135`](../../../../../raw/code/graphcast/graphcast/solar_radiation_test.py#L135)
  - `test_missing_coordinate_raises_value_error(self)` — [`L63`](../../../../../raw/code/graphcast/graphcast/solar_radiation_test.py#L63)
  - `test_missing_dim_raises_value_error(self)` — [`L50`](../../../../../raw/code/graphcast/graphcast/solar_radiation_test.py#L50)
  - `test_shape_multiple_timestamps(self)` — [`L78`](../../../../../raw/code/graphcast/graphcast/solar_radiation_test.py#L78)
  - `test_shape_single_timestamp(self)` — [`L98`](../../../../../raw/code/graphcast/graphcast/solar_radiation_test.py#L98)
- uses (calls/refs, reference-scoped): [`get_toa_incident_solar_radiation`](solar_radiation.md#get_toa_incident_solar_radiation), [`get_toa_incident_solar_radiation_for_xarray`](solar_radiation.md#get_toa_incident_solar_radiation_for_xarray), [`_get_grid_lat_lon_coords`](solar_radiation_test.md#_get_grid_lat_lon_coords)

## Functions
- `_get_grid_lat_lon_coords(num_lat: int, num_lon: int)` — [`L26`](../../../../../raw/code/graphcast/graphcast/solar_radiation_test.py#L26) — Generates a linear latitude-longitude grid of the given size.

