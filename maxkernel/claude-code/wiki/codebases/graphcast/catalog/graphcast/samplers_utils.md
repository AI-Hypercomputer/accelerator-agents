---
title: 'Module: graphcast/samplers_utils.py'
type: catalog
provenance: extracted
module: graphcast/samplers_utils.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.samplers_utils`/
symbols:
  Grid.inverse_transform: Grid#inverse_transform().
  sample: sample().
  _ArrayGrid.with_lat_lon: _ArrayGrid#with_lat_lon().
  Grid.for_nodal_data: Grid#for_nodal_data().
  Grid.__init__: Grid#__init__().
  Grid.to_nodal: Grid#to_nodal().
  spherical_white_noise_like.spherical_white_noise_like_dataarray: spherical_white_noise_like().spherical_white_noise_like_dataarray().
  Grid._total_wavenumber_coords: Grid#_total_wavenumber_coords.
  spherical_white_noise_like: spherical_white_noise_like().
  _ArrayGrid.nodal_axes: _ArrayGrid#nodal_axes().
  Grid.total_wavenumber_coords: Grid#total_wavenumber_coords().
  Grid.longitude_wavenumber_coords: Grid#longitude_wavenumber_coords().
  _ArrayGrid.modal_axes: _ArrayGrid#modal_axes().
  _ArrayGrid.to_nodal: _ArrayGrid#to_nodal().
  noise_schedule: noise_schedule().
  apply_stochastic_churn: apply_stochastic_churn().
  _ArrayGrid: _ArrayGrid#
  _ArrayGrid._grid: _ArrayGrid#_grid().
  Grid: Grid#
  Grid._longitude_wavenumber_coords: Grid#_longitude_wavenumber_coords.
  EARTH_CIRCUMFERENCE_KM: EARTH_CIRCUMFERENCE_KM.
  _verify_nodal_axes: _verify_nodal_axes().
  rho_inverse_cdf: rho_inverse_cdf().
  EARTH_RADIUS_KM: EARTH_RADIUS_KM.
  _infer_latitude_spacing: _infer_latitude_spacing().
  Grid._underlying: Grid#_underlying.
  Grid._lat_coords: Grid#_lat_coords.
  Grid._lon_coords: Grid#_lon_coords.
  tree_where: tree_where().
  stochastic_churn_rate_schedule: stochastic_churn_rate_schedule().
  _ArrayGrid.longitude_wavenumbers: _ArrayGrid#longitude_wavenumbers.
  _ArrayGrid.total_wavenumbers: _ArrayGrid#total_wavenumbers.
  _ArrayGrid.longitude_nodes: _ArrayGrid#longitude_nodes.
  _ArrayGrid.latitude_nodes: _ArrayGrid#latitude_nodes.
  _ArrayGrid.latitude_spacing: _ArrayGrid#latitude_spacing.
---
# Module: [`graphcast/samplers_utils.py`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py)

## Classes
### `Grid`
- def: [`graphcast/samplers_utils.py:149`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L149)
- doc: xarray wrapper around _ArrayGrid.
- signature: `class Grid:`
- members:
  - `for_nodal_data(cls, nodal_data: xarray.DataArray)` — [`L153`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L153) — A Grid for use with a given shape of nodal (lat/lon grid) data.
  - `inverse_transform(modal: xarray.DataArray)` — [`L231`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L231)
  - `longitude_wavenumber_coords(self)` — [`L209`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L209) — Coords that must be used for 'longitude_wavenumber' dimension.
  - `to_nodal(self, modal_data: xarray.DataArray)` — [`L216`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L216) — Applies the inverse spherical harmonic transform.
  - `total_wavenumber_coords(self)` — [`L201`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L201) — Coords that must be used for 'total_wavenumber' dimension.
- protocol/private: `__init__`[`L186`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L186), `_lat_coords`[`L195`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L195), `_lon_coords`[`L196`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L196), `_longitude_wavenumber_coords`[`L197`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L197), `_total_wavenumber_coords`[`L197`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L197), `_underlying`[`L191`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L191)
- uses (calls/refs, reference-scoped): [`map_structure`](xarray_tree.md#map_structure), [`with_lat_lon`](samplers_utils.md#_ArrayGrid.with_lat_lon), [`apply_ufunc`](xarray_jax.md#apply_ufunc), [`nodal_axes`](samplers_utils.md#_ArrayGrid.nodal_axes), [`modal_axes`](samplers_utils.md#_ArrayGrid.modal_axes), [`to_nodal`](samplers_utils.md#_ArrayGrid.to_nodal), [`_ArrayGrid`](samplers_utils.md#_ArrayGrid), [`_verify_nodal_axes`](samplers_utils.md#_verify_nodal_axes)
- used by: [`sample`](samplers_utils.md#sample)

### `_ArrayGrid`
- def: [`graphcast/samplers_utils.py:42`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L42)
- doc: A class that performs operations and transformations in the spectral basis.
- signature: `class _ArrayGrid:`
- members:
  - `modal_axes(self)` — [`L112`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L112) — Longitudinal and total wavenumbers (m, l) of the modal basis.
  - `nodal_axes(self)` — [`L107`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L107) — Longitude and sin(latitude) coordinates of the nodal basis.
  - `to_nodal(self, x: chex.Array)` — [`L116`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L116) — Maps `x` from a modal to nodal representation.
  - `with_lat_lon(cls, lat: np.ndarray, lon: np.ndarray)` — [`L60`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L60) — _ArrayGrid for use with data in specified lat/lon grid (in degrees).
  - `latitude_nodes` — [`L56`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L56)
  - `latitude_spacing` — [`L57`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L57)
  - `longitude_nodes` — [`L55`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L55)
  - `longitude_wavenumbers` — [`L53`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L53)
  - `total_wavenumbers` — [`L54`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L54)
- protocol/private: `_grid`[`L100`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L100)
- uses (calls/refs, reference-scoped): [`_verify_nodal_axes`](samplers_utils.md#_verify_nodal_axes), [`_infer_latitude_spacing`](samplers_utils.md#_infer_latitude_spacing)
- used by: [`inverse_transform`](samplers_utils.md#Grid.inverse_transform), [`for_nodal_data`](samplers_utils.md#Grid.for_nodal_data), [`__init__`](samplers_utils.md#Grid.__init__), [`_total_wavenumber_coords`](samplers_utils.md#Grid._total_wavenumber_coords)

## Functions
- `_infer_latitude_spacing(lat: np.ndarray)` — [`L121`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L121) — Infers the type of latitude spacing given the latitude.
- `_verify_nodal_axes(lat_coords: np.ndarray, lon_coords: np.ndarray, nodal_axes: Tuple[np.ndarray, np.ndarray])` — [`L136`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L136)
- `apply_stochastic_churn(x: Any, noise_level: jax.typing.ArrayLike, stochastic_churn_rate: jax.typing.ArrayLike, noise_level_inflation_factor: jax.typing.ArrayLike)` — [`L418`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L418) — Returns x at higher noise level, and the higher noise level itself.
- `noise_schedule(max_noise_level: float = 80, min_noise_level: float = 0.002, num_noise_levels: int = 30, rho: float = 7)` — [`L379`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L379) — Computes a descending noise schedule for sampling, ending with zero.
- `rho_inverse_cdf(min_value: float, max_value: float, rho: float, cdf: Any)` — [`L334`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L334) — Quantiles of rho distribution used for noise levels at sampling time.
- `sample(key: jnp.ndarray, power_spectrum: xarray.DataArray, template: xarray.DataArray, grid: Optional[Grid] = None)` — [`L250`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L250) — Samples Gaussian Process noise on a sphere, with a given power spectrum.
- `spherical_white_noise_like(template: xarray.Dataset)` — [`L319`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L319) — Samples isotropic mean 0 variance 1 white noise on the sphere.
- `spherical_white_noise_like_dataarray(data_array: xarray.DataArray)` — [`L321`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L321)
- `stochastic_churn_rate_schedule(noise_levels: np.ndarray, stochastic_churn_rate: float = 0, churn_min_noise_level: float = 0.05, churn_max_noise_level: float = 50)` — [`L399`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L399) — Computes a stochastic churn rate for each noise level.
- `tree_where(cond: jnp.ndarray, xs: Any, ys: Any)` — [`L370`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L370) — Like jnp.where but works with trees for xs and ys (but not for cond).

## Module values
- `EARTH_CIRCUMFERENCE_KM` — [`L38`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L38)
- `EARTH_RADIUS_KM` — [`L34`](../../../../../raw/code/graphcast/graphcast/samplers_utils.py#L34)

