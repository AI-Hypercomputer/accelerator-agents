---
title: 'Module: graphcast/dpm_solver_plus_plus_2s.py'
type: catalog
provenance: extracted
module: graphcast/dpm_solver_plus_plus_2s.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.dpm_solver_plus_plus_2s`/Sampler#
symbols:
  Sampler.body_fn: body_fn().
  Sampler.__call__: __call__().
  Sampler.denoiser: denoiser().
  Sampler._per_step_churn_rates: _per_step_churn_rates.
  Sampler.__init__: __init__().
  Sampler._noise_levels: _noise_levels.
  Sampler: ''
  Sampler.init_noise: init_noise().
  Sampler._stochastic_churn: _stochastic_churn.
  Sampler._noise_level_inflation_factor: _noise_level_inflation_factor.
---
# Module: [`graphcast/dpm_solver_plus_plus_2s.py`](../../../../../raw/code/graphcast/graphcast/dpm_solver_plus_plus_2s.py)

## Classes
### `Sampler`  ·  implements/extends Sampler
- def: [`graphcast/dpm_solver_plus_plus_2s.py:28`](../../../../../raw/code/graphcast/graphcast/dpm_solver_plus_plus_2s.py#L28)
- doc: Sampling using DPM-Solver++ 2S from \[1\].
- signature: `class Sampler(base.Sampler):`
- members:
  - `__init__(self, denoiser: denoisers_base.Denoiser, max_noise_level: float, min_noise_level: float, num_noise_levels: int, rho: float, stochastic_churn_rate: float, churn_min_noise_level: float, churn_max_noise_level: float, noise_level_inflation_factor: float)` — [`L45`](../../../../../raw/code/graphcast/graphcast/dpm_solver_plus_plus_2s.py#L45) — Initializes the sampler.
  - `body_fn(i: jnp.ndarray, x: xarray.Dataset)` — [`L114`](../../../../../raw/code/graphcast/graphcast/dpm_solver_plus_plus_2s.py#L114) — One iteration of the sampling algorithm.
  - `denoiser(noise_level: jnp.ndarray, x: xarray.Dataset)` — [`L102`](../../../../../raw/code/graphcast/graphcast/dpm_solver_plus_plus_2s.py#L102) — Computes D(x, sigma, y).
  - `init_noise(template)` — [`L125`](../../../../../raw/code/graphcast/graphcast/dpm_solver_plus_plus_2s.py#L125)
- protocol/private: `__call__`[`L91`](../../../../../raw/code/graphcast/graphcast/dpm_solver_plus_plus_2s.py#L91), `_noise_level_inflation_factor`[`L89`](../../../../../raw/code/graphcast/graphcast/dpm_solver_plus_plus_2s.py#L89), `_noise_levels`[`L83`](../../../../../raw/code/graphcast/graphcast/dpm_solver_plus_plus_2s.py#L83), `_per_step_churn_rates`[`L86`](../../../../../raw/code/graphcast/graphcast/dpm_solver_plus_plus_2s.py#L86), `_stochastic_churn`[`L85`](../../../../../raw/code/graphcast/graphcast/dpm_solver_plus_plus_2s.py#L85)
- uses (calls/refs, reference-scoped): [`DataArray`](xarray_jax.md#DataArray), [`infer_floating_dtype`](casting.md#infer_floating_dtype), [`Denoiser`](denoisers_base.md#Denoiser), [`__init__`](samplers_base.md#Sampler.__init__), [`spherical_white_noise_like`](samplers_utils.md#spherical_white_noise_like), [`_denoiser`](samplers_base.md#Sampler._denoiser), [`Sampler`](samplers_base.md#Sampler), [`apply_stochastic_churn`](samplers_utils.md#apply_stochastic_churn), [`noise_schedule`](samplers_utils.md#noise_schedule), [`stochastic_churn_rate_schedule`](samplers_utils.md#stochastic_churn_rate_schedule), [`tree_where`](samplers_utils.md#tree_where)
- used by: [`__call__`](gencast.md#GenCast.__call__), [`Sampler`](samplers_base.md#Sampler)

