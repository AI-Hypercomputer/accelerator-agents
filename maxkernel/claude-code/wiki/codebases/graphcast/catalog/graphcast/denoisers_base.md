---
title: 'Module: graphcast/denoisers_base.py'
type: catalog
provenance: extracted
module: graphcast/denoisers_base.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.denoisers_base`/Denoiser#
symbols:
  Denoiser: ''
  Denoiser.__call__: __call__().
---
# Module: [`graphcast/denoisers_base.py`](../../../../../raw/code/graphcast/graphcast/denoisers_base.py)

## Classes
### `Denoiser`  ·  implements/extends Protocol
- def: [`graphcast/denoisers_base.py:28`](../../../../../raw/code/graphcast/graphcast/denoisers_base.py#L28)
- doc: A denoising model that conditions on inputs as well as noise level.
- signature: `class Denoiser(Protocol):`
- members:
  - `__call__(self, inputs: xarray.Dataset, noisy_targets: xarray.Dataset, noise_levels: xarray.DataArray, forcings: Optional[xarray.Dataset] = None, **kwargs)` — [`L31`](../../../../../raw/code/graphcast/graphcast/denoisers_base.py#L31) — Computes denoised targets from noisy targets.
- uses (calls/refs, reference-scoped): [`Denoiser`](denoiser.md#Denoiser)
- used by: [`__init__`](samplers_base.md#Sampler.__init__), [`__init__`](dpm_solver_plus_plus_2s.md#Sampler.__init__), [`_denoiser`](samplers_base.md#Sampler._denoiser), [`Denoiser`](denoiser.md#Denoiser)

