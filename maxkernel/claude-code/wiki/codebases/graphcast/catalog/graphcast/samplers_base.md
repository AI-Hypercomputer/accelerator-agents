---
title: 'Module: graphcast/samplers_base.py'
type: catalog
provenance: extracted
module: graphcast/samplers_base.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.samplers_base`/Sampler#
symbols:
  Sampler.__init__: __init__().
  Sampler._denoiser: _denoiser.
  Sampler: ''
  Sampler.__call__: __call__().
---
# Module: [`graphcast/samplers_base.py`](../../../../../raw/code/graphcast/graphcast/samplers_base.py)

## Classes
### `Sampler`  ·  implements/extends ABC
- def: [`graphcast/samplers_base.py:23`](../../../../../raw/code/graphcast/graphcast/samplers_base.py#L23)
- doc: A sampling algorithm for a denoising diffusion model.
- signature: `class Sampler(abc.ABC):`
- members:
  - `__call__(self, inputs: xarray.Dataset, targets_template: xarray.Dataset, forcings: Optional[xarray.Dataset] = None, **kwargs)` — [`L41`](../../../../../raw/code/graphcast/graphcast/samplers_base.py#L41) — Draws a sample using self._denoiser. Contract like Predictor.__call__.
  - `__init__(self, denoiser: denoisers_base.Denoiser)` — [`L31`](../../../../../raw/code/graphcast/graphcast/samplers_base.py#L31) — Constructs Sampler.
- protocol/private: `_denoiser`[`L29`](../../../../../raw/code/graphcast/graphcast/samplers_base.py#L29)
- uses (calls/refs, reference-scoped): [`Denoiser`](denoisers_base.md#Denoiser), [`Sampler`](dpm_solver_plus_plus_2s.md#Sampler)
- used by: [`denoiser`](dpm_solver_plus_plus_2s.md#Sampler.denoiser), [`__init__`](dpm_solver_plus_plus_2s.md#Sampler.__init__), [`Sampler`](dpm_solver_plus_plus_2s.md#Sampler)

