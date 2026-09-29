---
title: 'Module: graphcast/gencast.py'
type: catalog
provenance: extracted
module: graphcast/gencast.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.gencast`/
symbols:
  GenCast.loss: GenCast#loss().
  GenCast.__init__: GenCast#__init__().
  TASK: TASK.
  GenCast._preconditioned_denoiser: GenCast#_preconditioned_denoiser().
  GenCast.__call__: GenCast#__call__().
  GenCast.loss_and_predictions: GenCast#loss_and_predictions().
  GenCast._noise_config: GenCast#_noise_config.
  GenCast._denoiser: GenCast#_denoiser.
  GenCast._loss_weighting: GenCast#_loss_weighting().
  GenCast._sampler: GenCast#_sampler.
  CheckPoint.task_config: CheckPoint#task_config.
  CheckPoint.denoiser_architecture_config: CheckPoint#denoiser_architecture_config.
  CheckPoint.sampler_config: CheckPoint#sampler_config.
  CheckPoint.noise_config: CheckPoint#noise_config.
  CheckPoint.noise_encoder_config: CheckPoint#noise_encoder_config.
  GenCast: GenCast#
  SamplerConfig: SamplerConfig#
  NoiseConfig: NoiseConfig#
  GenCast._sampler_config: GenCast#_sampler_config.
  GenCast._c_out: GenCast#_c_out().
  TARGET_SURFACE_VARS: TARGET_SURFACE_VARS.
  TARGET_SURFACE_NO_PRECIP_VARS: TARGET_SURFACE_NO_PRECIP_VARS.
  NoiseConfig.training_noise_level_rho: NoiseConfig#training_noise_level_rho.
  NoiseConfig.training_max_noise_level: NoiseConfig#training_max_noise_level.
  NoiseConfig.training_min_noise_level: NoiseConfig#training_min_noise_level.
  GenCast._c_in: GenCast#_c_in().
  GenCast._c_skip: GenCast#_c_skip().
  SamplerConfig.max_noise_level: SamplerConfig#max_noise_level.
  SamplerConfig.min_noise_level: SamplerConfig#min_noise_level.
  SamplerConfig.num_noise_levels: SamplerConfig#num_noise_levels.
  SamplerConfig.rho: SamplerConfig#rho.
  SamplerConfig.stochastic_churn_rate: SamplerConfig#stochastic_churn_rate.
  SamplerConfig.churn_min_noise_level: SamplerConfig#churn_min_noise_level.
  SamplerConfig.churn_max_noise_level: SamplerConfig#churn_max_noise_level.
  SamplerConfig.noise_level_inflation_factor: SamplerConfig#noise_level_inflation_factor.
  CheckPoint: CheckPoint#
  CheckPoint.description: CheckPoint#description.
  CheckPoint.license: CheckPoint#license.
  CheckPoint.params: CheckPoint#params.
---
# Module: [`graphcast/gencast.py`](../../../../../raw/code/graphcast/graphcast/gencast.py)

## Classes
### `CheckPoint`
- def: [`graphcast/gencast.py:119`](../../../../../raw/code/graphcast/graphcast/gencast.py#L119)
- signature: `class CheckPoint:`
- members:
  - `denoiser_architecture_config` — [`L124`](../../../../../raw/code/graphcast/graphcast/gencast.py#L124)
  - `description` — [`L120`](../../../../../raw/code/graphcast/graphcast/gencast.py#L120)
  - `license` — [`L121`](../../../../../raw/code/graphcast/graphcast/gencast.py#L121)
  - `noise_config` — [`L126`](../../../../../raw/code/graphcast/graphcast/gencast.py#L126)
  - `noise_encoder_config` — [`L127`](../../../../../raw/code/graphcast/graphcast/gencast.py#L127)
  - `params` — [`L122`](../../../../../raw/code/graphcast/graphcast/gencast.py#L122)
  - `sampler_config` — [`L125`](../../../../../raw/code/graphcast/graphcast/gencast.py#L125)
  - `task_config` — [`L123`](../../../../../raw/code/graphcast/graphcast/gencast.py#L123)
- uses (calls/refs, reference-scoped): [`TaskConfig`](graphcast.md#TaskConfig), [`DenoiserArchitectureConfig`](denoiser.md#DenoiserArchitectureConfig), [`NoiseEncoderConfig`](denoiser.md#NoiseEncoderConfig), [`NoiseConfig`](gencast.md#NoiseConfig), [`SamplerConfig`](gencast.md#SamplerConfig)

### `GenCast`  ·  implements/extends Predictor
- def: [`graphcast/gencast.py:130`](../../../../../raw/code/graphcast/graphcast/gencast.py#L130)
- doc: Predictor for a denoising diffusion model following the framework of \[1\].
- signature: `class GenCast(predictor_base.Predictor):`
- members:
  - `__init__(self, task_config: graphcast.TaskConfig, denoiser_architecture_config: denoiser.DenoiserArchitectureConfig, sampler_config: Optional[SamplerConfig] = None, noise_config: Optional[NoiseConfig] = None, noise_encoder_config: Optional[denoiser.NoiseEncoderConfig] = None)` — [`L145`](../../../../../raw/code/graphcast/graphcast/gencast.py#L145) — Constructs GenCast.
  - `_c_in(self, noise_scale: xarray.DataArray)` — [`L177`](../../../../../raw/code/graphcast/graphcast/gencast.py#L177) — Scaling applied to the noisy targets input to the underlying network.
  - `_c_out(self, noise_scale: xarray.DataArray)` — [`L181`](../../../../../raw/code/graphcast/graphcast/gencast.py#L181) — Scaling applied to the underlying network's raw outputs.
  - `_c_skip(self, noise_scale: xarray.DataArray)` — [`L185`](../../../../../raw/code/graphcast/graphcast/gencast.py#L185) — Scaling applied to the skip connection.
  - `_loss_weighting(self, noise_scale: xarray.DataArray)` — [`L189`](../../../../../raw/code/graphcast/graphcast/gencast.py#L189) — The loss weighting \lambda(\sigma) from the paper.
  - `_preconditioned_denoiser(self, inputs: xarray.Dataset, noisy_targets: xarray.Dataset, noise_levels: xarray.DataArray, forcings: Optional[xarray.Dataset] = None, **kwargs)` — [`L193`](../../../../../raw/code/graphcast/graphcast/gencast.py#L193) — The preconditioned denoising function D from the paper (Eqn 7).
  - `loss(self, inputs: xarray.Dataset, targets: xarray.Dataset, forcings: Optional[xarray.Dataset] = None)` — [`L218`](../../../../../raw/code/graphcast/graphcast/gencast.py#L218) — documented in [graphcast-casting](../../concepts/graphcast-casting.md)
  - `loss_and_predictions(self, inputs: xarray.Dataset, targets: xarray.Dataset, forcings: Optional[xarray.Dataset] = None)` — [`L210`](../../../../../raw/code/graphcast/graphcast/gencast.py#L210)
- protocol/private: `__call__`[`L271`](../../../../../raw/code/graphcast/graphcast/gencast.py#L271), `_denoiser`[`L168`](../../../../../raw/code/graphcast/graphcast/gencast.py#L168), `_noise_config`[`L175`](../../../../../raw/code/graphcast/graphcast/gencast.py#L175), `_sampler`[`L174`](../../../../../raw/code/graphcast/graphcast/gencast.py#L174), `_sampler_config`[`L172`](../../../../../raw/code/graphcast/graphcast/gencast.py#L172)
- uses (calls/refs, reference-scoped): [`DataArray`](xarray_jax.md#DataArray), [`Predictor`](predictor_base.md#Predictor), [`LossAndDiagnostics`](predictor_base.md#LossAndDiagnostics), [`weighted_mse_per_level`](losses.md#weighted_mse_per_level), [`TaskConfig`](graphcast.md#TaskConfig), [`infer_floating_dtype`](casting.md#infer_floating_dtype), [`spherical_white_noise_like`](samplers_utils.md#spherical_white_noise_like), [`DenoiserArchitectureConfig`](denoiser.md#DenoiserArchitectureConfig), [`NoiseEncoderConfig`](denoiser.md#NoiseEncoderConfig), [`ALL_ATMOSPHERIC_VARS`](graphcast.md#ALL_ATMOSPHERIC_VARS), [`target_variables`](graphcast.md#TaskConfig.target_variables), [`Denoiser`](denoiser.md#Denoiser), [`Sampler`](dpm_solver_plus_plus_2s.md#Sampler), [`NoiseConfig`](gencast.md#NoiseConfig), [`SamplerConfig`](gencast.md#SamplerConfig), [`node_output_size`](denoiser.md#DenoiserArchitectureConfig.node_output_size), [`pressure_levels`](graphcast.md#TaskConfig.pressure_levels), [`rho_inverse_cdf`](samplers_utils.md#rho_inverse_cdf), [`training_max_noise_level`](gencast.md#NoiseConfig.training_max_noise_level), [`training_min_noise_level`](gencast.md#NoiseConfig.training_min_noise_level), [`training_noise_level_rho`](gencast.md#NoiseConfig.training_noise_level_rho)
- used by: [`Predictor`](predictor_base.md#Predictor), [`loss`](predictor_base.md#Predictor.loss), [`loss_and_predictions`](predictor_base.md#Predictor.loss_and_predictions)

### `NoiseConfig`
- def: [`graphcast/gencast.py:112`](../../../../../raw/code/graphcast/graphcast/gencast.py#L112)
- signature: `class NoiseConfig:`
- members:
  - `training_max_noise_level` — [`L114`](../../../../../raw/code/graphcast/graphcast/gencast.py#L114)
  - `training_min_noise_level` — [`L115`](../../../../../raw/code/graphcast/graphcast/gencast.py#L115)
  - `training_noise_level_rho` — [`L113`](../../../../../raw/code/graphcast/graphcast/gencast.py#L113)
- used by: [`loss`](gencast.md#GenCast.loss), [`__init__`](gencast.md#GenCast.__init__), [`noise_config`](gencast.md#CheckPoint.noise_config)

### `SamplerConfig`
- def: [`graphcast/gencast.py:75`](../../../../../raw/code/graphcast/graphcast/gencast.py#L75)
- doc: Configures the sampler used to draw samples from GenCast.
- signature: `class SamplerConfig:`
- members:
  - `churn_max_noise_level` — [`L107`](../../../../../raw/code/graphcast/graphcast/gencast.py#L107)
  - `churn_min_noise_level` — [`L106`](../../../../../raw/code/graphcast/graphcast/gencast.py#L106)
  - `max_noise_level` — [`L100`](../../../../../raw/code/graphcast/graphcast/gencast.py#L100)
  - `min_noise_level` — [`L101`](../../../../../raw/code/graphcast/graphcast/gencast.py#L101)
  - `noise_level_inflation_factor` — [`L108`](../../../../../raw/code/graphcast/graphcast/gencast.py#L108)
  - `num_noise_levels` — [`L102`](../../../../../raw/code/graphcast/graphcast/gencast.py#L102)
  - `rho` — [`L103`](../../../../../raw/code/graphcast/graphcast/gencast.py#L103)
  - `stochastic_churn_rate` — [`L105`](../../../../../raw/code/graphcast/graphcast/gencast.py#L105)
- used by: [`__init__`](gencast.md#GenCast.__init__), [`sampler_config`](gencast.md#CheckPoint.sampler_config)

## Module values
- `TARGET_SURFACE_NO_PRECIP_VARS` — [`L48`](../../../../../raw/code/graphcast/graphcast/gencast.py#L48)
- `TARGET_SURFACE_VARS` — [`L39`](../../../../../raw/code/graphcast/graphcast/gencast.py#L39)
- `TASK` — [`L57`](../../../../../raw/code/graphcast/graphcast/gencast.py#L57)

