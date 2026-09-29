---
title: 'Module: src/maxdiffusion/schedulers/scheduling_ddpm_flax.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/schedulers/scheduling_ddpm_flax.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.schedulers.scheduling_ddpm_flax`/
symbols:
  FlaxDDPMScheduler.step: FlaxDDPMScheduler#step().
  FlaxDDPMScheduler._get_variance: FlaxDDPMScheduler#_get_variance().
  FlaxDDPMScheduler.create_state: FlaxDDPMScheduler#create_state().
  DDPMSchedulerState.common: DDPMSchedulerState#common.
  DDPMSchedulerState: DDPMSchedulerState#
  FlaxDDPMScheduler.add_noise: FlaxDDPMScheduler#add_noise().
  FlaxDDPMScheduler.get_velocity: FlaxDDPMScheduler#get_velocity().
  FlaxDDPMSchedulerOutput: FlaxDDPMSchedulerOutput#
  FlaxDDPMScheduler: FlaxDDPMScheduler#
  FlaxDDPMScheduler.dtype: FlaxDDPMScheduler#dtype.
  FlaxDDPMScheduler.random_variance: FlaxDDPMScheduler#random_variance().
  FlaxDDPMScheduler.__init__: FlaxDDPMScheduler#__init__().
  FlaxDDPMScheduler.set_timesteps: FlaxDDPMScheduler#set_timesteps().
  DDPMSchedulerState.create: DDPMSchedulerState#create().
  FlaxDDPMScheduler._compatibles: FlaxDDPMScheduler#_compatibles.
  FlaxDDPMScheduler.scale_model_input: FlaxDDPMScheduler#scale_model_input().
  FlaxDDPMScheduler.__len__: FlaxDDPMScheduler#__len__().
  FlaxDDPMSchedulerOutput.state: FlaxDDPMSchedulerOutput#state.
  DDPMSchedulerState.init_noise_sigma: DDPMSchedulerState#init_noise_sigma.
  DDPMSchedulerState.timesteps: DDPMSchedulerState#timesteps.
  DDPMSchedulerState.num_inference_steps: DDPMSchedulerState#num_inference_steps.
  FlaxDDPMScheduler.has_state: FlaxDDPMScheduler#has_state().
---
# Module: [`src/maxdiffusion/schedulers/scheduling_ddpm_flax.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddpm_flax.py)

## Classes
### `DDPMSchedulerState`
- def: [`src/maxdiffusion/schedulers/scheduling_ddpm_flax.py:36`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddpm_flax.py#L36)
- signature: `class DDPMSchedulerState:`
- members:
  - `create(cls, common: CommonSchedulerState, init_noise_sigma: jnp.ndarray, timesteps: jnp.ndarray)` — [`L45`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddpm_flax.py#L45)
  - `common` — [`L37`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddpm_flax.py#L37)
  - `init_noise_sigma` — [`L40`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddpm_flax.py#L40)
  - `num_inference_steps` — [`L42`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddpm_flax.py#L42)
  - `timesteps` — [`L41`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddpm_flax.py#L41)
- uses (calls/refs, reference-scoped): [`CommonSchedulerState`](scheduling_utils_flax.md#CommonSchedulerState)
- used by: [`step`](scheduling_ddpm_flax.md#FlaxDDPMScheduler.step), [`_get_variance`](scheduling_ddpm_flax.md#FlaxDDPMScheduler._get_variance), [`create_state`](scheduling_ddpm_flax.md#FlaxDDPMScheduler.create_state), [`add_noise`](scheduling_ddpm_flax.md#FlaxDDPMScheduler.add_noise), [`get_velocity`](scheduling_ddpm_flax.md#FlaxDDPMScheduler.get_velocity), [`FlaxDDPMSchedulerOutput`](scheduling_ddpm_flax.md#FlaxDDPMSchedulerOutput), [`set_timesteps`](scheduling_ddpm_flax.md#FlaxDDPMScheduler.set_timesteps), [`scale_model_input`](scheduling_ddpm_flax.md#FlaxDDPMScheduler.scale_model_input)

### `FlaxDDPMScheduler`  ·  implements/extends ConfigMixin, FlaxSchedulerMixin
- def: [`src/maxdiffusion/schedulers/scheduling_ddpm_flax.py:54`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddpm_flax.py#L54)
- doc: Denoising diffusion probabilistic models (DDPMs) explores the connections between denoising score matching and
- signature: `class FlaxDDPMScheduler(FlaxSchedulerMixin, ConfigMixin):`
- members:
  - `add_noise(self, state: DDPMSchedulerState, original_samples: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray)` — [`L284`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddpm_flax.py#L284)
  - `create_state(self, common: Optional[CommonSchedulerState] = None)` — [`L111`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddpm_flax.py#L111)
  - `get_velocity(self, state: DDPMSchedulerState, sample: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray)` — [`L293`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddpm_flax.py#L293)
  - `has_state(self)` — [`L92`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddpm_flax.py#L92)
  - `random_variance()` — [`L270`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddpm_flax.py#L270)
  - `scale_model_input(self, state: DDPMSchedulerState, sample: jnp.ndarray, timestep: Optional[int] = None)` — [`L126`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddpm_flax.py#L126) — Args:
  - `set_timesteps(self, state: DDPMSchedulerState, num_inference_steps: int, shape: Tuple = ())` — [`L138`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddpm_flax.py#L138) — Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.
  - `step(self, state: DDPMSchedulerState, model_output: jnp.ndarray, timestep: int, sample: jnp.ndarray, key: Optional[jax.Array] = None, return_dict: bool = True)` — [`L199`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddpm_flax.py#L199) — Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
  - `dtype` — [`L89`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddpm_flax.py#L89)
- protocol/private: `__init__`[`L96`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddpm_flax.py#L96), `__len__`[`L302`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddpm_flax.py#L302), `_compatibles`[`L87`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddpm_flax.py#L87), `_get_variance`[`L166`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddpm_flax.py#L166)
- uses (calls/refs, reference-scoped): [`config`](../configuration_utils.md#ConfigMixin.config), [`ConfigMixin`](../configuration_utils.md#ConfigMixin), [`FlaxSchedulerMixin`](scheduling_utils_flax.md#FlaxSchedulerMixin), [`CommonSchedulerState`](scheduling_utils_flax.md#CommonSchedulerState), [`register_to_config`](../configuration_utils.md#register_to_config), [`alphas_cumprod`](scheduling_utils_flax.md#CommonSchedulerState.alphas_cumprod), [`common`](scheduling_ddpm_flax.md#DDPMSchedulerState.common), [`FlaxKarrasDiffusionSchedulers`](scheduling_utils_flax.md#FlaxKarrasDiffusionSchedulers), [`add_noise_common`](scheduling_utils_flax.md#add_noise_common), [`create`](scheduling_utils_flax.md#CommonSchedulerState.create), [`DDPMSchedulerState`](scheduling_ddpm_flax.md#DDPMSchedulerState), [`get_velocity_common`](scheduling_utils_flax.md#get_velocity_common), [`FlaxDDPMSchedulerOutput`](scheduling_ddpm_flax.md#FlaxDDPMSchedulerOutput), [`prev_sample`](scheduling_utils_flax.md#FlaxSchedulerOutput.prev_sample), [`betas`](scheduling_utils_flax.md#CommonSchedulerState.betas), [`create`](scheduling_ddpm_flax.md#DDPMSchedulerState.create), [`alphas`](scheduling_utils_flax.md#CommonSchedulerState.alphas), [`state`](scheduling_ddpm_flax.md#FlaxDDPMSchedulerOutput.state)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`ConfigMixin`](../configuration_utils.md#ConfigMixin), [`_import_structure`](__init__.md#_import_structure), [`FlaxSchedulerMixin`](scheduling_utils_flax.md#FlaxSchedulerMixin)

### `FlaxDDPMSchedulerOutput`  ·  implements/extends FlaxSchedulerOutput
- def: [`src/maxdiffusion/schedulers/scheduling_ddpm_flax.py:50`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddpm_flax.py#L50)
- signature: `class FlaxDDPMSchedulerOutput(FlaxSchedulerOutput):`
- members:
  - `state` — [`L51`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddpm_flax.py#L51)
- uses (calls/refs, reference-scoped): [`FlaxSchedulerOutput`](scheduling_utils_flax.md#FlaxSchedulerOutput), [`DDPMSchedulerState`](scheduling_ddpm_flax.md#DDPMSchedulerState)
- used by: [`FlaxSchedulerOutput`](scheduling_utils_flax.md#FlaxSchedulerOutput), [`step`](scheduling_ddpm_flax.md#FlaxDDPMScheduler.step)

