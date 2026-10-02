---
title: 'Module: src/maxdiffusion/schedulers/scheduling_ddim_flax.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/schedulers/scheduling_ddim_flax.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.schedulers.scheduling_ddim_flax`/
symbols:
  FlaxDDIMScheduler.step: FlaxDDIMScheduler#step().
  FlaxDDIMScheduler.create_state: FlaxDDIMScheduler#create_state().
  FlaxDDIMScheduler: FlaxDDIMScheduler#
  FlaxDDIMScheduler._get_variance: FlaxDDIMScheduler#_get_variance().
  DDIMSchedulerState: DDIMSchedulerState#
  DDIMSchedulerState.common: DDIMSchedulerState#common.
  FlaxDDIMScheduler.add_noise: FlaxDDIMScheduler#add_noise().
  FlaxDDIMScheduler.get_velocity: FlaxDDIMScheduler#get_velocity().
  FlaxDDIMSchedulerOutput: FlaxDDIMSchedulerOutput#
  FlaxDDIMScheduler.__init__: FlaxDDIMScheduler#__init__().
  FlaxDDIMScheduler.set_timesteps: FlaxDDIMScheduler#set_timesteps().
  DDIMSchedulerState.create: DDIMSchedulerState#create().
  FlaxDDIMScheduler.dtype: FlaxDDIMScheduler#dtype.
  FlaxDDIMScheduler._compatibles: FlaxDDIMScheduler#_compatibles.
  FlaxDDIMScheduler.scale_model_input: FlaxDDIMScheduler#scale_model_input().
  FlaxDDIMScheduler.__len__: FlaxDDIMScheduler#__len__().
  DDIMSchedulerState.final_alpha_cumprod: DDIMSchedulerState#final_alpha_cumprod.
  DDIMSchedulerState.num_inference_steps: DDIMSchedulerState#num_inference_steps.
  FlaxDDIMSchedulerOutput.state: FlaxDDIMSchedulerOutput#state.
  DDIMSchedulerState.init_noise_sigma: DDIMSchedulerState#init_noise_sigma.
  DDIMSchedulerState.timesteps: DDIMSchedulerState#timesteps.
  FlaxDDIMScheduler.has_state: FlaxDDIMScheduler#has_state().
---
# Module: [`src/maxdiffusion/schedulers/scheduling_ddim_flax.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddim_flax.py)

## Classes
### `DDIMSchedulerState`
- def: [`src/maxdiffusion/schedulers/scheduling_ddim_flax.py:36`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddim_flax.py#L36)
- signature: `class DDIMSchedulerState:`
- members:
  - `create(cls, common: CommonSchedulerState, final_alpha_cumprod: jnp.ndarray, init_noise_sigma: jnp.ndarray, timesteps: jnp.ndarray)` — [`L46`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddim_flax.py#L46)
  - `common` — [`L37`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddim_flax.py#L37)
  - `final_alpha_cumprod` — [`L38`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddim_flax.py#L38)
  - `init_noise_sigma` — [`L41`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddim_flax.py#L41)
  - `num_inference_steps` — [`L43`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddim_flax.py#L43)
  - `timesteps` — [`L42`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddim_flax.py#L42)
- uses (calls/refs, reference-scoped): [`CommonSchedulerState`](scheduling_utils_flax.md#CommonSchedulerState)
- used by: [`step`](scheduling_ddim_flax.md#FlaxDDIMScheduler.step), [`create_state`](scheduling_ddim_flax.md#FlaxDDIMScheduler.create_state), [`_get_variance`](scheduling_ddim_flax.md#FlaxDDIMScheduler._get_variance), [`add_noise`](scheduling_ddim_flax.md#FlaxDDIMScheduler.add_noise), [`get_velocity`](scheduling_ddim_flax.md#FlaxDDIMScheduler.get_velocity), [`FlaxDDIMSchedulerOutput`](scheduling_ddim_flax.md#FlaxDDIMSchedulerOutput), [`set_timesteps`](scheduling_ddim_flax.md#FlaxDDIMScheduler.set_timesteps), [`scale_model_input`](scheduling_ddim_flax.md#FlaxDDIMScheduler.scale_model_input)

### `FlaxDDIMScheduler`  ·  implements/extends ConfigMixin, FlaxSchedulerMixin
- def: [`src/maxdiffusion/schedulers/scheduling_ddim_flax.py:66`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddim_flax.py#L66)
- doc: Denoising diffusion implicit models is a scheduler that extends the denoising procedure introduced in denoising
- signature: `class FlaxDDIMScheduler(FlaxSchedulerMixin, ConfigMixin):`
- members:
  - `add_noise(self, state: DDIMSchedulerState, original_samples: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray)` — [`L282`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddim_flax.py#L282)
  - `create_state(self, common: Optional[CommonSchedulerState] = None)` — [`L128`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddim_flax.py#L128) — documented in [maxdiffusion-schedulers-scheduling_utils_flax](../../../../concepts/maxdiffusion-schedulers-scheduling_utils_flax.md)
  - `get_velocity(self, state: DDIMSchedulerState, sample: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray)` — [`L291`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddim_flax.py#L291)
  - `has_state(self)` — [`L109`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddim_flax.py#L109)
  - `scale_model_input(self, state: DDIMSchedulerState, sample: jnp.ndarray, timestep: Optional[int] = None)` — [`L150`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddim_flax.py#L150) — Args:
  - `set_timesteps(self, state: DDIMSchedulerState, num_inference_steps: int, shape: Tuple = ())` — [`L162`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddim_flax.py#L162) — Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.
  - `step(self, state: DDIMSchedulerState, model_output: jnp.ndarray, timestep: int, sample: jnp.ndarray, eta: float = 0, return_dict: bool = True)` — [`L198`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddim_flax.py#L198) — Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
  - `dtype` — [`L106`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddim_flax.py#L106)
- protocol/private: `__init__`[`L113`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddim_flax.py#L113), `__len__`[`L300`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddim_flax.py#L300), `_compatibles`[`L104`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddim_flax.py#L104), `_get_variance`[`L188`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddim_flax.py#L188)
- uses (calls/refs, reference-scoped): [`config`](../configuration_utils.md#ConfigMixin.config), [`ConfigMixin`](../configuration_utils.md#ConfigMixin), [`FlaxSchedulerMixin`](scheduling_utils_flax.md#FlaxSchedulerMixin), [`CommonSchedulerState`](scheduling_utils_flax.md#CommonSchedulerState), [`register_to_config`](../configuration_utils.md#register_to_config), [`alphas_cumprod`](scheduling_utils_flax.md#CommonSchedulerState.alphas_cumprod), [`FlaxKarrasDiffusionSchedulers`](scheduling_utils_flax.md#FlaxKarrasDiffusionSchedulers), [`add_noise_common`](scheduling_utils_flax.md#add_noise_common), [`create`](scheduling_utils_flax.md#CommonSchedulerState.create), [`DDIMSchedulerState`](scheduling_ddim_flax.md#DDIMSchedulerState), [`get_velocity_common`](scheduling_utils_flax.md#get_velocity_common), [`common`](scheduling_ddim_flax.md#DDIMSchedulerState.common), [`FlaxDDIMSchedulerOutput`](scheduling_ddim_flax.md#FlaxDDIMSchedulerOutput), [`prev_sample`](scheduling_utils_flax.md#FlaxSchedulerOutput.prev_sample), [`create`](scheduling_ddim_flax.md#DDIMSchedulerState.create), [`final_alpha_cumprod`](scheduling_ddim_flax.md#DDIMSchedulerState.final_alpha_cumprod), [`num_inference_steps`](scheduling_ddim_flax.md#DDIMSchedulerState.num_inference_steps), [`state`](scheduling_ddim_flax.md#FlaxDDIMSchedulerOutput.state)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`ConfigMixin`](../configuration_utils.md#ConfigMixin), [`_import_structure`](__init__.md#_import_structure), [`FlaxSchedulerMixin`](scheduling_utils_flax.md#FlaxSchedulerMixin), [`__init__`](../pipelines/controlnet/pipeline_flax_controlnet_sdxl.md#FlaxStableDiffusionXLControlNetPipeline.__init__), [`__init__`](../pipelines/controlnet/pipeline_flax_controlnet.md#FlaxStableDiffusionControlNetPipeline.__init__), [`__init__`](../pipelines/stable_diffusion/pipeline_flax_stable_diffusion.md#FlaxStableDiffusionPipeline.__init__), [`__init__`](../pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.md#FlaxStableDiffusionXLPipeline.__init__)

### `FlaxDDIMSchedulerOutput`  ·  implements/extends FlaxSchedulerOutput
- def: [`src/maxdiffusion/schedulers/scheduling_ddim_flax.py:62`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddim_flax.py#L62)
- signature: `class FlaxDDIMSchedulerOutput(FlaxSchedulerOutput):`
- members:
  - `state` — [`L63`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_ddim_flax.py#L63)
- uses (calls/refs, reference-scoped): [`FlaxSchedulerOutput`](scheduling_utils_flax.md#FlaxSchedulerOutput), [`DDIMSchedulerState`](scheduling_ddim_flax.md#DDIMSchedulerState)
- used by: [`FlaxSchedulerOutput`](scheduling_utils_flax.md#FlaxSchedulerOutput), [`step`](scheduling_ddim_flax.md#FlaxDDIMScheduler.step)

