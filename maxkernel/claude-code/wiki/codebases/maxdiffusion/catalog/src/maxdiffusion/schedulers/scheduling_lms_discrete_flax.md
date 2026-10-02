---
title: 'Module: src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.schedulers.scheduling_lms_discrete_flax`/
symbols:
  FlaxLMSDiscreteScheduler.step: FlaxLMSDiscreteScheduler#step().
  FlaxLMSDiscreteScheduler: FlaxLMSDiscreteScheduler#
  FlaxLMSDiscreteScheduler.create_state: FlaxLMSDiscreteScheduler#create_state().
  FlaxLMSDiscreteScheduler.set_timesteps: FlaxLMSDiscreteScheduler#set_timesteps().
  LMSDiscreteSchedulerState: LMSDiscreteSchedulerState#
  FlaxLMSDiscreteScheduler.get_lms_coefficient: FlaxLMSDiscreteScheduler#get_lms_coefficient().
  LMSDiscreteSchedulerState.sigmas: LMSDiscreteSchedulerState#sigmas.
  FlaxLMSDiscreteScheduler.scale_model_input: FlaxLMSDiscreteScheduler#scale_model_input().
  FlaxLMSDiscreteScheduler.add_noise: FlaxLMSDiscreteScheduler#add_noise().
  FlaxLMSSchedulerOutput: FlaxLMSSchedulerOutput#
  FlaxLMSDiscreteScheduler.__init__: FlaxLMSDiscreteScheduler#__init__().
  LMSDiscreteSchedulerState.common: LMSDiscreteSchedulerState#common.
  LMSDiscreteSchedulerState.derivatives: LMSDiscreteSchedulerState#derivatives.
  FlaxLMSDiscreteScheduler.dtype: FlaxLMSDiscreteScheduler#dtype.
  LMSDiscreteSchedulerState.create: LMSDiscreteSchedulerState#create().
  FlaxLMSDiscreteScheduler.lms_derivative: FlaxLMSDiscreteScheduler#lms_derivative().
  FlaxLMSDiscreteScheduler._compatibles: FlaxLMSDiscreteScheduler#_compatibles.
  FlaxLMSDiscreteScheduler.__len__: FlaxLMSDiscreteScheduler#__len__().
  LMSDiscreteSchedulerState.timesteps: LMSDiscreteSchedulerState#timesteps.
  LMSDiscreteSchedulerState.num_inference_steps: LMSDiscreteSchedulerState#num_inference_steps.
  FlaxLMSSchedulerOutput.state: FlaxLMSSchedulerOutput#state.
  LMSDiscreteSchedulerState.init_noise_sigma: LMSDiscreteSchedulerState#init_noise_sigma.
  FlaxLMSDiscreteScheduler.has_state: FlaxLMSDiscreteScheduler#has_state().
---
# Module: [`src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py)

## Classes
### `FlaxLMSDiscreteScheduler`  ·  implements/extends ConfigMixin, FlaxSchedulerMixin
- def: [`src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py:55`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py#L55)
- doc: Linear Multistep Scheduler for discrete beta schedules. Based on the original k-diffusion implementation by
- signature: `class FlaxLMSDiscreteScheduler(FlaxSchedulerMixin, ConfigMixin):`
- members:
  - `add_noise(self, state: LMSDiscreteSchedulerState, original_samples: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray)` — [`L260`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py#L260)
  - `create_state(self, common: Optional[CommonSchedulerState] = None)` — [`L104`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py#L104) — documented in [maxdiffusion-schedulers-scheduling_utils_flax](../../../../concepts/maxdiffusion-schedulers-scheduling_utils_flax.md)
  - `get_lms_coefficient(self, state: LMSDiscreteSchedulerState, order, t, current_order)` — [`L143`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py#L143) — Compute a linear multistep coefficient.
  - `has_state(self)` — [`L88`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py#L88)
  - `lms_derivative(tau)` — [`L153`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py#L153)
  - `scale_model_input(self, state: LMSDiscreteSchedulerState, sample: jnp.ndarray, timestep: int)` — [`L121`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py#L121) — Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the K-LMS algorithm.
  - `set_timesteps(self, state: LMSDiscreteSchedulerState, num_inference_steps: int, shape: Tuple = ())` — [`L165`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py#L165) — Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.
  - `step(self, state: LMSDiscreteSchedulerState, model_output: jnp.ndarray, timestep: int, sample: jnp.ndarray, order: int = 4, return_dict: bool = True)` — [`L201`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py#L201) — Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
  - `dtype` — [`L85`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py#L85)
- protocol/private: `__init__`[`L92`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py#L92), `__len__`[`L274`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py#L274), `_compatibles`[`L83`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py#L83)
- uses (calls/refs, reference-scoped): [`config`](../configuration_utils.md#ConfigMixin.config), [`ConfigMixin`](../configuration_utils.md#ConfigMixin), [`FlaxSchedulerMixin`](scheduling_utils_flax.md#FlaxSchedulerMixin), [`CommonSchedulerState`](scheduling_utils_flax.md#CommonSchedulerState), [`register_to_config`](../configuration_utils.md#register_to_config), [`alphas_cumprod`](scheduling_utils_flax.md#CommonSchedulerState.alphas_cumprod), [`FlaxKarrasDiffusionSchedulers`](scheduling_utils_flax.md#FlaxKarrasDiffusionSchedulers), [`create`](scheduling_utils_flax.md#CommonSchedulerState.create), [`LMSDiscreteSchedulerState`](scheduling_lms_discrete_flax.md#LMSDiscreteSchedulerState), [`sigmas`](scheduling_lms_discrete_flax.md#LMSDiscreteSchedulerState.sigmas), [`FlaxLMSSchedulerOutput`](scheduling_lms_discrete_flax.md#FlaxLMSSchedulerOutput), [`broadcast_to_shape_from_left`](scheduling_utils_flax.md#broadcast_to_shape_from_left), [`prev_sample`](scheduling_utils_flax.md#FlaxSchedulerOutput.prev_sample), [`common`](scheduling_lms_discrete_flax.md#LMSDiscreteSchedulerState.common), [`derivatives`](scheduling_lms_discrete_flax.md#LMSDiscreteSchedulerState.derivatives), [`create`](scheduling_lms_discrete_flax.md#LMSDiscreteSchedulerState.create), [`num_inference_steps`](scheduling_lms_discrete_flax.md#LMSDiscreteSchedulerState.num_inference_steps), [`state`](scheduling_lms_discrete_flax.md#FlaxLMSSchedulerOutput.state), [`timesteps`](scheduling_lms_discrete_flax.md#LMSDiscreteSchedulerState.timesteps)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`ConfigMixin`](../configuration_utils.md#ConfigMixin), [`_import_structure`](__init__.md#_import_structure), [`FlaxSchedulerMixin`](scheduling_utils_flax.md#FlaxSchedulerMixin), [`__init__`](../pipelines/controlnet/pipeline_flax_controlnet_sdxl.md#FlaxStableDiffusionXLControlNetPipeline.__init__), [`__init__`](../pipelines/controlnet/pipeline_flax_controlnet.md#FlaxStableDiffusionControlNetPipeline.__init__), [`__init__`](../pipelines/stable_diffusion/pipeline_flax_stable_diffusion.md#FlaxStableDiffusionPipeline.__init__), [`__init__`](../pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.md#FlaxStableDiffusionXLPipeline.__init__)

### `FlaxLMSSchedulerOutput`  ·  implements/extends FlaxSchedulerOutput
- def: [`src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py:51`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py#L51)
- signature: `class FlaxLMSSchedulerOutput(FlaxSchedulerOutput):`
- members:
  - `state` — [`L52`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py#L52)
- uses (calls/refs, reference-scoped): [`FlaxSchedulerOutput`](scheduling_utils_flax.md#FlaxSchedulerOutput), [`LMSDiscreteSchedulerState`](scheduling_lms_discrete_flax.md#LMSDiscreteSchedulerState)
- used by: [`FlaxSchedulerOutput`](scheduling_utils_flax.md#FlaxSchedulerOutput), [`step`](scheduling_lms_discrete_flax.md#FlaxLMSDiscreteScheduler.step)

### `LMSDiscreteSchedulerState`
- def: [`src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py:33`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py#L33)
- signature: `class LMSDiscreteSchedulerState:`
- members:
  - `create(cls, common: CommonSchedulerState, init_noise_sigma: jnp.ndarray, timesteps: jnp.ndarray, sigmas: jnp.ndarray)` — [`L46`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py#L46)
  - `common` — [`L34`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py#L34)
  - `derivatives` — [`L43`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py#L43)
  - `init_noise_sigma` — [`L37`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py#L37)
  - `num_inference_steps` — [`L40`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py#L40)
  - `sigmas` — [`L39`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py#L39)
  - `timesteps` — [`L38`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_lms_discrete_flax.py#L38)
- uses (calls/refs, reference-scoped): [`CommonSchedulerState`](scheduling_utils_flax.md#CommonSchedulerState)
- used by: [`step`](scheduling_lms_discrete_flax.md#FlaxLMSDiscreteScheduler.step), [`create_state`](scheduling_lms_discrete_flax.md#FlaxLMSDiscreteScheduler.create_state), [`set_timesteps`](scheduling_lms_discrete_flax.md#FlaxLMSDiscreteScheduler.set_timesteps), [`get_lms_coefficient`](scheduling_lms_discrete_flax.md#FlaxLMSDiscreteScheduler.get_lms_coefficient), [`add_noise`](scheduling_lms_discrete_flax.md#FlaxLMSDiscreteScheduler.add_noise), [`scale_model_input`](scheduling_lms_discrete_flax.md#FlaxLMSDiscreteScheduler.scale_model_input), [`FlaxLMSSchedulerOutput`](scheduling_lms_discrete_flax.md#FlaxLMSSchedulerOutput), [`lms_derivative`](scheduling_lms_discrete_flax.md#FlaxLMSDiscreteScheduler.lms_derivative)

