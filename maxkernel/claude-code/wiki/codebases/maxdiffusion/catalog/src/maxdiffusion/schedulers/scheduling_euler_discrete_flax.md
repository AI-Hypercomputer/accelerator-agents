---
title: 'Module: src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.schedulers.scheduling_euler_discrete_flax`/
symbols:
  FlaxEulerDiscreteScheduler.step: FlaxEulerDiscreteScheduler#step().
  FlaxEulerDiscreteScheduler.create_state: FlaxEulerDiscreteScheduler#create_state().
  FlaxEulerDiscreteScheduler.set_timesteps: FlaxEulerDiscreteScheduler#set_timesteps().
  FlaxEulerDiscreteScheduler.add_noise: FlaxEulerDiscreteScheduler#add_noise().
  FlaxEulerDiscreteScheduler: FlaxEulerDiscreteScheduler#
  EulerDiscreteSchedulerState: EulerDiscreteSchedulerState#
  FlaxEulerDiscreteScheduler.scale_model_input: FlaxEulerDiscreteScheduler#scale_model_input().
  FlaxEulerDiscreteSchedulerOutput: FlaxEulerDiscreteSchedulerOutput#
  FlaxEulerDiscreteScheduler.__init__: FlaxEulerDiscreteScheduler#__init__().
  EulerDiscreteSchedulerState.common: EulerDiscreteSchedulerState#common.
  EulerDiscreteSchedulerState.sigmas: EulerDiscreteSchedulerState#sigmas.
  FlaxEulerDiscreteScheduler.dtype: FlaxEulerDiscreteScheduler#dtype.
  EulerDiscreteSchedulerState.create: EulerDiscreteSchedulerState#create().
  EulerDiscreteSchedulerState.timesteps: EulerDiscreteSchedulerState#timesteps.
  FlaxEulerDiscreteScheduler._compatibles: FlaxEulerDiscreteScheduler#_compatibles.
  FlaxEulerDiscreteScheduler.__len__: FlaxEulerDiscreteScheduler#__len__().
  EulerDiscreteSchedulerState.num_inference_steps: EulerDiscreteSchedulerState#num_inference_steps.
  FlaxEulerDiscreteSchedulerOutput.state: FlaxEulerDiscreteSchedulerOutput#state.
  EulerDiscreteSchedulerState.init_noise_sigma: EulerDiscreteSchedulerState#init_noise_sigma.
  FlaxEulerDiscreteScheduler.has_state: FlaxEulerDiscreteScheduler#has_state().
---
# Module: [`src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py)

## Classes
### `EulerDiscreteSchedulerState`
- def: [`src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py:32`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py#L32)
- signature: `class EulerDiscreteSchedulerState:`
- members:
  - `create(cls, common: CommonSchedulerState, init_noise_sigma: jnp.ndarray, timesteps: jnp.ndarray, sigmas: jnp.ndarray)` — [`L42`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py#L42)
  - `common` — [`L33`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py#L33)
  - `init_noise_sigma` — [`L36`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py#L36)
  - `num_inference_steps` — [`L39`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py#L39)
  - `sigmas` — [`L38`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py#L38)
  - `timesteps` — [`L37`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py#L37)
- uses (calls/refs, reference-scoped): [`CommonSchedulerState`](scheduling_utils_flax.md#CommonSchedulerState)
- used by: [`step`](scheduling_euler_discrete_flax.md#FlaxEulerDiscreteScheduler.step), [`create_state`](scheduling_euler_discrete_flax.md#FlaxEulerDiscreteScheduler.create_state), [`set_timesteps`](scheduling_euler_discrete_flax.md#FlaxEulerDiscreteScheduler.set_timesteps), [`add_noise`](scheduling_euler_discrete_flax.md#FlaxEulerDiscreteScheduler.add_noise), [`scale_model_input`](scheduling_euler_discrete_flax.md#FlaxEulerDiscreteScheduler.scale_model_input), [`FlaxEulerDiscreteSchedulerOutput`](scheduling_euler_discrete_flax.md#FlaxEulerDiscreteSchedulerOutput)

### `FlaxEulerDiscreteScheduler`  ·  implements/extends ConfigMixin, FlaxSchedulerMixin
- def: [`src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py:51`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py#L51)
- doc: Euler scheduler (Algorithm 2) from Karras et al. (2022) https://arxiv.org/abs/2206.00364. . Based on the original
- signature: `class FlaxEulerDiscreteScheduler(FlaxSchedulerMixin, ConfigMixin):`
- members:
  - `add_noise(self, state: EulerDiscreteSchedulerState, original_samples: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray, flux: bool = False)` — [`L251`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py#L251)
  - `create_state(self, common: Optional[CommonSchedulerState] = None)` — [`L102`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py#L102)
  - `has_state(self)` — [`L85`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py#L85)
  - `scale_model_input(self, state: EulerDiscreteSchedulerState, sample: jnp.ndarray, timestep: int)` — [`L124`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py#L124) — Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.
  - `set_timesteps(self, state: EulerDiscreteSchedulerState, num_inference_steps: int, shape: Tuple = (), timestep_spacing: str = "")` — [`L146`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py#L146) — Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.
  - `step(self, state: EulerDiscreteSchedulerState, model_output: jnp.ndarray, timestep: int, sample: jnp.ndarray, return_dict: bool = True)` — [`L195`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py#L195) — Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
  - `dtype` — [`L82`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py#L82)
- protocol/private: `__init__`[`L89`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py#L89), `__len__`[`L272`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py#L272), `_compatibles`[`L80`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py#L80)
- uses (calls/refs, reference-scoped): [`config`](../configuration_utils.md#ConfigMixin.config), [`ConfigMixin`](../configuration_utils.md#ConfigMixin), [`FlaxSchedulerMixin`](scheduling_utils_flax.md#FlaxSchedulerMixin), [`CommonSchedulerState`](scheduling_utils_flax.md#CommonSchedulerState), [`register_to_config`](../configuration_utils.md#register_to_config), [`alphas_cumprod`](scheduling_utils_flax.md#CommonSchedulerState.alphas_cumprod), [`FlaxKarrasDiffusionSchedulers`](scheduling_utils_flax.md#FlaxKarrasDiffusionSchedulers), [`create`](scheduling_utils_flax.md#CommonSchedulerState.create), [`EulerDiscreteSchedulerState`](scheduling_euler_discrete_flax.md#EulerDiscreteSchedulerState), [`FlaxEulerDiscreteSchedulerOutput`](scheduling_euler_discrete_flax.md#FlaxEulerDiscreteSchedulerOutput), [`broadcast_to_shape_from_left`](scheduling_utils_flax.md#broadcast_to_shape_from_left), [`prev_sample`](scheduling_utils_flax.md#FlaxSchedulerOutput.prev_sample), [`common`](scheduling_euler_discrete_flax.md#EulerDiscreteSchedulerState.common), [`sigmas`](scheduling_euler_discrete_flax.md#EulerDiscreteSchedulerState.sigmas), [`create`](scheduling_euler_discrete_flax.md#EulerDiscreteSchedulerState.create), [`timesteps`](scheduling_euler_discrete_flax.md#EulerDiscreteSchedulerState.timesteps), [`num_inference_steps`](scheduling_euler_discrete_flax.md#EulerDiscreteSchedulerState.num_inference_steps), [`state`](scheduling_euler_discrete_flax.md#FlaxEulerDiscreteSchedulerOutput.state)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`ConfigMixin`](../configuration_utils.md#ConfigMixin), [`_import_structure`](__init__.md#_import_structure), [`FlaxSchedulerMixin`](scheduling_utils_flax.md#FlaxSchedulerMixin), [`create_scheduler`](../trainers/flux_trainer.md#FluxTrainer.create_scheduler)

### `FlaxEulerDiscreteSchedulerOutput`  ·  implements/extends FlaxSchedulerOutput
- def: [`src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py:47`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py#L47)
- signature: `class FlaxEulerDiscreteSchedulerOutput(FlaxSchedulerOutput):`
- members:
  - `state` — [`L48`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py#L48)
- uses (calls/refs, reference-scoped): [`FlaxSchedulerOutput`](scheduling_utils_flax.md#FlaxSchedulerOutput), [`EulerDiscreteSchedulerState`](scheduling_euler_discrete_flax.md#EulerDiscreteSchedulerState)
- used by: [`FlaxSchedulerOutput`](scheduling_utils_flax.md#FlaxSchedulerOutput), [`step`](scheduling_euler_discrete_flax.md#FlaxEulerDiscreteScheduler.step)

