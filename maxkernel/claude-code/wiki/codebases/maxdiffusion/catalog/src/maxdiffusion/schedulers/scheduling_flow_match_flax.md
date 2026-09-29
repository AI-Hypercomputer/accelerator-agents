---
title: 'Module: src/maxdiffusion/schedulers/scheduling_flow_match_flax.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/schedulers/scheduling_flow_match_flax.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.schedulers.scheduling_flow_match_flax`/Fl
symbols:
  FlaxFlowMatchScheduler.step: axFlowMatchScheduler#step().
  FlowMatchSchedulerState: owMatchSchedulerState#
  FlaxFlowMatchScheduler.set_timesteps: axFlowMatchScheduler#set_timesteps().
  FlaxFlowMatchScheduler.set_timesteps_ltx2: axFlowMatchScheduler#set_timesteps_ltx2().
  FlaxFlowMatchScheduler.add_noise: axFlowMatchScheduler#add_noise().
  FlaxFlowMatchScheduler._find_timestep_id: axFlowMatchScheduler#_find_timestep_id().
  FlaxFlowMatchScheduler.return_to_timestep: axFlowMatchScheduler#return_to_timestep().
  FlaxFlowMatchScheduler.training_weight: axFlowMatchScheduler#training_weight().
  FlowMatchSchedulerState.sigmas: owMatchSchedulerState#sigmas.
  FlowMatchSchedulerState.timesteps: owMatchSchedulerState#timesteps.
  FlaxFlowMatchScheduler: axFlowMatchScheduler#
  FlaxFlowMatchScheduler.dtype: axFlowMatchScheduler#dtype.
  FlaxFlowMatchScheduler.__init__: axFlowMatchScheduler#__init__().
  FlaxFlowMatchScheduler.create_state: axFlowMatchScheduler#create_state().
  FlaxFlowMatchScheduler.apply_flow_match: axFlowMatchScheduler#apply_flow_match().
  FlaxFlowMatchSchedulerOutput: axFlowMatchSchedulerOutput#
  FlaxFlowMatchScheduler.get_next_sigma: axFlowMatchScheduler#get_next_sigma().
  FlaxFlowMatchScheduler.get_final_sigma: axFlowMatchScheduler#get_final_sigma().
  FlaxFlowMatchScheduler._calculate_training_weights: axFlowMatchScheduler#_calculate_training_weights().
  FlaxFlowMatchSchedulerOutput.state: axFlowMatchSchedulerOutput#state.
  FlaxFlowMatchScheduler.sample_timesteps: axFlowMatchScheduler#sample_timesteps().
  FlaxFlowMatchScheduler.__len__: axFlowMatchScheduler#__len__().
  FlowMatchSchedulerState.linear_timesteps_weights: owMatchSchedulerState#linear_timesteps_weights.
  FlowMatchSchedulerState.create: owMatchSchedulerState#create().
  FlowMatchSchedulerState.training: owMatchSchedulerState#training.
  FlowMatchSchedulerState.num_inference_steps: owMatchSchedulerState#num_inference_steps.
  FlaxFlowMatchScheduler.has_state: axFlowMatchScheduler#has_state().
  FlaxFlowMatchScheduler.training_target: axFlowMatchScheduler#training_target().
---
# Module: [`src/maxdiffusion/schedulers/scheduling_flow_match_flax.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py)

## Classes
### `FlaxFlowMatchScheduler`  ·  implements/extends ConfigMixin, FlaxSchedulerMixin
- def: [`src/maxdiffusion/schedulers/scheduling_flow_match_flax.py:71`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L71)
- doc: FlaxFlowMatchScheduler is a JAX/Flax conversion of a scheduler used for training video generation models like
- signature: `class FlaxFlowMatchScheduler(FlaxSchedulerMixin, ConfigMixin):`
- members:
  - `_calculate_training_weights(self, timesteps: jnp.ndarray, num_inference_steps: int)` — [`L230`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L230) — Calculates the training weight for a given timestep.
  - `_find_timestep_id(self, state: FlowMatchSchedulerState, timestep: jnp.ndarray)` — [`L276`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L276) — Finds the index of the closest timestep in the scheduler's `timesteps` array.
  - `add_noise(self, state: FlowMatchSchedulerState, original_samples: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray)` — [`L354`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L354) — Adds noise to the original samples according to the flow matching schedule.
  - `apply_flow_match(self, noise: jnp.ndarray, batch_images: jnp.ndarray, timesteps: jnp.ndarray)` — [`L251`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L251) — Apply flow match to the batch of images.
  - `create_state(self)` — [`L100`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L100) — Creates the initial state for the scheduler.
  - `get_final_sigma()` — [`L323`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L323)
  - `get_next_sigma()` — [`L320`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L320)
  - `has_state(self)` — [`L83`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L83)
  - `return_to_timestep(self, state: FlowMatchSchedulerState, timestep: jnp.ndarray, sample: jnp.ndarray, sample_stablized: jnp.ndarray)` — [`L341`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L341) — Calculates the model output required to go from a stabilized sample back to the original sample.
  - `sample_timesteps(self, timestep_rng, batch_size)` — [`L239`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L239)
  - `set_timesteps(self, state: FlowMatchSchedulerState, num_inference_steps: int = 100, shape: Tuple = None, denoising_strength: float = 1, training: bool = False, shift: Optional[float] = None)` — [`L105`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L105) — Sets the discrete timesteps used for the diffusion chain.
  - `set_timesteps_ltx2(self, state: FlowMatchSchedulerState, num_inference_steps: int = 100, shape: Tuple = None, denoising_strength: float = 1, training: bool = False, shift: Optional[float] = None, timesteps: Optional[jnp.ndarray] = None, sigmas: Optional[jnp.ndarray] = None)` — [`L164`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L164) — LTX-2 specific logic for set_timesteps that correctly applies exponential dynamic shifting.
  - `step(self, state: FlowMatchSchedulerState, model_output: jnp.ndarray, timestep: jnp.ndarray, sample: jnp.ndarray, to_final: bool = False, return_dict: bool = True)` — [`L287`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L287) — Propagates the sample with the flow matching scheduler.
  - `training_target(self, sample: jnp.ndarray, noise: jnp.ndarray, *args, **kwargs)` — [`L389`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L389) — Calculates the training target. For flow matching, this is typically the velocity, `x_1 - x_0`,
  - `training_weight(self, state: FlowMatchSchedulerState, timestep: jnp.ndarray)` — [`L397`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L397) — Calculates the training weight for a given timestep.
  - `dtype` — [`L80`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L80)
- protocol/private: `__init__`[`L87`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L87), `__len__`[`L403`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L403)
- uses (calls/refs, reference-scoped): [`config`](../configuration_utils.md#ConfigMixin.config), [`ConfigMixin`](../configuration_utils.md#ConfigMixin), [`FlaxSchedulerMixin`](scheduling_utils_flax.md#FlaxSchedulerMixin), [`register_to_config`](../configuration_utils.md#register_to_config), [`FlowMatchSchedulerState`](scheduling_flow_match_flax.md#FlowMatchSchedulerState), [`sigmas`](scheduling_flow_match_flax.md#FlowMatchSchedulerState.sigmas), [`timesteps`](scheduling_flow_match_flax.md#FlowMatchSchedulerState.timesteps), [`FlaxFlowMatchSchedulerOutput`](scheduling_flow_match_flax.md#FlaxFlowMatchSchedulerOutput), [`create`](scheduling_flow_match_flax.md#FlowMatchSchedulerState.create), [`linear_timesteps_weights`](scheduling_flow_match_flax.md#FlowMatchSchedulerState.linear_timesteps_weights)
- used by: [`ConfigMixin`](../configuration_utils.md#ConfigMixin), [`FlaxSchedulerMixin`](scheduling_utils_flax.md#FlaxSchedulerMixin)

### `FlaxFlowMatchSchedulerOutput`  ·  implements/extends FlaxSchedulerOutput
- def: [`src/maxdiffusion/schedulers/scheduling_flow_match_flax.py:57`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L57)
- doc: Output class for the JAX FlowMatchScheduler's step function.
- signature: `class FlaxFlowMatchSchedulerOutput(FlaxSchedulerOutput):`
- members:
  - `state` — [`L68`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L68)
- uses (calls/refs, reference-scoped): [`FlaxSchedulerOutput`](scheduling_utils_flax.md#FlaxSchedulerOutput), [`FlowMatchSchedulerState`](scheduling_flow_match_flax.md#FlowMatchSchedulerState)
- used by: [`FlaxSchedulerOutput`](scheduling_utils_flax.md#FlaxSchedulerOutput), [`step`](scheduling_flow_match_flax.md#FlaxFlowMatchScheduler.step)

### `FlowMatchSchedulerState`
- def: [`src/maxdiffusion/schedulers/scheduling_flow_match_flax.py:34`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L34)
- doc: Data class to hold the mutable state of the FlaxFlowMatchScheduler.
- signature: `class FlowMatchSchedulerState:`
- members:
  - `create(cls)` — [`L46`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L46)
  - `linear_timesteps_weights` — [`L41`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L41)
  - `num_inference_steps` — [`L43`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L43)
  - `sigmas` — [`L39`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L39)
  - `timesteps` — [`L40`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L40)
  - `training` — [`L42`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py#L42)
- used by: [`step`](scheduling_flow_match_flax.md#FlaxFlowMatchScheduler.step), [`add_noise`](scheduling_flow_match_flax.md#FlaxFlowMatchScheduler.add_noise), [`set_timesteps`](scheduling_flow_match_flax.md#FlaxFlowMatchScheduler.set_timesteps), [`set_timesteps_ltx2`](scheduling_flow_match_flax.md#FlaxFlowMatchScheduler.set_timesteps_ltx2), [`_find_timestep_id`](scheduling_flow_match_flax.md#FlaxFlowMatchScheduler._find_timestep_id), [`return_to_timestep`](scheduling_flow_match_flax.md#FlaxFlowMatchScheduler.return_to_timestep), [`training_weight`](scheduling_flow_match_flax.md#FlaxFlowMatchScheduler.training_weight), [`create_state`](scheduling_flow_match_flax.md#FlaxFlowMatchScheduler.create_state), [`get_next_sigma`](scheduling_flow_match_flax.md#FlaxFlowMatchScheduler.get_next_sigma), [`state`](scheduling_flow_match_flax.md#FlaxFlowMatchSchedulerOutput.state)

