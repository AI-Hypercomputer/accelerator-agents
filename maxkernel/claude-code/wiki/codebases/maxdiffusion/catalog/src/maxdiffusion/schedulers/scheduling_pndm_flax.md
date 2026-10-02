---
title: 'Module: src/maxdiffusion/schedulers/scheduling_pndm_flax.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/schedulers/scheduling_pndm_flax.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.schedulers.scheduling_pndm_flax`/
symbols:
  FlaxPNDMScheduler.step_prk: FlaxPNDMScheduler#step_prk().
  FlaxPNDMScheduler.step: FlaxPNDMScheduler#step().
  FlaxPNDMScheduler.step_plms: FlaxPNDMScheduler#step_plms().
  FlaxPNDMScheduler.create_state: FlaxPNDMScheduler#create_state().
  PNDMSchedulerState.ets: PNDMSchedulerState#ets.
  PNDMSchedulerState: PNDMSchedulerState#
  FlaxPNDMScheduler: FlaxPNDMScheduler#
  FlaxPNDMScheduler._get_prev_sample: FlaxPNDMScheduler#_get_prev_sample().
  PNDMSchedulerState.counter: PNDMSchedulerState#counter.
  FlaxPNDMScheduler.set_timesteps: FlaxPNDMScheduler#set_timesteps().
  FlaxPNDMSchedulerOutput: FlaxPNDMSchedulerOutput#
  FlaxPNDMScheduler.__init__: FlaxPNDMScheduler#__init__().
  FlaxPNDMScheduler.add_noise: FlaxPNDMScheduler#add_noise().
  PNDMSchedulerState.num_inference_steps: PNDMSchedulerState#num_inference_steps.
  PNDMSchedulerState.cur_model_output: PNDMSchedulerState#cur_model_output.
  FlaxPNDMScheduler.dtype: FlaxPNDMScheduler#dtype.
  PNDMSchedulerState.common: PNDMSchedulerState#common.
  PNDMSchedulerState.cur_sample: PNDMSchedulerState#cur_sample.
  PNDMSchedulerState.create: PNDMSchedulerState#create().
  FlaxPNDMScheduler.pndm_order: FlaxPNDMScheduler#pndm_order.
  FlaxPNDMScheduler._compatibles: FlaxPNDMScheduler#_compatibles.
  FlaxPNDMScheduler.scale_model_input: FlaxPNDMScheduler#scale_model_input().
  FlaxPNDMScheduler.__len__: FlaxPNDMScheduler#__len__().
  PNDMSchedulerState.prk_timesteps: PNDMSchedulerState#prk_timesteps.
  PNDMSchedulerState.final_alpha_cumprod: PNDMSchedulerState#final_alpha_cumprod.
  FlaxPNDMSchedulerOutput.state: FlaxPNDMSchedulerOutput#state.
  PNDMSchedulerState.init_noise_sigma: PNDMSchedulerState#init_noise_sigma.
  PNDMSchedulerState.timesteps: PNDMSchedulerState#timesteps.
  PNDMSchedulerState.plms_timesteps: PNDMSchedulerState#plms_timesteps.
  FlaxPNDMScheduler.has_state: FlaxPNDMScheduler#has_state().
---
# Module: [`src/maxdiffusion/schedulers/scheduling_pndm_flax.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py)

## Classes
### `FlaxPNDMScheduler`  ·  implements/extends ConfigMixin, FlaxSchedulerMixin
- def: [`src/maxdiffusion/schedulers/scheduling_pndm_flax.py:73`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L73)
- doc: Pseudo numerical methods for diffusion models (PNDM) proposes using more advanced ODE integration techniques,
- signature: `class FlaxPNDMScheduler(FlaxSchedulerMixin, ConfigMixin):`
- members:
  - `add_noise(self, state: PNDMSchedulerState, original_samples: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray)` — [`L482`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L482)
  - `create_state(self, common: Optional[CommonSchedulerState] = None)` — [`L143`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L143)
  - `has_state(self)` — [`L119`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L119)
  - `scale_model_input(self, state: PNDMSchedulerState, sample: jnp.ndarray, timestep: Optional[int] = None)` — [`L220`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L220) — Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
  - `set_timesteps(self, state: PNDMSchedulerState, num_inference_steps: int, shape: Tuple)` — [`L165`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L165) — Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.
  - `step(self, state: PNDMSchedulerState, model_output: jnp.ndarray, timestep: int, sample: jnp.ndarray, return_dict: bool = True)` — [`L235`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L235) — Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
  - `step_plms(self, state: PNDMSchedulerState, model_output: jnp.ndarray, timestep: int, sample: jnp.ndarray)` — [`L352`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L352) — Step function propagating the sample with the linear multi-step method. This has one forward pass with multiple
  - `step_prk(self, state: PNDMSchedulerState, model_output: jnp.ndarray, timestep: int, sample: jnp.ndarray)` — [`L288`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L288) — Step function propagating the sample with the Runge-Kutta method. RK takes 4 forward passes to approximate the
  - `dtype` — [`L115`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L115)
  - `pndm_order` — [`L116`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L116)
- protocol/private: `__init__`[`L123`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L123), `__len__`[`L491`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L491), `_compatibles`[`L113`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L113), `_get_prev_sample`[`L443`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L443)
- uses (calls/refs, reference-scoped): [`config`](../configuration_utils.md#ConfigMixin.config), [`ConfigMixin`](../configuration_utils.md#ConfigMixin), [`FlaxSchedulerMixin`](scheduling_utils_flax.md#FlaxSchedulerMixin), [`CommonSchedulerState`](scheduling_utils_flax.md#CommonSchedulerState), [`register_to_config`](../configuration_utils.md#register_to_config), [`alphas_cumprod`](scheduling_utils_flax.md#CommonSchedulerState.alphas_cumprod), [`ets`](scheduling_pndm_flax.md#PNDMSchedulerState.ets), [`PNDMSchedulerState`](scheduling_pndm_flax.md#PNDMSchedulerState), [`counter`](scheduling_pndm_flax.md#PNDMSchedulerState.counter), [`FlaxKarrasDiffusionSchedulers`](scheduling_utils_flax.md#FlaxKarrasDiffusionSchedulers), [`add_noise_common`](scheduling_utils_flax.md#add_noise_common), [`create`](scheduling_utils_flax.md#CommonSchedulerState.create), [`FlaxPNDMSchedulerOutput`](scheduling_pndm_flax.md#FlaxPNDMSchedulerOutput), [`cur_model_output`](scheduling_pndm_flax.md#PNDMSchedulerState.cur_model_output), [`num_inference_steps`](scheduling_pndm_flax.md#PNDMSchedulerState.num_inference_steps), [`prev_sample`](scheduling_utils_flax.md#FlaxSchedulerOutput.prev_sample), [`common`](scheduling_pndm_flax.md#PNDMSchedulerState.common), [`cur_sample`](scheduling_pndm_flax.md#PNDMSchedulerState.cur_sample), [`create`](scheduling_pndm_flax.md#PNDMSchedulerState.create), [`prk_timesteps`](scheduling_pndm_flax.md#PNDMSchedulerState.prk_timesteps), [`final_alpha_cumprod`](scheduling_pndm_flax.md#PNDMSchedulerState.final_alpha_cumprod), [`state`](scheduling_pndm_flax.md#FlaxPNDMSchedulerOutput.state)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`ConfigMixin`](../configuration_utils.md#ConfigMixin), [`_import_structure`](__init__.md#_import_structure), [`FlaxSchedulerMixin`](scheduling_utils_flax.md#FlaxSchedulerMixin), [`__init__`](../pipelines/controlnet/pipeline_flax_controlnet_sdxl.md#FlaxStableDiffusionXLControlNetPipeline.__init__), [`__init__`](../pipelines/controlnet/pipeline_flax_controlnet.md#FlaxStableDiffusionControlNetPipeline.__init__), [`__init__`](../pipelines/stable_diffusion/pipeline_flax_stable_diffusion.md#FlaxStableDiffusionPipeline.__init__), [`__init__`](../pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.md#FlaxStableDiffusionXLPipeline.__init__)

### `FlaxPNDMSchedulerOutput`  ·  implements/extends FlaxSchedulerOutput
- def: [`src/maxdiffusion/schedulers/scheduling_pndm_flax.py:69`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L69)
- signature: `class FlaxPNDMSchedulerOutput(FlaxSchedulerOutput):`
- members:
  - `state` — [`L70`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L70)
- uses (calls/refs, reference-scoped): [`FlaxSchedulerOutput`](scheduling_utils_flax.md#FlaxSchedulerOutput), [`PNDMSchedulerState`](scheduling_pndm_flax.md#PNDMSchedulerState)
- used by: [`FlaxSchedulerOutput`](scheduling_utils_flax.md#FlaxSchedulerOutput), [`step_prk`](scheduling_pndm_flax.md#FlaxPNDMScheduler.step_prk), [`step`](scheduling_pndm_flax.md#FlaxPNDMScheduler.step), [`step_plms`](scheduling_pndm_flax.md#FlaxPNDMScheduler.step_plms)

### `PNDMSchedulerState`
- def: [`src/maxdiffusion/schedulers/scheduling_pndm_flax.py:35`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L35)
- signature: `class PNDMSchedulerState:`
- members:
  - `create(cls, common: CommonSchedulerState, final_alpha_cumprod: jnp.ndarray, init_noise_sigma: jnp.ndarray, timesteps: jnp.ndarray)` — [`L53`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L53)
  - `common` — [`L36`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L36)
  - `counter` — [`L48`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L48)
  - `cur_model_output` — [`L47`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L47)
  - `cur_sample` — [`L49`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L49)
  - `ets` — [`L50`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L50)
  - `final_alpha_cumprod` — [`L37`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L37)
  - `init_noise_sigma` — [`L40`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L40)
  - `num_inference_steps` — [`L42`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L42)
  - `plms_timesteps` — [`L44`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L44)
  - `prk_timesteps` — [`L43`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L43)
  - `timesteps` — [`L41`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_pndm_flax.py#L41)
- uses (calls/refs, reference-scoped): [`CommonSchedulerState`](scheduling_utils_flax.md#CommonSchedulerState)
- used by: [`_import_structure`](../pipelines/stable_diffusion/__init__.md#_import_structure), [`_import_structure`](../pipelines/stable_diffusion_xl/__init__.md#_import_structure), [`step_prk`](scheduling_pndm_flax.md#FlaxPNDMScheduler.step_prk), [`step`](scheduling_pndm_flax.md#FlaxPNDMScheduler.step), [`step_plms`](scheduling_pndm_flax.md#FlaxPNDMScheduler.step_plms), [`create_state`](scheduling_pndm_flax.md#FlaxPNDMScheduler.create_state), [`_get_prev_sample`](scheduling_pndm_flax.md#FlaxPNDMScheduler._get_prev_sample), [`set_timesteps`](scheduling_pndm_flax.md#FlaxPNDMScheduler.set_timesteps), [`FlaxPNDMSchedulerOutput`](scheduling_pndm_flax.md#FlaxPNDMSchedulerOutput), [`add_noise`](scheduling_pndm_flax.md#FlaxPNDMScheduler.add_noise), [`scale_model_input`](scheduling_pndm_flax.md#FlaxPNDMScheduler.scale_model_input)

