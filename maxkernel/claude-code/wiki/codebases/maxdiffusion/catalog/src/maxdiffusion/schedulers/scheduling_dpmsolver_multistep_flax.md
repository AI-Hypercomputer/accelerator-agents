---
title: 'Module: src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.schedulers.scheduling_dpmsolver_multistep_flax`/
symbols:
  FlaxDPMSolverMultistepScheduler.step: FlaxDPMSolverMultistepScheduler#step().
  FlaxDPMSolverMultistepScheduler.create_state: FlaxDPMSolverMultistepScheduler#create_state().
  DPMSolverMultistepSchedulerState: DPMSolverMultistepSchedulerState#
  FlaxDPMSolverMultistepScheduler.step_1: FlaxDPMSolverMultistepScheduler#step_1().
  FlaxDPMSolverMultistepScheduler.step_23: FlaxDPMSolverMultistepScheduler#step_23().
  FlaxDPMSolverMultistepScheduler.step_2: FlaxDPMSolverMultistepScheduler#step_2().
  FlaxDPMSolverMultistepScheduler.step_3: FlaxDPMSolverMultistepScheduler#step_3().
  FlaxDPMSolverMultistepScheduler: FlaxDPMSolverMultistepScheduler#
  FlaxDPMSolverMultistepScheduler.dpm_solver_first_order_update: FlaxDPMSolverMultistepScheduler#dpm_solver_first_order_update().
  FlaxDPMSolverMultistepScheduler.multistep_dpm_solver_second_order_update: FlaxDPMSolverMultistepScheduler#multistep_dpm_solver_second_order_update().
  FlaxDPMSolverMultistepScheduler.multistep_dpm_solver_third_order_update: FlaxDPMSolverMultistepScheduler#multistep_dpm_solver_third_order_update().
  DPMSolverMultistepSchedulerState.timesteps: DPMSolverMultistepSchedulerState#timesteps.
  FlaxDPMSolverMultistepScheduler.convert_model_output: FlaxDPMSolverMultistepScheduler#convert_model_output().
  DPMSolverMultistepSchedulerState.alpha_t: DPMSolverMultistepSchedulerState#alpha_t.
  DPMSolverMultistepSchedulerState.sigma_t: DPMSolverMultistepSchedulerState#sigma_t.
  DPMSolverMultistepSchedulerState.lambda_t: DPMSolverMultistepSchedulerState#lambda_t.
  FlaxDPMSolverMultistepScheduler.set_timesteps: FlaxDPMSolverMultistepScheduler#set_timesteps().
  FlaxDPMSolverMultistepScheduler.add_noise: FlaxDPMSolverMultistepScheduler#add_noise().
  FlaxDPMSolverMultistepSchedulerOutput: FlaxDPMSolverMultistepSchedulerOutput#
  DPMSolverMultistepSchedulerState.lower_order_nums: DPMSolverMultistepSchedulerState#lower_order_nums.
  FlaxDPMSolverMultistepScheduler.__init__: FlaxDPMSolverMultistepScheduler#__init__().
  DPMSolverMultistepSchedulerState.model_outputs: DPMSolverMultistepSchedulerState#model_outputs.
  FlaxDPMSolverMultistepScheduler.dtype: FlaxDPMSolverMultistepScheduler#dtype.
  DPMSolverMultistepSchedulerState.common: DPMSolverMultistepSchedulerState#common.
  DPMSolverMultistepSchedulerState.create: DPMSolverMultistepSchedulerState#create().
  DPMSolverMultistepSchedulerState.prev_timestep: DPMSolverMultistepSchedulerState#prev_timestep.
  DPMSolverMultistepSchedulerState.cur_sample: DPMSolverMultistepSchedulerState#cur_sample.
  FlaxDPMSolverMultistepScheduler._compatibles: FlaxDPMSolverMultistepScheduler#_compatibles.
  FlaxDPMSolverMultistepScheduler.scale_model_input: FlaxDPMSolverMultistepScheduler#scale_model_input().
  FlaxDPMSolverMultistepScheduler.__len__: FlaxDPMSolverMultistepScheduler#__len__().
  DPMSolverMultistepSchedulerState.num_inference_steps: DPMSolverMultistepSchedulerState#num_inference_steps.
  FlaxDPMSolverMultistepSchedulerOutput.state: FlaxDPMSolverMultistepSchedulerOutput#state.
  DPMSolverMultistepSchedulerState.init_noise_sigma: DPMSolverMultistepSchedulerState#init_noise_sigma.
  FlaxDPMSolverMultistepScheduler.has_state: FlaxDPMSolverMultistepScheduler#has_state().
---
# Module: [`src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py)

## Classes
### `DPMSolverMultistepSchedulerState`
- def: [`src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py:35`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L35)
- signature: `class DPMSolverMultistepSchedulerState:`
- members:
  - `create(cls, common: CommonSchedulerState, alpha_t: jnp.ndarray, sigma_t: jnp.ndarray, lambda_t: jnp.ndarray, init_noise_sigma: jnp.ndarray, timesteps: jnp.ndarray)` — [`L53`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L53)
  - `alpha_t` — [`L37`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L37)
  - `common` — [`L36`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L36)
  - `cur_sample` — [`L50`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L50)
  - `init_noise_sigma` — [`L42`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L42)
  - `lambda_t` — [`L39`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L39)
  - `lower_order_nums` — [`L48`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L48)
  - `model_outputs` — [`L47`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L47)
  - `num_inference_steps` — [`L44`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L44)
  - `prev_timestep` — [`L49`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L49)
  - `sigma_t` — [`L38`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L38)
  - `timesteps` — [`L43`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L43)
- uses (calls/refs, reference-scoped): [`CommonSchedulerState`](scheduling_utils_flax.md#CommonSchedulerState)
- used by: [`step`](scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler.step), [`create_state`](scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler.create_state), [`step_1`](scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler.step_1), [`step_2`](scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler.step_2), [`step_23`](scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler.step_23), [`step_3`](scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler.step_3), [`dpm_solver_first_order_update`](scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler.dpm_solver_first_order_update), [`multistep_dpm_solver_second_order_update`](scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler.multistep_dpm_solver_second_order_update), [`multistep_dpm_solver_third_order_update`](scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler.multistep_dpm_solver_third_order_update), [`convert_model_output`](scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler.convert_model_output), [`add_noise`](scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler.add_noise), [`set_timesteps`](scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler.set_timesteps), [`FlaxDPMSolverMultistepSchedulerOutput`](scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepSchedulerOutput), [`scale_model_input`](scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler.scale_model_input)

### `FlaxDPMSolverMultistepScheduler`  ·  implements/extends ConfigMixin, FlaxSchedulerMixin
- def: [`src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py:77`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L77)
- doc: DPM-Solver (and the improved version DPM-Solver++) is a fast dedicated high-order solver for diffusion ODEs with
- signature: `class FlaxDPMSolverMultistepScheduler(FlaxSchedulerMixin, ConfigMixin):`
- members:
  - `add_noise(self, state: DPMSolverMultistepSchedulerState, original_samples: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray)` — [`L617`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L617)
  - `convert_model_output(self, state: DPMSolverMultistepSchedulerState, model_output: jnp.ndarray, timestep: int, sample: jnp.ndarray)` — [`L253`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L253) — Convert the model output to the corresponding type that the algorithm (DPM-Solver / DPM-Solver++) needs.
  - `create_state(self, common: Optional[CommonSchedulerState] = None)` — [`L174`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L174)
  - `dpm_solver_first_order_update(self, state: DPMSolverMultistepSchedulerState, model_output: jnp.ndarray, timestep: int, prev_timestep: int, sample: jnp.ndarray)` — [`L321`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L321) — One step for the first-order DPM-Solver (equivalent to DDIM).
  - `has_state(self)` — [`L150`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L150)
  - `multistep_dpm_solver_second_order_update(self, state: DPMSolverMultistepSchedulerState, model_output_list: jnp.ndarray, timestep_list: List[int], prev_timestep: int, sample: jnp.ndarray)` — [`L356`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L356) — One step for the second-order multistep DPM-Solver.
  - `multistep_dpm_solver_third_order_update(self, state: DPMSolverMultistepSchedulerState, model_output_list: jnp.ndarray, timestep_list: List[int], prev_timestep: int, sample: jnp.ndarray)` — [`L410`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L410) — One step for the third-order multistep DPM-Solver.
  - `scale_model_input(self, state: DPMSolverMultistepSchedulerState, sample: jnp.ndarray, timestep: Optional[int] = None)` — [`L599`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L599) — Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
  - `set_timesteps(self, state: DPMSolverMultistepSchedulerState, num_inference_steps: int, shape: Tuple)` — [`L203`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L203) — Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.
  - `step(self, state: DPMSolverMultistepSchedulerState, model_output: jnp.ndarray, timestep: int, sample: jnp.ndarray, return_dict: bool = True)` — [`L466`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L466) — Predict the sample at the previous timestep by DPM-Solver. Core function to propagate the diffusion process
  - `step_1(state: DPMSolverMultistepSchedulerState)` — [`L510`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L510)
  - `step_2(state: DPMSolverMultistepSchedulerState)` — [`L520`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L520)
  - `step_23(state: DPMSolverMultistepSchedulerState)` — [`L519`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L519)
  - `step_3(state: DPMSolverMultistepSchedulerState)` — [`L530`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L530)
  - `dtype` — [`L147`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L147)
- protocol/private: `__init__`[`L154`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L154), `__len__`[`L626`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L626), `_compatibles`[`L145`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L145)
- uses (calls/refs, reference-scoped): [`config`](../configuration_utils.md#ConfigMixin.config), [`ConfigMixin`](../configuration_utils.md#ConfigMixin), [`FlaxSchedulerMixin`](scheduling_utils_flax.md#FlaxSchedulerMixin), [`CommonSchedulerState`](scheduling_utils_flax.md#CommonSchedulerState), [`register_to_config`](../configuration_utils.md#register_to_config), [`alphas_cumprod`](scheduling_utils_flax.md#CommonSchedulerState.alphas_cumprod), [`DPMSolverMultistepSchedulerState`](scheduling_dpmsolver_multistep_flax.md#DPMSolverMultistepSchedulerState), [`FlaxKarrasDiffusionSchedulers`](scheduling_utils_flax.md#FlaxKarrasDiffusionSchedulers), [`add_noise_common`](scheduling_utils_flax.md#add_noise_common), [`timesteps`](scheduling_dpmsolver_multistep_flax.md#DPMSolverMultistepSchedulerState.timesteps), [`create`](scheduling_utils_flax.md#CommonSchedulerState.create), [`alpha_t`](scheduling_dpmsolver_multistep_flax.md#DPMSolverMultistepSchedulerState.alpha_t), [`sigma_t`](scheduling_dpmsolver_multistep_flax.md#DPMSolverMultistepSchedulerState.sigma_t), [`lambda_t`](scheduling_dpmsolver_multistep_flax.md#DPMSolverMultistepSchedulerState.lambda_t), [`FlaxDPMSolverMultistepSchedulerOutput`](scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepSchedulerOutput), [`prev_sample`](scheduling_utils_flax.md#FlaxSchedulerOutput.prev_sample), [`lower_order_nums`](scheduling_dpmsolver_multistep_flax.md#DPMSolverMultistepSchedulerState.lower_order_nums), [`model_outputs`](scheduling_dpmsolver_multistep_flax.md#DPMSolverMultistepSchedulerState.model_outputs), [`common`](scheduling_dpmsolver_multistep_flax.md#DPMSolverMultistepSchedulerState.common), [`create`](scheduling_dpmsolver_multistep_flax.md#DPMSolverMultistepSchedulerState.create), [`cur_sample`](scheduling_dpmsolver_multistep_flax.md#DPMSolverMultistepSchedulerState.cur_sample), [`prev_timestep`](scheduling_dpmsolver_multistep_flax.md#DPMSolverMultistepSchedulerState.prev_timestep), [`num_inference_steps`](scheduling_dpmsolver_multistep_flax.md#DPMSolverMultistepSchedulerState.num_inference_steps), [`state`](scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepSchedulerOutput.state)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`ConfigMixin`](../configuration_utils.md#ConfigMixin), [`_import_structure`](__init__.md#_import_structure), [`FlaxSchedulerMixin`](scheduling_utils_flax.md#FlaxSchedulerMixin), [`__init__`](../pipelines/controlnet/pipeline_flax_controlnet_sdxl.md#FlaxStableDiffusionXLControlNetPipeline.__init__), [`__init__`](../pipelines/controlnet/pipeline_flax_controlnet.md#FlaxStableDiffusionControlNetPipeline.__init__), [`__init__`](../pipelines/stable_diffusion/pipeline_flax_stable_diffusion.md#FlaxStableDiffusionPipeline.__init__), [`__init__`](../pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.md#FlaxStableDiffusionXLPipeline.__init__)

### `FlaxDPMSolverMultistepSchedulerOutput`  ·  implements/extends FlaxSchedulerOutput
- def: [`src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py:73`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L73)
- signature: `class FlaxDPMSolverMultistepSchedulerOutput(FlaxSchedulerOutput):`
- members:
  - `state` — [`L74`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_dpmsolver_multistep_flax.py#L74)
- uses (calls/refs, reference-scoped): [`FlaxSchedulerOutput`](scheduling_utils_flax.md#FlaxSchedulerOutput), [`DPMSolverMultistepSchedulerState`](scheduling_dpmsolver_multistep_flax.md#DPMSolverMultistepSchedulerState)
- used by: [`FlaxSchedulerOutput`](scheduling_utils_flax.md#FlaxSchedulerOutput), [`step`](scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler.step)

