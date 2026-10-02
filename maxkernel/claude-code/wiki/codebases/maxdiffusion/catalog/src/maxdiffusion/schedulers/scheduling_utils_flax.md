---
title: 'Module: src/maxdiffusion/schedulers/scheduling_utils_flax.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/schedulers/scheduling_utils_flax.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.schedulers.scheduling_utils_flax`/
symbols:
  FlaxSchedulerMixin: FlaxSchedulerMixin#
  FlaxSchedulerOutput: FlaxSchedulerOutput#
  CommonSchedulerState: CommonSchedulerState#
  CommonSchedulerState.alphas_cumprod: CommonSchedulerState#alphas_cumprod.
  FlaxKarrasDiffusionSchedulers: FlaxKarrasDiffusionSchedulers#
  add_noise_common: add_noise_common().
  CommonSchedulerState.create: CommonSchedulerState#create().
  get_sqrt_alpha_prod: get_sqrt_alpha_prod().
  get_velocity_common: get_velocity_common().
  broadcast_to_shape_from_left: broadcast_to_shape_from_left().
  FlaxSchedulerOutput.prev_sample: FlaxSchedulerOutput#prev_sample.
  CommonSchedulerState.betas: CommonSchedulerState#betas.
  FlaxSchedulerMixin._get_compatibles: FlaxSchedulerMixin#_get_compatibles().
  betas_for_alpha_bar: betas_for_alpha_bar().
  SCHEDULER_CONFIG_NAME: SCHEDULER_CONFIG_NAME.
  FlaxSchedulerMixin.config_name: FlaxSchedulerMixin#config_name.
  FlaxSchedulerMixin.compatibles: FlaxSchedulerMixin#compatibles().
  betas_for_alpha_bar.alpha_bar: betas_for_alpha_bar().alpha_bar().
  FlaxSchedulerMixin._compatibles: FlaxSchedulerMixin#_compatibles.
  FlaxSchedulerMixin.from_pretrained: FlaxSchedulerMixin#from_pretrained().
  rescale_betas_zero_snr: rescale_betas_zero_snr().
  CommonSchedulerState.alphas: CommonSchedulerState#alphas.
  FlaxKarrasDiffusionSchedulers.FlaxDDIMScheduler: FlaxKarrasDiffusionSchedulers#FlaxDDIMScheduler.
  FlaxKarrasDiffusionSchedulers.FlaxDDPMScheduler: FlaxKarrasDiffusionSchedulers#FlaxDDPMScheduler.
  FlaxKarrasDiffusionSchedulers.FlaxPNDMScheduler: FlaxKarrasDiffusionSchedulers#FlaxPNDMScheduler.
  FlaxKarrasDiffusionSchedulers.FlaxLMSDiscreteScheduler: FlaxKarrasDiffusionSchedulers#FlaxLMSDiscreteScheduler.
  FlaxKarrasDiffusionSchedulers.FlaxDPMSolverMultistepScheduler: FlaxKarrasDiffusionSchedulers#FlaxDPMSolverMultistepScheduler.
  FlaxSchedulerMixin.ignore_for_config: FlaxSchedulerMixin#ignore_for_config.
  FlaxSchedulerMixin.has_compatibles: FlaxSchedulerMixin#has_compatibles.
  FlaxSchedulerMixin.save_pretrained: FlaxSchedulerMixin#save_pretrained().
---
# Module: [`src/maxdiffusion/schedulers/scheduling_utils_flax.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py)

## Classes
### `CommonSchedulerState`
- def: [`src/maxdiffusion/schedulers/scheduling_utils_flax.py:249`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L249) — documented in [maxdiffusion-schedulers-scheduling_utils_flax](../../../../concepts/maxdiffusion-schedulers-scheduling_utils_flax.md)
- signature: `class CommonSchedulerState:`
- members:
  - `create(cls, scheduler)` — [`L255`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L255)
  - `alphas` — [`L250`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L250)
  - `alphas_cumprod` — [`L252`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L252) — documented in [maxdiffusion-schedulers-scheduling_utils_flax](../../../../concepts/maxdiffusion-schedulers-scheduling_utils_flax.md)
  - `betas` — [`L251`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L251)
- uses (calls/refs, reference-scoped): [`betas_for_alpha_bar`](scheduling_utils_flax.md#betas_for_alpha_bar), [`rescale_betas_zero_snr`](scheduling_utils_flax.md#rescale_betas_zero_snr)
- used by: [`step`](scheduling_ddpm_flax.md#FlaxDDPMScheduler.step), [`step`](scheduling_ddim_flax.md#FlaxDDIMScheduler.step), [`create_state`](scheduling_ddim_flax.md#FlaxDDIMScheduler.create_state), [`create_state`](scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler.create_state), [`create_state`](scheduling_euler_discrete_flax.md#FlaxEulerDiscreteScheduler.create_state), [`create_state`](scheduling_pndm_flax.md#FlaxPNDMScheduler.create_state), [`_get_variance`](scheduling_ddpm_flax.md#FlaxDDPMScheduler._get_variance), [`create_state`](scheduling_ddpm_flax.md#FlaxDDPMScheduler.create_state), [`create_state`](scheduling_lms_discrete_flax.md#FlaxLMSDiscreteScheduler.create_state), [`_get_prev_sample`](scheduling_pndm_flax.md#FlaxPNDMScheduler._get_prev_sample), [`common`](scheduling_ddpm_flax.md#DDPMSchedulerState.common), [`add_noise_common`](scheduling_utils_flax.md#add_noise_common), [`set_timesteps`](scheduling_euler_discrete_flax.md#FlaxEulerDiscreteScheduler.set_timesteps), [`set_timesteps`](scheduling_lms_discrete_flax.md#FlaxLMSDiscreteScheduler.set_timesteps), [`_get_variance`](scheduling_ddim_flax.md#FlaxDDIMScheduler._get_variance), [`get_sqrt_alpha_prod`](scheduling_utils_flax.md#get_sqrt_alpha_prod), [`get_velocity_common`](scheduling_utils_flax.md#get_velocity_common), [`common`](scheduling_ddim_flax.md#DDIMSchedulerState.common), [`common`](scheduling_pndm_flax.md#PNDMSchedulerState.common), [`common`](scheduling_euler_discrete_flax.md#EulerDiscreteSchedulerState.common), [`common`](scheduling_lms_discrete_flax.md#LMSDiscreteSchedulerState.common), [`common`](scheduling_dpmsolver_multistep_flax.md#DPMSolverMultistepSchedulerState.common), [`create`](scheduling_ddim_flax.md#DDIMSchedulerState.create), [`create`](scheduling_ddpm_flax.md#DDPMSchedulerState.create), [`create`](scheduling_dpmsolver_multistep_flax.md#DPMSolverMultistepSchedulerState.create), [`create`](scheduling_euler_discrete_flax.md#EulerDiscreteSchedulerState.create), [`create`](scheduling_lms_discrete_flax.md#LMSDiscreteSchedulerState.create), [`create`](scheduling_pndm_flax.md#PNDMSchedulerState.create)

### `FlaxKarrasDiffusionSchedulers`  ·  implements/extends Enum
- def: [`src/maxdiffusion/schedulers/scheduling_utils_flax.py:34`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L34)
- signature: `class FlaxKarrasDiffusionSchedulers(Enum):`
- members:
  - `FlaxDDIMScheduler` — [`L35`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L35)
  - `FlaxDDPMScheduler` — [`L36`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L36)
  - `FlaxDPMSolverMultistepScheduler` — [`L39`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L39)
  - `FlaxLMSDiscreteScheduler` — [`L38`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L38)
  - `FlaxPNDMScheduler` — [`L37`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L37)
- used by: [`_import_structure`](__init__.md#_import_structure), [`_compatibles`](scheduling_ddim_flax.md#FlaxDDIMScheduler._compatibles), [`_compatibles`](scheduling_ddpm_flax.md#FlaxDDPMScheduler._compatibles), [`_compatibles`](scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler._compatibles), [`_compatibles`](scheduling_euler_discrete_flax.md#FlaxEulerDiscreteScheduler._compatibles), [`_compatibles`](scheduling_lms_discrete_flax.md#FlaxLMSDiscreteScheduler._compatibles), [`_compatibles`](scheduling_pndm_flax.md#FlaxPNDMScheduler._compatibles)

### `FlaxSchedulerMixin`  ·  implements/extends PushToHubMixin
- def: [`src/maxdiffusion/schedulers/scheduling_utils_flax.py:56`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L56) — documented in [maxdiffusion-schedulers-scheduling_utils_flax](../../../../concepts/maxdiffusion-schedulers-scheduling_utils_flax.md)
- doc: Mixin containing common functions for the schedulers.
- signature: `class FlaxSchedulerMixin(PushToHubMixin):`
- members:
  - `compatibles(self)` — [`L169`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L169) — Returns all schedulers that are compatible with this scheduler
  - `from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None, subfolder: Optional[str] = None, return_unused_kwargs=False, **kwargs)` — [`L72`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L72) — Instantiate a Scheduler class from a pre-defined JSON-file.
  - `save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs)` — [`L151`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L151) — Save a scheduler configuration object to the directory `save_directory`, so that it can be re-loaded using the
  - `config_name` — [`L66`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L66)
  - `has_compatibles` — [`L69`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L69)
  - `ignore_for_config` — [`L67`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L67)
- protocol/private: `_compatibles`[`L68`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L68), `_get_compatibles`[`L179`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L179)
- uses (calls/refs, reference-scoped): [`PushToHubMixin`](../utils/hub_utils.md#PushToHubMixin), [`FlaxDDIMScheduler`](scheduling_ddim_flax.md#FlaxDDIMScheduler), [`FlaxDPMSolverMultistepScheduler`](scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler), [`FlaxLMSDiscreteScheduler`](scheduling_lms_discrete_flax.md#FlaxLMSDiscreteScheduler), [`FlaxPNDMScheduler`](scheduling_pndm_flax.md#FlaxPNDMScheduler), [`FlaxEulerDiscreteScheduler`](scheduling_euler_discrete_flax.md#FlaxEulerDiscreteScheduler), [`FlaxDDPMScheduler`](scheduling_ddpm_flax.md#FlaxDDPMScheduler), [`FlaxFlowMatchScheduler`](scheduling_flow_match_flax.md#FlaxFlowMatchScheduler), [`SCHEDULER_CONFIG_NAME`](scheduling_utils_flax.md#SCHEDULER_CONFIG_NAME)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`_import_structure`](__init__.md#_import_structure), [`from_pretrained`](../pipelines/pipeline_flax_utils.md#FlaxDiffusionPipeline.from_pretrained), [`PushToHubMixin`](../utils/hub_utils.md#PushToHubMixin), [`FlaxDDIMScheduler`](scheduling_ddim_flax.md#FlaxDDIMScheduler), [`FlaxDPMSolverMultistepScheduler`](scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler), [`FlaxLMSDiscreteScheduler`](scheduling_lms_discrete_flax.md#FlaxLMSDiscreteScheduler), [`FlaxPNDMScheduler`](scheduling_pndm_flax.md#FlaxPNDMScheduler), [`FlaxEulerDiscreteScheduler`](scheduling_euler_discrete_flax.md#FlaxEulerDiscreteScheduler), [`FlaxDDPMScheduler`](scheduling_ddpm_flax.md#FlaxDDPMScheduler), [`create_scheduler`](../trainers/flux_trainer.md#FluxTrainer.create_scheduler), [`FlaxFlowMatchScheduler`](scheduling_flow_match_flax.md#FlaxFlowMatchScheduler)

### `FlaxSchedulerOutput`  ·  implements/extends BaseOutput
- def: [`src/maxdiffusion/schedulers/scheduling_utils_flax.py:43`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L43) — documented in [maxdiffusion-schedulers-scheduling_utils_flax](../../../../concepts/maxdiffusion-schedulers-scheduling_utils_flax.md)
- doc: Base class for the scheduler's step function output.
- signature: `class FlaxSchedulerOutput(BaseOutput):`
- members:
  - `prev_sample` — [`L53`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L53)
- uses (calls/refs, reference-scoped): [`BaseOutput`](../utils/outputs.md#BaseOutput), [`FlaxPNDMSchedulerOutput`](scheduling_pndm_flax.md#FlaxPNDMSchedulerOutput), [`FlaxDDIMSchedulerOutput`](scheduling_ddim_flax.md#FlaxDDIMSchedulerOutput), [`FlaxDDPMSchedulerOutput`](scheduling_ddpm_flax.md#FlaxDDPMSchedulerOutput), [`FlaxDPMSolverMultistepSchedulerOutput`](scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepSchedulerOutput), [`FlaxEulerDiscreteSchedulerOutput`](scheduling_euler_discrete_flax.md#FlaxEulerDiscreteSchedulerOutput), [`FlaxLMSSchedulerOutput`](scheduling_lms_discrete_flax.md#FlaxLMSSchedulerOutput), [`FlaxFlowMatchSchedulerOutput`](scheduling_flow_match_flax.md#FlaxFlowMatchSchedulerOutput)
- used by: [`BaseOutput`](../utils/outputs.md#BaseOutput), [`_import_structure`](__init__.md#_import_structure), [`step`](scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler.step), [`step`](scheduling_ddpm_flax.md#FlaxDDPMScheduler.step), [`step`](scheduling_ddim_flax.md#FlaxDDIMScheduler.step), [`step`](scheduling_pndm_flax.md#FlaxPNDMScheduler.step), [`step`](scheduling_lms_discrete_flax.md#FlaxLMSDiscreteScheduler.step), [`step`](scheduling_euler_discrete_flax.md#FlaxEulerDiscreteScheduler.step), [`FlaxPNDMSchedulerOutput`](scheduling_pndm_flax.md#FlaxPNDMSchedulerOutput), [`FlaxDDIMSchedulerOutput`](scheduling_ddim_flax.md#FlaxDDIMSchedulerOutput), [`FlaxDDPMSchedulerOutput`](scheduling_ddpm_flax.md#FlaxDDPMSchedulerOutput), [`FlaxDPMSolverMultistepSchedulerOutput`](scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepSchedulerOutput), [`FlaxEulerDiscreteSchedulerOutput`](scheduling_euler_discrete_flax.md#FlaxEulerDiscreteSchedulerOutput), [`FlaxLMSSchedulerOutput`](scheduling_lms_discrete_flax.md#FlaxLMSSchedulerOutput), [`FlaxFlowMatchSchedulerOutput`](scheduling_flow_match_flax.md#FlaxFlowMatchSchedulerOutput)

## Functions
- `add_noise_common(state: CommonSchedulerState, original_samples: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray)` — [`L306`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L306) — documented in [maxdiffusion-schedulers-scheduling_utils_flax](../../../../concepts/maxdiffusion-schedulers-scheduling_utils_flax.md)
- `alpha_bar(time_step)` — [`L209`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L209)
- `betas_for_alpha_bar(num_diffusion_timesteps: int, max_beta=0.999, dtype=jnp.float32)` — [`L191`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L191) — Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
- `broadcast_to_shape_from_left(x: jnp.ndarray, shape: Tuple[int])` — [`L186`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L186)
- `get_sqrt_alpha_prod(state: CommonSchedulerState, original_samples: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray)` — [`L290`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L290) — documented in [maxdiffusion-schedulers-scheduling_utils_flax](../../../../concepts/maxdiffusion-schedulers-scheduling_utils_flax.md)
- `get_velocity_common(state: CommonSchedulerState, sample: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray)` — [`L312`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L312) — documented in [maxdiffusion-schedulers-scheduling_utils_flax](../../../../concepts/maxdiffusion-schedulers-scheduling_utils_flax.md)
- `rescale_betas_zero_snr(betas)` — [`L220`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L220) — Rescales betas to have a zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)

## Module values
- `SCHEDULER_CONFIG_NAME` — [`L27`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/schedulers/scheduling_utils_flax.py#L27)

