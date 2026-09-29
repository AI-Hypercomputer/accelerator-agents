---
title: 'Module: src/maxdiffusion/pipelines/stable_diffusion/pipeline_flax_stable_diffusion.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/pipelines/stable_diffusion/pipeline_flax_stable_diffusion.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.pipelines.stable_diffusion.pipeline_flax_stable_diffusion`/
symbols:
  FlaxStableDiffusionPipeline.__init__: FlaxStableDiffusionPipeline#__init__().
  FlaxStableDiffusionPipeline.__call__: FlaxStableDiffusionPipeline#__call__().
  FlaxStableDiffusionPipeline.dtype: FlaxStableDiffusionPipeline#dtype.
  FlaxStableDiffusionPipeline._generate: FlaxStableDiffusionPipeline#_generate().
  FlaxStableDiffusionPipeline: FlaxStableDiffusionPipeline#
  FlaxStableDiffusionPipeline.vae_scale_factor: FlaxStableDiffusionPipeline#vae_scale_factor.
  logger: logger.
  FlaxStableDiffusionPipeline.loop_body: FlaxStableDiffusionPipeline#loop_body().
  DEBUG: DEBUG.
  EXAMPLE_DOC_STRING: EXAMPLE_DOC_STRING.
  _p_generate: _p_generate().
  FlaxStableDiffusionPipeline.prepare_inputs: FlaxStableDiffusionPipeline#prepare_inputs().
  _p_get_has_nsfw_concepts: _p_get_has_nsfw_concepts().
  unshard: unshard().
---
# Module: [`src/maxdiffusion/pipelines/stable_diffusion/pipeline_flax_stable_diffusion.py`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/pipeline_flax_stable_diffusion.py)

## Classes
### `FlaxStableDiffusionPipeline`  ·  implements/extends FlaxDiffusionPipeline, StableDiffusionLoraLoaderMixin
- def: [`src/maxdiffusion/pipelines/stable_diffusion/pipeline_flax_stable_diffusion.py:77`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/pipeline_flax_stable_diffusion.py#L77)
- doc: Flax-based pipeline for text-to-image generation using Stable Diffusion.
- signature: `class FlaxStableDiffusionPipeline(FlaxDiffusionPipeline, StableDiffusionLoraLoaderMixin):`
- members:
  - `__call__(self, prompt_ids: jnp.array, params: Union[Dict, FrozenDict], prng_seed: jax.Array, num_inference_steps: int = 50, height: Optional[int] = None, width: Optional[int] = None, guidance_scale: Union[float, jnp.ndarray] = 7.5, latents: jnp.ndarray = None, neg_prompt_ids: jnp.ndarray = None, return_dict: bool = True, jit: bool = False)` — [`L245`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/pipeline_flax_stable_diffusion.py#L245) — The call function to the pipeline for generation.
  - `loop_body(step, args)` — [`L196`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/pipeline_flax_stable_diffusion.py#L196)
  - `prepare_inputs(self, prompt: Union[str, List[str]])` — [`L135`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/pipeline_flax_stable_diffusion.py#L135)
  - `dtype` — [`L109`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/pipeline_flax_stable_diffusion.py#L109)
  - `vae_scale_factor` — [`L133`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/pipeline_flax_stable_diffusion.py#L133)
- protocol/private: `__init__`[`L99`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/pipeline_flax_stable_diffusion.py#L99), `_generate`[`L148`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/pipeline_flax_stable_diffusion.py#L148)
- uses (calls/refs, reference-scoped): [`config`](../../configuration_utils.md#ConfigMixin.config), [`FlaxDiffusionPipeline`](../pipeline_flax_utils.md#FlaxDiffusionPipeline), [`FlaxAutoencoderKL`](../../models/vae_flax.md#FlaxAutoencoderKL), [`StableDiffusionLoraLoaderMixin`](../../loaders/lora_pipeline.md#StableDiffusionLoraLoaderMixin), [`FlaxDDIMScheduler`](../../schedulers/scheduling_ddim_flax.md#FlaxDDIMScheduler), [`FlaxDPMSolverMultistepScheduler`](../../schedulers/scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler), [`FlaxLMSDiscreteScheduler`](../../schedulers/scheduling_lms_discrete_flax.md#FlaxLMSDiscreteScheduler), [`FlaxPNDMScheduler`](../../schedulers/scheduling_pndm_flax.md#FlaxPNDMScheduler), [`FlaxUNet2DConditionModel`](../../models/unet_2d_condition_flax.md#FlaxUNet2DConditionModel), [`deprecate`](../../utils/deprecation_utils.md#deprecate), [`register_modules`](../pipeline_flax_utils.md#FlaxDiffusionPipeline.register_modules), [`_internal_dict`](../../configuration_utils.md#ConfigMixin._internal_dict), [`FlaxStableDiffusionPipelineOutput`](pipeline_output.md#FlaxStableDiffusionPipelineOutput), [`replace_example_docstring`](../../utils/doc_utils.md#replace_example_docstring), [`DEBUG`](pipeline_flax_stable_diffusion.md#DEBUG), [`EXAMPLE_DOC_STRING`](pipeline_flax_stable_diffusion.md#EXAMPLE_DOC_STRING), [`_p_generate`](pipeline_flax_stable_diffusion.md#_p_generate)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`_import_structure`](__init__.md#_import_structure), [`_import_structure`](../__init__.md#_import_structure), [`FlaxDiffusionPipeline`](../pipeline_flax_utils.md#FlaxDiffusionPipeline), [`StableDiffusionLoraLoaderMixin`](../../loaders/lora_pipeline.md#StableDiffusionLoraLoaderMixin)

## Functions
- `_p_generate(pipe, prompt_ids, params, prng_seed, num_inference_steps, height, width, guidance_scale, latents, neg_prompt_ids)` — [`L356`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/pipeline_flax_stable_diffusion.py#L356)
- `_p_get_has_nsfw_concepts(pipe, features, params)` — [`L382`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/pipeline_flax_stable_diffusion.py#L382)
- `unshard(x: jnp.ndarray)` — [`L386`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/pipeline_flax_stable_diffusion.py#L386)

## Module values
- `DEBUG` — [`L41`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/pipeline_flax_stable_diffusion.py#L41)
- `EXAMPLE_DOC_STRING` — [`L43`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/pipeline_flax_stable_diffusion.py#L43)
- `logger` — [`L38`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/pipeline_flax_stable_diffusion.py#L38)

