---
title: 'Module: src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.pipelines.stable_diffusion_xl.pipeline_flax_stable_diffusion_xl`/
symbols:
  FlaxStableDiffusionXLPipeline.__init__: FlaxStableDiffusionXLPipeline#__init__().
  FlaxStableDiffusionXLPipeline._generate: FlaxStableDiffusionXLPipeline#_generate().
  FlaxStableDiffusionXLPipeline.__call__: FlaxStableDiffusionXLPipeline#__call__().
  FlaxStableDiffusionXLPipeline: FlaxStableDiffusionXLPipeline#
  FlaxStableDiffusionXLPipeline.vae_scale_factor: FlaxStableDiffusionXLPipeline#vae_scale_factor.
  FlaxStableDiffusionXLPipeline.dtype: FlaxStableDiffusionXLPipeline#dtype.
  FlaxStableDiffusionXLPipeline.get_jit_generate: FlaxStableDiffusionXLPipeline#get_jit_generate().
  FlaxStableDiffusionXLPipeline.get_embeddings: FlaxStableDiffusionXLPipeline#get_embeddings().
  FlaxStableDiffusionXLPipeline.loop_body: FlaxStableDiffusionXLPipeline#loop_body().
  DEBUG: DEBUG.
  FlaxStableDiffusionXLPipeline._get_add_time_ids: FlaxStableDiffusionXLPipeline#_get_add_time_ids().
  _p_generate: _p_generate().
  logger: logger.
  FlaxStableDiffusionXLPipeline.prepare_inputs: FlaxStableDiffusionXLPipeline#prepare_inputs().
---
# Module: [`src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.py`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.py)

## Classes
### `FlaxStableDiffusionXLPipeline`  ·  implements/extends FlaxDiffusionPipeline, StableDiffusionLoraLoaderMixin
- def: [`src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.py:43`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.py#L43)
- signature: `class FlaxStableDiffusionXLPipeline(FlaxDiffusionPipeline, StableDiffusionLoraLoaderMixin):`
- members:
  - `get_embeddings(self, prompt_ids: jnp.array, params)` — [`L146`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.py#L146)
  - `get_jit_generate(self, data_sharding, params_sharding)` — [`L284`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.py#L284)
  - `loop_body(step, args)` — [`L228`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.py#L228)
  - `prepare_inputs(self, prompt: Union[str, List[str]])` — [`L69`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.py#L69)
  - `dtype` — [`L57`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.py#L57)
  - `vae_scale_factor` — [`L67`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.py#L67)
- protocol/private: `__call__`[`L87`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.py#L87), `__init__`[`L45`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.py#L45), `_generate`[`L166`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.py#L166), `_get_add_time_ids`[`L161`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.py#L161)
- uses (calls/refs, reference-scoped): [`FlaxDiffusionPipeline`](../pipeline_flax_utils.md#FlaxDiffusionPipeline), [`FlaxAutoencoderKL`](../../models/vae_flax.md#FlaxAutoencoderKL), [`StableDiffusionLoraLoaderMixin`](../../loaders/lora_pipeline.md#StableDiffusionLoraLoaderMixin), [`FlaxDDIMScheduler`](../../schedulers/scheduling_ddim_flax.md#FlaxDDIMScheduler), [`FlaxDPMSolverMultistepScheduler`](../../schedulers/scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler), [`FlaxLMSDiscreteScheduler`](../../schedulers/scheduling_lms_discrete_flax.md#FlaxLMSDiscreteScheduler), [`FlaxPNDMScheduler`](../../schedulers/scheduling_pndm_flax.md#FlaxPNDMScheduler), [`FlaxUNet2DConditionModel`](../../models/unet_2d_condition_flax.md#FlaxUNet2DConditionModel), [`register_modules`](../pipeline_flax_utils.md#FlaxDiffusionPipeline.register_modules), [`FlaxStableDiffusionXLPipelineOutput`](pipeline_output.md#FlaxStableDiffusionXLPipelineOutput), [`DEBUG`](pipeline_flax_stable_diffusion_xl.md#DEBUG), [`_p_generate`](pipeline_flax_stable_diffusion_xl.md#_p_generate)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`_import_structure`](../__init__.md#_import_structure), [`_import_structure`](__init__.md#_import_structure), [`FlaxDiffusionPipeline`](../pipeline_flax_utils.md#FlaxDiffusionPipeline), [`StableDiffusionLoraLoaderMixin`](../../loaders/lora_pipeline.md#StableDiffusionLoraLoaderMixin)

## Functions
- `_p_generate(pipe, prompt_ids, params, prng_seed, num_inference_steps, height, width, guidance_scale, latents, neg_prompt_ids, return_latents)` — [`L301`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.py#L301)

## Module values
- `DEBUG` — [`L40`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.py#L40)
- `logger` — [`L37`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.py#L37)

