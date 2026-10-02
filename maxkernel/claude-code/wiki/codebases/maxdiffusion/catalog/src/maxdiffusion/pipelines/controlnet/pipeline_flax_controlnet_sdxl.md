---
title: 'Module: src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet_sdxl.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet_sdxl.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.pipelines.controlnet.pipeline_flax_controlnet_sdxl`/
symbols:
  FlaxStableDiffusionXLControlNetPipeline.__init__: FlaxStableDiffusionXLControlNetPipeline#__init__().
  FlaxStableDiffusionXLControlNetPipeline._generate: FlaxStableDiffusionXLControlNetPipeline#_generate().
  FlaxStableDiffusionXLControlNetPipeline.__call__: FlaxStableDiffusionXLControlNetPipeline#__call__().
  FlaxStableDiffusionXLControlNetPipeline: FlaxStableDiffusionXLControlNetPipeline#
  preprocess: preprocess().
  FlaxStableDiffusionXLControlNetPipeline.dtype: FlaxStableDiffusionXLControlNetPipeline#dtype.
  FlaxStableDiffusionXLControlNetPipeline.prepare_image_inputs: FlaxStableDiffusionXLControlNetPipeline#prepare_image_inputs().
  FlaxStableDiffusionXLControlNetPipeline.get_jit_generate: FlaxStableDiffusionXLControlNetPipeline#get_jit_generate().
  FlaxStableDiffusionXLControlNetPipeline.vae_scale_factor: FlaxStableDiffusionXLControlNetPipeline#vae_scale_factor.
  FlaxStableDiffusionXLControlNetPipeline.get_embeddings: FlaxStableDiffusionXLControlNetPipeline#get_embeddings().
  FlaxStableDiffusionXLControlNetPipeline.loop_body: FlaxStableDiffusionXLControlNetPipeline#loop_body().
  DEBUG: DEBUG.
  FlaxStableDiffusionXLControlNetPipeline._get_add_time_ids: FlaxStableDiffusionXLControlNetPipeline#_get_add_time_ids().
  _p_generate: _p_generate().
  logger: logger.
  FlaxStableDiffusionXLControlNetPipeline.prepare_text_inputs: FlaxStableDiffusionXLControlNetPipeline#prepare_text_inputs().
---
# Module: [`src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet_sdxl.py`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet_sdxl.py)

## Classes
### `FlaxStableDiffusionXLControlNetPipeline`  ·  implements/extends FlaxDiffusionPipeline
- def: [`src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet_sdxl.py:42`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet_sdxl.py#L42)
- signature: `class FlaxStableDiffusionXLControlNetPipeline(FlaxDiffusionPipeline):`
- members:
  - `get_embeddings(self, prompt_ids: jnp.array, params)` — [`L163`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet_sdxl.py#L163)
  - `get_jit_generate(self, data_sharding, params_sharding)` — [`L317`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet_sdxl.py#L317)
  - `loop_body(step, args)` — [`L248`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet_sdxl.py#L248)
  - `prepare_image_inputs(self, image: Union[Image.Image, List[Image.Image]])` — [`L89`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet_sdxl.py#L89)
  - `prepare_text_inputs(self, prompt: Union[str, List[str]])` — [`L71`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet_sdxl.py#L71)
  - `dtype` — [`L57`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet_sdxl.py#L57)
  - `vae_scale_factor` — [`L69`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet_sdxl.py#L69)
- protocol/private: `__call__`[`L100`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet_sdxl.py#L100), `__init__`[`L44`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet_sdxl.py#L44), `_generate`[`L183`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet_sdxl.py#L183), `_get_add_time_ids`[`L178`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet_sdxl.py#L178)
- uses (calls/refs, reference-scoped): [`FlaxDiffusionPipeline`](../pipeline_flax_utils.md#FlaxDiffusionPipeline), [`FlaxAutoencoderKL`](../../models/vae_flax.md#FlaxAutoencoderKL), [`FlaxDDIMScheduler`](../../schedulers/scheduling_ddim_flax.md#FlaxDDIMScheduler), [`FlaxDPMSolverMultistepScheduler`](../../schedulers/scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler), [`FlaxLMSDiscreteScheduler`](../../schedulers/scheduling_lms_discrete_flax.md#FlaxLMSDiscreteScheduler), [`FlaxPNDMScheduler`](../../schedulers/scheduling_pndm_flax.md#FlaxPNDMScheduler), [`FlaxUNet2DConditionModel`](../../models/unet_2d_condition_flax.md#FlaxUNet2DConditionModel), [`FlaxControlNetModel`](../../models/controlnet_flax.md#FlaxControlNetModel), [`register_modules`](../pipeline_flax_utils.md#FlaxDiffusionPipeline.register_modules), [`FlaxStableDiffusionXLPipelineOutput`](../stable_diffusion_xl/pipeline_output.md#FlaxStableDiffusionXLPipelineOutput), [`preprocess`](pipeline_flax_controlnet_sdxl.md#preprocess), [`DEBUG`](pipeline_flax_controlnet_sdxl.md#DEBUG), [`_p_generate`](pipeline_flax_controlnet_sdxl.md#_p_generate)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`_import_structure`](../__init__.md#_import_structure), [`FlaxDiffusionPipeline`](../pipeline_flax_utils.md#FlaxDiffusionPipeline), [`_import_structure`](__init__.md#_import_structure)

## Functions
- `_p_generate(pipe, prompt_ids, image, params, prng_seed, num_inference_steps, guidance_scale, latents, neg_prompt_ids, return_latents, controlnet_conditioning_scale)` — [`L334`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet_sdxl.py#L334)
- `preprocess(image, dtype)` — [`L361`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet_sdxl.py#L361)

## Module values
- `DEBUG` — [`L39`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet_sdxl.py#L39)
- `logger` — [`L36`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet_sdxl.py#L36)

