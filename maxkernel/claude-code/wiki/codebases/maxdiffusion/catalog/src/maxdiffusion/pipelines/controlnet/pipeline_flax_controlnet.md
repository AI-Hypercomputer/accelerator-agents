---
title: 'Module: src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.pipelines.controlnet.pipeline_flax_controlnet`/
symbols:
  FlaxStableDiffusionControlNetPipeline.__init__: FlaxStableDiffusionControlNetPipeline#__init__().
  FlaxStableDiffusionControlNetPipeline.__call__: FlaxStableDiffusionControlNetPipeline#__call__().
  FlaxStableDiffusionControlNetPipeline._generate: FlaxStableDiffusionControlNetPipeline#_generate().
  FlaxStableDiffusionControlNetPipeline: FlaxStableDiffusionControlNetPipeline#
  preprocess: preprocess().
  logger: logger.
  FlaxStableDiffusionControlNetPipeline.dtype: FlaxStableDiffusionControlNetPipeline#dtype.
  FlaxStableDiffusionControlNetPipeline.prepare_image_inputs: FlaxStableDiffusionControlNetPipeline#prepare_image_inputs().
  FlaxStableDiffusionControlNetPipeline.vae_scale_factor: FlaxStableDiffusionControlNetPipeline#vae_scale_factor.
  FlaxStableDiffusionControlNetPipeline.loop_body: FlaxStableDiffusionControlNetPipeline#loop_body().
  DEBUG: DEBUG.
  EXAMPLE_DOC_STRING: EXAMPLE_DOC_STRING.
  _p_generate: _p_generate().
  FlaxStableDiffusionControlNetPipeline.prepare_text_inputs: FlaxStableDiffusionControlNetPipeline#prepare_text_inputs().
  _p_get_has_nsfw_concepts: _p_get_has_nsfw_concepts().
  unshard: unshard().
---
# Module: [`src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet.py`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet.py)

## Classes
### `FlaxStableDiffusionControlNetPipeline`  ·  implements/extends FlaxDiffusionPipeline
- def: [`src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet.py:107`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet.py#L107)
- doc: Flax-based pipeline for text-to-image generation using Stable Diffusion with ControlNet Guidance.
- signature: `class FlaxStableDiffusionControlNetPipeline(FlaxDiffusionPipeline):`
- members:
  - `__call__(self, prompt_ids: jnp.ndarray, image: jnp.ndarray, params: Union[Dict, FrozenDict], prng_seed: jax.Array, num_inference_steps: int = 50, guidance_scale: Union[float, jnp.ndarray] = 7.5, latents: jnp.ndarray = None, neg_prompt_ids: jnp.ndarray = None, controlnet_conditioning_scale: Union[float, jnp.ndarray] = 1, return_dict: bool = True, jit: bool = False)` — [`L290`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet.py#L290) — The call function to the pipeline for generation.
  - `loop_body(step, args)` — [`L228`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet.py#L228)
  - `prepare_image_inputs(self, image: Union[Image.Image, List[Image.Image]])` — [`L169`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet.py#L169)
  - `prepare_text_inputs(self, prompt: Union[str, List[str]])` — [`L155`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet.py#L155)
  - `dtype` — [`L148`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet.py#L148)
  - `vae_scale_factor` — [`L153`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet.py#L153)
- protocol/private: `__init__`[`L137`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet.py#L137), `_generate`[`L180`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet.py#L180)
- uses (calls/refs, reference-scoped): [`FlaxDiffusionPipeline`](../pipeline_flax_utils.md#FlaxDiffusionPipeline), [`FlaxAutoencoderKL`](../../models/vae_flax.md#FlaxAutoencoderKL), [`FlaxDDIMScheduler`](../../schedulers/scheduling_ddim_flax.md#FlaxDDIMScheduler), [`FlaxDPMSolverMultistepScheduler`](../../schedulers/scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler), [`FlaxLMSDiscreteScheduler`](../../schedulers/scheduling_lms_discrete_flax.md#FlaxLMSDiscreteScheduler), [`FlaxPNDMScheduler`](../../schedulers/scheduling_pndm_flax.md#FlaxPNDMScheduler), [`FlaxUNet2DConditionModel`](../../models/unet_2d_condition_flax.md#FlaxUNet2DConditionModel), [`FlaxControlNetModel`](../../models/controlnet_flax.md#FlaxControlNetModel), [`register_modules`](../pipeline_flax_utils.md#FlaxDiffusionPipeline.register_modules), [`FlaxStableDiffusionPipelineOutput`](../stable_diffusion/pipeline_output.md#FlaxStableDiffusionPipelineOutput), [`replace_example_docstring`](../../utils/doc_utils.md#replace_example_docstring), [`preprocess`](pipeline_flax_controlnet.md#preprocess), [`DEBUG`](pipeline_flax_controlnet.md#DEBUG), [`EXAMPLE_DOC_STRING`](pipeline_flax_controlnet.md#EXAMPLE_DOC_STRING), [`_p_generate`](pipeline_flax_controlnet.md#_p_generate)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`_import_structure`](../__init__.md#_import_structure), [`FlaxDiffusionPipeline`](../pipeline_flax_utils.md#FlaxDiffusionPipeline), [`_import_structure`](__init__.md#_import_structure)

## Functions
- `_p_generate(pipe, prompt_ids, image, params, prng_seed, num_inference_steps, guidance_scale, latents, neg_prompt_ids, controlnet_conditioning_scale)` — [`L412`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet.py#L412)
- `_p_get_has_nsfw_concepts(pipe, features, params)` — [`L438`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet.py#L438)
- `preprocess(image, dtype)` — [`L449`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet.py#L449)
- `unshard(x: jnp.ndarray)` — [`L442`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet.py#L442)

## Module values
- `DEBUG` — [`L40`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet.py#L40)
- `EXAMPLE_DOC_STRING` — [`L42`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet.py#L42)
- `logger` — [`L37`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/controlnet/pipeline_flax_controlnet.py#L37)

