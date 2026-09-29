---
title: 'Module: src/maxdiffusion/pipelines/stable_diffusion/pipeline_output.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/pipelines/stable_diffusion/pipeline_output.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.pipelines.stable_diffusion.pipeline_output`/FlaxStableDiffusionPipelineOutput#
symbols:
  FlaxStableDiffusionPipelineOutput: ''
  FlaxStableDiffusionPipelineOutput.images: images.
  FlaxStableDiffusionPipelineOutput.nsfw_content_detected: nsfw_content_detected.
---
# Module: [`src/maxdiffusion/pipelines/stable_diffusion/pipeline_output.py`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/pipeline_output.py)

## Classes
### `FlaxStableDiffusionPipelineOutput`  ·  implements/extends BaseOutput
- def: [`src/maxdiffusion/pipelines/stable_diffusion/pipeline_output.py:25`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/pipeline_output.py#L25)
- doc: Output class for Flax-based Stable Diffusion pipelines.
- signature: `class FlaxStableDiffusionPipelineOutput(BaseOutput):`
- members:
  - `images` — [`L37`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/pipeline_output.py#L37)
  - `nsfw_content_detected` — [`L38`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion/pipeline_output.py#L38)
- uses (calls/refs, reference-scoped): [`BaseOutput`](../../utils/outputs.md#BaseOutput)
- used by: [`_import_structure`](__init__.md#_import_structure), [`BaseOutput`](../../utils/outputs.md#BaseOutput), [`__call__`](pipeline_flax_stable_diffusion.md#FlaxStableDiffusionPipeline.__call__), [`__call__`](../controlnet/pipeline_flax_controlnet.md#FlaxStableDiffusionControlNetPipeline.__call__)

