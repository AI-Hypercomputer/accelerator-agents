---
title: 'Module: src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_output.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_output.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.pipelines.stable_diffusion_xl.pipeline_output`/
symbols:
  FlaxStableDiffusionXLPipelineOutput: FlaxStableDiffusionXLPipelineOutput#
  StableDiffusionXLPipelineOutput: StableDiffusionXLPipelineOutput#
  StableDiffusionXLPipelineOutput.images: StableDiffusionXLPipelineOutput#images.
  FlaxStableDiffusionXLPipelineOutput.images: FlaxStableDiffusionXLPipelineOutput#images.
---
# Module: [`src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_output.py`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_output.py)

## Classes
### `FlaxStableDiffusionXLPipelineOutput`  ·  implements/extends BaseOutput
- def: [`src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_output.py:42`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_output.py#L42)
- doc: Output class for Flax Stable Diffusion XL pipelines.
- signature: `class FlaxStableDiffusionXLPipelineOutput(BaseOutput):`
- members:
  - `images` — [`L51`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_output.py#L51)
- uses (calls/refs, reference-scoped): [`BaseOutput`](../../utils/outputs.md#BaseOutput)
- used by: [`BaseOutput`](../../utils/outputs.md#BaseOutput), [`_import_structure`](__init__.md#_import_structure), [`__call__`](pipeline_flax_stable_diffusion_xl.md#FlaxStableDiffusionXLPipeline.__call__), [`__call__`](../controlnet/pipeline_flax_controlnet_sdxl.md#FlaxStableDiffusionXLControlNetPipeline.__call__)

### `StableDiffusionXLPipelineOutput`  ·  implements/extends BaseOutput
- def: [`src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_output.py:25`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_output.py#L25)
- doc: Output class for Stable Diffusion pipelines.
- signature: `class StableDiffusionXLPipelineOutput(BaseOutput):`
- members:
  - `images` — [`L35`](../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/stable_diffusion_xl/pipeline_output.py#L35)
- uses (calls/refs, reference-scoped): [`BaseOutput`](../../utils/outputs.md#BaseOutput), [`is_flax_available`](../../utils/import_utils.md#is_flax_available)
- used by: [`BaseOutput`](../../utils/outputs.md#BaseOutput)

