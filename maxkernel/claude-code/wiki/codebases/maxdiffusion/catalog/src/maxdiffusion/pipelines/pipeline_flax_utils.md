---
title: 'Module: src/maxdiffusion/pipelines/pipeline_flax_utils.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/pipelines/pipeline_flax_utils.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.pipelines.pipeline_flax_utils`/
symbols:
  FlaxDiffusionPipeline.from_pretrained: FlaxDiffusionPipeline#from_pretrained().
  FlaxDiffusionPipeline: FlaxDiffusionPipeline#
  FlaxDiffusionPipeline.save_pretrained: FlaxDiffusionPipeline#save_pretrained().
  FlaxDiffusionPipeline.register_modules: FlaxDiffusionPipeline#register_modules().
  library: library.
  FlaxDiffusionPipeline._progress_bar_config: FlaxDiffusionPipeline#_progress_bar_config.
  LOADABLE_CLASSES: LOADABLE_CLASSES.
  logger: logger.
  FlaxDiffusionPipeline.components: FlaxDiffusionPipeline#components().
  FlaxImagePipelineOutput: FlaxImagePipelineOutput#
  FlaxDiffusionPipeline.set_progress_bar_config: FlaxDiffusionPipeline#set_progress_bar_config().
  ALL_IMPORTABLE_CLASSES: ALL_IMPORTABLE_CLASSES.
  import_flax_or_no_model: import_flax_or_no_model().
  FlaxDiffusionPipeline._get_signature_keys: FlaxDiffusionPipeline#_get_signature_keys().
  FlaxDiffusionPipeline.config_name: FlaxDiffusionPipeline#config_name.
  FlaxDiffusionPipeline.load_module: FlaxDiffusionPipeline#load_module().
  INDEX_FILE: INDEX_FILE.
  FlaxImagePipelineOutput.images: FlaxImagePipelineOutput#images.
  FlaxDiffusionPipeline.numpy_to_pil: FlaxDiffusionPipeline#numpy_to_pil().
  FlaxDiffusionPipeline.progress_bar: FlaxDiffusionPipeline#progress_bar().
---
# Module: [`src/maxdiffusion/pipelines/pipeline_flax_utils.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/pipeline_flax_utils.py)

## Classes
### `FlaxDiffusionPipeline`  ·  implements/extends ConfigMixin, PushToHubMixin
- def: [`src/maxdiffusion/pipelines/pipeline_flax_utils.py:100`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/pipeline_flax_utils.py#L100)
- doc: Base class for Flax-based pipelines.
- signature: `class FlaxDiffusionPipeline(ConfigMixin, PushToHubMixin):`
- members:
  - `components(self)` — [`L556`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/pipeline_flax_utils.py#L556) — The `self.components` property can be useful to run different pipelines with the same weights and
  - `from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs)` — [`L227`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/pipeline_flax_utils.py#L227) — Instantiate a Flax-based diffusion pipeline from pretrained pipeline weights.
  - `load_module(name, value)` — [`L408`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/pipeline_flax_utils.py#L408)
  - `numpy_to_pil(images)` — [`L591`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/pipeline_flax_utils.py#L591) — Convert a NumPy image or a batch of images to a PIL image.
  - `progress_bar(self, iterable)` — [`L607`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/pipeline_flax_utils.py#L607)
  - `register_modules(self, **kwargs)` — [`L117`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/pipeline_flax_utils.py#L117)
  - `save_pretrained(self, save_directory: Union[str, os.PathLike], params: Union[Dict, FrozenDict], push_to_hub: bool = False, **kwargs)` — [`L150`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/pipeline_flax_utils.py#L150) — Save all saveable variables of the pipeline to a directory. A pipeline variable can be saved and loaded if its
  - `set_progress_bar_config(self, **kwargs)` — [`L615`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/pipeline_flax_utils.py#L615)
  - `config_name` — [`L115`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/pipeline_flax_utils.py#L115)
- protocol/private: `_get_signature_keys`[`L548`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/pipeline_flax_utils.py#L548), `_progress_bar_config`[`L609`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/pipeline_flax_utils.py#L609)
- uses (calls/refs, reference-scoped): [`config`](../configuration_utils.md#ConfigMixin.config), [`ConfigMixin`](../configuration_utils.md#ConfigMixin), [`FlaxModelMixin`](../models/modeling_flax_utils.md#FlaxModelMixin), [`FlaxSchedulerMixin`](../schedulers/scheduling_utils_flax.md#FlaxSchedulerMixin), [`http_user_agent`](../utils/hub_utils.md#http_user_agent), [`PushToHubMixin`](../utils/hub_utils.md#PushToHubMixin), [`load_config`](../configuration_utils.md#ConfigMixin.load_config), [`extract_init_dict`](../configuration_utils.md#ConfigMixin.extract_init_dict), [`DIFFUSERS_CACHE`](../utils/constants.md#DIFFUSERS_CACHE), [`FLAX_WEIGHTS_NAME`](../utils/constants.md#FLAX_WEIGHTS_NAME), [`save_config`](../configuration_utils.md#ConfigMixin.save_config), [`register_to_config`](../configuration_utils.md#ConfigMixin.register_to_config), [`FlaxStableDiffusionPipeline`](stable_diffusion/pipeline_flax_stable_diffusion.md#FlaxStableDiffusionPipeline), [`FlaxStableDiffusionXLPipeline`](stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.md#FlaxStableDiffusionXLPipeline), [`CONFIG_NAME`](../utils/constants.md#CONFIG_NAME), [`_upload_folder`](../utils/hub_utils.md#PushToHubMixin._upload_folder), [`LOADABLE_CLASSES`](pipeline_flax_utils.md#LOADABLE_CLASSES), [`FlaxStableDiffusionControlNetPipeline`](controlnet/pipeline_flax_controlnet.md#FlaxStableDiffusionControlNetPipeline), [`FlaxStableDiffusionXLControlNetPipeline`](controlnet/pipeline_flax_controlnet_sdxl.md#FlaxStableDiffusionXLControlNetPipeline), [`logger`](pipeline_flax_utils.md#logger), `repo_id`, [`SCHEDULER_CONFIG_NAME`](../schedulers/scheduling_utils_flax.md#SCHEDULER_CONFIG_NAME), [`ALL_IMPORTABLE_CLASSES`](pipeline_flax_utils.md#ALL_IMPORTABLE_CLASSES), [`import_flax_or_no_model`](pipeline_flax_utils.md#import_flax_or_no_model)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`ConfigMixin`](../configuration_utils.md#ConfigMixin), [`_import_structure`](__init__.md#_import_structure), [`PushToHubMixin`](../utils/hub_utils.md#PushToHubMixin), [`dtype`](stable_diffusion/pipeline_flax_stable_diffusion.md#FlaxStableDiffusionPipeline.dtype), [`FlaxStableDiffusionPipeline`](stable_diffusion/pipeline_flax_stable_diffusion.md#FlaxStableDiffusionPipeline), [`FlaxStableDiffusionXLPipeline`](stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.md#FlaxStableDiffusionXLPipeline), [`FlaxStableDiffusionControlNetPipeline`](controlnet/pipeline_flax_controlnet.md#FlaxStableDiffusionControlNetPipeline), [`FlaxStableDiffusionXLControlNetPipeline`](controlnet/pipeline_flax_controlnet_sdxl.md#FlaxStableDiffusionXLControlNetPipeline), [`dtype`](controlnet/pipeline_flax_controlnet_sdxl.md#FlaxStableDiffusionXLControlNetPipeline.dtype), [`dtype`](controlnet/pipeline_flax_controlnet.md#FlaxStableDiffusionControlNetPipeline.dtype), [`dtype`](stable_diffusion_xl/pipeline_flax_stable_diffusion_xl.md#FlaxStableDiffusionXLPipeline.dtype)

### `FlaxImagePipelineOutput`  ·  implements/extends BaseOutput
- def: [`src/maxdiffusion/pipelines/pipeline_flax_utils.py:87`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/pipeline_flax_utils.py#L87)
- doc: Output class for image pipelines.
- signature: `class FlaxImagePipelineOutput(BaseOutput):`
- members:
  - `images` — [`L97`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/pipeline_flax_utils.py#L97)
- uses (calls/refs, reference-scoped): [`BaseOutput`](../utils/outputs.md#BaseOutput)
- used by: [`BaseOutput`](../utils/outputs.md#BaseOutput)

## Functions
- `import_flax_or_no_model(module, class_name)` — [`L73`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/pipeline_flax_utils.py#L73)

## Module values
- `ALL_IMPORTABLE_CLASSES` — [`L68`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/pipeline_flax_utils.py#L68)
- `INDEX_FILE` — [`L46`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/pipeline_flax_utils.py#L46)
- `LOADABLE_CLASSES` — [`L52`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/pipeline_flax_utils.py#L52)
- `library` — [`L69`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/pipeline_flax_utils.py#L69)
- `logger` — [`L49`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/pipelines/pipeline_flax_utils.py#L49)

