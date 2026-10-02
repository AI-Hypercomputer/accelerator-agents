---
title: 'Module: src/maxdiffusion/utils/outputs.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/utils/outputs.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.utils.outputs`/
symbols:
  BaseOutput: BaseOutput#
  is_tensor: is_tensor().
  BaseOutput.__getitem__: BaseOutput#__getitem__().
  BaseOutput.to_tuple: BaseOutput#to_tuple().
  BaseOutput.__post_init__: BaseOutput#__post_init__().
  BaseOutput.__delitem__: BaseOutput#__delitem__().
  BaseOutput.setdefault: BaseOutput#setdefault().
  BaseOutput.pop: BaseOutput#pop().
  BaseOutput.update: BaseOutput#update().
  BaseOutput.__setattr__: BaseOutput#__setattr__().
  BaseOutput.__setitem__: BaseOutput#__setitem__().
  BaseOutput.__reduce__: BaseOutput#__reduce__().
---
# Module: [`src/maxdiffusion/utils/outputs.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/outputs.py)

## Classes
### `BaseOutput`  ·  implements/extends OrderedDict
- def: [`src/maxdiffusion/utils/outputs.py:40`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/outputs.py#L40)
- doc: Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
- signature: `class BaseOutput(OrderedDict):`
- members:
  - `pop(self, *args, **kwargs)` — [`L79`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/outputs.py#L79)
  - `setdefault(self, *args, **kwargs)` — [`L76`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/outputs.py#L76)
  - `to_tuple(self)` — [`L111`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/outputs.py#L111) — Convert self to a tuple containing all the attributes/keys that are not `None`.
  - `update(self, *args, **kwargs)` — [`L82`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/outputs.py#L82)
- protocol/private: `__delitem__`[`L73`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/outputs.py#L73), `__getitem__`[`L85`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/outputs.py#L85), `__post_init__`[`L54`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/outputs.py#L54), `__reduce__`[`L104`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/outputs.py#L104), `__setattr__`[`L92`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/outputs.py#L92), `__setitem__`[`L98`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/outputs.py#L98)
- uses (calls/refs, reference-scoped): [`FlaxSchedulerOutput`](../schedulers/scheduling_utils_flax.md#FlaxSchedulerOutput), [`FlaxDecoderOutput`](../models/vae_flax.md#FlaxDecoderOutput), [`FlaxAutoencoderKLOutput`](../models/vae_flax.md#FlaxAutoencoderKLOutput), [`FlaxStableDiffusionPipelineOutput`](../pipelines/stable_diffusion/pipeline_output.md#FlaxStableDiffusionPipelineOutput), [`FlaxStableDiffusionXLPipelineOutput`](../pipelines/stable_diffusion_xl/pipeline_output.md#FlaxStableDiffusionXLPipelineOutput), [`FlaxControlNetOutput`](../models/controlnet_flax.md#FlaxControlNetOutput), [`FlaxUNet2DConditionOutput`](../models/unet_2d_condition_flax.md#FlaxUNet2DConditionOutput), [`FlaxAutoencoderKLOutput`](../models/ltx2/autoencoder_kl_ltx2_audio.md#FlaxAutoencoderKLOutput), [`FlaxDecoderOutput`](../models/ltx2/autoencoder_kl_ltx2_audio.md#FlaxDecoderOutput), [`Transformer2DModelOutput`](../models/flux/transformers/transformer_flux_flax.md#Transformer2DModelOutput), [`FlaxImagePipelineOutput`](../pipelines/pipeline_flax_utils.md#FlaxImagePipelineOutput), [`StableDiffusionXLPipelineOutput`](../pipelines/stable_diffusion_xl/pipeline_output.md#StableDiffusionXLPipelineOutput)
- used by: [`FlaxSchedulerOutput`](../schedulers/scheduling_utils_flax.md#FlaxSchedulerOutput), [`FlaxDecoderOutput`](../models/vae_flax.md#FlaxDecoderOutput), [`FlaxAutoencoderKLOutput`](../models/vae_flax.md#FlaxAutoencoderKLOutput), [`FlaxStableDiffusionPipelineOutput`](../pipelines/stable_diffusion/pipeline_output.md#FlaxStableDiffusionPipelineOutput), [`FlaxStableDiffusionXLPipelineOutput`](../pipelines/stable_diffusion_xl/pipeline_output.md#FlaxStableDiffusionXLPipelineOutput), [`FlaxControlNetOutput`](../models/controlnet_flax.md#FlaxControlNetOutput), [`FlaxUNet2DConditionOutput`](../models/unet_2d_condition_flax.md#FlaxUNet2DConditionOutput), [`FlaxAutoencoderKLOutput`](../models/ltx2/autoencoder_kl_ltx2_audio.md#FlaxAutoencoderKLOutput), [`FlaxDecoderOutput`](../models/ltx2/autoencoder_kl_ltx2_audio.md#FlaxDecoderOutput), [`Transformer2DModelOutput`](../models/flux/transformers/transformer_flux_flax.md#Transformer2DModelOutput), [`FlaxImagePipelineOutput`](../pipelines/pipeline_flax_utils.md#FlaxImagePipelineOutput), [`StableDiffusionXLPipelineOutput`](../pipelines/stable_diffusion_xl/pipeline_output.md#StableDiffusionXLPipelineOutput)

## Functions
- `is_tensor(x)` — [`L27`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/utils/outputs.py#L27) — Tests if `x` is a `torch.Tensor` or `np.ndarray`.

