---
title: 'Module: src/maxdiffusion/models/modeling_flax_utils.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/modeling_flax_utils.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.modeling_flax_utils`/
symbols:
  FlaxModelMixin: FlaxModelMixin#
  get_activation: get_activation().
  FlaxModelMixin.mesh: FlaxModelMixin#mesh.
  FlaxModelMixin.init_weights: FlaxModelMixin#init_weights().
  FlaxModelMixin.save_pretrained: FlaxModelMixin#save_pretrained().
  FlaxModelMixin._cast_floating_to: FlaxModelMixin#_cast_floating_to().
  FlaxModelMixin.from_pretrained: FlaxModelMixin#from_pretrained().
  logger: logger.
  FlaxModelMixin.config_name: FlaxModelMixin#config_name.
  FlaxModelMixin.to_bf16: FlaxModelMixin#to_bf16().
  FlaxModelMixin.to_fp32: FlaxModelMixin#to_fp32().
  FlaxModelMixin.to_fp16: FlaxModelMixin#to_fp16().
  FlaxModelMixin.conditional_cast: FlaxModelMixin#conditional_cast().
  _ACTIVATIONS: _ACTIVATIONS.
  FlaxModelMixin._automatically_saved_args: FlaxModelMixin#_automatically_saved_args.
  FlaxModelMixin._flax_internal_args: FlaxModelMixin#_flax_internal_args.
  FlaxModelMixin._from_config: FlaxModelMixin#_from_config().
---
# Module: [`src/maxdiffusion/models/modeling_flax_utils.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_utils.py)

## Classes
### `FlaxModelMixin`  ·  implements/extends PushToHubMixin
- def: [`src/maxdiffusion/models/modeling_flax_utils.py:63`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_utils.py#L63)
- doc: Base class for all Flax models.
- signature: `class FlaxModelMixin(PushToHubMixin):`
- members:
  - `_cast_floating_to(self, params: Union[Dict, FrozenDict], dtype: jnp.dtype, mask: Any = None)` — [`L84`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_utils.py#L84) — Helper method to cast floating-point values of given parameter `PyTree` to given `dtype`.
  - `_from_config(cls, config, **kwargs)` — [`L78`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_utils.py#L78) — All context managers that the model should be initialized under go here.
  - `conditional_cast(param)` — [`L90`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_utils.py#L90)
  - `from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], dtype: jnp.dtype = jnp.float32, *model_args, **kwargs)` — [`L217`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_utils.py#L217) — Instantiate a pretrained Flax model from a pretrained model configuration.
  - `init_weights(self, rng: jax.Array, eval_only: bool = False)` — [`L213`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_utils.py#L213)
  - `save_pretrained(self, save_directory: Union[str, os.PathLike], params: Union[Dict, FrozenDict], is_main_process: bool = True, push_to_hub: bool = False, **kwargs)` — [`L460`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_utils.py#L460) — Save a model and its configuration file to a directory so that it can be reloaded using the
  - `to_bf16(self, params: Union[Dict, FrozenDict], mask: Any = None)` — [`L108`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_utils.py#L108) — Cast the floating-point `params` to `jax.numpy.bfloat16`. This returns a new `params` tree and does not cast
  - `to_fp16(self, params: Union[Dict, FrozenDict], mask: Any = None)` — [`L174`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_utils.py#L174) — Cast the floating-point `params` to `jax.numpy.float16`. This returns a new `params` tree and does not cast the
  - `to_fp32(self, params: Union[Dict, FrozenDict], mask: Any = None)` — [`L147`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_utils.py#L147) — Cast the floating-point `params` to `jax.numpy.float32`. This method can be used to explicitly convert the
  - `config_name` — [`L73`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_utils.py#L73)
  - `mesh` — [`L326`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_utils.py#L326)
- protocol/private: `_automatically_saved_args`[`L74`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_utils.py#L74), `_flax_internal_args`[`L75`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_utils.py#L75)
- uses (calls/refs, reference-scoped): [`__version__`](../__init__.md#__version__), [`is_torch_available`](../utils/import_utils.md#is_torch_available), [`PushToHubMixin`](../utils/hub_utils.md#PushToHubMixin), [`FlaxAutoencoderKL`](vae_flax.md#FlaxAutoencoderKL), [`init_weights`](controlnet_flax.md#FlaxControlNetModel.init_weights), [`init_weights`](unet_2d_condition_flax.md#FlaxUNet2DConditionModel.init_weights), [`WEIGHTS_NAME`](../utils/constants.md#WEIGHTS_NAME), [`DIFFUSERS_CACHE`](../utils/constants.md#DIFFUSERS_CACHE), [`FlaxUNet2DConditionModel`](unet_2d_condition_flax.md#FlaxUNet2DConditionModel), [`FLAX_WEIGHTS_NAME`](../utils/constants.md#FLAX_WEIGHTS_NAME), [`FlaxControlNetModel`](controlnet_flax.md#FlaxControlNetModel), [`load_state_dict`](modeling_utils.md#load_state_dict), [`FluxTransformer2DModel`](flux/transformers/transformer_flux_flax.md#FluxTransformer2DModel), [`SAFETENSORS_WEIGHTS_NAME`](../utils/constants.md#SAFETENSORS_WEIGHTS_NAME), [`WanModel`](wan/transformers/transformer_wan.md#WanModel), [`HUGGINGFACE_CO_RESOLVE_ENDPOINT`](../utils/constants.md#HUGGINGFACE_CO_RESOLVE_ENDPOINT), [`CONFIG_NAME`](../utils/constants.md#CONFIG_NAME), [`convert_pytorch_state_dict_to_flax`](modeling_flax_pytorch_utils.md#convert_pytorch_state_dict_to_flax), [`_upload_folder`](../utils/hub_utils.md#PushToHubMixin._upload_folder), [`WanAnimateTransformer3DModel`](wan/transformers/transformer_wan_animate.md#WanAnimateTransformer3DModel), [`AutoencoderKLWan`](wan/autoencoder_kl_wan.md#AutoencoderKLWan), [`AutoencoderKLWan2p2`](wan/autoencoder_kl_wan_2p2.md#AutoencoderKLWan2p2), [`FlaxAutoencoderKLLTX2Audio`](ltx2/autoencoder_kl_ltx2_audio.md#FlaxAutoencoderKLLTX2Audio), [`LTX2VideoAutoencoderKL`](ltx2/autoencoder_kl_ltx2.md#LTX2VideoAutoencoderKL), [`init_weights`](vae_flax.md#FlaxAutoencoderKL.init_weights), `repo_id`, [`logger`](modeling_flax_utils.md#logger), [`init_weights`](flux/transformers/transformer_flux_flax.md#FluxTransformer2DModel.init_weights), [`init_weights`](wan/transformers/transformer_wan_animate.md#WanAnimateTransformer3DModel.init_weights)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`from_pretrained`](../pipelines/pipeline_flax_utils.md#FlaxDiffusionPipeline.from_pretrained), [`SPATIOTEMPORAL`](ltx2/autoencoder_kl_ltx2.md#DownsampleType.SPATIOTEMPORAL), [`PushToHubMixin`](../utils/hub_utils.md#PushToHubMixin), [`FlaxAutoencoderKL`](vae_flax.md#FlaxAutoencoderKL), [`FlaxUNet2DConditionModel`](unet_2d_condition_flax.md#FlaxUNet2DConditionModel), [`FlaxControlNetModel`](controlnet_flax.md#FlaxControlNetModel), [`FluxTransformer2DModel`](flux/transformers/transformer_flux_flax.md#FluxTransformer2DModel), [`WanModel`](wan/transformers/transformer_wan.md#WanModel), [`WanAnimateTransformer3DModel`](wan/transformers/transformer_wan_animate.md#WanAnimateTransformer3DModel), [`AutoencoderKLWan`](wan/autoencoder_kl_wan.md#AutoencoderKLWan), [`AutoencoderKLWan2p2`](wan/autoencoder_kl_wan_2p2.md#AutoencoderKLWan2p2), [`FlaxAutoencoderKLLTX2Audio`](ltx2/autoencoder_kl_ltx2_audio.md#FlaxAutoencoderKLLTX2Audio), [`LTX2VideoAutoencoderKL`](ltx2/autoencoder_kl_ltx2.md#LTX2VideoAutoencoderKL)

## Functions
- `get_activation(name: str)` — [`L56`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_utils.py#L56)

## Module values
- `_ACTIVATIONS` — [`L46`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_utils.py#L46)
- `logger` — [`L45`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_flax_utils.py#L45)

