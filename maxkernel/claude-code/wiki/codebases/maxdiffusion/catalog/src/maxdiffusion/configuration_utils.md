---
title: 'Module: src/maxdiffusion/configuration_utils.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/configuration_utils.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.configuration_utils`/
symbols:
  ConfigMixin.config: ConfigMixin#config().
  ConfigMixin: ConfigMixin#
  register_to_config: register_to_config().
  ConfigMixin.load_config: ConfigMixin#load_config().
  ConfigMixin.extract_init_dict: ConfigMixin#extract_init_dict().
  ConfigMixin.from_config: ConfigMixin#from_config().
  ConfigMixin.config_name: ConfigMixin#config_name.
  ConfigMixin.to_json_string: ConfigMixin#to_json_string().
  ConfigMixin.save_config: ConfigMixin#save_config().
  flax_register_to_config: flax_register_to_config().
  ConfigMixin.register_to_config: ConfigMixin#register_to_config().
  ConfigMixin._internal_dict: ConfigMixin#_internal_dict.
  logger: logger.
  ConfigMixin.__getattr__: ConfigMixin#__getattr__().
  ConfigMixin.get_config_dict: ConfigMixin#get_config_dict().
  ConfigMixin._get_init_keys: ConfigMixin#_get_init_keys().
  ConfigMixin.to_json_file: ConfigMixin#to_json_file().
  register_to_config.inner_init: register_to_config().inner_init().
  flax_register_to_config.init: flax_register_to_config().init().
  FrozenDict.__setattr__: FrozenDict#__setattr__().
  FrozenDict.__setitem__: FrozenDict#__setitem__().
  ConfigMixin.__repr__: ConfigMixin#__repr__().
  FrozenDict: FrozenDict#
  FrozenDict.__frozen: FrozenDict#__frozen.
  ConfigMixin.ignore_for_config: ConfigMixin#ignore_for_config.
  CustomEncoder: CustomEncoder#
  ConfigMixin.has_compatibles: ConfigMixin#has_compatibles.
  ConfigMixin._deprecated_kwargs: ConfigMixin#_deprecated_kwargs.
  ConfigMixin._dict_from_json_file: ConfigMixin#_dict_from_json_file().
  ConfigMixin.to_json_saveable: ConfigMixin#to_json_saveable().
  _re_configuration_file: _re_configuration_file.
  CustomEncoder.default: CustomEncoder#default().
  FrozenDict.__init__: FrozenDict#__init__().
  FrozenDict.__delitem__: FrozenDict#__delitem__().
  FrozenDict.setdefault: FrozenDict#setdefault().
  FrozenDict.pop: FrozenDict#pop().
  FrozenDict.update: FrozenDict#update().
---
# Module: [`src/maxdiffusion/configuration_utils.py`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py)

## Classes
### `ConfigMixin`
- def: [`src/maxdiffusion/configuration_utils.py:104`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L104) — documented in [maxdiffusion-configuration_utils](../../../concepts/maxdiffusion-configuration_utils.md)
- doc: Base class for all configuration classes. All configuration parameters are stored under `self.config`. Also
- signature: `class ConfigMixin:`
- members:
  - `__getattr__(self, name: str)` — [`L144`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L144) — The only reason we overwrite `getattr` here is to gracefully deprecate accessing
  - `config(self)` — [`L566`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L566) — Returns the config of the class as a frozen dictionary — documented in [maxdiffusion-configuration_utils](../../../concepts/maxdiffusion-configuration_utils.md)
  - `extract_init_dict(cls, config_dict, **kwargs)` — [`L471`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L471)
  - `from_config(cls, config: Union[FrozenDict, Dict[str, Any]] = None, return_unused_kwargs=False, **kwargs)` — [`L205`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L205) — Instantiate a Python class from a config dictionary.
  - `get_config_dict(cls, *args, **kwargs)` — [`L291`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L291)
  - `load_config(cls, pretrained_model_name_or_path: Union[str, os.PathLike], return_unused_kwargs=False, return_commit_hash=False, **kwargs)` — [`L300`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L300) — Load a model or scheduler configuration. — documented in [maxdiffusion-configuration_utils](../../../concepts/maxdiffusion-configuration_utils.md)
  - `register_to_config(self, **kwargs)` — [`L127`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L127)
  - `save_config(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs)` — [`L162`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L162) — Save a configuration object to the directory specified in `save_directory` so that it can be reloaded using the
  - `to_json_file(self, json_file_path: Union[str, os.PathLike])` — [`L621`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L621) — Save the configuration instance's parameters to a JSON file.
  - `to_json_saveable(value)` — [`L587`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L587)
  - `to_json_string(self)` — [`L575`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L575) — Serializes the configuration instance to a JSON string.
  - `config_name` — [`L121`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L121)
  - `has_compatibles` — [`L123`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L123)
  - `ignore_for_config` — [`L122`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L122)
- protocol/private: `__repr__`[`L562`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L562), `_deprecated_kwargs`[`L125`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L125), `_dict_from_json_file`[`L557`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L557), `_get_init_keys`[`L467`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L467), `_internal_dict`[`L142`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L142)
- uses (calls/refs, reference-scoped): [`__version__`](__init__.md#__version__), [`http_user_agent`](utils/hub_utils.md#http_user_agent), [`DummyObject`](utils/import_utils.md#DummyObject), [`FlaxDiffusionPipeline`](pipelines/pipeline_flax_utils.md#FlaxDiffusionPipeline), [`FlaxAutoencoderKL`](models/vae_flax.md#FlaxAutoencoderKL), [`DIFFUSERS_CACHE`](utils/constants.md#DIFFUSERS_CACHE), [`FlaxDDIMScheduler`](schedulers/scheduling_ddim_flax.md#FlaxDDIMScheduler), [`FlaxDPMSolverMultistepScheduler`](schedulers/scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler), [`FlaxLMSDiscreteScheduler`](schedulers/scheduling_lms_discrete_flax.md#FlaxLMSDiscreteScheduler), [`FlaxPNDMScheduler`](schedulers/scheduling_pndm_flax.md#FlaxPNDMScheduler), [`FlaxUNet2DConditionModel`](models/unet_2d_condition_flax.md#FlaxUNet2DConditionModel), [`FlaxControlNetModel`](models/controlnet_flax.md#FlaxControlNetModel), [`deprecate`](utils/deprecation_utils.md#deprecate), [`FluxTransformer2DModel`](models/flux/transformers/transformer_flux_flax.md#FluxTransformer2DModel), [`WanModel`](models/wan/transformers/transformer_wan.md#WanModel), [`FlaxEulerDiscreteScheduler`](schedulers/scheduling_euler_discrete_flax.md#FlaxEulerDiscreteScheduler), [`HUGGINGFACE_CO_RESOLVE_ENDPOINT`](utils/constants.md#HUGGINGFACE_CO_RESOLVE_ENDPOINT), [`FlaxDDPMScheduler`](schedulers/scheduling_ddpm_flax.md#FlaxDDPMScheduler), [`logger`](configuration_utils.md#logger), [`WanAnimateTransformer3DModel`](models/wan/transformers/transformer_wan_animate.md#WanAnimateTransformer3DModel), [`AutoencoderKLWan`](models/wan/autoencoder_kl_wan.md#AutoencoderKLWan), [`AutoencoderKLWan2p2`](models/wan/autoencoder_kl_wan_2p2.md#AutoencoderKLWan2p2), [`FlaxAutoencoderKLLTX2Audio`](models/ltx2/autoencoder_kl_ltx2_audio.md#FlaxAutoencoderKLLTX2Audio), [`FlaxFlowMatchScheduler`](schedulers/scheduling_flow_match_flax.md#FlaxFlowMatchScheduler), [`LTX2VideoAutoencoderKL`](models/ltx2/autoencoder_kl_ltx2.md#LTX2VideoAutoencoderKL), `repo_id`, [`extract_commit_hash`](utils/hub_utils.md#extract_commit_hash), [`FrozenDict`](configuration_utils.md#FrozenDict), [`CustomEncoder`](configuration_utils.md#CustomEncoder)
- used by: [`_import_structure`](__init__.md#_import_structure), [`__call__`](models/wan/transformers/transformer_wan_vace.md#WanVACEModel.__call__), [`from_pretrained`](pipelines/pipeline_flax_utils.md#FlaxDiffusionPipeline.from_pretrained), [`__call__`](models/controlnet_flax.md#FlaxControlNetModel.__call__), [`decoder`](models/vae_flax.md#FlaxAutoencoderKL.decoder), [`encoder`](models/vae_flax.md#FlaxAutoencoderKL.encoder), [`__call__`](models/wan/transformers/transformer_wan.md#WanModel.__call__), [`step`](schedulers/scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler.step), [`FlaxDiffusionPipeline`](pipelines/pipeline_flax_utils.md#FlaxDiffusionPipeline), [`step`](schedulers/scheduling_ddpm_flax.md#FlaxDDPMScheduler.step), [`step_prk`](schedulers/scheduling_pndm_flax.md#FlaxPNDMScheduler.step_prk), [`step`](schedulers/scheduling_ddim_flax.md#FlaxDDIMScheduler.step), [`step`](schedulers/scheduling_pndm_flax.md#FlaxPNDMScheduler.step), [`step_plms`](schedulers/scheduling_pndm_flax.md#FlaxPNDMScheduler.step_plms), [`SPATIOTEMPORAL`](models/ltx2/autoencoder_kl_ltx2.md#DownsampleType.SPATIOTEMPORAL), [`step`](schedulers/scheduling_lms_discrete_flax.md#FlaxLMSDiscreteScheduler.step), [`step`](schedulers/scheduling_euler_discrete_flax.md#FlaxEulerDiscreteScheduler.step), [`FlaxAutoencoderKL`](models/vae_flax.md#FlaxAutoencoderKL), [`create_state`](schedulers/scheduling_ddim_flax.md#FlaxDDIMScheduler.create_state), [`create_state`](schedulers/scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler.create_state), [`create_state`](schedulers/scheduling_euler_discrete_flax.md#FlaxEulerDiscreteScheduler.create_state), [`create_state`](schedulers/scheduling_pndm_flax.md#FlaxPNDMScheduler.create_state), [`init_weights`](models/controlnet_flax.md#FlaxControlNetModel.init_weights), [`init_weights`](models/unet_2d_condition_flax.md#FlaxUNet2DConditionModel.init_weights), [`_get_variance`](schedulers/scheduling_ddpm_flax.md#FlaxDDPMScheduler._get_variance), [`add_time_proj`](models/controlnet_flax.md#FlaxControlNetModel.add_time_proj), [`step_23`](schedulers/scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler.step_23), [`FlaxDDIMScheduler`](schedulers/scheduling_ddim_flax.md#FlaxDDIMScheduler), [`FlaxDPMSolverMultistepScheduler`](schedulers/scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler), [`FlaxLMSDiscreteScheduler`](schedulers/scheduling_lms_discrete_flax.md#FlaxLMSDiscreteScheduler), [`FlaxPNDMScheduler`](schedulers/scheduling_pndm_flax.md#FlaxPNDMScheduler), [`FlaxUNet2DConditionModel`](models/unet_2d_condition_flax.md#FlaxUNet2DConditionModel), [`create_state`](schedulers/scheduling_ddpm_flax.md#FlaxDDPMScheduler.create_state), [`create_state`](schedulers/scheduling_lms_discrete_flax.md#FlaxLMSDiscreteScheduler.create_state), [`_get_prev_sample`](schedulers/scheduling_pndm_flax.md#FlaxPNDMScheduler._get_prev_sample), [`FlaxControlNetModel`](models/controlnet_flax.md#FlaxControlNetModel), [`decode`](models/vae_flax.md#FlaxAutoencoderKL.decode), [`dpm_solver_first_order_update`](schedulers/scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler.dpm_solver_first_order_update), [`multistep_dpm_solver_second_order_update`](schedulers/scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler.multistep_dpm_solver_second_order_update), [`multistep_dpm_solver_third_order_update`](schedulers/scheduling_dpmsolver_multistep_flax.md#FlaxDPMSolverMultistepScheduler.multistep_dpm_solver_third_order_update)  (+41 more)

### `CustomEncoder`  ·  implements/extends JSONEncoder
- def: [`src/maxdiffusion/configuration_utils.py:52`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L52)
- doc: Custom JSON encoder to handle non-serializable types like JAX/Numpy dtypes.
- signature: `class CustomEncoder(json.JSONEncoder):`
- members:
  - `default(self, o)` — [`L57`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L57)
- used by: [`to_json_string`](configuration_utils.md#ConfigMixin.to_json_string)

### `FrozenDict`  ·  implements/extends OrderedDict
- def: [`src/maxdiffusion/configuration_utils.py:71`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L71)
- signature: `class FrozenDict(OrderedDict):`
- members:
  - `pop(self, *args, **kwargs)` — [`L87`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L87)
  - `setdefault(self, *args, **kwargs)` — [`L84`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L84)
  - `update(self, *args, **kwargs)` — [`L90`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L90)
- protocol/private: `__delitem__`[`L81`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L81), `__frozen`[`L79`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L79), `__init__`[`L73`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L73), `__setattr__`[`L93`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L93), `__setitem__`[`L98`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L98)
- used by: [`from_config`](configuration_utils.md#ConfigMixin.from_config), [`_internal_dict`](configuration_utils.md#ConfigMixin._internal_dict)

## Functions
- `flax_register_to_config(cls)` — [`L679`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L679)
- `init(self, *args, **kwargs)` — [`L683`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L683)
- `inner_init(self, *args, **kwargs)` — [`L643`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L643)
- `register_to_config(init)` — [`L633`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L633) — Decorator to apply on the init of classes inheriting from [`ConfigMixin`] so that all the arguments are — documented in [maxdiffusion-configuration_utils](../../../concepts/maxdiffusion-configuration_utils.md)

## Module values
- `_re_configuration_file` — [`L49`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L49)
- `logger` — [`L47`](../../../../../../raw/code/maxdiffusion/src/maxdiffusion/configuration_utils.py#L47)

