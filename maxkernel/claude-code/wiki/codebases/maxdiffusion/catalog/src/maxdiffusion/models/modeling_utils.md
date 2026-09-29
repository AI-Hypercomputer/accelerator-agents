---
title: 'Module: src/maxdiffusion/models/modeling_utils.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/modeling_utils.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.modeling_utils`/
symbols:
  ModelMixin.from_pretrained: ModelMixin#from_pretrained().
  logger: logger.
  ModelMixin.save_pretrained: ModelMixin#save_pretrained().
  ModelMixin.disable_adapters: ModelMixin#disable_adapters().
  ModelMixin.enable_adapters: ModelMixin#enable_adapters().
  ModelMixin.set_adapter: ModelMixin#set_adapter().
  load_state_dict: load_state_dict().
  ModelMixin.add_adapter: ModelMixin#add_adapter().
  _LOW_CPU_MEM_USAGE_DEFAULT: _LOW_CPU_MEM_USAGE_DEFAULT.
  ModelMixin._load_pretrained_model: ModelMixin#_load_pretrained_model().
  ModelMixin.recursive_find_attn_block: ModelMixin#recursive_find_attn_block().
  ModelMixin.active_adapters: ModelMixin#active_adapters().
  ModelMixin._hf_peft_config_loaded: ModelMixin#_hf_peft_config_loaded.
  _load_state_dict_into_model.load: _load_state_dict_into_model().load().
  ModelMixin.set_use_memory_efficient_attention_xformers: ModelMixin#set_use_memory_efficient_attention_xformers().
  ModelMixin.fn_recursive_set_mem_eff: ModelMixin#fn_recursive_set_mem_eff().
  get_parameter_device: get_parameter_device().
  get_parameter_dtype: get_parameter_dtype().
  _load_state_dict_into_model: _load_state_dict_into_model().
  ModelMixin: ModelMixin#
  ModelMixin.config_name: ModelMixin#config_name.
  ModelMixin.__getattr__: ModelMixin#__getattr__().
  ModelMixin.enable_gradient_checkpointing: ModelMixin#enable_gradient_checkpointing().
  ModelMixin.disable_gradient_checkpointing: ModelMixin#disable_gradient_checkpointing().
  ModelMixin.enable_xformers_memory_efficient_attention: ModelMixin#enable_xformers_memory_efficient_attention().
  ModelMixin.disable_xformers_memory_efficient_attention: ModelMixin#disable_xformers_memory_efficient_attention().
  ModelMixin.device: ModelMixin#device().
  ModelMixin.dtype: ModelMixin#dtype().
  ModelMixin._convert_deprecated_attention_blocks: ModelMixin#_convert_deprecated_attention_blocks().
  ModelMixin._temp_convert_self_to_deprecated_attention_blocks: ModelMixin#_temp_convert_self_to_deprecated_attention_blocks().
  ModelMixin._undo_temp_convert_self_to_deprecated_attention_blocks: ModelMixin#_undo_temp_convert_self_to_deprecated_attention_blocks().
  ModelMixin._supports_gradient_checkpointing: ModelMixin#_supports_gradient_checkpointing.
  ModelMixin._keys_to_ignore_on_load_unexpected: ModelMixin#_keys_to_ignore_on_load_unexpected.
  get_parameter_device.find_tensor_attributes: get_parameter_device().find_tensor_attributes().
  get_parameter_dtype.find_tensor_attributes: get_parameter_dtype().find_tensor_attributes().
  load_model_dict_into_meta: load_model_dict_into_meta().
  ModelMixin._find_mismatched_keys: ModelMixin#_find_mismatched_keys().
  ModelMixin._automatically_saved_args: ModelMixin#_automatically_saved_args.
  ModelMixin.__init__: ModelMixin#__init__().
  ModelMixin.is_gradient_checkpointing: ModelMixin#is_gradient_checkpointing().
  ModelMixin.num_parameters: ModelMixin#num_parameters().
---
# Module: [`src/maxdiffusion/models/modeling_utils.py`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py)

## Classes
### `ModelMixin`  ·  implements/extends Module, PushToHubMixin
- def: [`src/maxdiffusion/models/modeling_utils.py:183`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L183)
- doc: Base class for all models.
- signature: `class ModelMixin(torch.nn.Module, PushToHubMixin):`
- members:
  - `__getattr__(self, name: str)` — [`L202`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L202) — The only reason we overwrite `getattr` here is to gracefully deprecate accessing
  - `active_adapters(self)` — [`L428`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L428) — Gets the current list of active adapters of the model.
  - `add_adapter(self, adapter_config, adapter_name: str = "default")` — [`L301`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L301) — Adds a new adapter to the current model for training. If no adapter name is passed, a default name is assigned
  - `device(self)` — [`L1003`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L1003) — `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
  - `disable_adapters(self)` — [`L383`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L383) — Disable all adapters attached to the model and fallback to inference with the base model only.
  - `disable_gradient_checkpointing(self)` — [`L236`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L236) — Deactivates gradient checkpointing for the current model (may be referred to as *activation checkpointing* or
  - `disable_xformers_memory_efficient_attention(self)` — [`L295`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L295) — Disable memory efficient attention from [xFormers](https://facebookresearch.github.io/xformers/).
  - `dtype(self)` — [`L1011`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L1011) — `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
  - `enable_adapters(self)` — [`L405`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L405) — Enable adapters that are attached to the model. The model will use `self.active_adapters()` to retrieve the
  - `enable_gradient_checkpointing(self)` — [`L227`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L227) — Activates gradient checkpointing for the current model (may be referred to as *activation checkpointing* or
  - `enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None)` — [`L259`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L259) — Enable memory efficient attention from [xFormers](https://facebookresearch.github.io/xformers/).
  - `fn_recursive_set_mem_eff(module: torch.nn.Module)` — [`L248`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L248)
  - `from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs)` — [`L528`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L528) — Instantiate a pretrained PyTorch model from a pretrained model configuration.
  - `is_gradient_checkpointing(self)` — [`L221`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L221) — Whether gradient checkpointing is activated for this model or not.
  - `num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False)` — [`L1017`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L1017) — Get number of (trainable or non-embedding) parameters in the module.
  - `recursive_find_attn_block(name, module)` — [`L1056`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L1056)
  - `save_pretrained(self, save_directory: Union[str, os.PathLike], is_main_process: bool = True, save_function: Callable = None, safe_serialization: bool = True, variant: Optional[str] = None, push_to_hub: bool = False, **kwargs)` — [`L446`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L446) — Save a model and its configuration file to a directory so that it can be reloaded using the
  - `set_adapter(self, adapter_name: Union[str, List[str]])` — [`L334`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L334) — Sets a specific adapter by forcing the model to only use that adapter and disables the other adapters.
  - `set_use_memory_efficient_attention_xformers(self, valid: bool, attention_op: Optional[Callable] = None)` — [`L244`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L244)
  - `config_name` — [`L193`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L193)
- protocol/private: `__init__`[`L199`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L199), `_automatically_saved_args`[`L194`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L194), `_convert_deprecated_attention_blocks`[`L1053`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L1053), `_find_mismatched_keys`[`L928`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L928), `_hf_peft_config_loaded`[`L197`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L197), `_keys_to_ignore_on_load_unexpected`[`L196`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L196), `_load_pretrained_model`[`L906`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L906), `_supports_gradient_checkpointing`[`L195`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L195), `_temp_convert_self_to_deprecated_attention_blocks`[`L1097`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L1097), `_undo_temp_convert_self_to_deprecated_attention_blocks`[`L1124`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L1124)
- uses (calls/refs, reference-scoped): [`__version__`](../__init__.md#__version__), [`_get_model_file`](../utils/hub_utils.md#_get_model_file), [`PushToHubMixin`](../utils/hub_utils.md#PushToHubMixin), [`WEIGHTS_NAME`](../utils/constants.md#WEIGHTS_NAME), [`DIFFUSERS_CACHE`](../utils/constants.md#DIFFUSERS_CACHE), [`logger`](modeling_utils.md#logger), [`FLAX_WEIGHTS_NAME`](../utils/constants.md#FLAX_WEIGHTS_NAME), [`deprecate`](../utils/deprecation_utils.md#deprecate), [`is_torch_version`](../utils/import_utils.md#is_torch_version), [`load_state_dict`](modeling_utils.md#load_state_dict), [`SAFETENSORS_WEIGHTS_NAME`](../utils/constants.md#SAFETENSORS_WEIGHTS_NAME), [`_LOW_CPU_MEM_USAGE_DEFAULT`](modeling_utils.md#_LOW_CPU_MEM_USAGE_DEFAULT), [`check_peft_version`](../utils/peft_utils.md#check_peft_version), [`_add_variant`](../utils/hub_utils.md#_add_variant), [`MIN_PEFT_VERSION`](../utils/constants.md#MIN_PEFT_VERSION), [`is_accelerate_available`](../utils/import_utils.md#is_accelerate_available), [`CONFIG_NAME`](../utils/constants.md#CONFIG_NAME), [`HF_HUB_OFFLINE`](../utils/hub_utils.md#HF_HUB_OFFLINE), [`_upload_folder`](../utils/hub_utils.md#PushToHubMixin._upload_folder), `repo_id`, [`_load_state_dict_into_model`](modeling_utils.md#_load_state_dict_into_model), [`get_parameter_device`](modeling_utils.md#get_parameter_device), [`get_parameter_dtype`](modeling_utils.md#get_parameter_dtype), [`load_model_dict_into_meta`](modeling_utils.md#load_model_dict_into_meta)
- used by: [`PushToHubMixin`](../utils/hub_utils.md#PushToHubMixin)

## Functions
- `_load_state_dict_into_model(model_to_load, state_dict)` — [`L162`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L162)
- `find_tensor_attributes(module: torch.nn.Module)` — [`L72`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L72)
- `find_tensor_attributes(module: torch.nn.Module)` — [`L94`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L94)
- `get_parameter_device(parameter: torch.nn.Module)` — [`L65`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L65)
- `get_parameter_dtype(parameter: torch.nn.Module)` — [`L81`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L81)
- `load(module: torch.nn.Module, prefix="")` — [`L170`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L170)
- `load_model_dict_into_meta(model, state_dict, device=None, dtype=None, model_name_or_path=None)` — [`L136`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L136)
- `load_state_dict(checkpoint_file: Union[str, os.PathLike], variant: Optional[str] = None)` — [`L103`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L103) — Reads a checkpoint file, returning properly formatted errors if they arise.

## Module values
- `_LOW_CPU_MEM_USAGE_DEFAULT` — [`L54`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L54)
- `logger` — [`L50`](../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/modeling_utils.py#L50)

