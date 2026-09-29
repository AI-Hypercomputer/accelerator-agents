---
title: 'Module: easydel/utils/parameters_transformation.py'
type: catalog
provenance: extracted
module: easydel/utils/parameters_transformation.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.utils.parameters_transformation`/
symbols:
  StateDictConverter.easydel_to_torch: StateDictConverter#easydel_to_torch().
  StateDictConverter: StateDictConverter#
  StateDictConverter.process_tensor: StateDictConverter#process_tensor().
  TensorConverter.jax_to_pytorch: TensorConverter#jax_to_pytorch().
  StateDictConverter._base_huggingface_to_easydel: StateDictConverter#_base_huggingface_to_easydel().
  ModelConverter.easydel_to_huggingface: ModelConverter#easydel_to_huggingface().
  TensorConverter: TensorConverter#
  StateDictConverter.huggingface_to_easydel: StateDictConverter#huggingface_to_easydel().
  DtypeHandler.float_tensor_to_dtype: DtypeHandler#float_tensor_to_dtype().
  ModelConverter: ModelConverter#
  StateDictConverter.apply_moe_transformations: StateDictConverter#apply_moe_transformations().
  EASYDEL_PREFERRED_HOST_COPY.EASYDEL_PREFERRED_HOST_COPY: EASYDEL_PREFERRED_HOST_COPY.EASYDEL_PREFERRED_HOST_COPY.
  logger: logger.
  DtypeHandler: DtypeHandler#
  DtypeHandler.get_dtype: DtypeHandler#get_dtype().
  TensorConverter.convert_pytorch_to_jnp: TensorConverter#convert_pytorch_to_jnp().
  mem_ops: mem_ops.
  _preferred_host_copy_raw: _preferred_host_copy_raw.
  TensorConverter.get_torch: TensorConverter#get_torch().
  StateDictConverter.get_memory_bytes: StateDictConverter#get_memory_bytes().
  EASYDEL_PREFERRED_HOST_COPY_INDEX: EASYDEL_PREFERRED_HOST_COPY_INDEX.
  StateDictConverter.apply_moe_transformations_reverse: StateDictConverter#apply_moe_transformations_reverse().
  TensorConverter.pytorch_to_jax: TensorConverter#pytorch_to_jax().
  StateDictConverter.match_keywords: StateDictConverter#match_keywords().
---
# Module: [`easydel/utils/parameters_transformation.py`](../../../../../../raw/code/EasyDeL/easydel/utils/parameters_transformation.py)

## Classes
### `DtypeHandler`
- def: [`easydel/utils/parameters_transformation.py:59`](../../../../../../raw/code/EasyDeL/easydel/utils/parameters_transformation.py#L59)
- doc: Handles dtype conversions and operations.
- signature: `class DtypeHandler:`
- members:
  - `float_tensor_to_dtype(tensor: tp.Any, dtype: str | jnp.dtype | None)` — [`L94`](../../../../../../raw/code/EasyDeL/easydel/utils/parameters_transformation.py#L94) — Convert float tensor to specified dtype.
  - `get_dtype(dtype: str | jnp.dtype)` — [`L63`](../../../../../../raw/code/EasyDeL/easydel/utils/parameters_transformation.py#L63) — Convert string dtype representation to JAX dtype.
- used by: [`easydel_to_torch`](parameters_transformation.md#StateDictConverter.easydel_to_torch)

### `ModelConverter`
- def: [`easydel/utils/parameters_transformation.py:833`](../../../../../../raw/code/EasyDeL/easydel/utils/parameters_transformation.py#L833)
- doc: Handles model conversions between EasyDeL and HuggingFace formats.
- signature: `class ModelConverter:`
- members:
  - `easydel_to_huggingface(module: EasyDeLBaseModule, config: EasyDeLBaseConfig, base_huggingface_module: PreTrainedModel, base_huggingface_module_kwarguments: dict | None = None, dtype: jnp.dtype = jnp.float16, use_meta_torch: bool = True, reform_param: dict | None = None, **kw)` — [`L837`](../../../../../../raw/code/EasyDeL/easydel/utils/parameters_transformation.py#L837) — Convert an EasyDeL module to a HuggingFace ``PreTrainedModel``.
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](../infra/base_module.md#EasyDeLBaseModule), [`EasyDeLBaseConfig`](../infra/base_config.md#EasyDeLBaseConfig), [`easydel_to_torch`](parameters_transformation.md#StateDictConverter.easydel_to_torch), [`to_dict`](../infra/base_config.md#EasyDeLBaseConfig.to_dict), [`StateDictConverter`](parameters_transformation.md#StateDictConverter)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`to_torch`](../infra/base_module.md#EasyDeLBaseModule.to_torch)

### `StateDictConverter`
- def: [`easydel/utils/parameters_transformation.py:178`](../../../../../../raw/code/EasyDeL/easydel/utils/parameters_transformation.py#L178)
- doc: Handles conversion between PyTorch and EasyDeL state dictionaries.
- signature: `class StateDictConverter:`
- members:
  - `_base_huggingface_to_easydel(state_dict: dict[str, tp.Any], *, device: jax.Device | None = None, embedding_layer_names: list[str] | None = None, layernorm_names: list[str] | None = None, moe_block_names: list[str] | None = None, moe_names: list[str] | None = None, shard_fns: Mapping[tuple, tp.Callable] | None = None, dtype: jnp.dtype = jnp.float16, verbose: bool = True, callback: tp.Callable[[jax.Array, tuple], jax.Array] | None = None, remove_state_dict: bool = False, lm_head_name: str | None = None, uses_tie_word_embedding: bool = False, consolidated_moe_keys: set[str] | None = None, reform_param: dict | None = None, **kwargs)` — [`L282`](../../../../../../raw/code/EasyDeL/easydel/utils/parameters_transformation.py#L282) — Base conversion from a PyTorch state dict to EasyDeL nested dict format.
  - `apply_moe_transformations(state_dict: dict[str, tp.Any], moe_block_names: list[str] | None = None, moe_names: list[str] | None = None, moe_block_path: list[str] | None = None, moe_path: list[str] | None = None, tensor_transform: tp.Callable | None = None, reform_param: dict | None = None)` — [`L386`](../../../../../../raw/code/EasyDeL/easydel/utils/parameters_transformation.py#L386) — Transform MoE weights from HuggingFace format (separate experts) to EasyDel format (stacked experts).
  - `apply_moe_transformations_reverse(state_dict: dict[str, tp.Any], moe_block_names: list[str] | None = None, moe_names: list[str] | None = None, moe_block_path: list[str] | None = None, moe_path: list[str] | None = None, tensor_transform: tp.Callable | None = None)` — [`L603`](../../../../../../raw/code/EasyDeL/easydel/utils/parameters_transformation.py#L603) — Transform MoE weights from EasyDel format (stacked experts) to HuggingFace format (separate experts).
  - `easydel_to_torch(module: EasyDeLBaseModule, dtype: jnp.dtype | None = jnp.float16, **kwargs)` — [`L683`](../../../../../../raw/code/EasyDeL/easydel/utils/parameters_transformation.py#L683) — Convert an EasyDeL module's parameters to a PyTorch state dict.
  - `get_memory_bytes(device_idx)` — [`L355`](../../../../../../raw/code/EasyDeL/easydel/utils/parameters_transformation.py#L355)
  - `huggingface_to_easydel(state_dict: dict[str, tp.Any], *, device: jax.Device | None = None, embedding_layer_names: list[str] | None = None, layernorm_names: list[str] | None = None, moe_block_names: list[str] | None = None, moe_names: list[str] | None = None, moe_block_path: list[str] | None = None, moe_path: list[str] | None = None, shard_fns: Mapping[tuple, tp.Callable] | None = None, dtype: jnp.dtype = jnp.float16, verbose: bool = True, callback: tp.Callable[[jax.Array, tuple], jax.Array] | None = None, remove_state_dict: bool = False, lm_head_name: str | None = None, uses_tie_word_embedding: bool = False, reform_param: dict | None = None, **kwargs)` — [`L523`](../../../../../../raw/code/EasyDeL/easydel/utils/parameters_transformation.py#L523) — Convert a PyTorch state dict to EasyDeL format with MoE support.
  - `match_keywords(string: str, required: list[str], forbidden: list[str])` — [`L182`](../../../../../../raw/code/EasyDeL/easydel/utils/parameters_transformation.py#L182) — Check if string contains all required keywords and none of the forbidden ones.
  - `process_tensor(key: str, tensor: tp.Any, config: dict[str, tp.Any])` — [`L187`](../../../../../../raw/code/EasyDeL/easydel/utils/parameters_transformation.py#L187) — Process a single PyTorch tensor into EasyDeL format.
- uses (calls/refs, reference-scoped): [`EasyDeLBaseModule`](../infra/base_module.md#EasyDeLBaseModule), [`BaseMoeModule`](../layers/moe/_moe_module.md#BaseMoeModule), [`iter_module_search`](traversals.md#iter_module_search), [`flatten_dict`](traversals.md#flatten_dict), [`unflatten_dict`](traversals.md#unflatten_dict), [`param_dtype`](../infra/base_module.md#EasyDeLBaseModule.param_dtype), [`jax_to_pytorch`](parameters_transformation.md#TensorConverter.jax_to_pytorch), [`ParallelMoELinear`](../layers/linears/_linear_moe.md#ParallelMoELinear), [`TensorConverter`](parameters_transformation.md#TensorConverter), [`parameters`](../infra/base_module.md#EasyDeLBaseModule.parameters), [`DtypeHandler`](parameters_transformation.md#DtypeHandler), [`convert_pytorch_to_jnp`](parameters_transformation.md#TensorConverter.convert_pytorch_to_jnp), [`get_dtype`](parameters_transformation.md#DtypeHandler.get_dtype), [`logger`](parameters_transformation.md#logger), [`get_torch`](parameters_transformation.md#TensorConverter.get_torch)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`huggingface_to_easydel_sequential`](../infra/mixins/bridge.md#EasyBridgeMixin.huggingface_to_easydel_sequential), [`_convert_streaming_checkpoint_to_params`](../infra/mixins/bridge.md#EasyBridgeMixin._convert_streaming_checkpoint_to_params), [`_build_transform_fn`](../infra/base_module.md#EasyDeLBaseModule._build_transform_fn), [`easydel_to_huggingface`](parameters_transformation.md#ModelConverter.easydel_to_huggingface), [`_process_tensor`](../infra/mixins/bridge.md#EasyBridgeMixin._process_tensor), [`_process_and_write`](../infra/mixins/bridge.md#EasyBridgeMixin._process_and_write)

### `TensorConverter`
- def: [`easydel/utils/parameters_transformation.py:117`](../../../../../../raw/code/EasyDeL/easydel/utils/parameters_transformation.py#L117)
- doc: Handles tensor conversions between PyTorch and JAX.
- signature: `class TensorConverter:`
- members:
  - `convert_pytorch_to_jnp(tensor: tp.Any, dtype: jnp.dtype)` — [`L121`](../../../../../../raw/code/EasyDeL/easydel/utils/parameters_transformation.py#L121) — Convert PyTorch tensor to JAX array.
  - `get_torch()` — [`L130`](../../../../../../raw/code/EasyDeL/easydel/utils/parameters_transformation.py#L130) — Import and return torch module (cached).
  - `jax_to_pytorch(x: jax.Array)` — [`L137`](../../../../../../raw/code/EasyDeL/easydel/utils/parameters_transformation.py#L137) — Convert JAX array to PyTorch tensor.
  - `pytorch_to_jax(x: tp.Any)` — [`L173`](../../../../../../raw/code/EasyDeL/easydel/utils/parameters_transformation.py#L173) — Convert PyTorch tensor to JAX array.
- uses (calls/refs, reference-scoped): [`check_bool_flag`](helpers.md#check_bool_flag), [`EASYDEL_PREFERRED_HOST_COPY`](parameters_transformation.md#EASYDEL_PREFERRED_HOST_COPY.EASYDEL_PREFERRED_HOST_COPY), [`EASYDEL_PREFERRED_HOST_COPY_INDEX`](parameters_transformation.md#EASYDEL_PREFERRED_HOST_COPY_INDEX)
- used by: [`_import_structure`](../__init__.md#_import_structure), [`huggingface_to_easydel_sequential`](../infra/mixins/bridge.md#EasyBridgeMixin.huggingface_to_easydel_sequential), [`easydel_to_torch`](parameters_transformation.md#StateDictConverter.easydel_to_torch), [`process_tensor`](parameters_transformation.md#StateDictConverter.process_tensor)

## Module values
- `EASYDEL_PREFERRED_HOST_COPY` — [`L56`](../../../../../../raw/code/EasyDeL/easydel/utils/parameters_transformation.py#L56)
- `EASYDEL_PREFERRED_HOST_COPY_INDEX` — [`L50`](../../../../../../raw/code/EasyDeL/easydel/utils/parameters_transformation.py#L50)
- `_preferred_host_copy_raw` — [`L53`](../../../../../../raw/code/EasyDeL/easydel/utils/parameters_transformation.py#L53)
- `logger` — [`L49`](../../../../../../raw/code/EasyDeL/easydel/utils/parameters_transformation.py#L49)
- `mem_ops` — [`L48`](../../../../../../raw/code/EasyDeL/easydel/utils/parameters_transformation.py#L48)

