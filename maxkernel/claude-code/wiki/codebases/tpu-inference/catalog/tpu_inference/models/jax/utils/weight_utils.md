---
title: 'Module: tpu_inference/models/jax/utils/weight_utils.py'
type: catalog
provenance: extracted
module: tpu_inference/models/jax/utils/weight_utils.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.models.jax.utils.weight_utils`/
symbols:
  LoadableWithIterator: LoadableWithIterator#
  _load_and_shard_weight: _load_and_shard_weight().
  logger: logger.
  load_nnx_param_from_reshaped_torch: load_nnx_param_from_reshaped_torch().
  get_default_maps: get_default_maps().
  JaxAutoWeightsLoader._load_module: JaxAutoWeightsLoader#_load_module().
  shard_put: shard_put().
  BaseWeightLoader: BaseWeightLoader#
  load_hf_weights: load_hf_weights().
  LoadableWithIterator.load_weights: LoadableWithIterator#load_weights().
  JaxDummyModelLoader._process_weights_after_loading: JaxDummyModelLoader#_process_weights_after_loading().
  get_param: get_param().
  StandardWeightLoader: StandardWeightLoader#
  StandardWeightLoader.load_weights: StandardWeightLoader#load_weights().
  JaxDummyModelLoader.load_weights: JaxDummyModelLoader#load_weights().
  get_model_weights_files: get_model_weights_files().
  JaxAutoWeightsLoader.pooler_weights: JaxAutoWeightsLoader#pooler_weights.
  print_param_info: print_param_info().
  jax_array_from_reshaped_torch: jax_array_from_reshaped_torch().
  MetadataMap: MetadataMap#
  convert_torch_to_jax_with_view: convert_torch_to_jax_with_view().
  transfer_state_with_mappings: transfer_state_with_mappings().
  _load_hf_weights_on_thread: _load_hf_weights_on_thread().
  assign_and_shard_param: assign_and_shard_param().
  JaxAutoWeightsLoader._map_weights: JaxAutoWeightsLoader#_map_weights().
  model_weights_generator: model_weights_generator().
  _is_pp_missing_layer: _is_pp_missing_layer().
  JaxAutoWeightsLoader.load_weights: JaxAutoWeightsLoader#load_weights().
  BaseWeightLoader.get_weights_iterator: BaseWeightLoader#get_weights_iterator().
  JaxDummyModelLoader._load_dummy_weight_on_thread: JaxDummyModelLoader#_load_dummy_weight_on_thread().
  transpose_params: transpose_params().
  JaxAutoWeightsLoader: JaxAutoWeightsLoader#
  model_weights_single_file_generator: model_weights_single_file_generator().
  reshape_params: reshape_params().
  BaseWeightLoader.__init__: BaseWeightLoader#__init__().
  model_file_generator: model_file_generator().
  check_all_loaded: check_all_loaded().
  build_flat_dict: build_flat_dict().
  BaseWeightLoader.names_and_weights_generator: BaseWeightLoader#names_and_weights_generator.
  HF_WEIGHTS_FORMAT: HF_WEIGHTS_FORMAT.
  MetadataMap.reshape_map: MetadataMap#reshape_map.
  JaxAutoWeightsLoader.pytorch_pooler: JaxAutoWeightsLoader#pytorch_pooler.
  StandardWeightLoader.__init__: StandardWeightLoader#__init__().
  JaxAutoWeightsLoader.__init__: JaxAutoWeightsLoader#__init__().
  JaxAutoWeightsLoader._add_loadable_non_param_tensors: JaxAutoWeightsLoader#_add_loadable_non_param_tensors().
  MetadataMap.name_map: MetadataMap#name_map.
  MetadataMap.transpose_map: MetadataMap#transpose_map.
  MetadataMap.bias_reshape_map: MetadataMap#bias_reshape_map.
  MetadataMap.pad_map: MetadataMap#pad_map.
  MetadataMap.bias_pad_map: MetadataMap#bias_pad_map.
  StandardWeightLoader.vllm_config: StandardWeightLoader#vllm_config.
  StandardWeightLoader.mesh: StandardWeightLoader#mesh.
  JaxAutoWeightsLoader._process_weights_after_loading_per_module: JaxAutoWeightsLoader#_process_weights_after_loading_per_module.
  DTYPE_VIEW_MAP: DTYPE_VIEW_MAP.
  get_param_and_sharding: get_param_and_sharding().
  check_all_loaded._check: check_all_loaded()._check().
  BaseWeightLoader.vllm_config: BaseWeightLoader#vllm_config.
  JaxAutoWeightsLoader._packed_remap: JaxAutoWeightsLoader#_packed_remap().
  JaxAutoWeightsLoader._route: JaxAutoWeightsLoader#_route().
  JaxDummyModelLoader: JaxDummyModelLoader#
---
# Module: [`tpu_inference/models/jax/utils/weight_utils.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py)

## Classes
### `BaseWeightLoader`
- def: [`tpu_inference/models/jax/utils/weight_utils.py:682`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L682)
- signature: `class BaseWeightLoader:`
- members:
  - `get_weights_iterator(self)` — [`L692`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L692)
  - `names_and_weights_generator` — [`L686`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L686)
  - `vllm_config` — [`L685`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L685)
- protocol/private: `__init__`[`L684`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L684)
- uses (calls/refs, reference-scoped): [`StandardWeightLoader`](weight_utils.md#StandardWeightLoader), [`model_weights_generator`](weight_utils.md#model_weights_generator), [`EagleLlama3WeightLoader`](../llama_eagle3.md#EagleLlama3WeightLoader), [`Llama4WeightLoader`](../llama4.md#Llama4WeightLoader), [`LlamaGuard4WeightLoader`](../llama_guard_4.md#LlamaGuard4WeightLoader)
- used by: [`load_weights`](../llama4.md#Llama4WeightLoader.load_weights), [`pp_missing_layers`](../llama_guard_4.md#LlamaGuard4WeightLoader.pp_missing_layers), [`StandardWeightLoader`](weight_utils.md#StandardWeightLoader), [`resolve_model_architecture`](../../common/model_loader.md#resolve_model_architecture), [`EagleLlama3WeightLoader`](../llama_eagle3.md#EagleLlama3WeightLoader), [`Llama4WeightLoader`](../llama4.md#Llama4WeightLoader), [`LlamaGuard4WeightLoader`](../llama_guard_4.md#LlamaGuard4WeightLoader), [`__init__`](../llama4.md#Llama4WeightLoader.__init__), [`__init__`](../llama_eagle3.md#EagleLlama3WeightLoader.__init__), [`__init__`](../llama_guard_4.md#LlamaGuard4WeightLoader.__init__), [`__init__`](weight_utils.md#StandardWeightLoader.__init__)

### `JaxAutoWeightsLoader`
- def: [`tpu_inference/models/jax/utils/weight_utils.py:870`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L870)
- doc: A weights loader for JAX models.
- signature: `class JaxAutoWeightsLoader(AutoWeightsLoader):`
- members:
  - `_add_loadable_non_param_tensors(self, module: JaxModule, child_params: dict[str, Any])` — [`L987`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L987) — Add tensor names that are not in the model params that may be in the
  - `_load_module(self, base_prefix: str, module: JaxModule, weights: Iterable)` — [`L995`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L995) — Load weights into the JAX module, performing prefix adjustments and interception.
  - `_packed_remap(self)` — [`L927`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L927) — Derive (fused_name, shard_name, shard_id) tuples for merged linears.
  - `load_weights(self, weights: Iterable, **kwargs)` — [`L945`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L945) — Route packed (e.g. fused gate_up_proj) checkpoint weights, then
  - `pooler_weights` — [`L879`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L879)
  - `pytorch_pooler` — [`L878`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L878)
- protocol/private: `__init__`[`L873`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L873), `_map_weights`[`L1008`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L1008), `_process_weights_after_loading_per_module`[`L924`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L924), `_route`[`L966`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L966)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../../layers/__init__.md#JaxModule), [`QuantizeMethodBase`](../../../layers/jax/quantization/__init__.md#QuantizeMethodBase), [`logger`](weight_utils.md#logger), [`load_nnx_param_from_reshaped_torch`](weight_utils.md#load_nnx_param_from_reshaped_torch), [`named_children`](../../../layers/__init__.md#JaxModule.named_children), [`named_parameters`](../../../layers/__init__.md#JaxModule.named_parameters), [`process_weights_after_loading`](../../../layers/jax/quantization/__init__.md#QuantizeMethodBase.process_weights_after_loading), [`LAYOUT_Q_PROJ_AS_NDH`](../../../envs.md#LAYOUT_Q_PROJ_AS_NDH)
- used by: [`load_weights`](../deepseek_v3.md#DeepseekV3ForCausalLM.load_weights), [`load_weights`](weight_utils.md#LoadableWithIterator.load_weights), [`load_weights`](../gemma4_mtp.md#Gemma4MTPForCausalLM.load_weights)

### `JaxDummyModelLoader`
- def: [`tpu_inference/models/jax/utils/weight_utils.py:1081`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L1081)
- doc: A dummy weights loader for flax_nnx models.
- signature: `class JaxDummyModelLoader(DummyModelLoader):`
- members:
  - `_process_weights_after_loading(self, module: JaxModule | JaxModuleList)` — [`L1149`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L1149) — Recursively call process_weights_after_loading if any.
  - `load_weights(self, model: JaxModule, model_config: ModelConfig)` — [`L1088`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L1088)
- protocol/private: `_load_dummy_weight_on_thread`[`L1093`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L1093)
- uses (calls/refs, reference-scoped): [`JaxModule`](../../../layers/__init__.md#JaxModule), [`QuantizeMethodBase`](../../../layers/jax/quantization/__init__.md#QuantizeMethodBase), [`logger`](weight_utils.md#logger), [`JaxModuleList`](../../../layers/__init__.md#JaxModuleList), [`cpu_mesh_context`](../../../layers/common/utils.md#cpu_mesh_context), [`named_children`](../../../layers/__init__.md#JaxModule.named_children), [`named_parameters`](../../../layers/__init__.md#JaxModule.named_parameters), [`process_weights_after_loading`](../../../layers/jax/quantization/__init__.md#QuantizeMethodBase.process_weights_after_loading), [`assign_and_shard_param`](weight_utils.md#assign_and_shard_param)

### `LoadableWithIterator`
- def: [`tpu_inference/models/jax/utils/weight_utils.py:1058`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L1058)
- doc: Mixin for models that support loading weights with an iterator.
- signature: `class LoadableWithIterator:`
- members:
  - `load_weights(self, weights: Iterable[tuple[str, torch.Tensor]])` — [`L1064`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L1064)
- uses (calls/refs, reference-scoped): [`load_weights`](../deepseek_v3.md#DeepseekV3ForCausalLM.load_weights), [`load_weights`](../gemma4_mtp.md#Gemma4MTPForCausalLM.load_weights), [`Gemma4ForCausalLM`](../gemma4.md#Gemma4ForCausalLM), [`load_weights`](weight_utils.md#JaxAutoWeightsLoader.load_weights), [`load_weights`](../gemma4.md#Gemma4ForCausalLM.load_weights), [`DeepseekV3ForCausalLM`](../deepseek_v3.md#DeepseekV3ForCausalLM), [`Gemma4ForConditionalGeneration`](../gemma4_mm.md#Gemma4ForConditionalGeneration), [`Gemma4MTPForCausalLM`](../gemma4_mtp.md#Gemma4MTPForCausalLM), [`Qwen2ForCausalLM`](../qwen2.md#Qwen2ForCausalLM), [`Qwen3ForCausalLM`](../qwen3.md#Qwen3ForCausalLM), [`Qwen3MoeForCausalLM`](../qwen3_moe.md#Qwen3MoeForCausalLM), [`JaxAutoWeightsLoader`](weight_utils.md#JaxAutoWeightsLoader), [`load_weights`](../gemma4_mm.md#Gemma4ForConditionalGeneration.load_weights)
- used by: [`_get_nnx_model`](../../common/model_loader.md#_get_nnx_model), [`load_weights`](../deepseek_v3.md#DeepseekV3ForCausalLM.load_weights), [`Gemma4ForCausalLM`](../gemma4.md#Gemma4ForCausalLM), [`load_weights`](../gemma4.md#Gemma4ForCausalLM.load_weights), [`DeepseekV3ForCausalLM`](../deepseek_v3.md#DeepseekV3ForCausalLM), [`Gemma4ForConditionalGeneration`](../gemma4_mm.md#Gemma4ForConditionalGeneration), [`Gemma4MTPForCausalLM`](../gemma4_mtp.md#Gemma4MTPForCausalLM), [`Qwen2ForCausalLM`](../qwen2.md#Qwen2ForCausalLM), [`Qwen3ForCausalLM`](../qwen3.md#Qwen3ForCausalLM), [`Qwen3MoeForCausalLM`](../qwen3_moe.md#Qwen3MoeForCausalLM), [`load_weights`](../gemma4_mm.md#Gemma4ForConditionalGeneration.load_weights)

### `MetadataMap`
- def: [`tpu_inference/models/jax/utils/weight_utils.py:64`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L64)
- signature: `class MetadataMap:`
- members:
  - `bias_pad_map` — [`L70`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L70)
  - `bias_reshape_map` — [`L68`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L68)
  - `name_map` — [`L65`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L65)
  - `pad_map` — [`L69`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L69)
  - `reshape_map` — [`L67`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L67)
  - `transpose_map` — [`L66`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L66)
- used by: [`_load_and_shard_weight`](weight_utils.md#_load_and_shard_weight), [`get_default_maps`](weight_utils.md#get_default_maps), [`load_hf_weights`](weight_utils.md#load_hf_weights), [`load_weights`](weight_utils.md#StandardWeightLoader.load_weights), [`_load_hf_weights_on_thread`](weight_utils.md#_load_hf_weights_on_thread), [`update_reshape_map_for_eagle3`](../llama_eagle3.md#update_reshape_map_for_eagle3)

### `StandardWeightLoader`  ·  implements/extends BaseWeightLoader
- def: [`tpu_inference/models/jax/utils/weight_utils.py:701`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L701)
- signature: `class StandardWeightLoader(BaseWeightLoader):`
- members:
  - `load_weights(self, model: nnx.Module, mappings: dict | MetadataMap, keep_hf_weight_suffix_when_match: list[str] = [])` — [`L708`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L708) — Calls the generic load_hf_weights utility, passing the correct
  - `mesh` — [`L706`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L706)
  - `vllm_config` — [`L705`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L705)
- protocol/private: `__init__`[`L703`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L703)
- uses (calls/refs, reference-scoped): [`get_default_maps`](weight_utils.md#get_default_maps), [`BaseWeightLoader`](weight_utils.md#BaseWeightLoader), [`load_hf_weights`](weight_utils.md#load_hf_weights), [`MetadataMap`](weight_utils.md#MetadataMap), [`__init__`](weight_utils.md#BaseWeightLoader.__init__)
- used by: [`BaseWeightLoader`](weight_utils.md#BaseWeightLoader), [`pp_missing_layers`](../qwen2_5_vl.md#Qwen2_5_VLForConditionalGeneration.pp_missing_layers), [`load_weights`](../llama3.md#LlamaForCausalLM.load_weights), [`WeightLoader`](../llama3.md#LlamaForCausalLM.WeightLoader), [`WeightLoader`](../qwen2_5_vl.md#Qwen2_5_VLForConditionalGeneration.WeightLoader), [`WeightLoader`](../gemma4_mm.md#Gemma4ForConditionalGeneration.WeightLoader), [`WeightLoader`](../gemma4_mtp.md#Gemma4MTPForCausalLM.WeightLoader), [`WeightLoader`](../gemma4.md#Gemma4ForCausalLM.WeightLoader), [`WeightLoader`](../qwen2.md#Qwen2ForCausalLM.WeightLoader)

## Functions
- `_check(x: Any)` — [`L597`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L597)
- `_is_pp_missing_layer(hf_key: str, pp_missing_layers: list[str])` — [`L464`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L464)
- `_load_and_shard_weight(vllm_config, params: nnx.State, shardings: Any, metadata_map: MetadataMap, mesh: Mesh, hf_key: str, hf_weight: jax.Array, keep_hf_weight_suffix_when_match: list[str], keep_original_dtype_keys_regex: list[str] | None = None, pp_missing_layers: list[str] | None = None)` — [`L322`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L322)
- `_load_hf_weights_on_thread(vllm_config: VllmConfig, params: nnx.State, metadata_map: MetadataMap, mesh: Mesh, weights_file: str, keep_hf_weight_suffix_when_match: list[str], filter_regex: Optional[str] = None, keep_original_dtype_keys_regex: Optional[list[str]] = None, pp_missing_layers: list[str] | None = None)` — [`L472`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L472) — Loads weights from a single weights file.
- `assign_and_shard_param(jax_param: nnx.Param, jax_weight: jax.Array, param_name: str = "Unknown", mesh: Optional[Mesh] = None)` — [`L769`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L769) — Distributes a JAX array across devices according to the `nnx.Param`'s sharding metadata, assigns it to the parameter, and marks it as loaded.
- `build_flat_dict(flat_state, mappings)` — [`L605`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L605) — Build a new flat dictionary from the flat state using the provided mappings.
- `check_all_loaded(params: nnx.State)` — [`L595`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L595)
- `convert_torch_to_jax_with_view(loaded_weight: torch.Tensor, cast_type: jnp.dtype)` — [`L125`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L125) — Converts a PyTorch tensor to a JAX array by reinterpreting its
- `get_default_maps(model_config, mesh: Mesh, name_map: dict[str, str])` — [`L252`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L252) — Load weights from one model weights file to the model, run on single thread.
- `get_model_weights_files(model_name_or_path: str, download_dir: Optional[str])` — [`L141`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L141) — Helper to get weight files and their location.
- `get_param(params: nnx.State, path: str)` — [`L190`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L190)
- `get_param_and_sharding(params: nnx.State, shardings: Any, path: str)` — [`L204`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L204)
- `jax_array_from_reshaped_torch(torch_weight: torch.Tensor, *, reshape_dims: Optional[tuple[int, ...]] = None, permute_dims: Optional[tuple[int, ...]] = None)` — [`L742`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L742) — Convert a torch.Tensor to a jax.Array with reshaping and transposing.
- `load_hf_weights(vllm_config: VllmConfig, model: nnx.Module, metadata_map: MetadataMap, mesh: Mesh, filter_regex: Optional[str] = None, is_draft_model: bool = False, keep_original_dtype_keys_regex: Optional[list[str]] = None, pp_missing_layers: list[str] | None = None, keep_hf_weight_suffix_when_match: list[str] = [])` — [`L505`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L505) — Load weights into a JAX model from either an iterator or files.
- `load_nnx_param_from_reshaped_torch(jax_param: nnx.Param, torch_weight: torch.Tensor, *, reshape_dims: Optional[tuple[int, ...]] = None, permute_dims: Optional[tuple[int, ...]] = None, param_name: str = "Unknown")` — [`L798`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L798) — Load a nnx.Param from a torch.Tensor with reshaping and transposing.
- `model_file_generator(model_name_or_path: str, download_dir: Optional[str])` — [`L105`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L105)
- `model_weights_generator(model_name_or_path: str, framework: str, filter_regex: Optional[str] = None, download_dir: Optional[str] = None)` — [`L113`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L113)
- `model_weights_single_file_generator(weights_file: str, framework: str, filter_regex: Optional[str] = None)` — [`L170`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L170)
- `print_param_info(param: nnx.Param, name: str)` — [`L76`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L76)
- `reshape_params(param_key: str, param_tensor: jax.Array, shape_map)` — [`L92`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L92)
- `shard_put(x: jax.Array, shardings, mesh: jax.sharding.Mesh | None = None)` — [`L222`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L222)
- `transfer_state_with_mappings(src_state, tgt_state, mappings, transpose_keys=None, shard=None)` — [`L634`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L634) — Transfer state from src_state to tgt_state using the provided mappings.
- `transpose_params(param_key: str, param_tensor: jax.Array, transpose_map)` — [`L85`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L85)

## Module values
- `DTYPE_VIEW_MAP` — [`L56`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L56)
- `HF_WEIGHTS_FORMAT` — [`L54`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L54)
- `logger` — [`L52`](../../../../../../../../raw/code/tpu-inference/tpu_inference/models/jax/utils/weight_utils.py#L52)

