---
title: 'Module: easydel/infra/mixins/bridge.py'
type: catalog
provenance: extracted
module: easydel/infra/mixins/bridge.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.infra.mixins.bridge`/
symbols:
  EasyBridgeMixin.huggingface_to_easydel_sequential: EasyBridgeMixin#huggingface_to_easydel_sequential().
  EasyBridgeMixin._from_torch_pretrained: EasyBridgeMixin#_from_torch_pretrained().
  EasyBridgeMixin._load_model_weights: EasyBridgeMixin#_load_model_weights().
  EasyBridgeMixin.from_pretrained: EasyBridgeMixin#from_pretrained().
  EasyBridgeMixin._resolve_streaming_checkpoint: EasyBridgeMixin#_resolve_streaming_checkpoint().
  logger: logger.
  EasyBridgeMixin.get_torch_loader: EasyBridgeMixin#get_torch_loader().
  EasyBridgeMixin._model_card: EasyBridgeMixin#_model_card().
  EasyBridgeMixin._convert_streaming_checkpoint_to_params: EasyBridgeMixin#_convert_streaming_checkpoint_to_params().
  EasyBridgeMixin._save_model_files: EasyBridgeMixin#_save_model_files().
  EasyBridgeMixin._model_task: EasyBridgeMixin#_model_task.
  _build_safe_checkpoint_partition_rules: _build_safe_checkpoint_partition_rules().
  CANDIDATE_FILENAMES: CANDIDATE_FILENAMES.
  _rebuild_lora_modules_from_checkpoint: _rebuild_lora_modules_from_checkpoint().
  EasyBridgeMixin.save_pretrained: EasyBridgeMixin#save_pretrained().
  _parse_torch_load_options: _parse_torch_load_options().
  EasyBridgeMixin._hf_download: EasyBridgeMixin#_hf_download().
  EasyBridgeMixin._find_local: EasyBridgeMixin#_find_local().
  EasyBridgeMixin._process_tensor: EasyBridgeMixin#_process_tensor().
  EasyBridgeMixin._ensure_moe_group_ready: EasyBridgeMixin#_ensure_moe_group_ready().
  EasyBridgeMixin._process_and_write: EasyBridgeMixin#_process_and_write().
  SAFE_WEIGHTS_NAME: SAFE_WEIGHTS_NAME.
  EasyBridgeMixin._with_resolved_shard: EasyBridgeMixin#_with_resolved_shard().
  _normalize_quantization_config: _normalize_quantization_config().
  EasyBridgeMixin._save_configs: EasyBridgeMixin#_save_configs().
  WEIGHTS_NAME: WEIGHTS_NAME.
  SAFE_WEIGHTS_INDEX_NAME: SAFE_WEIGHTS_INDEX_NAME.
  _load_generation_config: _load_generation_config().
  EasyBridgeMixin.push_to_hub: EasyBridgeMixin#push_to_hub().
  _save_generation_config: _save_generation_config().
  EasyBridgeMixin._upload_modified_files: EasyBridgeMixin#_upload_modified_files().
  EasyBridgeMixin._run_streaming_conversion: EasyBridgeMixin#_run_streaming_conversion().
  WEIGHTS_INDEX_NAME: WEIGHTS_INDEX_NAME.
  TENSORSTORE_INDEX_NAME: TENSORSTORE_INDEX_NAME.
  EasyBridgeMixin._resolve_shard: EasyBridgeMixin#_resolve_shard().
  EasyBridgeMixin._write_tensor: EasyBridgeMixin#_write_tensor().
  EasyBridgeMixin: EasyBridgeMixin#
  EasyBridgeMixin._pick_first_existing: EasyBridgeMixin#_pick_first_existing().
  EasyBridgeMixin._register_moe_key: EasyBridgeMixin#_register_moe_key().
  EasyBridgeMixin.config: EasyBridgeMixin#config.
  EasyBridgeMixin._convert: EasyBridgeMixin#_convert().
  EasyBridgeMixin._chunk_shape_for: EasyBridgeMixin#_chunk_shape_for().
  TorchLoadOptions.hub_kwargs: TorchLoadOptions#hub_kwargs.
  StreamingCheckpointInfo: StreamingCheckpointInfo#
  EasyBridgeMixin._clear: EasyBridgeMixin#_clear().
  EasyBridgeMixin._strip_or_keep_subfolder: EasyBridgeMixin#_strip_or_keep_subfolder().
  _sanitize_partition_spec_for_shape: _sanitize_partition_spec_for_shape().
  EasyBridgeMixin._resolve_upload_num_threads: EasyBridgeMixin#_resolve_upload_num_threads().
  EasyBridgeMixin.callback: EasyBridgeMixin#callback().
  EasyBridgeMixin._load_full_torch_checkpoint: EasyBridgeMixin#_load_full_torch_checkpoint().
  EasyBridgeMixin._write_checkpoint_index: EasyBridgeMixin#_write_checkpoint_index().
  GENERATION_CONFIG_NAME: GENERATION_CONFIG_NAME.
  TorchLoadOptions: TorchLoadOptions#
  TorchLoadOptions.mode: TorchLoadOptions#mode.
  IMAGE_PROCESSOR_NAME: IMAGE_PROCESSOR_NAME.
  EasyBridgeMixin.config_class: EasyBridgeMixin#config_class.
  _checkpoint_cpu_device: _checkpoint_cpu_device().
  _normalize_checkpoint_leaf_to_jax: _normalize_checkpoint_leaf_to_jax().
  TorchLoadOptions.streaming_cache: TorchLoadOptions#streaming_cache.
  TorchLoadOptions.streaming_tmp_dir: TorchLoadOptions#streaming_tmp_dir.
  StreamingCheckpointInfo.ckpt_weight_format: StreamingCheckpointInfo#ckpt_weight_format.
  StreamingCheckpointInfo.ckpt_key_to_filename: StreamingCheckpointInfo#ckpt_key_to_filename.
  StreamingCheckpointInfo.ckpt_filename_to_path: StreamingCheckpointInfo#ckpt_filename_to_path.
  StreamingCheckpointInfo.ed_config: StreamingCheckpointInfo#ed_config.
  StreamingCheckpointInfo.generation_config: StreamingCheckpointInfo#generation_config.
  StreamingCheckpointInfo.pretrained_model_name_or_path: StreamingCheckpointInfo#pretrained_model_name_or_path.
  StreamingCheckpointInfo.resolve_shard: StreamingCheckpointInfo#resolve_shard.
  EasyBridgeMixin.can_generate: EasyBridgeMixin#can_generate().
  EasyBridgeMixin._load_index: EasyBridgeMixin#_load_index().
  EasyBridgeMixin._tensorstore_path_for_params: EasyBridgeMixin#_tensorstore_path_for_params().
  EasyBridgeMixin._finalize_moe_group: EasyBridgeMixin#_finalize_moe_group().
  FLAX_WEIGHTS_NAME: FLAX_WEIGHTS_NAME.
  ED_SAFE_WEIGHTS_INDEX_NAME: ED_SAFE_WEIGHTS_INDEX_NAME.
  FEATURE_EXTRACTOR_NAME: FEATURE_EXTRACTOR_NAME.
  _mesh_partition_product: _mesh_partition_product().
  _tree_key_to_path: _tree_key_to_path().
  _path_match_variants: _path_match_variants().
  _collect_lora_checkpoint_paths: _collect_lora_checkpoint_paths().
  _strip_trailing_separators: _strip_trailing_separators().
  EasyBridgeMixin._model_type: EasyBridgeMixin#_model_type.
  EasyBridgeMixin._apply_shard_fn: EasyBridgeMixin#_apply_shard_fn().
  EasyBridgeMixin._maybe_shard_and_callback: EasyBridgeMixin#_maybe_shard_and_callback().
  EasyBridgeMixin._iter_keys_single_file: EasyBridgeMixin#_iter_keys_single_file().
  TF2_WEIGHTS_NAME: TF2_WEIGHTS_NAME.
  TF2_WEIGHTS_INDEX_NAME: TF2_WEIGHTS_INDEX_NAME.
  TF_WEIGHTS_NAME: TF_WEIGHTS_NAME.
  FLAX_WEIGHTS_INDEX_NAME: FLAX_WEIGHTS_INDEX_NAME.
  CONFIG_NAME: CONFIG_NAME.
  PROCESSOR_NAME: PROCESSOR_NAME.
  CHAT_TEMPLATE_NAME: CHAT_TEMPLATE_NAME.
  MODEL_CARD_NAME: MODEL_CARD_NAME.
  EasyBridgeMixin.hf_torch_auto_loader: EasyBridgeMixin#hf_torch_auto_loader.
  EasyBridgeMixin.base_model_prefix: EasyBridgeMixin#base_model_prefix.
---
# Module: [`easydel/infra/mixins/bridge.py`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py)

## Classes
### `EasyBridgeMixin`  ·  implements/extends PushToHubMixin
- def: [`easydel/infra/mixins/bridge.py:557`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L557)
- doc: Mixin class for adding bridging functionalities like saving, loading, and pushing models to Hugging Face Hub.
- signature: `class EasyBridgeMixin(PushToHubMixin):`
- members:
  - `_convert_streaming_checkpoint_to_params(cls, model, ckpt_info: StreamingCheckpointInfo, hf_config: tp.Any, load_options: TorchLoadOptions, shard_fns: dict[tuple, tp.Callable] | None, callback: tp.Callable | None, device: tp.Any | None, clear_fn: tp.Callable[[], None])` — [`L1900`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L1900) — Convert streaming PyTorch checkpoint to flattened JAX params dict.
  - `_from_torch_pretrained(cls, pretrained_model_name_or_path: str, device: jax.Device | None = None, dtype: jax.numpy.dtype = jax.numpy.float32, param_dtype: jax.numpy.dtype = jax.numpy.float32, precision: jax.lax.Precision | None = None, sharding_axis_dims: collections.abc.Sequence[int] = (1, -1, 1, 1, 1), sharding_dcn_axis_dims: collections.abc.Sequence[int] | None = None, sharding_axis_names: collections.abc.Sequence[str] = ("dp", "fsdp", "ep", "tp", "sp"), partition_axis: PartitionAxis | None = None, shard_fns: collections.abc.Mapping[tuple, tp.Callable] | dict | None = None, backend: EasyDeLBackends | None = None, platform: EasyDeLPlatforms | None = None, config_kwargs: EasyDeLBaseConfigDict | None = None, auto_shard_model: bool = True, partition_rules: tuple[tuple[str, PartitionSpec], ...] | None = None, quantization_config: QuantizationConfig | None = None, apply_quantization: bool = False, verbose: bool = True, **kwargs)` — [`L1500`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L1500) — Loads an EasyDeL model from a PyTorch pretrained checkpoint.
  - `_load_full_torch_checkpoint(cls, pretrained_model_name_or_path: str, config_class: type, hub_kwargs: dict[str, tp.Any], torch_loader: tp.Any, clear_fn: tp.Callable[[], None], kwargs: dict[str, tp.Any])` — [`L2223`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L2223) — Load full PyTorch checkpoint into memory (for torch_load_mode='full').
  - `_load_model_weights(cls, resolved_archive_file: str | None, model: EasyDeLBaseModule, param_dtype: jnp.dtype, mesh: jax.sharding.Mesh, shard_fns: dict[str, tp.Callable] | None, quantization_config: QuantizationConfig | None, apply_quantization: bool, verbose: bool)` — [`L923`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L923) — Loads model weights from a checkpoint file.
  - `_model_card(self, name: str, repo_id: str)` — [`L569`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L569) — Generates a model card (README.md) for the given model.
  - `_resolve_shard(fname: str, cache_dir_override: str | None = None)` — [`L1802`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L1802) — Unified shard resolution: local first, then HF download.
  - `_resolve_streaming_checkpoint(cls, pretrained_model_name_or_path: str, config_class: type, hub_kwargs: dict[str, tp.Any], kwargs: dict[str, tp.Any])` — [`L1726`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L1726) — Resolve checkpoint files for streaming mode, returning checkpoint info.
  - `_resolve_upload_num_threads(upload_num_threads: int | None, operation_count: int)` — [`L907`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L907) — Resolve Hub upload concurrency with a conservative automatic default.
  - `_save_model_files(self, save_directory: ePathLike, float_dtype=None, *, step: int | None = None)` — [`L600`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L600) — Saves the model's configuration, weights, and potentially the generation config to the specified directory.
  - `_upload_modified_files(self, working_dir: str | os.PathLike, repo_id: str, files_timestamps: dict[str, float], commit_message: str | None = None, token: bool | str | None = None, create_pr: bool = False, revision: str | None = None, commit_description: str | None = None, upload_num_threads: int | None = None)` — [`L814`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L814) — Uploads all modified files under `working_dir` to `repo_id`, at arbitrary depth, based on `files_timestamps`.
  - `callback(x, p)` — [`L1006`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L1006)
  - `can_generate(cls)` — [`L806`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L806) — Checks if the model can generate sequences with `.generate()`.
  - `from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike | None, sharding_axis_dims: collections.abc.Sequence[int] = (1, -1, 1, 1, 1), sharding_dcn_axis_dims: collections.abc.Sequence[int] | None = None, sharding_axis_names: collections.abc.Sequence[str] = ("dp", "fsdp", "ep", "tp", "sp"), partition_axis: PartitionAxis | None = None, dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.bfloat16, precision: jax.lax.PrecisionLike | None = None, config_kwargs: dict[str, tp.Any] | None = None, partition_rules: tuple[tuple[str, PartitionSpec]] | None = None, backend: EasyDeLBackends | None = None, platform: EasyDeLPlatforms | None = "jax", shard_fns: dict[str, tp.Callable] | None = None, auto_shard_model: bool = True, verbose: bool = True, mismatch_allowed: bool = True, *model_args, config: EasyDeLBaseConfig | str | os.PathLike | None = None, cache_dir: str | os.PathLike | None = None, force_download: bool = False, local_files_only: bool = False, token: str | bool | None = None, revision: str = "main", quantization_config: QuantizationConfig | None = None, apply_quantization: bool = False, **kwargs)` — [`L1174`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L1174) — Loads an EasyDeL model from a pretrained model or path.
  - `get_torch_loader(cls)` — [`L2922`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L2922) — Gets the appropriate HuggingFace AutoModel class for loading PyTorch weights.
  - `huggingface_to_easydel_sequential(cls, pretrained_model_name_or_path: str, save_directory: str | os.PathLike, *, output_repo_id: str | None = None, dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.bfloat16, precision: jax.lax.Precision | None = None, sharding_axis_dims: collections.abc.Sequence[int] = (1, -1, 1, 1, 1), sharding_dcn_axis_dims: collections.abc.Sequence[int] | None = None, sharding_axis_names: collections.abc.Sequence[str] = ("dp", "fsdp", "ep", "tp", "sp"), partition_axis: PartitionAxis | None = None, backend: EasyDeLBackends | None = None, platform: EasyDeLPlatforms | None = None, config_kwargs: EasyDeLBaseConfigDict | None = None, partition_rules: tuple[tuple[str, PartitionSpec], ...] | None = None, trust_remote_code: bool = False, torch_streaming_cache: str = "temp", torch_streaming_tmp_dir: str | None = None, tensorstore_chunk_bytes: int = 2147483648, verbose: bool = True, **kwargs)` — [`L2269`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L2269) — Convert a HuggingFace PyTorch checkpoint to an EasyDeL checkpoint sequentially.
  - `push_to_hub(self, repo_id: str, use_temp_dir: bool | None = None, commit_message: str | None = None, private: bool | None = None, token: bool | str | None = None, create_pr: bool = False, float_dtype: jnp.dtype | None = None, verbose: bool = True, mismatch_allowed: bool = True, revision: str | None = None, commit_description: str | None = None, upload_num_threads: int | None = None)` — [`L730`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L730) — Pushes the model to the Hugging Face Hub.
  - `save_pretrained(self, save_directory: str | os.PathLike, push_to_hub: bool = False, token: str | bool | None = None, float_dtype: jnp.dtype | None = None, step: int | None = None, upload_num_threads: int | None = None, **kwargs)` — [`L666`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L666) — Saves the model, its configuration, and optionally pushes it to the Hugging Face Hub.
  - `base_model_prefix` — [`L565`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L565)
  - `config` — [`L562`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L562)
  - `config_class` — [`L564`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L564)
  - `hf_torch_auto_loader` — [`L563`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L563)
- protocol/private: `_apply_shard_fn`[`L996`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L996), `_chunk_shape_for`[`L2623`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L2623), `_clear`[`L1578`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L1578), `_convert`[`L1137`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L1137), `_ensure_moe_group_ready`[`L2763`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L2763), `_finalize_moe_group`[`L2810`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L2810), `_find_local`[`L1792`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L1792), `_hf_download`[`L1778`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L1778), `_iter_keys_single_file`[`L2727`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L2727), `_load_index`[`L1809`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L1809), `_maybe_shard_and_callback`[`L2030`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L2030), `_model_task`[`L566`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L566), `_model_type`[`L567`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L567), `_pick_first_existing`[`L1348`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L1348), `_process_and_write`[`L2814`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L2814), `_process_tensor`[`L2037`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L2037), `_register_moe_key`[`L1963`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L1963), `_run_streaming_conversion`[`L2069`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L2069), `_save_configs`[`L2687`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L2687), `_strip_or_keep_subfolder`[`L1770`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L1770), `_tensorstore_path_for_params`[`L2641`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L2641), `_with_resolved_shard`[`L2045`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L2045), `_write_checkpoint_index`[`L2671`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L2671), `_write_tensor`[`L2648`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L2648)
- uses (calls/refs, reference-scoped): [`config`](../base_module.md#EasyDeLBaseModule.config), [`EasyDeLBaseModule`](../base_module.md#EasyDeLBaseModule), [`TaskType`](../factory.md#TaskType), [`EasyDeLBaseConfig`](../base_config.md#EasyDeLBaseConfig), [`EasyDeLGradientCheckPointers`](../etils.md#EasyDeLGradientCheckPointers), [`CAUSAL_LM`](../factory.md#TaskType.CAUSAL_LM), [`generation_config`](generation.md#EasyGenerationMixin.generation_config), [`BASE_MODULE`](../factory.md#TaskType.BASE_MODULE), [`QuantizationConfig`](../../layers/quantization/_configs.md#QuantizationConfig), [`SEQUENCE_CLASSIFICATION`](../factory.md#TaskType.SEQUENCE_CLASSIFICATION), [`mesh`](../base_module.md#EasyDeLBaseModule.mesh), [`add_basic_configurations`](../base_config.md#EasyDeLBaseConfig.add_basic_configurations), [`QuantizationType`](../../layers/quantization/_configs.md#QuantizationType), [`NONE`](../etils.md#EasyDeLGradientCheckPointers.NONE), [`EasyQuantizer`](../../layers/quantization/_quants.md#EasyQuantizer), [`iter_module_search`](../../utils/traversals.md#iter_module_search), [`flatten_dict`](../../utils/traversals.md#flatten_dict), [`resolve_ejkernel_quant_params`](../../layers/quantization/_configs.md#resolve_ejkernel_quant_params), [`IMAGE_TEXT_TO_TEXT`](../factory.md#TaskType.IMAGE_TEXT_TO_TEXT), [`logger`](bridge.md#logger), [`gradient_checkpointing`](../base_config.md#EasyDeLBaseConfig.gradient_checkpointing), [`from_pretrained`](../../modules/auto/auto_configuration.md#AutoEasyDeLConfig.from_pretrained), [`_get_partition_rules`](../base_module.md#EasyDeLBaseModule._get_partition_rules), [`get_modules_by_type`](../../modules/auto/auto_configuration.md#get_modules_by_type), [`from_config`](../../modules/auto/auto_configuration.md#AutoShardAndGatherFunctions.from_config), [`EasyDeLBackends`](../etils.md#EasyDeLBackends), [`EasyDeLBaseConfigDict`](../base_config.md#EasyDeLBaseConfigDict), [`unflatten_dict`](../../utils/traversals.md#unflatten_dict), [`param_dtype`](../base_module.md#EasyDeLBaseModule.param_dtype), [`EasyDeLPlatforms`](../etils.md#EasyDeLPlatforms), [`apply_quantization`](../../layers/quantization/_quants.md#EasyQuantizer.apply_quantization), [`mesh`](../base_config.md#EasyDeLBaseConfig.mesh), [`graphtree_params_shape`](../base_module.md#EasyDeLBaseModule.graphtree_params_shape), [`dtype`](../../layers/quantization/_configs.md#QuantizationConfig.dtype), [`_build_safe_checkpoint_partition_rules`](bridge.md#_build_safe_checkpoint_partition_rules), [`from_pretrained`](../../modules/auto/auto_configuration.md#AutoShardAndGatherFunctions.from_pretrained), [`shard_model`](../base_module.md#EasyDeLBaseModule.shard_model), [`BASE_VISION`](../factory.md#TaskType.BASE_VISION), [`quantize`](../base_module.md#EasyDeLBaseModule.quantize), [`CANDIDATE_FILENAMES`](bridge.md#CANDIDATE_FILENAMES)  (+65 more)
- used by: [`EasyDeLBaseModule`](../base_module.md#EasyDeLBaseModule), [`save_state`](../base_state.md#EasyDeLState.save_state), [`to_torch`](../base_module.md#EasyDeLBaseModule.to_torch), [`_model_task`](../base_state.md#EasyDeLState.load_state._BaseModuleLoader._model_task), [`_model_task`](../../modules/auto/auto_modeling.md#BaseAutoEasyModel._from_easydel_params.Base._model_task), [`_model_task`](../../modules/auto/auto_modeling.md#BaseAutoEasyModel._from_torch_pretrained.Base._model_task)

### `StreamingCheckpointInfo`
- def: [`easydel/infra/mixins/bridge.py:413`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L413)
- doc: Metadata for streaming PyTorch checkpoint conversion.
- signature: `class StreamingCheckpointInfo:`
- members:
  - `ckpt_filename_to_path` — [`L418`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L418)
  - `ckpt_key_to_filename` — [`L417`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L417)
  - `ckpt_weight_format` — [`L416`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L416)
  - `ed_config` — [`L419`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L419)
  - `generation_config` — [`L420`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L420)
  - `pretrained_model_name_or_path` — [`L421`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L421)
  - `resolve_shard` — [`L422`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L422)
- used by: [`_from_torch_pretrained`](bridge.md#EasyBridgeMixin._from_torch_pretrained), [`_resolve_streaming_checkpoint`](bridge.md#EasyBridgeMixin._resolve_streaming_checkpoint), [`_convert_streaming_checkpoint_to_params`](bridge.md#EasyBridgeMixin._convert_streaming_checkpoint_to_params)

### `TorchLoadOptions`
- def: [`easydel/infra/mixins/bridge.py:403`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L403)
- doc: Options for torch_load_mode in _from_torch_pretrained.
- signature: `class TorchLoadOptions:`
- members:
  - `hub_kwargs` — [`L409`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L409)
  - `mode` — [`L406`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L406)
  - `streaming_cache` — [`L407`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L407)
  - `streaming_tmp_dir` — [`L408`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L408)
- used by: [`_from_torch_pretrained`](bridge.md#EasyBridgeMixin._from_torch_pretrained), [`_convert_streaming_checkpoint_to_params`](bridge.md#EasyBridgeMixin._convert_streaming_checkpoint_to_params), [`_parse_torch_load_options`](bridge.md#_parse_torch_load_options)

## Functions
- `_build_safe_checkpoint_partition_rules(*, model: EasyDeLBaseModule, mesh: jax.sharding.Mesh, partition_rules: tuple[tuple[str, PartitionSpec], ...] | list[tuple[str, PartitionSpec]] | None)` — [`L352`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L352) — Create path-specific override rules for non-divisible shardings.
- `_checkpoint_cpu_device()` — [`L228`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L228) — Resolve a host CPU device for checkpoint-time array normalization.
- `_collect_lora_checkpoint_paths(flat_state: dict[tuple[tp.Any, ...], tp.Any])` — [`L250`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L250) — Extract LoRA adapter locations from a flattened checkpoint state.
- `_load_generation_config(pretrained_model_name_or_path: str | os.PathLike, *, subfolder: str = "", trust_remote_code: bool | None = None, log_missing: bool = False, **hf_kwargs)` — [`L451`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L451) — Load generation config with ePath-first lookup (supports GCS paths).
- `_mesh_partition_product(mesh: jax.sharding.Mesh, axis_spec: object)` — [`L156`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L156) — Return shard multiplicity implied by a PartitionSpec entry.
- `_normalize_checkpoint_leaf_to_jax(value: tp.Any, *, cpu_device: jax.Device | None)` — [`L237`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L237) — Convert NumPy leaves into JAX arrays without consuming accelerator HBM.
- `_normalize_quantization_config(quantization_config: QuantizationConfig | dict[str, tp.Any] | None)` — [`L539`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L539)
- `_parse_torch_load_options(kwargs: dict[str, tp.Any])` — [`L503`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L503) — Parse torch_load_mode related kwargs and return a TorchLoadOptions struct.
- `_path_match_variants(path: str)` — [`L210`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L210) — Build exact-match path variants for different checkpoint key prefixes.
- `_rebuild_lora_modules_from_checkpoint(model: EasyDeLBaseModule, flat_state: dict[tuple[tp.Any, ...], tp.Any])` — [`L279`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L279) — Recreate LoRA wrappers before merging a LoRA checkpoint into a model.
- `_sanitize_partition_spec_for_shape(spec: PartitionSpec, shape: tuple[int, ...], mesh: jax.sharding.Mesh)` — [`L176`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L176) — Drop non-divisible sharding axes for a concrete tensor shape.
- `_save_generation_config(generation_config: tp.Any, save_directory: ePathLike)` — [`L425`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L425) — Persist generation config with EasyDeL path handling (local and remote filesystems).
- `_strip_trailing_separators(path: str)` — [`L551`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L551)
- `_tree_key_to_path(key: tp.Any)` — [`L203`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L203)

## Module values
- `CANDIDATE_FILENAMES` — [`L147`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L147)
- `CHAT_TEMPLATE_NAME` — [`L143`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L143)
- `CONFIG_NAME` — [`L139`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L139)
- `ED_SAFE_WEIGHTS_INDEX_NAME` — [`L137`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L137)
- `FEATURE_EXTRACTOR_NAME` — [`L140`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L140)
- `FLAX_WEIGHTS_INDEX_NAME` — [`L134`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L134)
- `FLAX_WEIGHTS_NAME` — [`L128`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L128)
- `GENERATION_CONFIG_NAME` — [`L144`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L144)
- `IMAGE_PROCESSOR_NAME` — [`L141`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L141)
- `MODEL_CARD_NAME` — [`L145`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L145)
- `PROCESSOR_NAME` — [`L142`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L142)
- `SAFE_WEIGHTS_INDEX_NAME` — [`L136`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L136)
- `SAFE_WEIGHTS_NAME` — [`L135`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L135)
- `TENSORSTORE_INDEX_NAME` — [`L138`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L138)
- `TF2_WEIGHTS_INDEX_NAME` — [`L132`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L132)
- `TF2_WEIGHTS_NAME` — [`L131`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L131)
- `TF_WEIGHTS_NAME` — [`L133`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L133)
- `WEIGHTS_INDEX_NAME` — [`L130`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L130)
- `WEIGHTS_NAME` — [`L129`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L129)
- `logger` — [`L126`](../../../../../../../raw/code/EasyDeL/easydel/infra/mixins/bridge.py#L126)

