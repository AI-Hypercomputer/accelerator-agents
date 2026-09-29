---
title: 'Module: src/maxtext/configs/pyconfig_deprecated.py'
type: catalog
provenance: extracted
module: src/maxtext/configs/pyconfig_deprecated.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.configs.pyconfig_deprecated`/
symbols:
  validate_keys: validate_keys().
  _HyperParameters.user_init: _HyperParameters#user_init().
  _HyperParameters.__init__: _HyperParameters#__init__().
  validate_sparse_matmul_parallelism: validate_sparse_matmul_parallelism().
  set_and_validate_pipeline_config: set_and_validate_pipeline_config().
  _HyperParameters._load_config: _HyperParameters#_load_config().
  main_config: main_config.
  _lists_to_tuples: _lists_to_tuples().
  _HyperParameters._update_from_env_and_command_line: _HyperParameters#_update_from_env_and_command_line().
  _HyperParameters.update_model_vars: _HyperParameters#update_model_vars().
  validate_shard_expert_on_fsdp: validate_shard_expert_on_fsdp().
  initialize: initialize().
  yaml_key_to_env_key: yaml_key_to_env_key().
  _yaml_types_to_parser: _yaml_types_to_parser.
  _MAX_PREFIX: _MAX_PREFIX.
  _HyperParameters: _HyperParameters#
  using_expert_parallelism: using_expert_parallelism().
  calculate_global_batch_sizes: calculate_global_batch_sizes().
  get_num_target_devices: get_num_target_devices().
  _HyperParameters._validate_env_variables: _HyperParameters#_validate_env_variables().
  validate_ring_of_experts_parallelism: validate_ring_of_experts_parallelism().
  create_new_logical_axis_rules: create_new_logical_axis_rules().
  update_model_keys: update_model_keys().
  validate_and_update_keys: validate_and_update_keys().
  using_tensor_parallelism: using_tensor_parallelism().
  r: r.
  _BASE_CONFIG_ATTR: _BASE_CONFIG_ATTR.
  using_pipeline_parallelism: using_pipeline_parallelism().
  string_to_bool: string_to_bool().
  validate_compute_axis_order: validate_compute_axis_order().
  validate_shard_mode: validate_shard_mode().
  validate_kv_quant_axis: validate_kv_quant_axis().
  validate_attention_kernel: validate_attention_kernel().
  validate_attention_type: validate_attention_type().
  validate_moba_attention: validate_moba_attention().
  validate_attention_window_params: validate_attention_window_params().
  validate_profiler_type: validate_profiler_type().
  validate_periodic_profiler: validate_periodic_profiler().
  validate_model_call_mode: validate_model_call_mode().
  validate_prefill_and_target_lengths: validate_prefill_and_target_lengths().
  validate_rope_type: validate_rope_type().
  validate_expert_shard_attention_option: validate_expert_shard_attention_option().
  validate_vocab_tiling: validate_vocab_tiling().
  validate_rampup_batch_size: validate_rampup_batch_size().
  validate_context_parallel_strategy_ring: validate_context_parallel_strategy_ring().
  validate_tokenizer: validate_tokenizer().
  validate_constant_bound: validate_constant_bound().
  validate_quantization_methods: validate_quantization_methods().
  validate_tokamax_usage: validate_tokamax_usage().
  validate_data_input: validate_data_input().
  validate_llama4_config: validate_llama4_config().
  validate_model_name: validate_model_name().
  validate_multimodal_model_name: validate_multimodal_model_name().
  validate_no_keys_overwritten_twice: validate_no_keys_overwritten_twice().
  validate_and_assign_remat_tensors: validate_and_assign_remat_tensors().
  resolve_config_path: resolve_config_path().
  _HyperParameters.configure_gpt3_task: _HyperParameters#configure_gpt3_task().
  _HyperParameters.configure_c4_mlperf_task: _HyperParameters#configure_c4_mlperf_task().
  create_parallelisms_list: create_parallelisms_list().
  set_mu_dtype: set_mu_dtype().
  validate_and_set_hlo_dump_defaults: validate_and_set_hlo_dump_defaults().
  validate_multiple_slices: validate_multiple_slices().
  set_and_validate_pipeline_config.modify_activation_embed_and_logits_batch: set_and_validate_pipeline_config().modify_activation_embed_and_logits_batch().
  set_and_validate_pipeline_config.pipeline_first_axis: set_and_validate_pipeline_config().pipeline_first_axis().
  validate_deepseek_moe: validate_deepseek_moe().
  validate_mlp_dim: validate_mlp_dim().
  validate_gpt_oss_moe: validate_gpt_oss_moe().
  validate_ragged_dot: validate_ragged_dot().
  validate_optimizer_sharding_over_data: validate_optimizer_sharding_over_data().
  get_individual_scales: get_individual_scales().
  calculate_rampup_samples_and_steps: calculate_rampup_samples_and_steps().
  get_quantization_local_shard_count: get_quantization_local_shard_count().
  get_context_parallel_size: get_context_parallel_size().
  using_fsdp_and_transpose_parallelism: using_fsdp_and_transpose_parallelism().
  HyperParameters: HyperParameters#
  _HyperParameters.keys: _HyperParameters#keys.
  using_sequence_parallelism: using_sequence_parallelism().
  HyperParameters.__init__: HyperParameters#__init__().
  HyperParameters.__getattr__: HyperParameters#__getattr__().
  HyperParameters.__setattr__: HyperParameters#__setattr__().
  HyperParameters.get_keys: HyperParameters#get_keys().
  HyperParameters.tree_flatten: HyperParameters#tree_flatten().
  HyperParameters.tree_unflatten: HyperParameters#tree_unflatten().
---
# Module: [`src/maxtext/configs/pyconfig_deprecated.py`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py)

## Classes
### `HyperParameters`
- def: [`src/maxtext/configs/pyconfig_deprecated.py:1422`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1422)
- doc: Wrapper class to expose the configuration in a read-only manner.
- signature: `class HyperParameters:`
- members:
  - `get_keys(self)` — [`L1438`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1438)
  - `tree_flatten(self)` — [`L1441`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1441)
  - `tree_unflatten(cls, aux_data, children)` — [`L1445`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1445)
- protocol/private: `__getattr__`[`L1428`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1428), `__init__`[`L1425`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1425), `__setattr__`[`L1435`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1435)
- used by: [`initialize`](pyconfig_deprecated.md#initialize)

### `_HyperParameters`
- def: [`src/maxtext/configs/pyconfig_deprecated.py:555`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L555) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- signature: `class _HyperParameters:`
- members:
  - `_load_config(self, config_name: str)` — [`L620`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L620) — Loads the YAML config from a file using omegaconf.OmegaConf, and resolves inheritance. — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
  - `_update_from_env_and_command_line(self, raw_keys, raw_data_from_yaml, argv, **kwargs)` — [`L568`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L568) — Update model config from environment and command line using omegaconf.OmegaConf overrides. — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
  - `configure_c4_mlperf_task(raw_keys)` — [`L868`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L868) — dynamically configure based on training rules
  - `configure_gpt3_task(raw_keys)` — [`L852`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L852) — dynamically configure gpt3 task based on training rules
  - `update_model_vars(base_config_path, raw_keys, config_name: str, keys_from_env_and_command_line)` — [`L887`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L887) — Update model config variables — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
  - `user_init(raw_keys)` — [`L705`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L705) — Transformations between the config data and configs used at runtime — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
  - `keys` — [`L695`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L695)
- protocol/private: `__init__`[`L644`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L644), `_validate_env_variables`[`L559`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L559)
- uses (calls/refs, reference-scoped): [`validate_keys`](pyconfig_deprecated.md#validate_keys), [`set_and_validate_pipeline_config`](pyconfig_deprecated.md#set_and_validate_pipeline_config), [`_lists_to_tuples`](pyconfig_deprecated.md#_lists_to_tuples), [`_yaml_types_to_parser`](pyconfig_deprecated.md#_yaml_types_to_parser), [`yaml_key_to_env_key`](pyconfig_deprecated.md#yaml_key_to_env_key), [`_MAX_PREFIX`](pyconfig_deprecated.md#_MAX_PREFIX), [`calculate_global_batch_sizes`](pyconfig_deprecated.md#calculate_global_batch_sizes), [`get_num_target_devices`](pyconfig_deprecated.md#get_num_target_devices), [`validate_and_update_keys`](pyconfig_deprecated.md#validate_and_update_keys), [`_BASE_CONFIG_ATTR`](pyconfig_deprecated.md#_BASE_CONFIG_ATTR), [`calculate_rampup_samples_and_steps`](pyconfig_deprecated.md#calculate_rampup_samples_and_steps), [`create_parallelisms_list`](pyconfig_deprecated.md#create_parallelisms_list), [`get_individual_scales`](pyconfig_deprecated.md#get_individual_scales), [`get_quantization_local_shard_count`](pyconfig_deprecated.md#get_quantization_local_shard_count), [`resolve_config_path`](pyconfig_deprecated.md#resolve_config_path), [`set_mu_dtype`](pyconfig_deprecated.md#set_mu_dtype), [`validate_and_assign_remat_tensors`](pyconfig_deprecated.md#validate_and_assign_remat_tensors), [`validate_and_set_hlo_dump_defaults`](pyconfig_deprecated.md#validate_and_set_hlo_dump_defaults), [`validate_constant_bound`](pyconfig_deprecated.md#validate_constant_bound), [`validate_data_input`](pyconfig_deprecated.md#validate_data_input), [`validate_model_name`](pyconfig_deprecated.md#validate_model_name), [`validate_no_keys_overwritten_twice`](pyconfig_deprecated.md#validate_no_keys_overwritten_twice), [`validate_quantization_methods`](pyconfig_deprecated.md#validate_quantization_methods), [`validate_tokamax_usage`](pyconfig_deprecated.md#validate_tokamax_usage), [`validate_tokenizer`](pyconfig_deprecated.md#validate_tokenizer)
- used by: [`initialize`](pyconfig_deprecated.md#initialize)

## Functions
- `_lists_to_tuples(l: list[Any])` — [`L544`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L544) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `calculate_global_batch_sizes(per_device_batch_size, expansion_factor_real_data, num_devices, gradient_accumulation_steps)` — [`L1313`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1313) — Calculates target global batch size from target devices and per_device_batch — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `calculate_rampup_samples_and_steps(batch_size_start, batch_size_end, batch_size_increment, global_rampup_samples)` — [`L1335`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1335) — Calculate num of samples for each increment and num of steps for batch rampup — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `create_new_logical_axis_rules(old_logical_axis_rules, new_logical_axis_rules)` — [`L1243`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1243) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `create_parallelisms_list(raw_keys)` — [`L911`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L911) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `get_context_parallel_size(raw_keys)` — [`L1377`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1377) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `get_individual_scales(scale)` — [`L1286`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1286) — Choose appropriate scales for individual dimensions based on global scale — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `get_num_target_devices(raw_keys)` — [`L1356`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1356) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `get_quantization_local_shard_count(raw_keys)` — [`L1370`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1370) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `initialize(argv, **kwargs)` — [`L1449`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1449)
- `modify_activation_embed_and_logits_batch(logical_axis_rules)` — [`L1008`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1008)
- `pipeline_first_axis(raw_keys)` — [`L1021`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1021)
- `resolve_config_path(param: str)` — [`L549`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L549) — Resolve config path to auto rewrite to use new src folder.
- `set_and_validate_pipeline_config(raw_keys)` — [`L1000`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1000) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `set_mu_dtype(raw_keys)` — [`L945`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L945) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `string_to_bool(s: str)` — [`L51`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L51)
- `update_model_keys(raw_keys, model_keys, key)` — [`L1260`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1260) — Update `key` value in `raw_keys` from the value in `model_keys`. — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `using_expert_parallelism(raw_keys)` — [`L1402`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1402) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `using_fsdp_and_transpose_parallelism(raw_keys)` — [`L1412`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1412) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `using_pipeline_parallelism(raw_keys)` — [`L1385`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1385) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `using_sequence_parallelism(raw_keys)` — [`L1398`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1398)
- `using_tensor_parallelism(raw_keys)` — [`L1389`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1389) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_and_assign_remat_tensors(keys)` — [`L510`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L510) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_and_set_hlo_dump_defaults(raw_keys)` — [`L956`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L956) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_and_update_keys(raw_keys, model_keys, config_name: str)` — [`L1271`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1271) — Validate and update model specific config keys — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_attention_kernel(s: str)` — [`L101`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L101) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_attention_type(s: str)` — [`L114`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L114) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_attention_window_params(attention_type: str, chunk_attn_window_size: int, sliding_window_size: int)` — [`L125`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L125) — Validates window size parameters for attention types 'chunk' and 'local'. — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_compute_axis_order(s: str)` — [`L62`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L62) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_constant_bound(keys)` — [`L333`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L333) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_context_parallel_strategy_ring(context_parallel_size: int, context_parallel_strategy: str, hardware: str)` — [`L217`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L217) — Validates that ring context parallelism strategy is only used on GPU hardware. — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_data_input(keys)` — [`L359`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L359) — validate provided parameters for data input — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_deepseek_moe(raw_keys)` — [`L1147`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1147) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_expert_shard_attention_option(expert_shard_attention_option: str)` — [`L188`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L188) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_gpt_oss_moe(raw_keys)` — [`L1173`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1173) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_keys(keys)` — [`L226`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L226) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_kv_quant_axis(s: str, quantize_kvcache: bool)` — [`L93`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L93) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_llama4_config(keys: dict)` — [`L409`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L409) — Validates the following checks for Llama4 models: — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_mlp_dim(raw_keys)` — [`L1161`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1161) — Validates that MLP dimensions are consistent for fully MoE models. — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_moba_attention(moba, attention)` — [`L120`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L120) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_model_call_mode(s: str)` — [`L165`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L165) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_model_name(s: str)` — [`L429`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L429) — Validate provided model name.
- `validate_multimodal_model_name(s: str)` — [`L487`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L487) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_multiple_slices(raw_keys)` — [`L976`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L976) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_no_keys_overwritten_twice(keys1: list[str], keys2: list[str])` — [`L501`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L501) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_optimizer_sharding_over_data(raw_keys)` — [`L1234`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1234) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_periodic_profiler(profiler, profile_periodically_period, profiler_steps)` — [`L154`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L154) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_prefill_and_target_lengths(max_prefill_length: int, max_target_length: int)` — [`L171`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L171) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_profiler_type(s: str)` — [`L148`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L148) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_quantization_methods(keys)` — [`L344`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L344) — Validate quantization methods — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_ragged_dot(raw_keys)` — [`L1225`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1225)
- `validate_rampup_batch_size(batch_size_start, batch_size_end, batch_size_increment, global_rampup_samples)` — [`L202`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L202)
- `validate_ring_of_experts_parallelism(raw_keys)` — [`L1211`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1211) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_rope_type(rope_type: str)` — [`L182`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L182)
- `validate_shard_expert_on_fsdp(raw_keys)` — [`L1216`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1216) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_shard_mode(shard_mode: str, decoder_block: str, quantization: str)` — [`L68`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L68) — Validates sharding settings, raising ValueError for incompatible combinations.
- `validate_sparse_matmul_parallelism(raw_keys)` — [`L1180`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1180) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `validate_tokamax_usage(keys)` — [`L352`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L352) — Validate tokamax usage for gmm kernel
- `validate_tokenizer(keys)` — [`L327`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L327)
- `validate_vocab_tiling(num_vocab_tiling: int, per_device_batch_size: int, max_target_length: int, enable_nnx: bool)` — [`L196`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L196)
- `yaml_key_to_env_key(s: str)` — [`L47`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L47) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)

## Module values
- `_BASE_CONFIG_ATTR` — [`L44`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L44) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `_MAX_PREFIX` — [`L41`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L41) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `_yaml_types_to_parser` — [`L59`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L59) — documented in [maxtext-configs-pyconfig_deprecated](../../../../concepts/maxtext-configs-pyconfig_deprecated.md)
- `main_config` — [`L1456`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1456)
- `r` — [`L1458`](../../../../../../../raw/code/maxtext/src/maxtext/configs/pyconfig_deprecated.py#L1458)

