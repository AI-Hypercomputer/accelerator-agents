---
title: 'Module: python/sgl_jax/srt/configs/model_config.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/configs/model_config.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.configs.model_config`/
symbols:
  ModelConfig.hf_config: ModelConfig#hf_config.
  ModelConfig: ModelConfig#
  ModelConfig.from_server_args: ModelConfig#from_server_args().
  ModelConfig.hf_text_config: ModelConfig#hf_text_config.
  ModelConfig._resolve_quantization_config: ModelConfig#_resolve_quantization_config().
  ModelConfig.moe_backend: ModelConfig#moe_backend.
  ModelConfig.quantization_config: ModelConfig#quantization_config.
  ModelConfig.dtype_config: ModelConfig#dtype_config.
  logger: logger.
  ModelConfig.model_path: ModelConfig#model_path.
  ModelConfig.log_kv_heads_info: ModelConfig#log_kv_heads_info().
  ModelConfig.head_dim: ModelConfig#head_dim.
  ModelConfig.num_hidden_layers: ModelConfig#num_hidden_layers.
  ModelConfig.get_total_num_kv_heads: ModelConfig#get_total_num_kv_heads().
  ModelConfig.get_num_kv_heads: ModelConfig#get_num_kv_heads().
  ModelConfig.num_key_value_heads: ModelConfig#num_key_value_heads.
  ModelConfig.vocab_size: ModelConfig#vocab_size.
  ModelConfig.sliding_window: ModelConfig#sliding_window.
  ModelConfig.dtype: ModelConfig#dtype.
  ModelConfig.__init__: ModelConfig#__init__().
  ModelConfig.context_len: ModelConfig#context_len.
  MoEBackend: MoEBackend#
  ModelConfig._apply_model_specific_config: ModelConfig#_apply_model_specific_config().
  ModelConfig.num_attention_heads: ModelConfig#num_attention_heads.
  ModelConfig.final_logit_softcapping: ModelConfig#final_logit_softcapping.
  ModelConfig.is_generation: ModelConfig#is_generation.
  ModelConfig._original_hf_num_key_value_heads: ModelConfig#_original_hf_num_key_value_heads.
  ModelConfig._original_swa_num_key_value_heads: ModelConfig#_original_swa_num_key_value_heads.
  _assert_fused_moe_v2_supported: _assert_fused_moe_v2_supported().
  ModelConfig.model_override_args: ModelConfig#model_override_args.
  ModelConfig.attention_arch: ModelConfig#attention_arch.
  ModelConfig.get_total_num_kv_heads_with_replication: ModelConfig#get_total_num_kv_heads_with_replication().
  ModelConfig.hf_eos_token_id: ModelConfig#hf_eos_token_id.
  ModelConfig._get_hf_quant_config: ModelConfig#_get_hf_quant_config().
  ModelConfig.needs_kv_head_replication: ModelConfig#needs_kv_head_replication().
  ModelConfig.is_gqa_model: ModelConfig#is_gqa_model().
  ModelConfig.get_hybrid_layer_counts: ModelConfig#get_hybrid_layer_counts().
  MockModelConfig: MockModelConfig#
  ModelImpl: ModelImpl#
  ModelConfig.get_hf_eos_token_id: ModelConfig#get_hf_eos_token_id().
  ModelConfig.maybe_pull_model_tokenizer_from_remote: ModelConfig#maybe_pull_model_tokenizer_from_remote().
  _get_and_verify_dtype: _get_and_verify_dtype().
  ModelConfig.v_head_dim: ModelConfig#v_head_dim.
  ModelConfig._original_num_key_value_heads: ModelConfig#_original_num_key_value_heads.
  ModelConfig.get_original_kv_head_id: ModelConfig#get_original_kv_head_id().
  ModelConfig._parse_quant_hf_config: ModelConfig#_parse_quant_hf_config().
  ModelConfig.hf_generation_config: ModelConfig#hf_generation_config.
  ModelConfig.hidden_size: ModelConfig#hidden_size.
  ModelConfig.get_num_kv_head_replicas: ModelConfig#get_num_kv_head_replicas().
  ModelConfig.get_kv_padding_strategy: ModelConfig#get_kv_padding_strategy().
  ModelConfig.validate_tensor_parallel_config: ModelConfig#validate_tensor_parallel_config().
  need_attention_mask: need_attention_mask().
  ModelConfig.revision: ModelConfig#revision.
  ModelConfig.ep_size: ModelConfig#ep_size.
  ModelConfig.configure_for_tensor_parallel: ModelConfig#configure_for_tensor_parallel().
  ModelConfig.get_swa_weight_params: ModelConfig#get_swa_weight_params().
  AttentionArch: AttentionArch#
  MoEBackend.FUSED: MoEBackend#FUSED.
  ModelConfig.model_sub_dir: ModelConfig#model_sub_dir.
  ModelConfig.model_impl: ModelConfig#model_impl.
  MockModelConfig.get_num_kv_heads: MockModelConfig#get_num_kv_heads().
  ModelImpl.AUTO: ModelImpl#AUTO.
  ModelImpl.TRANSFORMERS: ModelImpl#TRANSFORMERS.
  MoEBackend.FUSED_V2: MoEBackend#FUSED_V2.
  MoEBackend.AUTO: MoEBackend#AUTO.
  ModelConfig.is_dynamic_fp8_act: ModelConfig#is_dynamic_fp8_act().
  MockModelConfig.num_kv_heads: MockModelConfig#num_kv_heads.
  MockModelConfig.context_len: MockModelConfig#context_len.
  AttentionArch.MLA: AttentionArch#MLA.
  AttentionArch.MHA: AttentionArch#MHA.
  MoEBackend.EPMOE: MoEBackend#EPMOE.
  _FUSED_MOE_V2_SUPPORTED_ARCHITECTURES: _FUSED_MOE_V2_SUPPORTED_ARCHITECTURES.
  _FORCED_FUSED_EP_MOE_ARCHS: _FORCED_FUSED_EP_MOE_ARCHS.
  is_generation_model: is_generation_model().
  ENCODER_ONLY_MODELS: ENCODER_ONLY_MODELS.
  MockModelConfig.num_heads: MockModelConfig#num_heads.
  MockModelConfig.head_dim: MockModelConfig#head_dim.
  MockModelConfig.num_hidden_layers: MockModelConfig#num_hidden_layers.
  ModelImpl.SGLANG: ModelImpl#SGLANG.
  ModelConfig.quantization: ModelConfig#quantization.
  ModelConfig.quantization_config_path: ModelConfig#quantization_config_path.
  ModelConfig.is_multimodal: ModelConfig#is_multimodal.
  ModelConfig.image_token_id: ModelConfig#image_token_id.
  multimodal_model_archs: multimodal_model_archs.
  MockModelConfig.__init__: MockModelConfig#__init__().
---
# Module: [`python/sgl_jax/srt/configs/model_config.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py)

## Classes
### `AttentionArch`  ·  implements/extends IntEnum
- def: [`python/sgl_jax/srt/configs/model_config.py:24`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L24)
- signature: `class AttentionArch(IntEnum):`
- members:
  - `MHA` — [`L26`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L26)
  - `MLA` — [`L25`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L25)
- used by: [`use_mla_backend`](../model_executor/model_runner.md#ModelRunner.use_mla_backend), [`attention_arch`](model_config.md#ModelConfig.attention_arch)

### `MoEBackend`  ·  implements/extends Enum, str
- def: [`python/sgl_jax/srt/configs/model_config.py:35`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L35)
- doc: Backend for Mixture of Experts computation.
- signature: `class MoEBackend(str, Enum):`
- members:
  - `AUTO` — [`L41`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L41)
  - `EPMOE` — [`L38`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L38)
  - `FUSED` — [`L39`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L39)
  - `FUSED_V2` — [`L40`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L40)
- used by: [`hf_config`](model_config.md#ModelConfig.hf_config), [`moe_backend`](model_config.md#ModelConfig.moe_backend), [`__init__`](model_config.md#ModelConfig.__init__), [`_assert_fused_moe_v2_supported`](model_config.md#_assert_fused_moe_v2_supported)

### `MockModelConfig`  ·  implements/extends ModelConfig
- def: [`python/sgl_jax/srt/configs/model_config.py:1011`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L1011)
- signature: `class MockModelConfig(ModelConfig):`
- members:
  - `get_num_kv_heads(self, tensor_parallel_size)` — [`L1026`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L1026)
  - `context_len` — [`L1023`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L1023)
  - `head_dim` — [`L1021`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L1021)
  - `num_heads` — [`L1020`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L1020)
  - `num_hidden_layers` — [`L1024`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L1024)
  - `num_kv_heads` — [`L1022`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L1022)
- protocol/private: `__init__`[`L1012`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L1012)
- uses (calls/refs, reference-scoped): [`ModelConfig`](model_config.md#ModelConfig)
- used by: [`ModelConfig`](model_config.md#ModelConfig), [`token_to_kv_pool`](../model_executor/model_runner.md#MockModelRunner.token_to_kv_pool), [`get_num_kv_heads`](model_config.md#ModelConfig.get_num_kv_heads), [`mesh`](../model_executor/model_runner.md#MockModelRunner.mesh), [`__init__`](../model_executor/model_runner.md#MockModelRunner.__init__), [`req_to_token_pool`](../model_executor/model_runner.md#MockModelRunner.req_to_token_pool), [`attention_tp_size`](../model_executor/model_runner.md#MockModelRunner.attention_tp_size), [`num_attn_heads`](../model_executor/model_runner.md#MockModelRunner.num_attn_heads), [`num_kv_heads`](../model_executor/model_runner.md#MockModelRunner.num_kv_heads)

### `ModelConfig`
- def: [`python/sgl_jax/srt/configs/model_config.py:69`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L69) — documented in [python-sgl_jax-srt-configs-model_config](../../../../../concepts/python-sgl_jax-srt-configs-model_config.md)
- signature: `class ModelConfig:`
- members:
  - `_apply_model_specific_config(self)` — [`L521`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L521) — Invoke the model class's optional `patch_model_config` hook so model
  - `_resolve_quantization_config(self)` — [`L318`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L318) — Resolve and unify quantization config from multiple sources. — documented in [python-sgl_jax-srt-configs-model_config](../../../../../concepts/python-sgl_jax-srt-configs-model_config.md)
  - `configure_for_tensor_parallel(self, tensor_parallel_size: int)` — [`L652`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L652) — Configure model config for tensor parallel execution with KV head replication.
  - `from_server_args(server_args: ServerArgs, model_path: str = None, model_revision: str = None, **kwargs)` — [`L541`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L541) — documented in [python-sgl_jax-srt-configs-model_config](../../../../../concepts/python-sgl_jax-srt-configs-model_config.md)
  - `get_hf_eos_token_id(self)` — [`L828`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L828)
  - `get_hybrid_layer_counts(self)` — [`L706`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L706) — Resolves the number of sliding window (SWA) and full attention layers.
  - `get_kv_padding_strategy(self)` — [`L748`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L748) — Returns the padding strategy for KV heads.
  - `get_num_kv_head_replicas(self, tensor_parallel_size: int)` — [`L633`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L633) — Returns the number of replicas for each original KV head.
  - `get_num_kv_heads(self, tensor_parallel_size)` — [`L616`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L616) — Returns the number of KV heads per TP size.
  - `get_original_kv_head_id(self, tp_rank: int, tensor_parallel_size: int)` — [`L695`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L695) — Determine which original KV head this device should use.
  - `get_swa_weight_params(self)` — [`L726`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L726) — Retrieves head dimensions, original checkpoint head counts, and target sharded head boundaries required for lazy weight loader tensor replication.
  - `get_total_num_kv_heads(self)` — [`L568`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L568) — Returns the total number of KV heads (original, not replicated).
  - `get_total_num_kv_heads_with_replication(self, tensor_parallel_size: int)` — [`L641`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L641) — Returns the total number of KV heads after replication.
  - `is_dynamic_fp8_act(cfg)` — [`L403`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L403)
  - `is_gqa_model(self)` — [`L702`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L702) — Returns True if this is a Grouped Query Attention model.
  - `log_kv_heads_info(self, tensor_parallel_size: int)` — [`L759`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L759) — Log KV heads configuration information during initialization.
  - `maybe_pull_model_tokenizer_from_remote(self)` — [`L846`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L846) — Pull the model config files to a temporary
  - `needs_kv_head_replication(self, tensor_parallel_size: int)` — [`L623`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L623) — Returns True if KV heads need to be replicated across devices.
  - `validate_tensor_parallel_config(self, tensor_parallel_size: int)` — [`L789`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L789) — Validate tensor parallel configuration constraints.
  - `attention_arch` — [`L261`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L261)
  - `context_len` — [`L243`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L243)
  - `dtype` — [`L221`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L221)
  - `dtype_config` — [`L224`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L224) — documented in [python-sgl_jax-srt-configs-model_config](../../../../../concepts/python-sgl_jax-srt-configs-model_config.md)
  - `ep_size` — [`L100`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L100)
  - `final_logit_softcapping` — [`L276`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L276)
  - `head_dim` — [`L254`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L254) — documented in [python-sgl_jax-srt-configs-model_config](../../../../../concepts/python-sgl_jax-srt-configs-model_config.md)
  - `hf_config` — [`L125`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L125) — documented in [python-sgl_jax-srt-configs-model_config](../../../../../concepts/python-sgl_jax-srt-configs-model_config.md)
  - `hf_eos_token_id` — [`L296`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L296)
  - `hf_generation_config` — [`L172`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L172)
  - `hf_text_config` — [`L179`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L179) — documented in [python-sgl_jax-srt-configs-model_config](../../../../../concepts/python-sgl_jax-srt-configs-model_config.md)
  - `hidden_size` — [`L273`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L273)
  - `image_token_id` — [`L301`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L301)
  - `is_generation` — [`L219`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L219)
  - `is_multimodal` — [`L220`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L220)
  - `model_impl` — [`L93`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L93)
  - `model_override_args` — [`L112`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L112)
  - `model_path` — [`L90`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L90)
  - `model_sub_dir` — [`L91`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L91)
  - `moe_backend` — [`L103`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L103) — documented in [python-sgl_jax-srt-configs-model_config](../../../../../concepts/python-sgl_jax-srt-configs-model_config.md)
  - `num_attention_heads` — [`L264`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L264)
  - `num_hidden_layers` — [`L274`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L274)
  - `num_key_value_heads` — [`L265`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L265)
  - `quantization` — [`L94`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L94)
  - `quantization_config` — [`L97`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L97) — documented in [python-sgl_jax-srt-configs-model_config](../../../../../concepts/python-sgl_jax-srt-configs-model_config.md)
  - `quantization_config_path` — [`L95`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L95)
  - `revision` — [`L92`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L92)
  - `sliding_window` — [`L180`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L180) — documented in [python-sgl_jax-srt-configs-model_config](../../../../../concepts/python-sgl_jax-srt-configs-model_config.md)
  - `v_head_dim` — [`L260`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L260)
  - `vocab_size` — [`L275`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L275)
- protocol/private: `__init__`[`L70`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L70), `_get_hf_quant_config`[`L305`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L305), `_original_hf_num_key_value_heads`[`L664`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L664), `_original_num_key_value_heads`[`L659`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L659), `_original_swa_num_key_value_heads`[`L685`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L685), `_parse_quant_hf_config`[`L799`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L799)
- uses (calls/refs, reference-scoped): [`ServerArgs`](../server_args.md#ServerArgs), [`get_bool_env_var`](../utils/common_utils.md#get_bool_env_var), [`logger`](model_config.md#logger), [`get_config`](../hf_transformers_utils.md#get_config), `architectures`, [`default_dtype`](dtype_config.md#DtypeConfig.default_dtype), [`MoEBackend`](model_config.md#MoEBackend), [`get_num_kv_heads_by_tp`](../utils/jax_utils.md#get_num_kv_heads_by_tp), [`model_path`](../server_args.md#ServerArgs.model_path), [`ModelRegistry`](../models/registry.md#ModelRegistry), [`resolve_model_cls`](../models/registry.md#_ModelRegistry.resolve_model_cls), [`QuantizationConfig`](quantization_config.md#QuantizationConfig), [`_assert_fused_moe_v2_supported`](model_config.md#_assert_fused_moe_v2_supported), [`DtypeConfig`](dtype_config.md#DtypeConfig), [`dtype_config`](../server_args.md#ServerArgs.dtype_config), [`download_from_hf`](../hf_transformers_utils.md#download_from_hf), [`ignored_layers`](quantization_config.md#QuantizationConfig.ignored_layers), [`multimodal`](../server_args.md#ServerArgs.multimodal), [`MockModelConfig`](model_config.md#MockModelConfig), [`ModelImpl`](model_config.md#ModelImpl), [`_get_and_verify_dtype`](model_config.md#_get_and_verify_dtype), [`from_path`](quantization_config.md#QuantizationConfig.from_path), [`is_remote_url`](../utils/common_utils.md#is_remote_url), [`is_static_checkpoint`](quantization_config.md#QuantizationConfig.is_static_checkpoint), [`moe_activation_dtype`](quantization_config.md#QuantizationConfig.moe_activation_dtype), [`moe_weight_dtype`](quantization_config.md#QuantizationConfig.moe_weight_dtype), [`linear_rules`](quantization_config.md#QuantizationConfig.linear_rules), [`weight_block_size`](quantization_config.md#QuantizationConfig.weight_block_size), [`get_context_length`](../hf_transformers_utils.md#get_context_length), [`get_generation_config`](../hf_transformers_utils.md#get_generation_config), [`get_original_kv_head_id`](../utils/jax_utils.md#get_original_kv_head_id), [`moe_backend`](../server_args.md#ServerArgs.moe_backend), [`revision`](../server_args.md#ServerArgs.revision), [`AttentionArch`](model_config.md#AttentionArch), [`FUSED`](model_config.md#MoEBackend.FUSED), [`get_hf_text_config`](../hf_transformers_utils.md#get_hf_text_config), [`json_model_override_args`](../server_args.md#ServerArgs.json_model_override_args), [`quantization`](../server_args.md#ServerArgs.quantization), [`trust_remote_code`](../server_args.md#ServerArgs.trust_remote_code), [`get_num_kv_heads`](model_config.md#MockModelConfig.get_num_kv_heads)  (+14 more)
- used by: [`handle_generate_request`](../managers/scheduler.md#Scheduler.handle_generate_request), `prepare_for_extend`, [`forward_batch_speculative_generation`](../speculative/base_worker.md#BaseSpecWorker.forward_batch_speculative_generation), [`forward_batch_generation`](../managers/tp_worker.md#ModelWorker.forward_batch_generation), [`init_new`](../model_executor/forward_batch_info.md#ForwardBatch.init_new), [`_init_pools`](../model_executor/model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin._init_pools), [`load_weights_from_safetensors`](../utils/weight_utils.md#WeightLoader.load_weights_from_safetensors), [`default_radix_cache_factory`](../mem_cache/registry.md#default_radix_cache_factory), [`add_input_logprob_return_values`](../managers/scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.add_input_logprob_return_values), [`build_kv_cache`](../mem_cache/kv_cache_builder.md#build_kv_cache), [`model`](../model_executor/model_runner.md#ModelRunner.model), `init_new`, [`max_running_requests`](../managers/tp_worker.md#ModelWorker.max_running_requests), [`cur_sampling_info`](../managers/tp_worker_overlap_thread.md#ModelWorkerClient.cur_sampling_info), [`grammar_queue`](../managers/scheduler.md#Scheduler.grammar_queue), [`_get_attention_backend`](../model_executor/model_runner.md#ModelRunner._get_attention_backend), [`compilation_manager`](../managers/tp_worker.md#ModelWorker.compilation_manager), [`load_model`](../model_executor/model_runner.md#ModelRunner.load_model), `prepare_for_idle`, [`_compute_cell_size`](../model_executor/model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin._compute_cell_size), [`_handle_single_weight`](../utils/weight_utils.md#WeightLoader._handle_single_weight), [`set_num_token_hybrid`](../model_executor/model_runner.md#ModelRunner.set_num_token_hybrid), [`max_total_num_tokens`](../managers/tp_worker.md#ModelWorker.max_total_num_tokens), [`num_kv_heads`](../utils/weight_utils.md#WeightLoader.num_kv_heads), [`_get_model`](../model_loader/loader.md#JAXModelLoader._get_model), [`_load_dummy_weights`](../utils/weight_utils.md#WeightLoader._load_dummy_weights), [`_resolve_max_num_reqs`](../model_executor/model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin._resolve_max_num_reqs), [`lora_manager`](../model_executor/model_runner.md#ModelRunner.lora_manager), [`apply_linear_quantization`](../utils/quantization/quantization_utils.md#apply_linear_quantization), [`model_config`](../managers/tp_worker.md#ModelWorker.model_config), [`_apply_kv_head_padding`](../utils/weight_utils.md#WeightLoader._apply_kv_head_padding), [`_spec_multi_layer`](../managers/scheduler.md#Scheduler._spec_multi_layer), [`get_model_architecture`](../model_loader/arch.md#get_model_architecture), [`init_spec_relay_buffers`](../speculative/base_worker.md#BaseSpecWorker.init_spec_relay_buffers), [`token_to_kv_pool`](../model_executor/model_runner.md#MockModelRunner.token_to_kv_pool), [`adjust_layer_num`](../model_executor/model_runner.md#ModelRunner.adjust_layer_num), [`dump_topk_ids`](../managers/tp_worker.md#ModelWorker.dump_topk_ids), [`resolve_transformers_arch`](../model_loader/arch.md#resolve_transformers_arch), [`num_effective_layers`](../model_executor/model_runner.md#ModelRunner.num_effective_layers), [`apply_moe_quantization`](../utils/quantization/quantization_utils.md#apply_moe_quantization)  (+52 more)

### `ModelImpl`  ·  implements/extends Enum, str
- def: [`python/sgl_jax/srt/configs/model_config.py:29`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L29)
- signature: `class ModelImpl(str, Enum):`
- members:
  - `AUTO` — [`L30`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L30)
  - `SGLANG` — [`L31`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L31)
  - `TRANSFORMERS` — [`L32`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L32)
- used by: [`get_model_architecture`](../model_loader/arch.md#get_model_architecture), [`resolve_transformers_arch`](../model_loader/arch.md#resolve_transformers_arch), [`__init__`](model_config.md#ModelConfig.__init__)

## Functions
- `_assert_fused_moe_v2_supported(moe_backend: MoEBackend, architectures: list[str])` — [`L59`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L59)
- `_get_and_verify_dtype(config: PretrainedConfig, dtype: str | jnp.dtype)` — [`L864`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L864)
- `is_generation_model(model_architectures: list[str], is_embedding: bool = False)` — [`L913`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L913)
- `need_attention_mask(model_architectures: list[str], is_embedding: bool = False)` — [`L994`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L994) — Determine if a model needs attention_mask for handling padding tokens.

## Module values
- `ENCODER_ONLY_MODELS` — [`L969`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L969)
- `_FORCED_FUSED_EP_MOE_ARCHS` — [`L56`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L56)
- `_FUSED_MOE_V2_SUPPORTED_ARCHITECTURES` — [`L44`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L44)
- `logger` — [`L21`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L21)
- `multimodal_model_archs` — [`L938`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/model_config.py#L938)

