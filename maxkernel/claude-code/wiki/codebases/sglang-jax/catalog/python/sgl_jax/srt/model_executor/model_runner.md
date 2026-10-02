---
title: 'Module: python/sgl_jax/srt/model_executor/model_runner.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/model_executor/model_runner.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.model_executor.model_runner`/
symbols:
  ModelRunner.model_config: ModelRunner#model_config.
  ModelRunner.model: ModelRunner#model.
  ModelRunner.server_args: ModelRunner#server_args.
  ModelRunner.init_routed_experts_capturer: ModelRunner#init_routed_experts_capturer().
  ModelRunner._get_attention_backend: ModelRunner#_get_attention_backend().
  ModelRunner.load_model: ModelRunner#load_model().
  ModelRunner.set_num_token_hybrid: ModelRunner#set_num_token_hybrid().
  ModelRunner: ModelRunner#
  ModelRunner.model_loader: ModelRunner#model_loader.
  ModelRunner.sampler: ModelRunner#sampler.
  ModelRunner.lora_manager: ModelRunner#lora_manager.
  ModelRunner.max_total_num_tokens: ModelRunner#max_total_num_tokens.
  ModelRunner.forward: ModelRunner#forward().
  ModelRunner.mesh: ModelRunner#mesh.
  ModelRunner._forward_raw: ModelRunner#_forward_raw().
  MockModelRunner.token_to_kv_pool: MockModelRunner#token_to_kv_pool.
  ModelRunner.adjust_layer_num: ModelRunner#adjust_layer_num().
  ModelRunner.num_effective_layers: ModelRunner#num_effective_layers.
  ModelRunner.swa_max_total_num_tokens: ModelRunner#swa_max_total_num_tokens.
  ModelRunner.attention_tp_size: ModelRunner#attention_tp_size.
  ModelRunner.attn_backend: ModelRunner#attn_backend.
  ModelRunner._sampler_step: ModelRunner#_sampler_step.
  ModelRunner.get_available_device_memory: ModelRunner#get_available_device_memory().
  ModelRunner._forward: ModelRunner#_forward().
  MockModelRunner.rngs: MockModelRunner#rngs.
  MockModelRunner.mesh: MockModelRunner#mesh.
  ModelRunner.page_size: ModelRunner#page_size.
  ModelRunner.use_mla_backend: ModelRunner#use_mla_backend.
  MockModelRunner.__init__: MockModelRunner#__init__().
  MockModelRunner.req_to_token_pool: MockModelRunner#req_to_token_pool.
  ModelRunner.spec_algorithm: ModelRunner#spec_algorithm.
  ModelRunner.sample: ModelRunner#sample().
  ModelRunner._has_softmax_dtype: ModelRunner#_has_softmax_dtype().
  MockModelRunner.attention_tp_size: MockModelRunner#attention_tp_size.
  ModelRunner.sliding_window_size: ModelRunner#sliding_window_size.
  ModelRunner.__init__: ModelRunner#__init__().
  ModelRunner.req_to_token_pool: ModelRunner#req_to_token_pool.
  ModelRunner.token_to_kv_pool_allocator: ModelRunner#token_to_kv_pool_allocator.
  ModelRunner.full_max_total_num_tokens: ModelRunner#full_max_total_num_tokens.
  ModelRunner.jitted_run_model: ModelRunner#jitted_run_model().
  ModelRunner.end_layer: ModelRunner#end_layer.
  ModelRunner.forward_idle: ModelRunner#forward_idle().
  ModelRunner.is_hybrid: ModelRunner#is_hybrid.
  logger: logger.
  ModelRunner.initialize_jit: ModelRunner#initialize_jit().
  ModelRunner._model_state_def: ModelRunner#_model_state_def.
  ModelRunner.is_hybrid_gdn: ModelRunner#is_hybrid_gdn().
  ModelRunner.is_draft_worker: ModelRunner#is_draft_worker.
  ModelRunner.num_kv_heads: ModelRunner#num_kv_heads.
  ModelRunner.dtype: ModelRunner#dtype.
  ModelRunner.num_attn_heads: ModelRunner#num_attn_heads.
  ModelRunner.max_padding: ModelRunner#max_padding.
  ModelRunner.run_model_wrapper: ModelRunner#run_model_wrapper().
  ModelRunner.jitted_sampler: ModelRunner#jitted_sampler.
  ModelRunner.jitted_compute_logprobs: ModelRunner#jitted_compute_logprobs.
  MockModelRunner.model_config: MockModelRunner#model_config.
  ModelRunner.device: ModelRunner#device.
  MockModelRunner: MockModelRunner#
  ModelRunner.ep_size: ModelRunner#ep_size.
  ModelRunner.use_sort_for_toppk_minp: ModelRunner#use_sort_for_toppk_minp.
  ModelRunner.initialize: ModelRunner#initialize().
  ModelRunner._sampler_base_rng: ModelRunner#_sampler_base_rng.
  ModelRunner.start_layer: ModelRunner#start_layer.
  ModelRunner.compute_logprobs: ModelRunner#compute_logprobs().
  ModelRunner.init_lora_manager: ModelRunner#init_lora_manager().
  MockModelRunner.tp_size: MockModelRunner#tp_size.
  MockModelRunner.dp_size: MockModelRunner#dp_size.
  MockModelRunner.num_kv_heads: MockModelRunner#num_kv_heads.
  MockModelRunner.num_attn_heads: MockModelRunner#num_attn_heads.
  MockModelRunner.max_total_num_tokens: MockModelRunner#max_total_num_tokens.
  ModelRunner.tp_size: ModelRunner#tp_size.
  ModelRunner.dp_size: ModelRunner#dp_size.
  ModelRunner.is_generation: ModelRunner#is_generation.
  ModelRunner.model_state_leaves: ModelRunner#model_state_leaves.
  ModelRunner.forward_pass_id: ModelRunner#forward_pass_id.
  _maybe_apply_recurrent_cow: _maybe_apply_recurrent_cow().
  ModelRunner.mem_fraction_static: ModelRunner#mem_fraction_static.
  ModelRunner.init_attention_backend: ModelRunner#init_attention_backend().
  MockModelRunner.kv_cache_dtype: MockModelRunner#kv_cache_dtype.
  MockModelRunner.page_size: MockModelRunner#page_size.
  ModelRunner.rngs: ModelRunner#rngs.
  ModelRunner._model_def: ModelRunner#_model_def.
  MockModelRunner.server_args: MockModelRunner#server_args.
  MockModelRunner.dtype: MockModelRunner#dtype.
  MockModelRunner.mem_fraction_static: MockModelRunner#mem_fraction_static.
  MockModelRunner.memory_pools: MockModelRunner#memory_pools.
---
# Module: [`python/sgl_jax/srt/model_executor/model_runner.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py)

## Classes
### `MockModelRunner`  ·  implements/extends ModelRunner
- def: [`python/sgl_jax/srt/model_executor/model_runner.py:713`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L713)
- signature: `class MockModelRunner(ModelRunner):`
- members:
  - `attention_tp_size` — [`L724`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L724)
  - `dp_size` — [`L723`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L723)
  - `dtype` — [`L737`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L737)
  - `kv_cache_dtype` — [`L741`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L741)
  - `max_total_num_tokens` — [`L740`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L740)
  - `mem_fraction_static` — [`L738`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L738)
  - `memory_pools` — [`L774`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L774)
  - `mesh` — [`L743`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L743)
  - `model_config` — [`L739`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L739)
  - `num_attn_heads` — [`L728`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L728)
  - `num_kv_heads` — [`L727`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L727)
  - `page_size` — [`L742`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L742)
  - `req_to_token_pool` — [`L757`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L757)
  - `rngs` — [`L729`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L729)
  - `server_args` — [`L721`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L721)
  - `token_to_kv_pool` — [`L763`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L763)
  - `tp_size` — [`L722`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L722)
- protocol/private: `__init__`[`L714`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L714)
- uses (calls/refs, reference-scoped): [`ServerArgs`](../server_args.md#ServerArgs), [`ModelConfig`](../configs/model_config.md#ModelConfig), [`ModelRunner`](model_runner.md#ModelRunner), [`ReqToTokenPool`](../mem_cache/memory_pool.md#ReqToTokenPool), [`dp_size`](../server_args.md#ServerArgs.dp_size), [`num_attention_heads`](../configs/model_config.md#ModelConfig.num_attention_heads), [`MHATokenToKVPool`](../mem_cache/memory_pool.md#MHATokenToKVPool), [`tp_size`](../server_args.md#ServerArgs.tp_size), [`get_total_num_kv_heads_with_replication`](../configs/model_config.md#ModelConfig.get_total_num_kv_heads_with_replication), [`MockModelConfig`](../configs/model_config.md#MockModelConfig), [`_build_non_hybrid_memory_pools`](model_runner_kv_cache_mixin.md#_build_non_hybrid_memory_pools), [`validate_tensor_parallel_config`](../configs/model_config.md#ModelConfig.validate_tensor_parallel_config), [`context_len`](../configs/model_config.md#MockModelConfig.context_len), [`num_kv_heads`](../configs/model_config.md#MockModelConfig.num_kv_heads), [`head_dim`](../configs/model_config.md#MockModelConfig.head_dim), [`num_heads`](../configs/model_config.md#MockModelConfig.num_heads), [`num_hidden_layers`](../configs/model_config.md#MockModelConfig.num_hidden_layers)
- used by: [`ModelRunner`](model_runner.md#ModelRunner), [`get_worker_info`](../managers/tp_worker.md#MockModelWorker.get_worker_info), [`max_running_requests`](../managers/tp_worker.md#MockModelWorker.max_running_requests), [`model_runner`](../managers/tp_worker.md#MockModelWorker.model_runner), [`max_total_num_tokens`](../managers/tp_worker.md#MockModelWorker.max_total_num_tokens), [`get_memory_pool`](../managers/tp_worker.md#MockModelWorker.get_memory_pool)

### `ModelRunner`  ·  implements/extends BaseModelRunner, ModelRunnerKVCacheMixin
- def: [`python/sgl_jax/srt/model_executor/model_runner.py:64`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L64)
- doc: ModelRunner runs the forward passes of the models.
- signature: `class ModelRunner(ModelRunnerKVCacheMixin, BaseModelRunner):`
- members:
  - `adjust_layer_num(self)` — [`L402`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L402) — For hybrid models, compute effective layer count accounting for
  - `compute_logprobs(self, logits, token_ids: jax.Array)` — [`L603`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L603)
  - `forward(self, forward_batch: ForwardBatch, logits_metadata: LogitsMetadata)` — [`L543`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L543)
  - `forward_idle(self, forward_batch: ForwardBatch, logits_metadata: LogitsMetadata)` — [`L536`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L536)
  - `get_available_device_memory(self)` — [`L291`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L291)
  - `init_attention_backend(self)` — [`L434`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L434) — Init attention kernel backend.
  - `init_lora_manager(self)` — [`L695`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L695) — Initialize LoRA manager for LoRA adapter support.
  - `init_routed_experts_capturer(self)` — [`L182`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L182) — documented in [python-sgl_jax-srt-model_executor-model_runner](../../../../../concepts/python-sgl_jax-srt-model_executor-model_runner.md)
  - `initialize(self)` — [`L135`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L135)
  - `initialize_jit(self)` — [`L202`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L202)
  - `is_hybrid_gdn(self)` — [`L428`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L428)
  - `jitted_compute_logprobs(mesh, logits, next_tokens)` — [`L266`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L266)
  - `jitted_run_model(model_def, model_state_def, model_state_leaves, forward_batch, memory_pools, logits_metadata)` — [`L229`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L229)
  - `jitted_sampler(sampler_def, sampler_state_def, sampler_state_leaves, use_sort_for_toppk_minp, rng_step, *args)` — [`L249`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L249)
  - `load_model(self)` — [`L316`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L316) — documented in [python-sgl_jax-srt-configs-model_config](../../../../../concepts/python-sgl_jax-srt-configs-model_config.md)
  - `run_model_wrapper(forward_batch, logits_metadata)` — [`L269`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L269)
  - `sample(self, logits_output: LogitsProcessorOutput, sampling_metadata: SamplingMetadata)` — [`L579`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L579) — Sample and compute logprobs and update logits_output.
  - `set_num_token_hybrid(self)` — [`L606`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L606) — documented in [python-sgl_jax-srt-model_executor-model_runner](../../../../../concepts/python-sgl_jax-srt-model_executor-model_runner.md)
  - `attention_tp_size` — [`L94`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L94)
  - `attn_backend` — [`L436`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L436)
  - `device` — [`L86`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L86)
  - `dp_size` — [`L93`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L93)
  - `dtype` — [`L387`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L387)
  - `end_layer` — [`L389`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L389)
  - `ep_size` — [`L98`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L98)
  - `forward_pass_id` — [`L108`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L108)
  - `full_max_total_num_tokens` — [`L668`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L668)
  - `is_draft_worker` — [`L83`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L83)
  - `is_generation` — [`L100`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L100)
  - `is_hybrid` — [`L104`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L104)
  - `jitted_compute_logprobs` — [`L289`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L289)
  - `jitted_run_model` — [`L279`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L279)
  - `jitted_sampler` — [`L281`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L281)
  - `lora_manager` — [`L699`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L699) — documented in [python-sgl_jax-srt-model_executor-model_runner](../../../../../concepts/python-sgl_jax-srt-model_executor-model_runner.md)
  - `max_padding` — [`L113`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L113)
  - `max_total_num_tokens` — [`L687`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L687)
  - `mem_fraction_static` — [`L85`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L85)
  - `mesh` — [`L87`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L87)
  - `model` — [`L344`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L344) — documented in [python-sgl_jax-srt-model_executor-model_runner](../../../../../concepts/python-sgl_jax-srt-model_executor-model_runner.md)
  - `model_config` — [`L84`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L84) — documented in [python-sgl_jax-srt-model_executor-model_runner](../../../../../concepts/python-sgl_jax-srt-model_executor-model_runner.md)
  - `model_loader` — [`L120`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L120) — documented in [python-sgl_jax-srt-model_executor-model_runner](../../../../../concepts/python-sgl_jax-srt-model_executor-model_runner.md)
  - `model_state_leaves` — [`L205`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L205)
  - `num_attn_heads` — [`L89`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L89)
  - `num_effective_layers` — [`L390`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L390)
  - `num_kv_heads` — [`L95`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L95)
  - `page_size` — [`L101`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L101)
  - `req_to_token_pool` — [`L102`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L102)
  - `rngs` — [`L90`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L90)
  - `sampler` — [`L149`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L149) — documented in [python-sgl_jax-srt-model_executor-model_runner](../../../../../concepts/python-sgl_jax-srt-model_executor-model_runner.md)
  - `server_args` — [`L99`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L99) — documented in [python-sgl_jax-srt-model_executor-model_runner](../../../../../concepts/python-sgl_jax-srt-model_executor-model_runner.md)
  - `sliding_window_size` — [`L386`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L386)
  - `spec_algorithm` — [`L106`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L106)
  - `start_layer` — [`L388`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L388)
  - `swa_max_total_num_tokens` — [`L669`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L669)
  - `token_to_kv_pool_allocator` — [`L103`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L103)
  - `tp_size` — [`L92`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L92)
  - `use_mla_backend` — [`L105`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L105)
  - `use_sort_for_toppk_minp` — [`L111`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L111)
- protocol/private: `__init__`[`L67`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L67), `_forward`[`L512`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L512), `_forward_raw`[`L555`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L555), `_get_attention_backend`[`L438`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L438), `_has_softmax_dtype`[`L439`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L439), `_model_def`[`L206`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L206), `_model_state_def`[`L207`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L207), `_sampler_base_rng`[`L166`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L166), `_sampler_step`[`L167`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L167)
- uses (calls/refs, reference-scoped): [`ServerArgs`](../server_args.md#ServerArgs), [`hf_config`](../configs/model_config.md#ModelConfig.hf_config), [`ModelConfig`](../configs/model_config.md#ModelConfig), [`ForwardBatch`](forward_batch_info.md#ForwardBatch), [`init_memory_pool`](model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin.init_memory_pool), [`precision_tracer`](../precision_tracer.md#precision_tracer), [`is_extend`](forward_batch_info.md#ForwardMode.is_extend), [`attn_backend_wrapper`](../layers/attention/hybrid_linear_attn_backend.md#attn_backend_wrapper), [`hf_text_config`](../configs/model_config.md#ModelConfig.hf_text_config), [`SpeculativeAlgorithm`](../speculative/spec_info.md#SpeculativeAlgorithm), [`get_bool_env_var`](../utils/common_utils.md#get_bool_env_var), [`ReqToTokenPool`](../mem_cache/memory_pool.md#ReqToTokenPool), [`start_batch_trace`](../precision_tracer.md#PrecisionTracer.start_batch_trace), [`LogitsProcessorOutput`](../layers/logits_processor.md#LogitsProcessorOutput), [`BaseTokenToKVPoolAllocator`](../mem_cache/allocator.md#BaseTokenToKVPoolAllocator), [`apply_linear_quantization`](../utils/quantization/quantization_utils.md#apply_linear_quantization), [`forward_mode`](forward_batch_info.md#ForwardBatch.forward_mode), [`moe_backend`](../configs/model_config.md#ModelConfig.moe_backend), [`quantization_config`](../configs/model_config.md#ModelConfig.quantization_config), [`dtype_config`](../configs/model_config.md#ModelConfig.dtype_config), [`is_decode`](forward_batch_info.md#ForwardMode.is_decode), [`get_model_loader`](../model_loader/loader.md#get_model_loader), `global_server_args_dict`, [`apply_moe_quantization`](../utils/quantization/quantization_utils.md#apply_moe_quantization), [`from_string`](../speculative/spec_info.md#SpeculativeAlgorithm.from_string), [`LogitsMetadata`](../layers/logits_processor.md#LogitsMetadata), [`SamplingMetadata`](../sampling/sampling_batch_info.md#SamplingMetadata), [`log_kv_heads_info`](../configs/model_config.md#ModelConfig.log_kv_heads_info), [`get_num_kv_heads`](../configs/model_config.md#ModelConfig.get_num_kv_heads), [`head_dim`](../configs/model_config.md#ModelConfig.head_dim), [`num_hidden_layers`](../configs/model_config.md#ModelConfig.num_hidden_layers), [`load_format`](../configs/load_config.md#LoadConfig.load_format), [`lora_paths`](../server_args.md#ServerArgs.lora_paths), `architectures`, [`dtype`](../configs/model_config.md#ModelConfig.dtype), [`sliding_window`](../configs/model_config.md#ModelConfig.sliding_window), [`LoadConfig`](../configs/load_config.md#LoadConfig), [`config_dict`](../configs/dtype_config.md#DtypeConfig.config_dict), [`page_size`](../server_args.md#ServerArgs.page_size), [`create`](../layers/routed_experts_capturer.md#RoutedExpertsCapturer.create)  (+74 more)
- used by: [`padding_for_decode`](../speculative/eagle_draft_worker.md#EagleDraftWorker.padding_for_decode), [`draft_extend_for_prefill`](../speculative/multi_layer_draft_worker.md#MultiLayerDraftWorker.draft_extend_for_prefill), [`forward_batch_generation`](../managers/tp_worker.md#ModelWorker.forward_batch_generation), [`init_new`](forward_batch_info.md#ForwardBatch.init_new), [`verify`](../speculative/base_worker.md#BaseSpecWorker.verify), [`draft_forward`](../speculative/eagle_draft_worker.md#EagleDraftWorker.draft_forward), [`draft_extend_for_prefill`](../speculative/eagle_draft_worker.md#EagleDraftWorker.draft_extend_for_prefill), [`_init_pools`](model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin._init_pools), [`model_runner`](../managers/tp_worker.md#ModelWorker.model_runner), [`draft_extend_for_decode`](../speculative/eagle_draft_worker.md#EagleDraftWorker.draft_extend_for_decode), [`init_memory_pool`](model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin.init_memory_pool), [`max_running_requests`](../managers/tp_worker.md#ModelWorker.max_running_requests), [`cur_sampling_info`](../managers/tp_worker_overlap_thread.md#ModelWorkerClient.cur_sampling_info), [`_precompile_decode`](compilation_manager.md#CompilationManager._precompile_decode), [`_precompile_extend`](compilation_manager.md#CompilationManager._precompile_extend), [`compilation_manager`](../managers/tp_worker.md#ModelWorker.compilation_manager), [`attn_backend_wrapper`](../layers/attention/hybrid_linear_attn_backend.md#attn_backend_wrapper), [`_worker`](../speculative/eagle_draft_worker.md#EagleDraftWorker._worker), [`_compute_cell_size`](model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin._compute_cell_size), [`max_total_num_tokens`](../managers/tp_worker.md#ModelWorker.max_total_num_tokens), [`get_worker_info`](../managers/tp_worker.md#ModelWorker.get_worker_info), [`_resolve_max_num_reqs`](model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin._resolve_max_num_reqs), [`handle_recurrent_cache`](model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin.handle_recurrent_cache), [`_worker`](../speculative/multi_layer_draft_worker.md#MultiLayerDraftWorker._worker), [`_share_embed_head`](../speculative/eagle_draft_worker.md#EagleDraftWorker._share_embed_head), [`linear_recurrent_config`](model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin.linear_recurrent_config), [`_maybe_wrap_hybrid_kv_pool`](model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin._maybe_wrap_hybrid_kv_pool), [`prepare_lora_batch`](../managers/tp_worker.md#ModelWorker.prepare_lora_batch), [`qwen3_5_hybrid_config`](model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin.qwen3_5_hybrid_config), [`_init_kv_cache_dtype`](model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin._init_kv_cache_dtype), [`_profile_available_bytes`](model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin._profile_available_bytes), [`kimi_linear_config`](model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin.kimi_linear_config), [`lightning_config`](model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin.lightning_config), [`logger`](forward_batch_info.md#logger), [`set_forward_metadata`](../managers/tp_worker.md#ModelWorker.set_forward_metadata), [`_kv_pool_layer_count`](model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin._kv_pool_layer_count), [`get_memory_pool`](../managers/tp_worker.md#ModelWorker.get_memory_pool), [`_apply_token_constraints`](model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin._apply_token_constraints), [`get_memory_pool`](../managers/tp_worker_overlap_thread.md#ModelWorkerClient.get_memory_pool), [`profile_max_num_token`](model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin.profile_max_num_token)  (+9 more)

## Functions
- `_maybe_apply_recurrent_cow(forward_batch, memory_pools)` — [`L49`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L49) — One-shot CoW: clone matched tree slots (src, 0 = skip) into running slots

## Module values
- `logger` — [`L46`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py#L46)

