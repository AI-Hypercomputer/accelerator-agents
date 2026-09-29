---
title: 'Module: python/sgl_jax/srt/managers/tp_worker.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/managers/tp_worker.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.managers.tp_worker`/
symbols:
  ModelWorker.forward_batch_generation: ModelWorker#forward_batch_generation().
  ModelWorker.model_runner: ModelWorker#model_runner.
  ModelWorker.max_running_requests: ModelWorker#max_running_requests.
  ModelWorker.compilation_manager: ModelWorker#compilation_manager.
  ModelWorker._materialize_logprobs_to_host: ModelWorker#_materialize_logprobs_to_host().
  ModelWorker.max_total_num_tokens: ModelWorker#max_total_num_tokens.
  ModelWorker.get_worker_info: ModelWorker#get_worker_info().
  MockModelWorker.get_worker_info: MockModelWorker#get_worker_info().
  ModelWorker._update_grammar_vocab_mask: ModelWorker#_update_grammar_vocab_mask().
  ModelWorker.model_config: ModelWorker#model_config.
  ModelWorker.dump_topk_ids: ModelWorker#dump_topk_ids().
  ModelWorker: ModelWorker#
  ModelWorker.run_precompile: ModelWorker#run_precompile().
  ModelWorker.prepare_lora_batch: ModelWorker#prepare_lora_batch().
  MockModelWorker.max_running_requests: MockModelWorker#max_running_requests.
  MockModelWorker.model_runner: MockModelWorker#model_runner.
  ModelWorker.get_precompile_paddings: ModelWorker#get_precompile_paddings().
  ModelWorker.get_max_padded_size: ModelWorker#get_max_padded_size().
  ModelWorker.set_forward_metadata: ModelWorker#set_forward_metadata().
  ModelWorker.dp_size: ModelWorker#dp_size.
  ModelWorker.mesh: ModelWorker#mesh.
  ModelWorker.get_memory_pool: ModelWorker#get_memory_pool().
  ModelWorker._sync_expert_ids_d2h_thread_func: ModelWorker#_sync_expert_ids_d2h_thread_func().
  ModelWorker.max_req_len: ModelWorker#max_req_len.
  MockModelWorker.forward_batch_generation: MockModelWorker#forward_batch_generation().
  MockModelWorker.model_config: MockModelWorker#model_config.
  MockModelWorker.max_total_num_tokens: MockModelWorker#max_total_num_tokens.
  ModelWorker._sync_experts_ids_d2h: ModelWorker#_sync_experts_ids_d2h().
  save_logits_with_txt: save_logits_with_txt().
  logger: logger.
  ModelWorker.random_seed: ModelWorker#random_seed.
  MockModelWorker.max_req_len: MockModelWorker#max_req_len.
  ModelWorker.speculative_algorithm: ModelWorker#speculative_algorithm.
  MockModelWorker.get_memory_pool: MockModelWorker#get_memory_pool().
  ModelWorker.max_req_input_len: ModelWorker#max_req_input_len.
  MockModelWorker.max_req_input_len: MockModelWorker#max_req_input_len.
  ModelWorker.page_size: ModelWorker#page_size.
  ModelWorker.chunked_prefill_size: ModelWorker#chunked_prefill_size.
  ModelWorker.max_padded_num_tokens: ModelWorker#max_padded_num_tokens.
  ModelWorker.sync_expert_ids_d2h_thread: ModelWorker#sync_expert_ids_d2h_thread.
  ModelWorker.sliding_window_size: ModelWorker#sliding_window_size().
  ModelWorker.is_hybrid: ModelWorker#is_hybrid().
  ModelWorker.get_tokens_per_layer_info: ModelWorker#get_tokens_per_layer_info().
  ModelWorker.get_pad_input_ids_func: ModelWorker#get_pad_input_ids_func().
  ModelWorker.max_prefill_tokens: ModelWorker#max_prefill_tokens.
  ModelWorker.get_model_runner: ModelWorker#get_model_runner().
  ModelWorker.__init__: ModelWorker#__init__().
  ModelWorker.need_prepare_lora_batch: ModelWorker#need_prepare_lora_batch.
  ModelWorker.device: ModelWorker#device.
  MockModelWorker.random_seed: MockModelWorker#random_seed.
  ModelWorker.server_args: ModelWorker#server_args.
  ModelWorker.tp_size: ModelWorker#tp_size.
  MockModelWorker.max_prefill_tokens: MockModelWorker#max_prefill_tokens.
  ModelWorker.gather: ModelWorker#gather().
  ModelWorker.lora_paths: ModelWorker#lora_paths.
  ModelWorker.max_loras_per_batch: ModelWorker#max_loras_per_batch.
  MockModelWorker.__init__: MockModelWorker#__init__().
  MockModelWorker.tp_size: MockModelWorker#tp_size.
  _iter_padded_input_logprob_reqs: _iter_padded_input_logprob_reqs().
  ModelWorker.sync_queue: ModelWorker#sync_queue.
  ModelWorker.worker: ModelWorker#worker.
  ModelWorker.max_padded_batch_size: ModelWorker#max_padded_batch_size.
  ModelWorker.parent_process: ModelWorker#parent_process.
  ModelWorker.precompile_params: ModelWorker#precompile_params.
  MockModelWorker: MockModelWorker#
  MockModelWorker.worker: MockModelWorker#worker.
---
# Module: [`python/sgl_jax/srt/managers/tp_worker.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py)

## Classes
### `MockModelWorker`
- def: [`python/sgl_jax/srt/managers/tp_worker.py:637`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L637)
- doc: A mock tensor parallel model worker.
- signature: `class MockModelWorker:`
- members:
  - `forward_batch_generation(self, _model_worker_batch: ModelWorkerBatch, _launch_done: threading.Event | None = None, _skip_sample: bool = False, _sampling_metadata: SamplingMetadata | None = None)` — [`L707`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L707)
  - `get_memory_pool(self)` — [`L704`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L704)
  - `get_worker_info(self)` — [`L690`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L690)
  - `max_prefill_tokens` — [`L666`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L666)
  - `max_req_input_len` — [`L684`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L684)
  - `max_req_len` — [`L680`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L680)
  - `max_running_requests` — [`L667`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L667)
  - `max_total_num_tokens` — [`L665`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L665)
  - `model_config` — [`L648`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L648)
  - `model_runner` — [`L658`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L658)
  - `random_seed` — [`L655`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L655)
  - `tp_size` — [`L645`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L645)
  - `worker` — [`L688`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L688)
- protocol/private: `__init__`[`L640`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L640)
- uses (calls/refs, reference-scoped): [`ServerArgs`](../server_args.md#ServerArgs), `ModelWorkerBatch`, [`ModelConfig`](../configs/model_config.md#ModelConfig), [`from_server_args`](../configs/model_config.md#ModelConfig.from_server_args), [`LogitsProcessorOutput`](../layers/logits_processor.md#LogitsProcessorOutput), [`next_token_logits`](../layers/logits_processor.md#LogitsProcessorOutput.next_token_logits), [`token_to_kv_pool`](../model_executor/model_runner.md#MockModelRunner.token_to_kv_pool), `global_server_args_dict`, [`SamplingMetadata`](../sampling/sampling_batch_info.md#SamplingMetadata), [`size`](../mem_cache/memory_pool.md#KVCache.size), [`context_len`](../configs/model_config.md#ModelConfig.context_len), [`req_to_token_pool`](../model_executor/model_runner.md#MockModelRunner.req_to_token_pool), [`dp_size`](../server_args.md#ServerArgs.dp_size), [`max_running_requests`](../server_args.md#ServerArgs.max_running_requests), [`model_path`](../server_args.md#ServerArgs.model_path), [`size`](../mem_cache/memory_pool.md#ReqToTokenPool.size), [`random_seed`](../server_args.md#ServerArgs.random_seed), [`tp_size`](../server_args.md#ServerArgs.tp_size), [`MockModelRunner`](../model_executor/model_runner.md#MockModelRunner), [`max_context_len`](../mem_cache/memory_pool.md#ReqToTokenPool.max_context_len), [`max_prefill_tokens`](../server_args.md#ServerArgs.max_prefill_tokens), [`max_total_num_tokens`](../model_executor/model_runner.md#MockModelRunner.max_total_num_tokens)

### `ModelWorker`
- def: [`python/sgl_jax/srt/managers/tp_worker.py:59`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L59)
- doc: A tensor parallel model worker.
- signature: `class ModelWorker:`
- members:
  - `_materialize_logprobs_to_host(self, logits_output: LogitsProcessorOutput, model_worker_batch: ModelWorkerBatch, selector: np.ndarray)` — [`L503`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L503) — Reorder + per-req split logprob tensors from device to host lists.
  - `dump_topk_ids(self, layers_topk_ids: list[jax.Array], model_worker_batch: ModelWorkerBatch)` — [`L594`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L594)
  - `forward_batch_generation(self, model_worker_batch: ModelWorkerBatch, launch_done: threading.Event | None = None, skip_sample: bool = False, sampling_metadata: SamplingMetadata = None, forward_metadata=None)` — [`L412`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L412) — documented in [python-sgl_jax-srt-model_executor-forward_batch_info](../../../../../concepts/python-sgl_jax-srt-model_executor-forward_batch_info.md)
  - `gather(arr, *, as_float=False)` — [`L521`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L521)
  - `get_max_padded_size(self)` — [`L305`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L305) — Calculate the max padded batch size and token nums.
  - `get_memory_pool(self)` — [`L388`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L388)
  - `get_model_runner(self)` — [`L339`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L339)
  - `get_pad_input_ids_func(self)` — [`L385`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L385)
  - `get_precompile_paddings(self)` — [`L332`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L332)
  - `get_tokens_per_layer_info(self)` — [`L371`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L371)
  - `get_worker_info(self)` — [`L348`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L348)
  - `is_hybrid(self)` — [`L368`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L368)
  - `prepare_lora_batch(self, model_worker_batch: ModelWorkerBatch)` — [`L342`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L342)
  - `run_precompile(self, future_token_ids_map=None)` — [`L290`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L290)
  - `set_forward_metadata(self, model_worker_batch: ModelWorkerBatch)` — [`L300`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L300)
  - `sliding_window_size(self)` — [`L364`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L364)
  - `chunked_prefill_size` — [`L120`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L120)
  - `compilation_manager` — [`L249`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L249)
  - `device` — [`L138`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L138)
  - `dp_size` — [`L73`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L73)
  - `lora_paths` — [`L85`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L85)
  - `max_loras_per_batch` — [`L86`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L86)
  - `max_padded_batch_size` — [`L244`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L244)
  - `max_padded_num_tokens` — [`L244`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L244)
  - `max_prefill_tokens` — [`L119`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L119)
  - `max_req_input_len` — [`L234`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L234)
  - `max_req_len` — [`L230`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L230)
  - `max_running_requests` — [`L158`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L158)
  - `max_total_num_tokens` — [`L141`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L141)
  - `mesh` — [`L104`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L104)
  - `model_config` — [`L89`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L89)
  - `model_runner` — [`L123`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L123)
  - `need_prepare_lora_batch` — [`L108`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L108)
  - `page_size` — [`L105`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L105)
  - `parent_process` — [`L269`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L269)
  - `precompile_params` — [`L82`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L82)
  - `random_seed` — [`L115`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L115)
  - `server_args` — [`L79`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L79)
  - `speculative_algorithm` — [`L76`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L76)
  - `sync_expert_ids_d2h_thread` — [`L271`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L271)
  - `sync_queue` — [`L270`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L270)
  - `tp_size` — [`L72`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L72)
  - `worker` — [`L242`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L242)
- protocol/private: `__init__`[`L62`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L62), `_sync_expert_ids_d2h_thread_func`[`L277`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L277), `_sync_experts_ids_d2h`[`L285`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L285), `_update_grammar_vocab_mask`[`L394`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L394)
- uses (calls/refs, reference-scoped): [`ServerArgs`](../server_args.md#ServerArgs), [`init_new`](../model_executor/forward_batch_info.md#ForwardBatch.init_new), `ModelWorkerBatch`, [`model_config`](../model_executor/model_runner.md#ModelRunner.model_config), [`model`](../model_executor/model_runner.md#ModelRunner.model), [`from_model_worker_batch`](../sampling/sampling_batch_info.md#SamplingMetadata.from_model_worker_batch), [`ModelConfig`](../configs/model_config.md#ModelConfig), [`ForwardBatch`](../model_executor/forward_batch_info.md#ForwardBatch), `forward_mode`, [`is_extend`](../model_executor/forward_batch_info.md#ForwardMode.is_extend), [`from_model_worker_batch`](../layers/logits_processor.md#LogitsMetadata.from_model_worker_batch), `sampling_info`, [`from_server_args`](../configs/model_config.md#ModelConfig.from_server_args), [`hf_text_config`](../configs/model_config.md#ModelConfig.hf_text_config), [`SpeculativeAlgorithm`](../speculative/spec_info.md#SpeculativeAlgorithm), [`ModelRunner`](../model_executor/model_runner.md#ModelRunner), [`ReqToTokenPool`](../mem_cache/memory_pool.md#ReqToTokenPool), [`LogitsProcessorOutput`](../layers/logits_processor.md#LogitsProcessorOutput), [`lora_manager`](../model_executor/model_runner.md#ModelRunner.lora_manager), [`forward_mode`](../model_executor/forward_batch_info.md#ForwardBatch.forward_mode), [`max_total_num_tokens`](../model_executor/model_runner.md#ModelRunner.max_total_num_tokens), [`forward`](../model_executor/model_runner.md#ModelRunner.forward), [`next_token_logits`](../layers/logits_processor.md#LogitsProcessorOutput.next_token_logits), [`moe_backend`](../configs/model_config.md#ModelConfig.moe_backend), [`linear_recurrent_config`](../model_executor/model_runner_kv_cache_mixin.md#ModelRunnerKVCacheMixin.linear_recurrent_config), [`is_decode`](../model_executor/forward_batch_info.md#ForwardMode.is_decode), `global_server_args_dict`, [`from_string`](../speculative/spec_info.md#SpeculativeAlgorithm.from_string), [`prepare_lora_batch`](../lora/lora_manager.md#LoRAManager.prepare_lora_batch), [`attn_backend`](../model_executor/model_runner.md#ModelRunner.attn_backend), [`LogitsMetadata`](../layers/logits_processor.md#LogitsMetadata), [`SamplingMetadata`](../sampling/sampling_batch_info.md#SamplingMetadata), [`get_forward_metadata`](../layers/attention/base_attn_backend.md#AttentionBackend.get_forward_metadata), [`next_token_logprobs`](../layers/logits_processor.md#LogitsProcessorOutput.next_token_logprobs), `temperatures`, `logits_indices_selector`, [`lora_paths`](../server_args.md#ServerArgs.lora_paths), `real_bs`, [`get_max_running_reqests`](../layers/attention/base_attn_backend.md#AttentionBackend.get_max_running_reqests), `get_original_input_len`  (+65 more)
- used by: [`padding_for_decode`](../speculative/eagle_draft_worker.md#EagleDraftWorker.padding_for_decode), [`forward_batch_speculative_generation`](../speculative/base_worker.md#BaseSpecWorker.forward_batch_speculative_generation), [`draft_extend_for_prefill`](../speculative/multi_layer_draft_worker.md#MultiLayerDraftWorker.draft_extend_for_prefill), [`verify`](../speculative/base_worker.md#BaseSpecWorker.verify), [`cur_sampling_info`](tp_worker_overlap_thread.md#ModelWorkerClient.cur_sampling_info), [`prepare_for_verify`](../speculative/eagle_util.md#EagleVerifyInput.prepare_for_verify), [`_worker`](../speculative/eagle_draft_worker.md#EagleDraftWorker._worker), [`mesh`](scheduler.md#Scheduler.mesh), [`worker`](tp_worker_overlap_thread.md#ModelWorkerClient.worker), [`draft_model_runner`](../speculative/eagle_draft_worker.md#EagleDraftWorker.draft_model_runner), [`_worker`](../speculative/multi_layer_draft_worker.md#MultiLayerDraftWorker._worker), [`init_spec_relay_buffers`](../speculative/base_worker.md#BaseSpecWorker.init_spec_relay_buffers), [`_share_embed_head`](../speculative/eagle_draft_worker.md#EagleDraftWorker._share_embed_head), [`forward_thread_func_`](tp_worker_overlap_thread.md#ModelWorkerClient.forward_thread_func_), [`forward_target_extend`](../speculative/base_worker.md#BaseSpecWorker.forward_target_extend), [`_workers`](../speculative/multi_layer_draft_worker.md#MultiLayerDraftWorker._workers), [`mesh`](../speculative/eagle_draft_worker.md#EagleDraftWorker.mesh), [`num_mtp_layers`](../speculative/multi_layer_draft_worker.md#MultiLayerDraftWorker.num_mtp_layers), [`target_worker`](../speculative/base_worker.md#BaseSpecWorker.target_worker), [`get_memory_pool`](tp_worker_overlap_thread.md#ModelWorkerClient.get_memory_pool), [`mesh`](../speculative/base_worker.md#BaseSpecWorker.mesh), [`hot_token_ids`](../speculative/eagle_draft_worker.md#EagleDraftWorker.hot_token_ids), [`__init__`](../speculative/eagle_worker.md#EAGLEWorker.__init__), [`_share_embed_head_one`](../speculative/multi_layer_draft_worker.md#MultiLayerDraftWorker._share_embed_head_one), [`run_precompile`](tp_worker_overlap_thread.md#ModelWorkerClient.run_precompile), [`__init__`](../speculative/multi_layer_eagle_worker.md#MultiLayerEAGLEWorker.__init__), [`max_running_requests`](tp_worker_overlap_thread.md#ModelWorkerClient.max_running_requests), [`__init__`](../speculative/base_worker.md#BaseSpecWorker.__init__), [`get_precompile_paddings`](tp_worker_overlap_thread.md#ModelWorkerClient.get_precompile_paddings), [`get_tokens_per_layer_info`](tp_worker_overlap_thread.md#ModelWorkerClient.get_tokens_per_layer_info), [`get_worker_info`](tp_worker_overlap_thread.md#ModelWorkerClient.get_worker_info), [`is_hybrid`](tp_worker_overlap_thread.md#ModelWorkerClient.is_hybrid), [`model_config`](tp_worker_overlap_thread.md#ModelWorkerClient.model_config), [`sliding_window_size`](tp_worker_overlap_thread.md#ModelWorkerClient.sliding_window_size), [`compilation_manager`](../speculative/eagle_draft_worker.md#EagleDraftWorker.compilation_manager), [`device`](tp_worker_overlap_thread.md#ModelWorkerClient.device), [`draft_model_runner`](../speculative/multi_layer_draft_worker.md#MultiLayerDraftWorker.draft_model_runner), [`get_kv_cache`](tp_worker_overlap_thread.md#ModelWorkerClient.get_kv_cache), [`get_max_padded_size`](tp_worker_overlap_thread.md#ModelWorkerClient.get_max_padded_size), [`get_max_padded_size`](../speculative/eagle_draft_worker.md#EagleDraftWorker.get_max_padded_size)  (+12 more)

## Functions
- `_iter_padded_input_logprob_reqs(model_worker_batch, padded_rows: int)` — [`L34`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L34) — Yield (slot, row_offset, pruned_len) per req, in DP-rank-then-req order.
- `save_logits_with_txt(arr: jax.Array, file_info: str, forward_mode: ForwardMode)` — [`L722`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L722)

## Module values
- `logger` — [`L31`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py#L31)

