---
title: 'Module: python/sgl_jax/srt/managers/tp_worker_overlap_thread.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/managers/tp_worker_overlap_thread.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.managers.tp_worker_overlap_thread`/
symbols:
  ModelWorkerClient.cur_sampling_info: ModelWorkerClient#cur_sampling_info.
  ModelWorkerClient.worker: ModelWorkerClient#worker.
  ModelWorkerClient.forward_thread_func_: ModelWorkerClient#forward_thread_func_().
  ModelWorkerClient.forward_batch_generation: ModelWorkerClient#forward_batch_generation().
  ModelWorkerClient.get_memory_pool: ModelWorkerClient#get_memory_pool().
  ModelWorkerClient.forward_thread_func: ModelWorkerClient#forward_thread_func().
  ModelWorkerClient.future_token_ids_map: ModelWorkerClient#future_token_ids_map.
  ModelWorkerClient.forward_thread: ModelWorkerClient#forward_thread.
  ModelWorkerClient.run_precompile: ModelWorkerClient#run_precompile().
  ModelWorkerClient.max_running_requests: ModelWorkerClient#max_running_requests.
  ModelWorkerClient.model_config: ModelWorkerClient#model_config().
  ModelWorkerClient.get_worker_info: ModelWorkerClient#get_worker_info().
  ModelWorkerClient.get_precompile_paddings: ModelWorkerClient#get_precompile_paddings().
  ModelWorkerClient.sliding_window_size: ModelWorkerClient#sliding_window_size().
  ModelWorkerClient.is_hybrid: ModelWorkerClient#is_hybrid().
  ModelWorkerClient.get_tokens_per_layer_info: ModelWorkerClient#get_tokens_per_layer_info().
  ModelWorkerClient.mesh: ModelWorkerClient#mesh.
  ModelWorkerClient.resolve_last_batch_result: ModelWorkerClient#resolve_last_batch_result().
  ModelWorkerClient.device: ModelWorkerClient#device.
  ModelWorkerClient.future_token_ids_ct: ModelWorkerClient#future_token_ids_ct.
  ModelWorkerClient.get_model_runner: ModelWorkerClient#get_model_runner().
  ModelWorkerClient.model_runner: ModelWorkerClient#model_runner().
  ModelWorkerClient.get_pad_input_ids_func: ModelWorkerClient#get_pad_input_ids_func().
  ModelWorkerClient.get_kv_cache: ModelWorkerClient#get_kv_cache().
  ModelWorkerClient.get_max_padded_size: ModelWorkerClient#get_max_padded_size().
  ModelWorkerClient.page_size: ModelWorkerClient#page_size().
  ModelWorkerClient.future_token_ids_limit: ModelWorkerClient#future_token_ids_limit.
  ModelWorkerClient.input_queue: ModelWorkerClient#input_queue.
  ModelWorkerClient.__init__: ModelWorkerClient#__init__().
  ModelWorkerClient.__delete__: ModelWorkerClient#__delete__().
  ModelWorkerClient: ModelWorkerClient#
  ModelWorkerClient.output_queue: ModelWorkerClient#output_queue.
  logger: logger.
  ModelWorkerClient.parent_process: ModelWorkerClient#parent_process.
  ModelWorkerClient.async_gather_fn: ModelWorkerClient#async_gather_fn.
---
# Module: [`python/sgl_jax/srt/managers/tp_worker_overlap_thread.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py)

## Classes
### `ModelWorkerClient`
- def: [`python/sgl_jax/srt/managers/tp_worker_overlap_thread.py:26`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L26)
- doc: A tensor parallel model worker.
- signature: `class ModelWorkerClient:`
- members:
  - `forward_batch_generation(self, model_worker_batch: ModelWorkerBatch, sampling_metadata: SamplingMetadata = None)` — [`L184`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L184)
  - `forward_thread_func(self)` — [`L96`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L96)
  - `forward_thread_func_(self)` — [`L104`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L104)
  - `get_kv_cache(self)` — [`L87`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L87)
  - `get_max_padded_size(self)` — [`L90`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L90)
  - `get_memory_pool(self)` — [`L81`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L81)
  - `get_model_runner(self)` — [`L64`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L64)
  - `get_pad_input_ids_func(self)` — [`L78`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L78)
  - `get_precompile_paddings(self)` — [`L93`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L93)
  - `get_tokens_per_layer_info(self)` — [`L255`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L255)
  - `get_worker_info(self)` — [`L75`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L75)
  - `is_hybrid(self)` — [`L252`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L252)
  - `model_config(self)` — [`L68`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L68)
  - `model_runner(self)` — [`L72`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L72)
  - `page_size(self)` — [`L244`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L244)
  - `resolve_last_batch_result(self, launch_done: threading.Event | None = None)` — [`L141`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L141) — This function is called to resolve the last batch result and
  - `run_precompile(self)` — [`L240`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L240)
  - `sliding_window_size(self)` — [`L248`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L248)
  - `async_gather_fn` — [`L62`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L62)
  - `cur_sampling_info` — [`L192`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L192)
  - `device` — [`L42`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L42)
  - `forward_thread` — [`L55`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L55)
  - `future_token_ids_ct` — [`L45`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L45)
  - `future_token_ids_limit` — [`L46`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L46)
  - `future_token_ids_map` — [`L47`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L47)
  - `input_queue` — [`L52`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L52)
  - `max_running_requests` — [`L41`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L41)
  - `mesh` — [`L48`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L48)
  - `output_queue` — [`L53`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L53)
  - `parent_process` — [`L60`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L60)
  - `worker` — [`L37`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L37)
- protocol/private: `__delete__`[`L258`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L258), `__init__`[`L29`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L29)
- uses (calls/refs, reference-scoped): [`ServerArgs`](../server_args.md#ServerArgs), [`forward_batch_generation`](tp_worker.md#ModelWorker.forward_batch_generation), [`init_new`](../model_executor/forward_batch_info.md#ForwardBatch.init_new), `ModelWorkerBatch`, [`model_runner`](tp_worker.md#ModelWorker.model_runner), `seq_lens`, [`from_model_worker_batch`](../sampling/sampling_batch_info.md#SamplingMetadata.from_model_worker_batch), [`ForwardBatch`](../model_executor/forward_batch_info.md#ForwardBatch), [`max_running_requests`](tp_worker.md#ModelWorker.max_running_requests), `sampling_info`, [`get_worker_info`](tp_worker.md#ModelWorker.get_worker_info), [`model_config`](tp_worker.md#ModelWorker.model_config), [`ModelWorker`](tp_worker.md#ModelWorker), [`run_precompile`](tp_worker.md#ModelWorker.run_precompile), [`prepare_lora_batch`](tp_worker.md#ModelWorker.prepare_lora_batch), [`attn_backend`](../model_executor/model_runner.md#ModelRunner.attn_backend), [`SamplingMetadata`](../sampling/sampling_batch_info.md#SamplingMetadata), [`get_forward_metadata`](../layers/attention/base_attn_backend.md#AttentionBackend.get_forward_metadata), [`get_precompile_paddings`](tp_worker.md#ModelWorker.get_precompile_paddings), [`vocab_size`](../configs/model_config.md#ModelConfig.vocab_size), [`get_max_padded_size`](tp_worker.md#ModelWorker.get_max_padded_size), [`enable_lora`](../server_args.md#ServerArgs.enable_lora), `forward_batch`, [`req_to_token_pool`](../model_executor/model_runner.md#ModelRunner.req_to_token_pool), [`token_to_kv_pool_allocator`](../model_executor/model_runner.md#ModelRunner.token_to_kv_pool_allocator), [`get_exception_traceback`](../../utils.md#get_exception_traceback), [`page_size`](tp_worker.md#ModelWorker.page_size), [`get_pad_input_ids_func`](tp_worker.md#ModelWorker.get_pad_input_ids_func), [`get_tokens_per_layer_info`](tp_worker.md#ModelWorker.get_tokens_per_layer_info), [`is_hybrid`](tp_worker.md#ModelWorker.is_hybrid), [`resolve_future_token_ids`](utils.md#resolve_future_token_ids), [`sliding_window_size`](tp_worker.md#ModelWorker.sliding_window_size), [`get_model_runner`](tp_worker.md#ModelWorker.get_model_runner), [`device`](tp_worker.md#ModelWorker.device), [`need_prepare_lora_batch`](tp_worker.md#ModelWorker.need_prepare_lora_batch), [`server_args`](tp_worker.md#ModelWorker.server_args), [`set_future_token_ids`](utils.md#set_future_token_ids), `update_penalties`, [`enable_single_process`](../server_args.md#ServerArgs.enable_single_process), [`logger`](tp_worker_overlap_thread.md#logger)
- used by: [`process_batch_result_decode`](scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.process_batch_result_decode), [`process_batch_result_prefill`](scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.process_batch_result_prefill), [`run_batch`](scheduler.md#Scheduler.run_batch), [`_request_dispatcher`](scheduler.md#Scheduler._request_dispatcher), [`mesh`](scheduler.md#Scheduler.mesh), [`process_batch_result`](scheduler.md#Scheduler.process_batch_result), [`token_to_kv_pool_allocator`](scheduler.md#Scheduler.token_to_kv_pool_allocator), [`_spec_multi_layer`](scheduler.md#Scheduler._spec_multi_layer), [`random_seed`](scheduler.md#Scheduler.random_seed), [`is_hybrid`](scheduler.md#Scheduler.is_hybrid), [`sliding_window_size`](scheduler.md#Scheduler.sliding_window_size), [`swa_tokens_per_layer`](scheduler.md#Scheduler.swa_tokens_per_layer)

## Module values
- `logger` — [`L23`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tp_worker_overlap_thread.py#L23)

