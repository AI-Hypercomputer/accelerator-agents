---
title: 'Module: python/sgl_jax/srt/precision_tracer.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/precision_tracer.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.precision_tracer`/
symbols:
  precision_tracer: precision_tracer.
  PrecisionTracer.start_trace: PrecisionTracer#start_trace().
  PrecisionTracer.record: PrecisionTracer#record().
  PrecisionTracer.start_batch_trace: PrecisionTracer#start_batch_trace().
  PrecisionTracerRecord.to_dict: PrecisionTracerRecord#to_dict().
  PrecisionTracer.stop_trace: PrecisionTracer#stop_trace().
  PrecisionTracer._calculate_tensor_pricision_info: PrecisionTracer#_calculate_tensor_pricision_info().
  logger: logger.
  PrecisionTracer._records: PrecisionTracer#_records.
  PrecisionTracer._trace_active: PrecisionTracer#_trace_active.
  PrecisionTracer.set_end_time_and_duration: PrecisionTracer#set_end_time_and_duration().
  PrecisionTracer.lock: PrecisionTracer#lock.
  PrecisionTracer.get_trace_active: PrecisionTracer#get_trace_active().
  PrecisionTracer.trace_callback: PrecisionTracer#trace_callback().
  PrecisionTracerRequestMetadata.to_dict: PrecisionTracerRequestMetadata#to_dict().
  PrecisionTracer._batch_requests_mapping: PrecisionTracer#_batch_requests_mapping.
  PrecisionTracer.set_request_status_to_completed: PrecisionTracer#set_request_status_to_completed().
  PrecisionTracer.add_request_to_batch_requests_mapping: PrecisionTracer#add_request_to_batch_requests_mapping().
  PrecisionTracer._trace_output_file: PrecisionTracer#_trace_output_file.
  PrecisionTracer.get_max_requests: PrecisionTracer#get_max_requests().
  PrecisionTracer.get_completed_requests_count: PrecisionTracer#get_completed_requests_count().
  PrecisionTracerRequestMetadata: PrecisionTracerRequestMetadata#
  PrecisionTracer._token_counters: PrecisionTracer#_token_counters.
  PrecisionTracerRequestMetadata.request_id: PrecisionTracerRequestMetadata#request_id.
  PrecisionTracer.get_request_counter: PrecisionTracer#get_request_counter().
  PrecisionTracer.add_request_counter: PrecisionTracer#add_request_counter().
  PrecisionTracer.add_completed_requests_count: PrecisionTracer#add_completed_requests_count().
  PrecisionTracer._last_forward_pass_id: PrecisionTracer#_last_forward_pass_id.
  PrecisionTracer.set_enable_precision_tracer: PrecisionTracer#set_enable_precision_tracer().
  PrecisionTracer.set_current_forward_pass_id: PrecisionTracer#set_current_forward_pass_id().
  PrecisionTracer._parse_layer_and_module: PrecisionTracer#_parse_layer_and_module().
  PrecisionTracer._verbose_logging_console: PrecisionTracer#_verbose_logging_console().
  PrecisionTracer._completed_requests_count: PrecisionTracer#_completed_requests_count.
  PrecisionTracerRequestMetadata.forward_mode: PrecisionTracerRequestMetadata#forward_mode.
  PrecisionTracer.jit_pure_callback_record: PrecisionTracer#jit_pure_callback_record().
  PrecisionTracer._request_counter: PrecisionTracer#_request_counter.
  PrecisionTracer._max_requests: PrecisionTracer#_max_requests.
  PrecisionTracer._save_tensor: PrecisionTracer#_save_tensor.
  PrecisionTracer._enable_precision_tracer: PrecisionTracer#_enable_precision_tracer.
  PrecisionTracer._current_batch_id: PrecisionTracer#_current_batch_id.
  PrecisionTracer._current_forward_pass_id: PrecisionTracer#_current_forward_pass_id.
  PrecisionTracer._compute_stats: PrecisionTracer#_compute_stats().
  PrecisionTracerRecord: PrecisionTracerRecord#
  PrecisionTracerRecord.end_time: PrecisionTracerRecord#end_time.
  TensorJSONEncoder.default: TensorJSONEncoder#default().
  PrecisionTracerRequestMetadata.input_hash: PrecisionTracerRequestMetadata#input_hash.
  PrecisionTracerRequestMetadata.input_len: PrecisionTracerRequestMetadata#input_len.
  PrecisionTracerRecord.start_time: PrecisionTracerRecord#start_time.
  PrecisionTracerRecord.duration: PrecisionTracerRecord#duration.
  PrecisionTracerRecord.precision_records: PrecisionTracerRecord#precision_records.
  PrecisionTracerRecord.status: PrecisionTracerRecord#status.
  PrecisionTracer._verbose_logging: PrecisionTracer#_verbose_logging.
  _is_jax_array: _is_jax_array().
  TensorJSONEncoder: TensorJSONEncoder#
  PrecisionTracerRecord.bid: PrecisionTracerRecord#bid.
  PrecisionTracerRecord.request_id: PrecisionTracerRecord#request_id.
  PrecisionTracerRecord.request_idx: PrecisionTracerRecord#request_idx.
  PrecisionTracerRecord.content_hash: PrecisionTracerRecord#content_hash.
  PrecisionTracerRecord.process_id: PrecisionTracerRecord#process_id.
  PrecisionTracer: PrecisionTracer#
  PrecisionTracer._traced_stats: PrecisionTracer#_traced_stats().
  PrecisionTracerRequestMetadata.__init__: PrecisionTracerRequestMetadata#__init__().
  PrecisionTracerRequestMetadata.request_input_ids: PrecisionTracerRequestMetadata#request_input_ids.
  PrecisionTracerRecord.__init__: PrecisionTracerRecord#__init__().
  PrecisionTracer.__init__: PrecisionTracer#__init__().
---
# Module: [`python/sgl_jax/srt/precision_tracer.py`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py)

## Classes
### `PrecisionTracer`
- def: [`python/sgl_jax/srt/precision_tracer.py:101`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L101)
- signature: `class PrecisionTracer:`
- members:
  - `add_completed_requests_count(self)` — [`L150`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L150)
  - `add_request_counter(self)` — [`L142`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L142)
  - `add_request_to_batch_requests_mapping(self, batch_id: int, request_metadata: PrecisionTracerRequestMetadata)` — [`L165`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L165)
  - `get_completed_requests_count(self)` — [`L146`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L146)
  - `get_max_requests(self)` — [`L134`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L134)
  - `get_request_counter(self)` — [`L138`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L138)
  - `get_trace_active(self)` — [`L130`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L130)
  - `jit_pure_callback_record(self, tensor: Any, name: str, stage: str, layer_id: int | None = None)` — [`L284`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L284)
  - `record(self, tensor: Any, name: str, stage: str = "")` — [`L309`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L309)
  - `set_current_forward_pass_id(self, forward_pass_id: int)` — [`L278`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L278) — Set the current forward pass ID for tracking inference steps
  - `set_enable_precision_tracer(self, enabled: bool)` — [`L126`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L126)
  - `set_end_time_and_duration(self, request_id: str)` — [`L158`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L158)
  - `set_request_status_to_completed(self, request_id: str)` — [`L154`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L154)
  - `start_batch_trace(self, batch_id: int)` — [`L247`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L247)
  - `start_trace(self, req_num: int | None = None, output_file: str | None = None, verbose_logging: bool = False, save_tensor: bool = False)` — [`L173`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L173)
  - `stop_trace(self)` — [`L219`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L219)
  - `trace_callback(tensor)` — [`L290`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L290)
  - `lock` — [`L103`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L103)
- protocol/private: `__init__`[`L102`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L102), `_batch_requests_mapping`[`L120`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L120), `_calculate_tensor_pricision_info`[`L416`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L416), `_completed_requests_count`[`L114`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L114), `_compute_stats`[`L548`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L548), `_current_batch_id`[`L118`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L118), `_current_forward_pass_id`[`L124`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L124), `_enable_precision_tracer`[`L109`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L109), `_last_forward_pass_id`[`L123`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L123), `_max_requests`[`L113`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L113), `_parse_layer_and_module`[`L494`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L494), `_records`[`L119`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L119), `_request_counter`[`L115`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L115), `_save_tensor`[`L110`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L110), `_token_counters`[`L122`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L122), `_trace_active`[`L106`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L106), `_trace_output_file`[`L107`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L107), `_traced_stats`[`L631`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L631), `_verbose_logging`[`L108`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L108), `_verbose_logging_console`[`L649`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L649)
- uses (calls/refs, reference-scoped): [`precision_tracer`](precision_tracer.md#precision_tracer), [`to_dict`](precision_tracer.md#PrecisionTracerRecord.to_dict), [`logger`](precision_tracer.md#logger), [`PrecisionTracerRequestMetadata`](precision_tracer.md#PrecisionTracerRequestMetadata), [`request_id`](precision_tracer.md#PrecisionTracerRequestMetadata.request_id), [`forward_mode`](precision_tracer.md#PrecisionTracerRequestMetadata.forward_mode), [`PrecisionTracerRecord`](precision_tracer.md#PrecisionTracerRecord), [`end_time`](precision_tracer.md#PrecisionTracerRecord.end_time), [`duration`](precision_tracer.md#PrecisionTracerRecord.duration), [`input_hash`](precision_tracer.md#PrecisionTracerRequestMetadata.input_hash), [`input_len`](precision_tracer.md#PrecisionTracerRequestMetadata.input_len), [`precision_records`](precision_tracer.md#PrecisionTracerRecord.precision_records), [`start_time`](precision_tracer.md#PrecisionTracerRecord.start_time), [`status`](precision_tracer.md#PrecisionTracerRecord.status), [`TensorJSONEncoder`](precision_tracer.md#TensorJSONEncoder)
- used by: [`process_batch_result_decode`](managers/scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.process_batch_result_decode), `get_model_worker_batch`, [`process_batch_result_prefill`](managers/scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.process_batch_result_prefill), [`precision_tracer`](precision_tracer.md#precision_tracer), [`set_internal_state`](managers/scheduler.md#Scheduler.set_internal_state), `_generate_trace_info`, [`model_loader`](model_executor/model_runner.md#ModelRunner.model_loader), [`forward`](model_executor/model_runner.md#ModelRunner.forward)

### `PrecisionTracerRecord`
- def: [`python/sgl_jax/srt/precision_tracer.py:60`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L60)
- signature: `class PrecisionTracerRecord:`
- members:
  - `to_dict(self)` — [`L86`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L86)
  - `bid` — [`L72`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L72)
  - `content_hash` — [`L83`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L83)
  - `duration` — [`L77`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L77)
  - `end_time` — [`L76`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L76)
  - `precision_records` — [`L78`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L78)
  - `process_id` — [`L84`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L84)
  - `request_id` — [`L73`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L73)
  - `request_idx` — [`L74`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L74)
  - `start_time` — [`L75`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L75)
  - `status` — [`L82`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L82)
- protocol/private: `__init__`[`L61`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L61)
- used by: [`start_trace`](precision_tracer.md#PrecisionTracer.start_trace), [`record`](precision_tracer.md#PrecisionTracer.record), [`start_batch_trace`](precision_tracer.md#PrecisionTracer.start_batch_trace), [`stop_trace`](precision_tracer.md#PrecisionTracer.stop_trace), [`_records`](precision_tracer.md#PrecisionTracer._records), [`set_end_time_and_duration`](precision_tracer.md#PrecisionTracer.set_end_time_and_duration), [`set_request_status_to_completed`](precision_tracer.md#PrecisionTracer.set_request_status_to_completed)

### `PrecisionTracerRequestMetadata`
- def: [`python/sgl_jax/srt/precision_tracer.py:43`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L43)
- signature: `class PrecisionTracerRequestMetadata:`
- members:
  - `to_dict(self)` — [`L51`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L51)
  - `forward_mode` — [`L49`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L49)
  - `input_hash` — [`L46`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L46)
  - `input_len` — [`L48`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L48)
  - `request_id` — [`L45`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L45)
  - `request_input_ids` — [`L47`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L47)
- protocol/private: `__init__`[`L44`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L44)
- used by: [`start_trace`](precision_tracer.md#PrecisionTracer.start_trace), [`record`](precision_tracer.md#PrecisionTracer.record), [`start_batch_trace`](precision_tracer.md#PrecisionTracer.start_batch_trace), `_generate_trace_info`, [`_calculate_tensor_pricision_info`](precision_tracer.md#PrecisionTracer._calculate_tensor_pricision_info), [`_batch_requests_mapping`](precision_tracer.md#PrecisionTracer._batch_requests_mapping), [`add_request_to_batch_requests_mapping`](precision_tracer.md#PrecisionTracer.add_request_to_batch_requests_mapping)

### `TensorJSONEncoder`  ·  implements/extends JSONEncoder
- def: [`python/sgl_jax/srt/precision_tracer.py:20`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L20)
- signature: `class TensorJSONEncoder(json.JSONEncoder):`
- members:
  - `default(self, obj)` — [`L21`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L21)
- uses (calls/refs, reference-scoped): [`_is_jax_array`](precision_tracer.md#_is_jax_array)
- used by: [`stop_trace`](precision_tracer.md#PrecisionTracer.stop_trace)

## Functions
- `_is_jax_array(obj)` — [`L16`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L16)

## Module values
- `logger` — [`L13`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L13)
- `precision_tracer` — [`L690`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/precision_tracer.py#L690)

