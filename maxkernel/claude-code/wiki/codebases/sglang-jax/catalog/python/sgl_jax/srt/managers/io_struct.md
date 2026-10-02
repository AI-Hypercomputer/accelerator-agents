---
title: 'Module: python/sgl_jax/srt/managers/io_struct.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/managers/io_struct.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.managers.io_struct`/
symbols:
  GenerateReqInput.__getitem__: GenerateReqInput#__getitem__().
  RpcReqInput: RpcReqInput#
  GenerateReqInput.parallel_sample_num: GenerateReqInput#parallel_sample_num.
  GenerateReqInput.normalize_batch_and_arguments: GenerateReqInput#normalize_batch_and_arguments().
  GenerateReqInput._normalize_single_inputs: GenerateReqInput#_normalize_single_inputs().
  GenerateReqInput._normalize_batch_inputs: GenerateReqInput#_normalize_batch_inputs().
  RpcReqOutput: RpcReqOutput#
  BaseReq.rid: BaseReq#rid.
  GenerateReqInput.input_ids: GenerateReqInput#input_ids.
  BaseReq: BaseReq#
  GenerateReqInput.text: GenerateReqInput#text.
  GenerateReqInput.sampling_params: GenerateReqInput#sampling_params.
  GenerateReqInput.lora_path: GenerateReqInput#lora_path.
  GenerateReqInput._normalize_logprob_params: GenerateReqInput#_normalize_logprob_params().
  ProfileReqOutput: ProfileReqOutput#
  GenerateReqInput._determine_batch_size: GenerateReqInput#_determine_batch_size().
  GenerateReqInput.rid: GenerateReqInput#rid.
  AbortReq: AbortReq#
  GenerateReqInput.token_ids_logprob: GenerateReqInput#token_ids_logprob.
  GenerateReqInput.input_embeds: GenerateReqInput#input_embeds.
  GenerateReqInput._expand_inputs: GenerateReqInput#_expand_inputs().
  ProfileReqOutput.success: ProfileReqOutput#success.
  ProfileReqOutput.message: ProfileReqOutput#message.
  MultimodalDataInputItem: MultimodalDataInputItem.
  TokenizedGenerateReqInput: TokenizedGenerateReqInput#
  GenerateReqInput.contains_mm_input: GenerateReqInput#contains_mm_input().
  has_valid_data: has_valid_data().
  GenerateReqInput.batch_size: GenerateReqInput#batch_size.
  GenerateReqInput.is_single: GenerateReqInput#is_single.
  MultimodalDataInputFormat: MultimodalDataInputFormat.
  TokenizedGenerateReqInput.dp_rank: TokenizedGenerateReqInput#dp_rank.
  ProfileReq: ProfileReq#
  TokenizedGenerateReqInput.mm_inputs: TokenizedGenerateReqInput#mm_inputs.
  AbortReq.abort_all: AbortReq#abort_all.
  GenerateReqInput._normalize_param: GenerateReqInput#_normalize_param().
  GenerateReqInput.return_logprob: GenerateReqInput#return_logprob.
  GenerateReqInput.logprob_start_len: GenerateReqInput#logprob_start_len.
  GenerateReqInput.top_logprobs_num: GenerateReqInput#top_logprobs_num.
  GenerateReqInput.return_routed_experts: GenerateReqInput#return_routed_experts.
  GenerateReqInput._normalize_rid: GenerateReqInput#_normalize_rid().
  GenerateReqInput._validate_inputs: GenerateReqInput#_validate_inputs().
  GenerateReqInput._normalize_sampling_params: GenerateReqInput#_normalize_sampling_params().
  GenerateReqInput._normalize_return_routed_experts: GenerateReqInput#_normalize_return_routed_experts().
  GenerateReqInput._normalize_lora_paths: GenerateReqInput#_normalize_lora_paths().
  PauseGenerationReqInput: PauseGenerationReqInput#
  ContinueGenerationReqInput: ContinueGenerationReqInput#
  FlushCacheReqInput: FlushCacheReqInput#
  SetInternalStateReq: SetInternalStateReq#
  FlushCacheReqOutput: FlushCacheReqOutput#
  GenerateReqInput: GenerateReqInput#
  GenerateReqInput.image_data: GenerateReqInput#image_data.
  SetInternalStateReqOutput: SetInternalStateReqOutput#
  PauseGenerationReqInput.mode: PauseGenerationReqInput#mode.
  GenerateReqInput.bootstrap_host: GenerateReqInput#bootstrap_host.
  GenerateReqInput.bootstrap_port: GenerateReqInput#bootstrap_port.
  GenerateReqInput.bootstrap_room: GenerateReqInput#bootstrap_room.
  GenerateReqInput.disagg_transfer_id: GenerateReqInput#disagg_transfer_id.
  ProfileReqType: ProfileReqType#
  ImageDataInputItem: ImageDataInputItem.
  GenerateReqInput.video_data: GenerateReqInput#video_data.
  GenerateReqInput.audio_data: GenerateReqInput#audio_data.
  GenerateReqInput._handle_parallel_sampling: GenerateReqInput#_handle_parallel_sampling().
  GenerateReqInput.lora_id: GenerateReqInput#lora_id.
  GetInternalStateReq: GetInternalStateReq#
  BaseReq.regenerate_rid: BaseReq#regenerate_rid().
  BatchTokenIDOut.finished_reasons: BatchTokenIDOut#finished_reasons.
  PauseGenerationReqInput.__post_init__: PauseGenerationReqInput#__post_init__().
  EmbeddingReqInput.image_data: EmbeddingReqInput#image_data.
  EmbeddingReqInput.video_data: EmbeddingReqInput#video_data.
  EmbeddingReqInput.audio_data: EmbeddingReqInput#audio_data.
  GenerateReqInput.regenerate_rid: GenerateReqInput#regenerate_rid().
  ReleaseMemoryOccupationReqInput: ReleaseMemoryOccupationReqInput#
  ResumeMemoryOccupationReqInput: ResumeMemoryOccupationReqInput#
  OpenSessionReqInput: OpenSessionReqInput#
  CloseSessionReqInput: CloseSessionReqInput#
  ConfigureLoggingReq: ConfigureLoggingReq#
  ReleaseMemoryOccupationReqOutput: ReleaseMemoryOccupationReqOutput#
  ResumeMemoryOccupationReqOutput: ResumeMemoryOccupationReqOutput#
  OpenSessionReqOutput: OpenSessionReqOutput#
  CloseSessionReqOutput: CloseSessionReqOutput#
  ConfigureLoggingReqOutput: ConfigureLoggingReqOutput#
  ParseFunctionCallReq: ParseFunctionCallReq#
  SeparateReasoningReqInput: SeparateReasoningReqInput#
  VertexGenerateReqInput: VertexGenerateReqInput#
  StartTraceReqInput: StartTraceReqInput#
  StartTraceReqInput.__post_init__: StartTraceReqInput#__post_init__().
  StopTraceReqInput: StopTraceReqInput#
  StopTraceReqInput.__post_init__: StopTraceReqInput#__post_init__().
  TraceStatusReqInput: TraceStatusReqInput#
  TraceStatusReqInput.__post_init__: TraceStatusReqInput#__post_init__().
  BatchTokenIDOut: BatchTokenIDOut#
  TokenizedGenerateReqInput.input_ids: TokenizedGenerateReqInput#input_ids.
  TokenizedGenerateReqInput.return_logprob: TokenizedGenerateReqInput#return_logprob.
  TokenizedGenerateReqInput.logprob_start_len: TokenizedGenerateReqInput#logprob_start_len.
  GenerateReqInput.stream: GenerateReqInput#stream.
  GenerateReqInput.return_text_in_logprobs: GenerateReqInput#return_text_in_logprobs.
  GetInternalStateReqOutput: GetInternalStateReqOutput#
  SetInternalStateReq.state_data: SetInternalStateReq#state_data.
  ProfileReq.output_dir: ProfileReq#output_dir.
  ProfileReq.num_steps: ProfileReq#num_steps.
  ProfileReq.host_tracer_level: ProfileReq#host_tracer_level.
  ProfileReq.python_tracer_level: ProfileReq#python_tracer_level.
  ProfileReq.profile_id: ProfileReq#profile_id.
  StartTraceReqInput.request_id: StartTraceReqInput#request_id.
  StopTraceReqInput.request_id: StopTraceReqInput#request_id.
  TraceStatusReqInput.request_id: TraceStatusReqInput#request_id.
  ImageData: ImageData#
  TokenizedGenerateReqInput.rid: TokenizedGenerateReqInput#rid.
  TokenizedGenerateReqInput.text: TokenizedGenerateReqInput#text.
  TokenizedGenerateReqInput.sampling_params: TokenizedGenerateReqInput#sampling_params.
  TokenizedGenerateReqInput.return_output_logprob_only: TokenizedGenerateReqInput#return_output_logprob_only.
  TokenizedGenerateReqInput.top_logprobs_num: TokenizedGenerateReqInput#top_logprobs_num.
  TokenizedGenerateReqInput.token_ids_logprob: TokenizedGenerateReqInput#token_ids_logprob.
  TokenizedGenerateReqInput.stream: TokenizedGenerateReqInput#stream.
  TokenizedGenerateReqInput.lora_id: TokenizedGenerateReqInput#lora_id.
  TokenizedGenerateReqInput.extra_key: TokenizedGenerateReqInput#extra_key.
  TokenizedGenerateReqInput.return_routed_experts: TokenizedGenerateReqInput#return_routed_experts.
  TokenizedGenerateReqInput.return_hidden_states: TokenizedGenerateReqInput#return_hidden_states.
  TokenizedGenerateReqInput.bootstrap_host: TokenizedGenerateReqInput#bootstrap_host.
  TokenizedGenerateReqInput.bootstrap_port: TokenizedGenerateReqInput#bootstrap_port.
  TokenizedGenerateReqInput.bootstrap_room: TokenizedGenerateReqInput#bootstrap_room.
  TokenizedGenerateReqInput.disagg_transfer_id: TokenizedGenerateReqInput#disagg_transfer_id.
  FlushCacheReqOutput.success: FlushCacheReqOutput#success.
  FlushCacheReqOutput.flushed_items: FlushCacheReqOutput#flushed_items.
  FlushCacheReqOutput.error_msg: FlushCacheReqOutput#error_msg.
  AudioDataInputItem: AudioDataInputItem.
  VideoDataInputItem: VideoDataInputItem.
  RpcReqInput.request_id: RpcReqInput#request_id.
  RpcReqOutput.request_id: RpcReqOutput#request_id.
  RpcReqOutput.success: RpcReqOutput#success.
  RpcReqOutput.error_msg: RpcReqOutput#error_msg.
  GetInternalStateReqOutput.internal_state: GetInternalStateReqOutput#internal_state.
  ProfileReqType.START_PROFILE: ProfileReqType#START_PROFILE.
  ProfileReqType.GET_STATUS: ProfileReqType#GET_STATUS.
  ProfileReq.start_step: ProfileReq#start_step.
  ProfileReq.profile_by_stage: ProfileReq#profile_by_stage.
  ProfileReq.profile_stages: ProfileReq#profile_stages.
  ImageData.url: ImageData#url.
  ImageData.detail: ImageData#detail.
  BaseReq.http_worker_ipc: BaseReq#http_worker_ipc.
  BatchStrOut: BatchStrOut#
  BatchStrOut.rids: BatchStrOut#rids.
  BatchStrOut.finished_reasons: BatchStrOut#finished_reasons.
  BatchStrOut.output_strs: BatchStrOut#output_strs.
  BatchStrOut.output_ids: BatchStrOut#output_ids.
  BatchStrOut.prompt_tokens: BatchStrOut#prompt_tokens.
  BatchStrOut.completion_tokens: BatchStrOut#completion_tokens.
  BatchStrOut.cached_tokens: BatchStrOut#cached_tokens.
  BatchStrOut.input_token_logprobs_val: BatchStrOut#input_token_logprobs_val.
  BatchStrOut.input_token_logprobs_idx: BatchStrOut#input_token_logprobs_idx.
  BatchStrOut.output_token_logprobs_val: BatchStrOut#output_token_logprobs_val.
  BatchStrOut.output_token_logprobs_idx: BatchStrOut#output_token_logprobs_idx.
  BatchStrOut.input_top_logprobs_val: BatchStrOut#input_top_logprobs_val.
  BatchStrOut.input_top_logprobs_idx: BatchStrOut#input_top_logprobs_idx.
  BatchStrOut.output_top_logprobs_val: BatchStrOut#output_top_logprobs_val.
  BatchStrOut.output_top_logprobs_idx: BatchStrOut#output_top_logprobs_idx.
  BatchStrOut.input_token_ids_logprobs_val: BatchStrOut#input_token_ids_logprobs_val.
  BatchStrOut.input_token_ids_logprobs_idx: BatchStrOut#input_token_ids_logprobs_idx.
  BatchStrOut.output_token_ids_logprobs_val: BatchStrOut#output_token_ids_logprobs_val.
  BatchStrOut.output_token_ids_logprobs_idx: BatchStrOut#output_token_ids_logprobs_idx.
  BatchStrOut.output_hidden_states: BatchStrOut#output_hidden_states.
  BatchStrOut.cache_miss_count: BatchStrOut#cache_miss_count.
  BatchStrOut.output_routed_experts: BatchStrOut#output_routed_experts.
  BatchTokenIDOut.rids: BatchTokenIDOut#rids.
  BatchTokenIDOut.decoded_texts: BatchTokenIDOut#decoded_texts.
  BatchTokenIDOut.decode_ids: BatchTokenIDOut#decode_ids.
  BatchTokenIDOut.read_offsets: BatchTokenIDOut#read_offsets.
  BatchTokenIDOut.output_ids: BatchTokenIDOut#output_ids.
  BatchTokenIDOut.skip_special_tokens: BatchTokenIDOut#skip_special_tokens.
  BatchTokenIDOut.spaces_between_special_tokens: BatchTokenIDOut#spaces_between_special_tokens.
  BatchTokenIDOut.no_stop_trim: BatchTokenIDOut#no_stop_trim.
  BatchTokenIDOut.prompt_tokens: BatchTokenIDOut#prompt_tokens.
  BatchTokenIDOut.completion_tokens: BatchTokenIDOut#completion_tokens.
  BatchTokenIDOut.cached_tokens: BatchTokenIDOut#cached_tokens.
  BatchTokenIDOut.input_token_logprobs_val: BatchTokenIDOut#input_token_logprobs_val.
  BatchTokenIDOut.input_token_logprobs_idx: BatchTokenIDOut#input_token_logprobs_idx.
  BatchTokenIDOut.output_token_logprobs_val: BatchTokenIDOut#output_token_logprobs_val.
  BatchTokenIDOut.output_token_logprobs_idx: BatchTokenIDOut#output_token_logprobs_idx.
  BatchTokenIDOut.input_top_logprobs_val: BatchTokenIDOut#input_top_logprobs_val.
  BatchTokenIDOut.input_top_logprobs_idx: BatchTokenIDOut#input_top_logprobs_idx.
  BatchTokenIDOut.output_top_logprobs_val: BatchTokenIDOut#output_top_logprobs_val.
  BatchTokenIDOut.output_top_logprobs_idx: BatchTokenIDOut#output_top_logprobs_idx.
  BatchTokenIDOut.input_token_ids_logprobs_val: BatchTokenIDOut#input_token_ids_logprobs_val.
  BatchTokenIDOut.input_token_ids_logprobs_idx: BatchTokenIDOut#input_token_ids_logprobs_idx.
  BatchTokenIDOut.output_token_ids_logprobs_val: BatchTokenIDOut#output_token_ids_logprobs_val.
  BatchTokenIDOut.output_token_ids_logprobs_idx: BatchTokenIDOut#output_token_ids_logprobs_idx.
  BatchTokenIDOut.output_hidden_states: BatchTokenIDOut#output_hidden_states.
  BatchTokenIDOut.output_hidden_states_for_mm: BatchTokenIDOut#output_hidden_states_for_mm.
  BatchTokenIDOut.cache_miss_count: BatchTokenIDOut#cache_miss_count.
  BatchTokenIDOut.output_routed_experts: BatchTokenIDOut#output_routed_experts.
  AbortReq.finished_reason: AbortReq#finished_reason.
  AbortReq.aborted_message: AbortReq#aborted_message.
  EmbeddingReqInput: EmbeddingReqInput#
  EmbeddingReqInput.rid: EmbeddingReqInput#rid.
  EmbeddingReqInput.text: EmbeddingReqInput#text.
  EmbeddingReqInput.input_ids: EmbeddingReqInput#input_ids.
  EmbeddingReqInput.normalize: EmbeddingReqInput#normalize.
  EmbeddingReqInput.extra_key: EmbeddingReqInput#extra_key.
  GenerateReqInput.return_output_logprob_only: GenerateReqInput#return_output_logprob_only.
  GenerateReqInput.extra_key: GenerateReqInput#extra_key.
  ReleaseMemoryOccupationReqInput.memory_size: ReleaseMemoryOccupationReqInput#memory_size.
  ResumeMemoryOccupationReqInput.memory_size: ResumeMemoryOccupationReqInput#memory_size.
  BatchEmbeddingOut: BatchEmbeddingOut#
  BatchEmbeddingOut.rids: BatchEmbeddingOut#rids.
  BatchEmbeddingOut.embeddings: BatchEmbeddingOut#embeddings.
  BatchEmbeddingOut.prompt_tokens: BatchEmbeddingOut#prompt_tokens.
  OpenSessionReqInput.session_id: OpenSessionReqInput#session_id.
  CloseSessionReqInput.session_id: CloseSessionReqInput#session_id.
  TokenizedEmbeddingReqInput: TokenizedEmbeddingReqInput#
  TokenizedEmbeddingReqInput.rid: TokenizedEmbeddingReqInput#rid.
  TokenizedEmbeddingReqInput.text: TokenizedEmbeddingReqInput#text.
  TokenizedEmbeddingReqInput.input_ids: TokenizedEmbeddingReqInput#input_ids.
  TokenizedEmbeddingReqInput.normalize: TokenizedEmbeddingReqInput#normalize.
  ConfigureLoggingReq.log_level: ConfigureLoggingReq#log_level.
  ConfigureLoggingReq.log_file: ConfigureLoggingReq#log_file.
  ProfileReqInput: ProfileReqInput#
  ProfileReqInput.output_dir: ProfileReqInput#output_dir.
  ProfileReqInput.start_step: ProfileReqInput#start_step.
  ProfileReqInput.num_steps: ProfileReqInput#num_steps.
  ProfileReqInput.host_tracer_level: ProfileReqInput#host_tracer_level.
  ProfileReqInput.python_tracer_level: ProfileReqInput#python_tracer_level.
  ProfileReqInput.stage_id: ProfileReqInput#stage_id.
  ProfileReqInput.profile_by_stage: ProfileReqInput#profile_by_stage.
  ProfileReqInput.profile_stages: ProfileReqInput#profile_stages.
  ProfileReqType.STOP_PROFILE: ProfileReqType#STOP_PROFILE.
  ProfileReq.stage_id: ProfileReq#stage_id.
  HealthCheckOutput: HealthCheckOutput#
  HealthCheckOutput.status: HealthCheckOutput#status.
  HealthCheckOutput.timestamp: HealthCheckOutput#timestamp.
  OpenSessionReqOutput.session_id: OpenSessionReqOutput#session_id.
  ParseFunctionCallReq.text: ParseFunctionCallReq#text.
  ParseFunctionCallReq.parser_type: ParseFunctionCallReq#parser_type.
  SeparateReasoningReqInput.text: SeparateReasoningReqInput#text.
  SeparateReasoningReqInput.reasoning_type: SeparateReasoningReqInput#reasoning_type.
  StartTraceReqInput.req_num: StartTraceReqInput#req_num.
  StartTraceReqInput.output_file: StartTraceReqInput#output_file.
  StartTraceReqInput.save_tensor: StartTraceReqInput#save_tensor.
---
# Module: [`python/sgl_jax/srt/managers/io_struct.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py)

## Classes
### `AbortReq`  ·  implements/extends BaseReq
- def: [`python/sgl_jax/srt/managers/io_struct.py:181`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L181)
- signature: `class AbortReq(BaseReq):`
- members:
  - `abort_all` — [`L183`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L183)
  - `aborted_message` — [`L186`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L186)
  - `finished_reason` — [`L185`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L185)
- uses (calls/refs, reference-scoped): [`BaseReq`](io_struct.md#BaseReq)
- used by: [`process_batch_result_decode`](scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.process_batch_result_decode), [`process_batch_result_prefill`](scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.process_batch_result_prefill), [`abort_request`](scheduler.md#Scheduler.abort_request), [`update_running_batch`](scheduler.md#Scheduler.update_running_batch), [`_request_dispatcher`](scheduler.md#Scheduler._request_dispatcher), [`_abort_decode_request`](../disaggregation/decode.md#SchedulerDisaggregationDecodeMixin._abort_decode_request), [`BaseReq`](io_struct.md#BaseReq)

### `BaseReq`
- def: [`python/sgl_jax/srt/managers/io_struct.py:36`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L36)
- signature: `class BaseReq:`
- members:
  - `regenerate_rid(self)` — [`L40`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L40) — Generate a new request ID and return it.
  - `http_worker_ipc` — [`L38`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L38)
  - `rid` — [`L37`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L37)
- uses (calls/refs, reference-scoped): [`AbortReq`](io_struct.md#AbortReq), [`ContinueGenerationReqInput`](io_struct.md#ContinueGenerationReqInput), [`FlushCacheReqInput`](io_struct.md#FlushCacheReqInput), [`PauseGenerationReqInput`](io_struct.md#PauseGenerationReqInput), [`FlushCacheReqOutput`](io_struct.md#FlushCacheReqOutput)
- used by: [`process_batch_result_decode`](scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.process_batch_result_decode), [`process_batch_result_prefill`](scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.process_batch_result_prefill), [`abort_request`](scheduler.md#Scheduler.abort_request), [`update_running_batch`](scheduler.md#Scheduler.update_running_batch), [`_abort_decode_request`](../disaggregation/decode.md#SchedulerDisaggregationDecodeMixin._abort_decode_request), [`flush_cache_wrapped`](scheduler.md#Scheduler.flush_cache_wrapped), [`AbortReq`](io_struct.md#AbortReq), [`ContinueGenerationReqInput`](io_struct.md#ContinueGenerationReqInput), [`FlushCacheReqInput`](io_struct.md#FlushCacheReqInput), [`PauseGenerationReqInput`](io_struct.md#PauseGenerationReqInput), [`FlushCacheReqOutput`](io_struct.md#FlushCacheReqOutput)

### `BatchEmbeddingOut`
- def: [`python/sgl_jax/srt/managers/io_struct.py:616`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L616)
- doc: Batch embedding output.
- signature: `class BatchEmbeddingOut:`
- members:
  - `embeddings` — [`L620`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L620)
  - `prompt_tokens` — [`L621`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L621)
  - `rids` — [`L619`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L619)

### `BatchStrOut`
- def: [`python/sgl_jax/srt/managers/io_struct.py:50`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L50)
- signature: `class BatchStrOut:`
- members:
  - `cache_miss_count` — [`L83`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L83)
  - `cached_tokens` — [`L63`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L63)
  - `completion_tokens` — [`L62`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L62)
  - `finished_reasons` — [`L54`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L54)
  - `input_token_ids_logprobs_idx` — [`L75`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L75)
  - `input_token_ids_logprobs_val` — [`L74`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L74)
  - `input_token_logprobs_idx` — [`L67`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L67)
  - `input_token_logprobs_val` — [`L66`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L66)
  - `input_top_logprobs_idx` — [`L71`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L71)
  - `input_top_logprobs_val` — [`L70`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L70)
  - `output_hidden_states` — [`L80`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L80)
  - `output_ids` — [`L58`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L58)
  - `output_routed_experts` — [`L86`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L86)
  - `output_strs` — [`L56`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L56)
  - `output_token_ids_logprobs_idx` — [`L77`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L77)
  - `output_token_ids_logprobs_val` — [`L76`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L76)
  - `output_token_logprobs_idx` — [`L69`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L69)
  - `output_token_logprobs_val` — [`L68`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L68)
  - `output_top_logprobs_idx` — [`L73`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L73)
  - `output_top_logprobs_val` — [`L72`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L72)
  - `prompt_tokens` — [`L61`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L61)
  - `rids` — [`L52`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L52)

### `BatchTokenIDOut`
- def: [`python/sgl_jax/srt/managers/io_struct.py:90`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L90)
- signature: `class BatchTokenIDOut:`
- members:
  - `cache_miss_count` — [`L129`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L129)
  - `cached_tokens` — [`L109`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L109)
  - `completion_tokens` — [`L108`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L108)
  - `decode_ids` — [`L97`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L97)
  - `decoded_texts` — [`L96`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L96)
  - `finished_reasons` — [`L94`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L94)
  - `input_token_ids_logprobs_idx` — [`L121`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L121)
  - `input_token_ids_logprobs_val` — [`L120`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L120)
  - `input_token_logprobs_idx` — [`L113`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L113)
  - `input_token_logprobs_val` — [`L112`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L112)
  - `input_top_logprobs_idx` — [`L117`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L117)
  - `input_top_logprobs_val` — [`L116`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L116)
  - `no_stop_trim` — [`L104`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L104)
  - `output_hidden_states` — [`L126`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L126)
  - `output_hidden_states_for_mm` — [`L127`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L127)
  - `output_ids` — [`L100`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L100)
  - `output_routed_experts` — [`L132`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L132)
  - `output_token_ids_logprobs_idx` — [`L123`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L123)
  - `output_token_ids_logprobs_val` — [`L122`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L122)
  - `output_token_logprobs_idx` — [`L115`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L115)
  - `output_token_logprobs_val` — [`L114`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L114)
  - `output_top_logprobs_idx` — [`L119`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L119)
  - `output_top_logprobs_val` — [`L118`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L118)
  - `prompt_tokens` — [`L107`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L107)
  - `read_offsets` — [`L98`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L98)
  - `rids` — [`L92`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L92)
  - `skip_special_tokens` — [`L102`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L102)
  - `spaces_between_special_tokens` — [`L103`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L103)
- uses (calls/refs, reference-scoped): `BaseFinishReason`
- used by: [`stream_output_generation`](scheduler_output_processor_mixin.md#SchedulerOutputProcessorMixin.stream_output_generation)

### `CloseSessionReqInput`  ·  implements/extends RpcReqInput
- def: [`python/sgl_jax/srt/managers/io_struct.py:633`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L633)
- doc: Request to close a session.
- signature: `class CloseSessionReqInput(RpcReqInput):`
- members:
  - `session_id` — [`L636`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L636)
- uses (calls/refs, reference-scoped): [`RpcReqInput`](io_struct.md#RpcReqInput)
- used by: [`RpcReqInput`](io_struct.md#RpcReqInput)

### `CloseSessionReqOutput`  ·  implements/extends RpcReqOutput
- def: [`python/sgl_jax/srt/managers/io_struct.py:758`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L758)
- doc: Output for CloseSessionReqInput.
- signature: `class CloseSessionReqOutput(RpcReqOutput):`
- uses (calls/refs, reference-scoped): [`RpcReqOutput`](io_struct.md#RpcReqOutput)
- used by: [`RpcReqOutput`](io_struct.md#RpcReqOutput)

### `ConfigureLoggingReq`  ·  implements/extends RpcReqInput
- def: [`python/sgl_jax/srt/managers/io_struct.py:651`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L651)
- doc: Request to configure logging.
- signature: `class ConfigureLoggingReq(RpcReqInput):`
- members:
  - `log_file` — [`L655`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L655)
  - `log_level` — [`L654`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L654)
- uses (calls/refs, reference-scoped): [`RpcReqInput`](io_struct.md#RpcReqInput)
- used by: [`RpcReqInput`](io_struct.md#RpcReqInput)

### `ConfigureLoggingReqOutput`  ·  implements/extends RpcReqOutput
- def: [`python/sgl_jax/srt/managers/io_struct.py:765`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L765)
- doc: Output for ConfigureLoggingReq.
- signature: `class ConfigureLoggingReqOutput(RpcReqOutput):`
- uses (calls/refs, reference-scoped): [`RpcReqOutput`](io_struct.md#RpcReqOutput)
- used by: [`RpcReqOutput`](io_struct.md#RpcReqOutput)

### `ContinueGenerationReqInput`  ·  implements/extends BaseReq
- def: [`python/sgl_jax/srt/managers/io_struct.py:218`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L218)
- signature: `class ContinueGenerationReqInput(BaseReq):`
- uses (calls/refs, reference-scoped): [`BaseReq`](io_struct.md#BaseReq)
- used by: [`_request_dispatcher`](scheduler.md#Scheduler._request_dispatcher), [`BaseReq`](io_struct.md#BaseReq), [`continue_generation`](scheduler.md#Scheduler.continue_generation)

### `EmbeddingReqInput`
- def: [`python/sgl_jax/srt/managers/io_struct.py:249`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L249)
- doc: Request input for embedding generation.
- signature: `class EmbeddingReqInput:`
- members:
  - `audio_data` — [`L264`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L264)
  - `extra_key` — [`L269`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L269)
  - `image_data` — [`L260`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L260)
  - `input_ids` — [`L266`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L266)
  - `normalize` — [`L267`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L267)
  - `rid` — [`L252`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L252)
  - `text` — [`L253`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L253)
  - `video_data` — [`L262`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L262)
- uses (calls/refs, reference-scoped): [`MultimodalDataInputFormat`](io_struct.md#MultimodalDataInputFormat)

### `FlushCacheReqInput`  ·  implements/extends BaseReq
- def: [`python/sgl_jax/srt/managers/io_struct.py:223`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L223)
- signature: `class FlushCacheReqInput(BaseReq):`
- uses (calls/refs, reference-scoped): [`BaseReq`](io_struct.md#BaseReq)
- used by: [`_request_dispatcher`](scheduler.md#Scheduler._request_dispatcher), [`flush_cache_wrapped`](scheduler.md#Scheduler.flush_cache_wrapped), [`BaseReq`](io_struct.md#BaseReq)

### `FlushCacheReqOutput`  ·  implements/extends BaseReq
- def: [`python/sgl_jax/srt/managers/io_struct.py:228`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L228)
- signature: `class FlushCacheReqOutput(BaseReq):`
- members:
  - `error_msg` — [`L231`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L231)
  - `flushed_items` — [`L230`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L230)
  - `success` — [`L229`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L229)
- uses (calls/refs, reference-scoped): [`BaseReq`](io_struct.md#BaseReq)
- used by: [`flush_cache_wrapped`](scheduler.md#Scheduler.flush_cache_wrapped), [`BaseReq`](io_struct.md#BaseReq)

### `GenerateReqInput`
- def: [`python/sgl_jax/srt/managers/io_struct.py:273`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L273)
- doc: Request input for text generation.
- signature: `class GenerateReqInput:`
- members:
  - `_determine_batch_size(self)` — [`L462`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L462) — Determine if this is a single example or a batch and the batch size. — documented in [python-sgl_jax-srt-managers-io_struct](../../../../../concepts/python-sgl_jax-srt-managers-io_struct.md)
  - `_expand_inputs(self, num)` — [`L431`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L431) — Expand the main inputs (text, input_ids, input_embeds) for parallel sampling. — documented in [python-sgl_jax-srt-managers-io_struct](../../../../../concepts/python-sgl_jax-srt-managers-io_struct.md)
  - `_handle_parallel_sampling(self)` — [`L388`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L388) — Handle parallel sampling parameters and adjust batch size if needed.
  - `_normalize_batch_inputs(self)` — [`L414`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L414) — Normalize inputs for a batch of examples, including parallel sampling expansion. — documented in [python-sgl_jax-srt-managers-io_struct](../../../../../concepts/python-sgl_jax-srt-managers-io_struct.md)
  - `_normalize_logprob_params(self, num)` — [`L506`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L506) — Normalize logprob-related parameters for batch processing. — documented in [python-sgl_jax-srt-managers-io_struct](../../../../../concepts/python-sgl_jax-srt-managers-io_struct.md)
  - `_normalize_lora_paths(self, num)` — [`L529`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L529) — Normalize LoRA paths for batch processing.
  - `_normalize_rid(self, num)` — [`L331`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L331) — Normalize request IDs for batch processing.
  - `_normalize_sampling_params(self, num)` — [`L453`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L453) — Normalize sampling parameters for batch processing.
  - `_normalize_single_inputs(self)` — [`L366`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L366) — Normalize inputs for a single example. — documented in [python-sgl_jax-srt-managers-io_struct](../../../../../concepts/python-sgl_jax-srt-managers-io_struct.md)
  - `_validate_inputs(self)` — [`L446`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L446) — Validate that the input configuration is valid. — documented in [python-sgl_jax-srt-managers-io_struct](../../../../../concepts/python-sgl_jax-srt-managers-io_struct.md)
  - `contains_mm_input(self)` — [`L324`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L324)
  - `normalize_batch_and_arguments(self)` — [`L348`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L348) — documented in [python-sgl_jax-srt-managers-io_struct](../../../../../concepts/python-sgl_jax-srt-managers-io_struct.md)
  - `regenerate_rid(self)` — [`L539`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L539) — Generate a new request ID and return it.
  - `audio_data` — [`L292`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L292)
  - `batch_size` — [`L276`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L276) — documented in [python-sgl_jax-srt-managers-io_struct](../../../../../concepts/python-sgl_jax-srt-managers-io_struct.md)
  - `bootstrap_host` — [`L319`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L319)
  - `bootstrap_port` — [`L320`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L320)
  - `bootstrap_room` — [`L321`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L321)
  - `disagg_transfer_id` — [`L322`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L322)
  - `extra_key` — [`L314`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L314)
  - `image_data` — [`L288`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L288)
  - `input_embeds` — [`L281`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L281) — documented in [python-sgl_jax-srt-managers-io_struct](../../../../../concepts/python-sgl_jax-srt-managers-io_struct.md)
  - `input_ids` — [`L279`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L279) — documented in [python-sgl_jax-srt-managers-io_struct](../../../../../concepts/python-sgl_jax-srt-managers-io_struct.md)
  - `is_single` — [`L297`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L297) — documented in [python-sgl_jax-srt-managers-io_struct](../../../../../concepts/python-sgl_jax-srt-managers-io_struct.md)
  - `logprob_start_len` — [`L301`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L301) — documented in [python-sgl_jax-srt-managers-io_struct](../../../../../concepts/python-sgl_jax-srt-managers-io_struct.md)
  - `lora_id` — [`L312`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L312)
  - `lora_path` — [`L310`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L310) — documented in [python-sgl_jax-srt-managers-io_struct](../../../../../concepts/python-sgl_jax-srt-managers-io_struct.md)
  - `parallel_sample_num` — [`L392`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L392) — documented in [python-sgl_jax-srt-managers-io_struct](../../../../../concepts/python-sgl_jax-srt-managers-io_struct.md)
  - `return_logprob` — [`L298`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L298) — documented in [python-sgl_jax-srt-managers-io_struct](../../../../../concepts/python-sgl_jax-srt-managers-io_struct.md)
  - `return_output_logprob_only` — [`L299`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L299)
  - `return_routed_experts` — [`L316`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L316) — documented in [python-sgl_jax-srt-managers-io_struct](../../../../../concepts/python-sgl_jax-srt-managers-io_struct.md)
  - `return_text_in_logprobs` — [`L307`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L307)
  - `rid` — [`L277`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L277) — documented in [python-sgl_jax-srt-managers-io_struct](../../../../../concepts/python-sgl_jax-srt-managers-io_struct.md)
  - `sampling_params` — [`L293`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L293) — documented in [python-sgl_jax-srt-managers-io_struct](../../../../../concepts/python-sgl_jax-srt-managers-io_struct.md)
  - `stream` — [`L296`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L296)
  - `text` — [`L278`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L278) — documented in [python-sgl_jax-srt-managers-io_struct](../../../../../concepts/python-sgl_jax-srt-managers-io_struct.md)
  - `token_ids_logprob` — [`L305`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L305) — documented in [python-sgl_jax-srt-managers-io_struct](../../../../../concepts/python-sgl_jax-srt-managers-io_struct.md)
  - `top_logprobs_num` — [`L303`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L303) — documented in [python-sgl_jax-srt-managers-io_struct](../../../../../concepts/python-sgl_jax-srt-managers-io_struct.md)
  - `video_data` — [`L290`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L290)
- protocol/private: `__getitem__`[`L544`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L544), `_normalize_param`[`L496`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L496), `_normalize_return_routed_experts`[`L490`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L490)
- uses (calls/refs, reference-scoped): [`has_valid_data`](io_struct.md#has_valid_data), [`MultimodalDataInputFormat`](io_struct.md#MultimodalDataInputFormat), [`VertexGenerateReqInput`](io_struct.md#VertexGenerateReqInput)
- used by: [`VertexGenerateReqInput`](io_struct.md#VertexGenerateReqInput)

### `GetInternalStateReq`
- def: [`python/sgl_jax/srt/managers/io_struct.py:660`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L660)
- signature: `class GetInternalStateReq:`
- used by: [`get_internal_state`](scheduler.md#Scheduler.get_internal_state), [`_request_dispatcher`](scheduler.md#Scheduler._request_dispatcher)

### `GetInternalStateReqOutput`
- def: [`python/sgl_jax/srt/managers/io_struct.py:665`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L665)
- signature: `class GetInternalStateReqOutput:`
- members:
  - `internal_state` — [`L666`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L666)
- used by: [`get_internal_state`](scheduler.md#Scheduler.get_internal_state)

### `HealthCheckOutput`
- def: [`python/sgl_jax/srt/managers/io_struct.py:729`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L729)
- doc: Health check output.
- signature: `class HealthCheckOutput:`
- members:
  - `status` — [`L732`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L732)
  - `timestamp` — [`L733`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L733)

### `ImageData`
- def: [`python/sgl_jax/srt/managers/io_struct.py:30`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L30)
- signature: `class ImageData:`
- members:
  - `detail` — [`L32`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L32)
  - `url` — [`L31`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L31)
- used by: [`ImageDataInputItem`](io_struct.md#ImageDataInputItem)

### `OpenSessionReqInput`  ·  implements/extends RpcReqInput
- def: [`python/sgl_jax/srt/managers/io_struct.py:626`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L626)
- doc: Request to open a session.
- signature: `class OpenSessionReqInput(RpcReqInput):`
- members:
  - `session_id` — [`L629`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L629)
- uses (calls/refs, reference-scoped): [`RpcReqInput`](io_struct.md#RpcReqInput)
- used by: [`RpcReqInput`](io_struct.md#RpcReqInput)

### `OpenSessionReqOutput`  ·  implements/extends RpcReqOutput
- def: [`python/sgl_jax/srt/managers/io_struct.py:751`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L751)
- doc: Output for OpenSessionReqInput.
- signature: `class OpenSessionReqOutput(RpcReqOutput):`
- members:
  - `session_id` — [`L754`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L754)
- uses (calls/refs, reference-scoped): [`RpcReqOutput`](io_struct.md#RpcReqOutput)
- used by: [`RpcReqOutput`](io_struct.md#RpcReqOutput)

### `ParseFunctionCallReq`  ·  implements/extends RpcReqInput
- def: [`python/sgl_jax/srt/managers/io_struct.py:780`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L780)
- doc: Request to parse function calls.
- signature: `class ParseFunctionCallReq(RpcReqInput):`
- members:
  - `parser_type` — [`L784`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L784)
  - `text` — [`L783`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L783)
- uses (calls/refs, reference-scoped): [`RpcReqInput`](io_struct.md#RpcReqInput)
- used by: [`RpcReqInput`](io_struct.md#RpcReqInput)

### `PauseGenerationReqInput`  ·  implements/extends BaseReq
- def: [`python/sgl_jax/srt/managers/io_struct.py:190`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L190)
- doc: Note that the PauseGenerationRequests is only supported in SGLang Server.
- signature: `class PauseGenerationReqInput(BaseReq):`
- members:
  - `mode` — [`L209`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L209)
- protocol/private: `__post_init__`[`L211`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L211)
- uses (calls/refs, reference-scoped): [`BaseReq`](io_struct.md#BaseReq)
- used by: [`_request_dispatcher`](scheduler.md#Scheduler._request_dispatcher), [`pause_generation`](scheduler.md#Scheduler.pause_generation), [`BaseReq`](io_struct.md#BaseReq)

### `ProfileReq`
- def: [`python/sgl_jax/srt/managers/io_struct.py:708`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L708)
- signature: `class ProfileReq:`
- members:
  - `host_tracer_level` — [`L713`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L713)
  - `num_steps` — [`L712`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L712)
  - `output_dir` — [`L710`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L710)
  - `profile_by_stage` — [`L717`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L717)
  - `profile_id` — [`L715`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L715)
  - `profile_stages` — [`L718`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L718)
  - `python_tracer_level` — [`L714`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L714)
  - `stage_id` — [`L716`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L716)
  - `start_step` — [`L711`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L711)
- uses (calls/refs, reference-scoped): [`ProfileReqType`](io_struct.md#ProfileReqType)
- used by: [`_request_dispatcher`](scheduler.md#Scheduler._request_dispatcher), [`_start_stage_profile`](scheduler_profiler_mixing.md#SchedulerProfilerMixin._start_stage_profile), [`profile`](scheduler_profiler_mixing.md#SchedulerProfilerMixin.profile)

### `ProfileReqInput`
- def: [`python/sgl_jax/srt/managers/io_struct.py:680`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L680)
- signature: `class ProfileReqInput:`
- members:
  - `host_tracer_level` — [`L689`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L689)
  - `num_steps` — [`L683`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L683)
  - `output_dir` — [`L681`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L681)
  - `profile_by_stage` — [`L697`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L697)
  - `profile_stages` — [`L698`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L698)
  - `python_tracer_level` — [`L693`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L693)
  - `stage_id` — [`L694`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L694)
  - `start_step` — [`L682`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L682)

### `ProfileReqOutput`
- def: [`python/sgl_jax/srt/managers/io_struct.py:722`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L722)
- signature: `class ProfileReqOutput:`
- members:
  - `message` — [`L724`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L724)
  - `success` — [`L723`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L723)
- used by: [`_start_stage_profile`](scheduler_profiler_mixing.md#SchedulerProfilerMixin._start_stage_profile), [`start_profile`](scheduler_profiler_mixing.md#SchedulerProfilerMixin.start_profile), [`stop_profile`](scheduler_profiler_mixing.md#SchedulerProfilerMixin.stop_profile), [`get_profile_status`](scheduler_profiler_mixing.md#SchedulerProfilerMixin.get_profile_status)

### `ProfileReqType`  ·  implements/extends Enum
- def: [`python/sgl_jax/srt/managers/io_struct.py:701`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L701)
- signature: `class ProfileReqType(Enum):`
- members:
  - `GET_STATUS` — [`L704`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L704)
  - `START_PROFILE` — [`L702`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L702)
  - `STOP_PROFILE` — [`L703`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L703)
- used by: [`profile`](scheduler_profiler_mixing.md#SchedulerProfilerMixin.profile), [`ProfileReq`](io_struct.md#ProfileReq)

### `ReleaseMemoryOccupationReqInput`  ·  implements/extends RpcReqInput
- def: [`python/sgl_jax/srt/managers/io_struct.py:601`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L601)
- doc: Request to release memory occupation.
- signature: `class ReleaseMemoryOccupationReqInput(RpcReqInput):`
- members:
  - `memory_size` — [`L604`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L604)
- uses (calls/refs, reference-scoped): [`RpcReqInput`](io_struct.md#RpcReqInput)
- used by: [`RpcReqInput`](io_struct.md#RpcReqInput)

### `ReleaseMemoryOccupationReqOutput`  ·  implements/extends RpcReqOutput
- def: [`python/sgl_jax/srt/managers/io_struct.py:737`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L737)
- doc: Output for ReleaseMemoryOccupationReqInput.
- signature: `class ReleaseMemoryOccupationReqOutput(RpcReqOutput):`
- uses (calls/refs, reference-scoped): [`RpcReqOutput`](io_struct.md#RpcReqOutput)
- used by: [`RpcReqOutput`](io_struct.md#RpcReqOutput)

### `ResumeMemoryOccupationReqInput`  ·  implements/extends RpcReqInput
- def: [`python/sgl_jax/srt/managers/io_struct.py:608`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L608)
- doc: Request to resume memory occupation.
- signature: `class ResumeMemoryOccupationReqInput(RpcReqInput):`
- members:
  - `memory_size` — [`L611`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L611)
- uses (calls/refs, reference-scoped): [`RpcReqInput`](io_struct.md#RpcReqInput)
- used by: [`RpcReqInput`](io_struct.md#RpcReqInput)

### `ResumeMemoryOccupationReqOutput`  ·  implements/extends RpcReqOutput
- def: [`python/sgl_jax/srt/managers/io_struct.py:744`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L744)
- doc: Output for ResumeMemoryOccupationReqInput.
- signature: `class ResumeMemoryOccupationReqOutput(RpcReqOutput):`
- uses (calls/refs, reference-scoped): [`RpcReqOutput`](io_struct.md#RpcReqOutput)
- used by: [`RpcReqOutput`](io_struct.md#RpcReqOutput)

### `RpcReqInput`
- def: [`python/sgl_jax/srt/managers/io_struct.py:585`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L585) — documented in [python-sgl_jax-srt-managers-io_struct](../../../../../concepts/python-sgl_jax-srt-managers-io_struct.md)
- doc: Base class for RPC request input.
- signature: `class RpcReqInput:`
- members:
  - `request_id` — [`L588`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L588)
- uses (calls/refs, reference-scoped): [`SetInternalStateReq`](io_struct.md#SetInternalStateReq), [`CloseSessionReqInput`](io_struct.md#CloseSessionReqInput), [`ConfigureLoggingReq`](io_struct.md#ConfigureLoggingReq), [`OpenSessionReqInput`](io_struct.md#OpenSessionReqInput), [`ParseFunctionCallReq`](io_struct.md#ParseFunctionCallReq), [`ReleaseMemoryOccupationReqInput`](io_struct.md#ReleaseMemoryOccupationReqInput), [`ResumeMemoryOccupationReqInput`](io_struct.md#ResumeMemoryOccupationReqInput), [`SeparateReasoningReqInput`](io_struct.md#SeparateReasoningReqInput), [`StartTraceReqInput`](io_struct.md#StartTraceReqInput), [`StopTraceReqInput`](io_struct.md#StopTraceReqInput), [`TraceStatusReqInput`](io_struct.md#TraceStatusReqInput)
- used by: [`set_internal_state`](scheduler.md#Scheduler.set_internal_state), [`SetInternalStateReq`](io_struct.md#SetInternalStateReq), [`CloseSessionReqInput`](io_struct.md#CloseSessionReqInput), [`ConfigureLoggingReq`](io_struct.md#ConfigureLoggingReq), [`OpenSessionReqInput`](io_struct.md#OpenSessionReqInput), [`ParseFunctionCallReq`](io_struct.md#ParseFunctionCallReq), [`ReleaseMemoryOccupationReqInput`](io_struct.md#ReleaseMemoryOccupationReqInput), [`ResumeMemoryOccupationReqInput`](io_struct.md#ResumeMemoryOccupationReqInput), [`SeparateReasoningReqInput`](io_struct.md#SeparateReasoningReqInput), [`StartTraceReqInput`](io_struct.md#StartTraceReqInput), [`StopTraceReqInput`](io_struct.md#StopTraceReqInput), [`TraceStatusReqInput`](io_struct.md#TraceStatusReqInput)

### `RpcReqOutput`
- def: [`python/sgl_jax/srt/managers/io_struct.py:592`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L592)
- doc: Base class for RPC request output.
- signature: `class RpcReqOutput:`
- members:
  - `error_msg` — [`L597`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L597)
  - `request_id` — [`L595`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L595)
  - `success` — [`L596`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L596)
- uses (calls/refs, reference-scoped): [`SetInternalStateReqOutput`](io_struct.md#SetInternalStateReqOutput), [`CloseSessionReqOutput`](io_struct.md#CloseSessionReqOutput), [`ConfigureLoggingReqOutput`](io_struct.md#ConfigureLoggingReqOutput), [`OpenSessionReqOutput`](io_struct.md#OpenSessionReqOutput), [`ReleaseMemoryOccupationReqOutput`](io_struct.md#ReleaseMemoryOccupationReqOutput), [`ResumeMemoryOccupationReqOutput`](io_struct.md#ResumeMemoryOccupationReqOutput)
- used by: [`set_internal_state`](scheduler.md#Scheduler.set_internal_state), [`SetInternalStateReqOutput`](io_struct.md#SetInternalStateReqOutput), [`CloseSessionReqOutput`](io_struct.md#CloseSessionReqOutput), [`ConfigureLoggingReqOutput`](io_struct.md#ConfigureLoggingReqOutput), [`OpenSessionReqOutput`](io_struct.md#OpenSessionReqOutput), [`ReleaseMemoryOccupationReqOutput`](io_struct.md#ReleaseMemoryOccupationReqOutput), [`ResumeMemoryOccupationReqOutput`](io_struct.md#ResumeMemoryOccupationReqOutput)

### `SeparateReasoningReqInput`  ·  implements/extends RpcReqInput
- def: [`python/sgl_jax/srt/managers/io_struct.py:788`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L788)
- doc: Request to separate reasoning.
- signature: `class SeparateReasoningReqInput(RpcReqInput):`
- members:
  - `reasoning_type` — [`L792`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L792)
  - `text` — [`L791`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L791)
- uses (calls/refs, reference-scoped): [`RpcReqInput`](io_struct.md#RpcReqInput)
- used by: [`RpcReqInput`](io_struct.md#RpcReqInput)

### `SetInternalStateReq`  ·  implements/extends RpcReqInput
- def: [`python/sgl_jax/srt/managers/io_struct.py:670`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L670)
- doc: Request to set internal state.
- signature: `class SetInternalStateReq(RpcReqInput):`
- members:
  - `state_data` — [`L673`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L673)
- uses (calls/refs, reference-scoped): [`RpcReqInput`](io_struct.md#RpcReqInput)
- used by: [`_request_dispatcher`](scheduler.md#Scheduler._request_dispatcher), [`set_internal_state`](scheduler.md#Scheduler.set_internal_state), [`RpcReqInput`](io_struct.md#RpcReqInput)

### `SetInternalStateReqOutput`  ·  implements/extends RpcReqOutput
- def: [`python/sgl_jax/srt/managers/io_struct.py:772`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L772)
- doc: Output for SetInternalStateReq.
- signature: `class SetInternalStateReqOutput(RpcReqOutput):`
- uses (calls/refs, reference-scoped): [`RpcReqOutput`](io_struct.md#RpcReqOutput)
- used by: [`set_internal_state`](scheduler.md#Scheduler.set_internal_state), [`RpcReqOutput`](io_struct.md#RpcReqOutput)

### `StartTraceReqInput`  ·  implements/extends RpcReqInput
- def: [`python/sgl_jax/srt/managers/io_struct.py:803`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L803)
- doc: Request to start precision tracing.
- signature: `class StartTraceReqInput(RpcReqInput):`
- members:
  - `output_file` — [`L807`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L807)
  - `req_num` — [`L806`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L806)
  - `request_id` — [`L808`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L808)
  - `save_tensor` — [`L809`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L809)
- protocol/private: `__post_init__`[`L811`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L811)
- uses (calls/refs, reference-scoped): [`RpcReqInput`](io_struct.md#RpcReqInput)
- used by: [`RpcReqInput`](io_struct.md#RpcReqInput)

### `StopTraceReqInput`  ·  implements/extends RpcReqInput
- def: [`python/sgl_jax/srt/managers/io_struct.py:817`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L817)
- doc: Request to stop precision tracing.
- signature: `class StopTraceReqInput(RpcReqInput):`
- members:
  - `request_id` — [`L820`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L820)
- protocol/private: `__post_init__`[`L822`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L822)
- uses (calls/refs, reference-scoped): [`RpcReqInput`](io_struct.md#RpcReqInput)
- used by: [`RpcReqInput`](io_struct.md#RpcReqInput)

### `TokenizedEmbeddingReqInput`
- def: [`python/sgl_jax/srt/managers/io_struct.py:640`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L640)
- doc: Tokenized embedding request input.
- signature: `class TokenizedEmbeddingReqInput:`
- members:
  - `input_ids` — [`L645`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L645)
  - `normalize` — [`L646`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L646)
  - `rid` — [`L643`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L643)
  - `text` — [`L644`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L644)

### `TokenizedGenerateReqInput`
- def: [`python/sgl_jax/srt/managers/io_struct.py:136`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L136)
- signature: `class TokenizedGenerateReqInput:`
- members:
  - `bootstrap_host` — [`L171`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L171)
  - `bootstrap_port` — [`L172`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L172)
  - `bootstrap_room` — [`L173`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L173)
  - `disagg_transfer_id` — [`L177`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L177)
  - `dp_rank` — [`L169`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L169)
  - `extra_key` — [`L161`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L161)
  - `input_ids` — [`L142`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L142)
  - `logprob_start_len` — [`L151`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L151)
  - `lora_id` — [`L159`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L159)
  - `mm_inputs` — [`L167`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L167)
  - `return_hidden_states` — [`L165`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L165)
  - `return_logprob` — [`L146`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L146)
  - `return_output_logprob_only` — [`L148`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L148)
  - `return_routed_experts` — [`L163`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L163)
  - `rid` — [`L138`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L138)
  - `sampling_params` — [`L144`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L144)
  - `stream` — [`L157`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L157)
  - `text` — [`L140`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L140)
  - `token_ids_logprob` — [`L155`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L155)
  - `top_logprobs_num` — [`L153`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L153)
- used by: [`handle_generate_request`](scheduler.md#Scheduler.handle_generate_request), [`_request_dispatcher`](scheduler.md#Scheduler._request_dispatcher), [`_estimate_req_tokens`](scheduler.md#Scheduler._estimate_req_tokens), [`select_dp_for_request`](scheduler.md#Scheduler.select_dp_for_request), [`_select_cache_aware_dp`](scheduler.md#Scheduler._select_cache_aware_dp), [`_get_input_token_len`](scheduler.md#Scheduler._get_input_token_len), [`pending_dp_reqs`](scheduler.md#Scheduler.pending_dp_reqs)

### `TraceStatusReqInput`  ·  implements/extends RpcReqInput
- def: [`python/sgl_jax/srt/managers/io_struct.py:828`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L828)
- doc: Request to get trace status.
- signature: `class TraceStatusReqInput(RpcReqInput):`
- members:
  - `request_id` — [`L831`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L831)
- protocol/private: `__post_init__`[`L833`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L833)
- uses (calls/refs, reference-scoped): [`RpcReqInput`](io_struct.md#RpcReqInput)
- used by: [`RpcReqInput`](io_struct.md#RpcReqInput)

### `VertexGenerateReqInput`  ·  implements/extends GenerateReqInput
- def: [`python/sgl_jax/srt/managers/io_struct.py:796`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L796)
- doc: Vertex AI compatible generate request input.
- signature: `class VertexGenerateReqInput(GenerateReqInput):`
- uses (calls/refs, reference-scoped): [`GenerateReqInput`](io_struct.md#GenerateReqInput)
- used by: [`GenerateReqInput`](io_struct.md#GenerateReqInput)

## Functions
- `has_valid_data(data)` — [`L20`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L20) — Check if data contains any valid content.

## Module values
- `AudioDataInputItem` — [`L237`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L237)
- `ImageDataInputItem` — [`L236`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L236)
- `MultimodalDataInputFormat` — [`L242`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L242)
- `MultimodalDataInputItem` — [`L240`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L240)
- `VideoDataInputItem` — [`L238`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/io_struct.py#L238)

