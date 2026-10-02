---
title: 'Module: easydel/inference/esurge/engine_types.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/engine_types.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.engine_types`/
symbols:
  EngineCoreEvent: EngineCoreEvent#
  FinishReason: FinishReason#
  EngineCoreOutputs: EngineCoreOutputs#
  EngineCoreEvent.new_event: EngineCoreEvent#new_event().
  EngineCoreOutput.finish_reason: EngineCoreOutput#finish_reason.
  EngineCoreOutput.finished: EngineCoreOutput#finished().
  EngineCoreOutputs.outputs: EngineCoreOutputs#outputs.
  EngineCoreEventType: EngineCoreEventType#
  EngineCoreOutput: EngineCoreOutput#
  EngineCoreRequest.sampling_params: EngineCoreRequest#sampling_params.
  EngineCoreOutput.new_prompt_logprobs_tensors: EngineCoreOutput#new_prompt_logprobs_tensors.
  EngineCoreOutput.events: EngineCoreOutput#events.
  EngineCoreOutputs.finished_requests: EngineCoreOutputs#finished_requests.
  EngineCoreOutput.new_logprobs: EngineCoreOutput#new_logprobs.
  UtilityOutput.result: UtilityOutput#result.
  EngineCoreOutputs.utility_output: EngineCoreOutputs#utility_output.
  EngineCoreOutputs.__post_init__: EngineCoreOutputs#__post_init__().
  FinishReason.LENGTH: FinishReason#LENGTH.
  EngineCoreRequest: EngineCoreRequest#
  EngineCoreOutput.request_id: EngineCoreOutput#request_id.
  EngineCoreOutput.new_token_ids: EngineCoreOutput#new_token_ids.
  EngineCoreOutputs.timestamp: EngineCoreOutputs#timestamp.
  FinishReason.STOP: FinishReason#STOP.
  FinishReason.ABORT: FinishReason#ABORT.
  EngineCoreRequest.request_id: EngineCoreRequest#request_id.
  EngineCoreRequest.prompt_token_ids: EngineCoreRequest#prompt_token_ids.
  EngineCoreRequest.eos_token_id: EngineCoreRequest#eos_token_id.
  EngineCoreRequest.arrival_time: EngineCoreRequest#arrival_time.
  EngineCoreRequest.client_index: EngineCoreRequest#client_index.
  EngineCoreRequest.priority: EngineCoreRequest#priority.
  EngineCoreRequest.pixel_values: EngineCoreRequest#pixel_values.
  EngineCoreRequest.image_grid_thw: EngineCoreRequest#image_grid_thw.
  EngineCoreRequest.pixel_values_videos: EngineCoreRequest#pixel_values_videos.
  EngineCoreRequest.video_grid_thw: EngineCoreRequest#video_grid_thw.
  EngineCoreOutput.stop_reason: EngineCoreOutput#stop_reason.
  EngineCoreOutput.num_cached_tokens: EngineCoreOutput#num_cached_tokens.
  UtilityResult: UtilityResult#
  UtilityOutput: UtilityOutput#
  FinishReason.__str__: FinishReason#__str__().
  EngineCoreRequest.data_parallel_rank: EngineCoreRequest#data_parallel_rank.
  EngineCoreRequest.current_wave: EngineCoreRequest#current_wave.
  EngineCoreEventType.QUEUED: EngineCoreEventType#QUEUED.
  EngineCoreEventType.SCHEDULED: EngineCoreEventType#SCHEDULED.
  EngineCoreEventType.PREEMPTED: EngineCoreEventType#PREEMPTED.
  EngineCoreEvent.timestamp: EngineCoreEvent#timestamp.
  UtilityResult.__init__: UtilityResult#__init__().
  UtilityResult.result: UtilityResult#result.
  UtilityOutput.call_id: UtilityOutput#call_id.
  UtilityOutput.failure_message: UtilityOutput#failure_message.
  EngineCoreOutputs.engine_index: EngineCoreOutputs#engine_index.
  EngineCoreOutputs.wave_complete: EngineCoreOutputs#wave_complete.
  EngineCoreOutputs.start_wave: EngineCoreOutputs#start_wave.
  EngineCoreRequestType: EngineCoreRequestType#
  EngineCoreRequestType.ADD: EngineCoreRequestType#ADD.
  EngineCoreRequestType.ABORT: EngineCoreRequestType#ABORT.
  EngineCoreRequestType.START_DP_WAVE: EngineCoreRequestType#START_DP_WAVE.
  EngineCoreRequestType.UTILITY: EngineCoreRequestType#UTILITY.
  EngineCoreRequestType.EXECUTOR_FAILED: EngineCoreRequestType#EXECUTOR_FAILED.
  ReconfigureDistributedRequest: ReconfigureDistributedRequest#
  ReconfigureDistributedRequest.new_data_parallel_size: ReconfigureDistributedRequest#new_data_parallel_size.
  ReconfigureDistributedRequest.new_data_parallel_rank: ReconfigureDistributedRequest#new_data_parallel_rank.
  ReconfigureDistributedRequest.new_data_parallel_rank_local: ReconfigureDistributedRequest#new_data_parallel_rank_local.
  ReconfigureDistributedRequest.new_data_parallel_master_ip: ReconfigureDistributedRequest#new_data_parallel_master_ip.
  ReconfigureDistributedRequest.new_data_parallel_master_port: ReconfigureDistributedRequest#new_data_parallel_master_port.
  ReconfigureRankType: ReconfigureRankType#
  ReconfigureRankType.KEEP_CURRENT_RANK: ReconfigureRankType#KEEP_CURRENT_RANK.
  ReconfigureRankType.SHUTDOWN_CURRENT_RANK: ReconfigureRankType#SHUTDOWN_CURRENT_RANK.
---
# Module: [`easydel/inference/esurge/engine_types.py`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py)

## Classes
### `EngineCoreEvent`  ·  implements/extends Struct
- def: [`easydel/inference/esurge/engine_types.py:165`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L165)
- doc: Timestamped engine core event.
- signature: `class EngineCoreEvent(msgspec.Struct):`
- members:
  - `new_event(cls, event_type: EngineCoreEventType, timestamp: float | None = None)` — [`L189`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L189) — Create a new engine event.
  - `timestamp` — [`L186`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L186)
- uses (calls/refs, reference-scoped): [`EngineCoreEventType`](engine_types.md#EngineCoreEventType)
- used by: [`record_event`](request.md#EngineRequest.record_event), [`events`](request.md#EngineRequest.events), [`take_events`](request.md#EngineRequest.take_events), [`events`](engine_types.md#EngineCoreOutput.events)

### `EngineCoreEventType`  ·  implements/extends IntEnum
- def: [`easydel/inference/esurge/engine_types.py:145`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L145)
- doc: Types of engine core events.
- signature: `class EngineCoreEventType(enum.IntEnum):`
- members:
  - `PREEMPTED` — [`L162`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L162)
  - `QUEUED` — [`L160`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L160)
  - `SCHEDULED` — [`L161`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L161)
- used by: [`record_event`](request.md#EngineRequest.record_event), [`EngineCoreEvent`](engine_types.md#EngineCoreEvent), [`new_event`](engine_types.md#EngineCoreEvent.new_event)

### `EngineCoreOutput`  ·  implements/extends Struct
- def: [`easydel/inference/esurge/engine_types.py:213`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L213)
- doc: Output from engine core processing.
- signature: `class EngineCoreOutput(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):`
- members:
  - `finished(self)` — [`L247`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L247) — Check if generation has finished.
  - `events` — [`L243`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L243)
  - `finish_reason` — [`L241`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L241)
  - `new_logprobs` — [`L239`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L239)
  - `new_prompt_logprobs_tensors` — [`L240`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L240)
  - `new_token_ids` — [`L238`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L238)
  - `num_cached_tokens` — [`L244`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L244)
  - `request_id` — [`L237`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L237)
  - `stop_reason` — [`L242`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L242)
- uses (calls/refs, reference-scoped): [`EngineCoreEvent`](engine_types.md#EngineCoreEvent), [`FinishReason`](engine_types.md#FinishReason), [`LogprobsLists`](outputs.md#LogprobsLists), [`LogprobsTensors`](outputs.md#LogprobsTensors)
- used by: [`update_from_output`](scheduler/scheduler.md#Scheduler.update_from_output), [`_process_engine_outputs`](mixins/parsing.md#EngineParsingMixin._process_engine_outputs), [`outputs`](engine_types.md#EngineCoreOutputs.outputs)

### `EngineCoreOutputs`  ·  implements/extends Struct
- def: [`easydel/inference/esurge/engine_types.py:302`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L302)
- doc: Batch of engine outputs.
- signature: `class EngineCoreOutputs(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):`
- members:
  - `__post_init__(self)` — [`L333`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L333) — Set timestamp to current monotonic time if not provided.
  - `engine_index` — [`L325`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L325)
  - `finished_requests` — [`L329`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L329)
  - `outputs` — [`L326`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L326)
  - `start_wave` — [`L331`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L331)
  - `timestamp` — [`L327`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L327)
  - `utility_output` — [`L328`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L328)
  - `wave_complete` — [`L330`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L330)
- uses (calls/refs, reference-scoped): [`EngineCoreOutput`](engine_types.md#EngineCoreOutput), [`UtilityOutput`](engine_types.md#UtilityOutput)
- used by: [`update_from_output`](scheduler/scheduler.md#Scheduler.update_from_output), [`_process_engine_outputs`](mixins/parsing.md#EngineParsingMixin._process_engine_outputs), [`update_from_output`](scheduler/interface.md#SchedulerInterface.update_from_output)

### `EngineCoreRequest`  ·  implements/extends Struct
- def: [`easydel/inference/esurge/engine_types.py:96`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L96)
- doc: Core request structure for engine processing.
- signature: `class EngineCoreRequest(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):`
- members:
  - `arrival_time` — [`L133`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L133)
  - `client_index` — [`L135`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L135)
  - `current_wave` — [`L136`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L136)
  - `data_parallel_rank` — [`L134`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L134)
  - `eos_token_id` — [`L132`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L132)
  - `image_grid_thw` — [`L140`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L140)
  - `pixel_values` — [`L139`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L139)
  - `pixel_values_videos` — [`L141`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L141)
  - `priority` — [`L137`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L137)
  - `prompt_token_ids` — [`L130`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L130)
  - `request_id` — [`L129`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L129)
  - `sampling_params` — [`L131`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L131)
  - `video_grid_thw` — [`L142`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L142)
- uses (calls/refs, reference-scoped): [`SamplingParams`](../sampling_params.md#SamplingParams)
- used by: [`from_engine_core_request`](request.md#EngineRequest.from_engine_core_request)

### `EngineCoreRequestType`  ·  implements/extends Enum
- def: [`easydel/inference/esurge/engine_types.py:339`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L339)
- doc: Engine request types defined as hex byte strings.
- signature: `class EngineCoreRequestType(enum.Enum):`
- members:
  - `ABORT` — [`L358`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L358)
  - `ADD` — [`L357`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L357)
  - `EXECUTOR_FAILED` — [`L362`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L362)
  - `START_DP_WAVE` — [`L359`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L359)
  - `UTILITY` — [`L360`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L360)

### `FinishReason`  ·  implements/extends IntEnum
- def: [`easydel/inference/esurge/engine_types.py:66`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L66)
- doc: Reason why text generation finished.
- signature: `class FinishReason(enum.IntEnum):`
- members:
  - `__str__(self)` — [`L87`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L87) — Return human-readable string representation.
  - `ABORT` — [`L85`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L85)
  - `LENGTH` — [`L84`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L84)
  - `STOP` — [`L83`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L83)
- used by: [`_FINISHED_REASON_MAP`](request.md#_FINISHED_REASON_MAP), [`get_finished_reason`](request.md#EngineRequest.get_finished_reason), [`get_finished_reason`](request.md#EngineRequestStatus.get_finished_reason), [`finish_reason`](engine_types.md#EngineCoreOutput.finish_reason)

### `ReconfigureDistributedRequest`  ·  implements/extends Struct
- def: [`easydel/inference/esurge/engine_types.py:365`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L365)
- doc: Request to reconfigure distributed processing setup.
- signature: `class ReconfigureDistributedRequest(msgspec.Struct):`
- members:
  - `new_data_parallel_master_ip` — [`L391`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L391)
  - `new_data_parallel_master_port` — [`L392`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L392)
  - `new_data_parallel_rank` — [`L389`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L389)
  - `new_data_parallel_rank_local` — [`L390`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L390)
  - `new_data_parallel_size` — [`L388`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L388)

### `ReconfigureRankType`  ·  implements/extends IntEnum
- def: [`easydel/inference/esurge/engine_types.py:395`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L395)
- doc: Rank type for reconfiguring distributed request.
- signature: `class ReconfigureRankType(enum.IntEnum):`
- members:
  - `KEEP_CURRENT_RANK` — [`L410`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L410)
  - `SHUTDOWN_CURRENT_RANK` — [`L411`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L411)

### `UtilityOutput`  ·  implements/extends Struct
- def: [`easydel/inference/esurge/engine_types.py:279`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L279)
- doc: Output from utility operations.
- signature: `class UtilityOutput(msgspec.Struct, array_like=True, gc=False):`
- members:
  - `call_id` — [`L296`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L296)
  - `failure_message` — [`L298`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L298)
  - `result` — [`L299`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L299)
- uses (calls/refs, reference-scoped): [`UtilityResult`](engine_types.md#UtilityResult)
- used by: [`utility_output`](engine_types.md#EngineCoreOutputs.utility_output)

### `UtilityResult`
- def: [`easydel/inference/esurge/engine_types.py:256`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L256)
- doc: Wrapper for special serialization/deserialization handling.
- signature: `class UtilityResult:`
- members:
  - `__init__(self, r: Any = None)` — [`L270`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L270) — Initialize with an optional result object.
  - `result` — [`L276`](../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/engine_types.py#L276)
- used by: [`result`](engine_types.md#UtilityOutput.result)

