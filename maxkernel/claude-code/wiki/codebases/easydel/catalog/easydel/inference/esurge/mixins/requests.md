---
title: 'Module: easydel/inference/esurge/mixins/requests.py'
type: catalog
provenance: extracted
module: easydel/inference/esurge/mixins/requests.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.esurge.mixins.requests`/
symbols:
  EngineRequestsMixin._add_request: EngineRequestsMixin#_add_request().
  EngineRequestsMixin.abort_request: EngineRequestsMixin#abort_request().
  EngineRequestsMixin._request_counter: EngineRequestsMixin#_request_counter.
  _set_requested_new: _set_requested_new().
  EngineRequestsMixin: EngineRequestsMixin#
  EngineRequestsMixin._configure_reasoning_parser_for_prompt: EngineRequestsMixin#_configure_reasoning_parser_for_prompt().
  EngineRequestsMixin._cleanup_detokenizer_state: EngineRequestsMixin#_cleanup_detokenizer_state().
  EngineRequestsMixin._get_requested_new: EngineRequestsMixin#_get_requested_new().
  EngineRequestsMixin.num_pending_requests: EngineRequestsMixin#num_pending_requests().
  EngineRequestsMixin.num_running_requests: EngineRequestsMixin#num_running_requests().
  EngineRequestsMixin._generate_request_id: EngineRequestsMixin#_generate_request_id().
---
# Module: [`easydel/inference/esurge/mixins/requests.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/requests.py)

## Classes
### `EngineRequestsMixin`
- def: [`easydel/inference/esurge/mixins/requests.py:42`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/requests.py#L42)
- doc: Mixin for request lifecycle management in the eSurge engine.
- signature: `class EngineRequestsMixin:`
- members:
  - `_add_request(self, request_id: str, prompt: str, sampling_params: SamplingParams, prompt_token_ids: list[int] | None = None, tool_parser_request: Any | None = None, pixel_values: Any | None = None, image_grid_thw: Any | None = None, pixel_values_videos: Any | None = None, video_grid_thw: Any | None = None, mm_features: list | None = None)` — [`L71`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/requests.py#L71) — Add a new request to the scheduler queue with intelligent context management.
  - `_cleanup_detokenizer_state(self)` — [`L594`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/requests.py#L594) — Attempt to clean up failed detokenizer states.
  - `_configure_reasoning_parser_for_prompt(self, reasoning_parser: Any | None, prompt_text: str, prompt_token_ids: Sequence[int])` — [`L57`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/requests.py#L57) — Configure prompt context on a request-scoped reasoning parser.
  - `_generate_request_id(self)` — [`L441`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/requests.py#L441) — Generate a unique request ID with overflow protection.
  - `abort_request(self, request_id: str)` — [`L458`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/requests.py#L458) — Abort an in-progress request.
  - `num_pending_requests(self)` — [`L629`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/requests.py#L629) — Get the number of requests waiting in queue.
  - `num_running_requests(self)` — [`L639`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/requests.py#L639) — Get the number of actively running requests.
- protocol/private: `_get_requested_new`[`L122`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/requests.py#L122), `_request_counter`[`L453`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/requests.py#L453)
- uses (calls/refs, reference-scoped): [`SamplingParams`](../../sampling_params.md#SamplingParams), [`logger`](../logger.md#logger), [`EngineRequest`](../request.md#EngineRequest), [`eSurge`](../esurge_engine.md#eSurge), [`EngineRequestStatus`](../request.md#EngineRequestStatus), [`RequestOutput`](../esurge_engine.md#RequestOutput), [`complete_request`](../metrics.md#MetricsCollector.complete_request), [`get_metrics_collector`](../metrics.md#get_metrics_collector), [`start_request`](../metrics.md#MetricsCollector.start_request), [`outputs`](../esurge_engine.md#RequestOutput.outputs), [`num_generated_tokens`](../esurge_engine.md#RequestOutput.num_generated_tokens), [`accumulated_text`](../esurge_engine.md#RequestOutput.accumulated_text), [`CompletionOutput`](../esurge_engine.md#CompletionOutput), [`finished`](../esurge_engine.md#RequestOutput.finished), [`log_metrics_summary`](../metrics.md#log_metrics_summary), [`tokens_per_second`](../esurge_engine.md#RequestOutput.tokens_per_second), [`first_token_time`](../esurge_engine.md#RequestOutput.first_token_time), [`processing_time`](../esurge_engine.md#RequestOutput.processing_time), [`stop_token_ids`](../../sampling_params.md#SamplingParams.stop_token_ids), [`_set_requested_new`](requests.md#_set_requested_new), [`prompt_token_ids`](../esurge_engine.md#RequestOutput.prompt_token_ids), [`request_id`](../esurge_engine.md#RequestOutput.request_id), [`_all_stop_token_ids`](../../sampling_params.md#SamplingParams._all_stop_token_ids), [`delta_text`](../esurge_engine.md#RequestOutput.delta_text), [`text`](../esurge_engine.md#CompletionOutput.text), [`DelegatingParser`](../../parsing/delegating_parser.md#DelegatingParser), [`time_spent_generating`](../esurge_engine.md#RequestOutput.time_spent_generating), [`token_ids`](../esurge_engine.md#CompletionOutput.token_ids), [`FINISHED_ABORTED`](../request.md#EngineRequestStatus.FINISHED_ABORTED), [`delta_seq`](../esurge_engine.md#RequestOutput.delta_seq), [`index`](../esurge_engine.md#CompletionOutput.index), [`prompt`](../esurge_engine.md#RequestOutput.prompt), [`raw_accumulated_text`](../esurge_engine.md#RequestOutput.raw_accumulated_text), [`raw_delta_text`](../esurge_engine.md#RequestOutput.raw_delta_text), [`truncate_tokens`](../utils.md#truncate_tokens), [`update_seq`](../esurge_engine.md#RequestOutput.update_seq)
- used by: [`generate_stream`](../server/api_server.md#eSurgeApiServer.generate_stream), [`eSurge`](../esurge_engine.md#eSurge), [`get_esurge`](../../../infra/mixins/generation.md#EasyGenerationMixin.get_esurge), [`_abort_non_stream_request_on_disconnect`](../server/api_server.md#eSurgeApiServer._abort_non_stream_request_on_disconnect), [`_abort_request_after_handler_cancel`](../server/api_server.md#eSurgeApiServer._abort_request_after_handler_cancel)

## Functions
- `_set_requested_new(sp, n: int)` — [`L34`](../../../../../../../../raw/code/EasyDeL/easydel/inference/esurge/mixins/requests.py#L34) — Set the max_tokens or max_new_tokens attribute on a SamplingParams object.

