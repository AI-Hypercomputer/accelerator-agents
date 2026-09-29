---
title: 'Module: easydel/inference/stream_protocol.py'
type: catalog
provenance: extracted
module: easydel/inference/stream_protocol.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.stream_protocol`/
symbols:
  ResponsesStreamAccumulator.finalize: ResponsesStreamAccumulator#finalize().
  ResponsesStreamAccumulator.add_output: ResponsesStreamAccumulator#add_output().
  iter_chat_completion_stream_responses: iter_chat_completion_stream_responses().
  ResponsesStreamAccumulator._ensure_message_item: ResponsesStreamAccumulator#_ensure_message_item().
  ResponsesStreamAccumulator._resolve_function_state: ResponsesStreamAccumulator#_resolve_function_state().
  build_responses_object: build_responses_object().
  ResponsesStreamAccumulator.initial_frames: ResponsesStreamAccumulator#initial_frames().
  ResponsesStreamAccumulator._ensure_reasoning_item: ResponsesStreamAccumulator#_ensure_reasoning_item().
  build_responses_function_call_items: build_responses_function_call_items().
  ResponsesStreamAccumulator._frame: ResponsesStreamAccumulator#_frame().
  ResponsesStreamAccumulator.message_state: ResponsesStreamAccumulator#message_state.
  iter_responses_stream_frames: iter_responses_stream_frames().
  ResponsesStreamAccumulator.reasoning_state: ResponsesStreamAccumulator#reasoning_state.
  coerce_stream_delta_message: coerce_stream_delta_message().
  build_responses_message_item: build_responses_message_item().
  build_responses_output_items: build_responses_output_items().
  build_responses_reasoning_item: build_responses_reasoning_item().
  ResponsesStreamAccumulator.reasoning_text_accum: ResponsesStreamAccumulator#reasoning_text_accum.
  responses_assistant_message_from_output_items: responses_assistant_message_from_output_items().
  ResponsesStreamAccumulator._primary_output: ResponsesStreamAccumulator#_primary_output().
  RequestOutputLike: RequestOutputLike#
  compute_stream_delta_text: compute_stream_delta_text().
  ResponsesStreamAccumulator.final_response_overrides: ResponsesStreamAccumulator#final_response_overrides.
  ResponsesStreamAccumulator.include_reasoning_summary: ResponsesStreamAccumulator#include_reasoning_summary.
  prompt_token_count_from_output: prompt_token_count_from_output().
  RequestOutputLike.outputs: RequestOutputLike#outputs.
  normalize_tool_calls: normalize_tool_calls().
  ResponsesStreamAccumulator.output_items_stream: ResponsesStreamAccumulator#output_items_stream.
  ResponsesStreamAccumulator.next_output_index: ResponsesStreamAccumulator#next_output_index.
  normalize_delta_tool_calls: normalize_delta_tool_calls().
  ResponsesStreamAccumulator.function_states: ResponsesStreamAccumulator#function_states.
  CompletionOutputLike.tool_calls: CompletionOutputLike#tool_calls.
  RequestOutputLike.tool_calls: RequestOutputLike#tool_calls.
  jsonify_tool_calls: jsonify_tool_calls().
  should_emit_responses_message_item: should_emit_responses_message_item().
  RequestOutputLike.delta_text: RequestOutputLike#delta_text.
  ResponsesStreamAccumulator.response_id: ResponsesStreamAccumulator#response_id.
  ResponsesStreamAccumulator.saw_function_call_delta: ResponsesStreamAccumulator#saw_function_call_delta.
  ResponsesStreamAccumulator.__init__: ResponsesStreamAccumulator#__init__().
  CompletionOutputLike: CompletionOutputLike#
  CompletionOutputLike.reasoning_content: CompletionOutputLike#reasoning_content.
  RequestOutputLike.num_generated_tokens: RequestOutputLike#num_generated_tokens.
  RequestOutputLike.first_token_time: RequestOutputLike#first_token_time.
  RequestOutputLike.delta_reasoning_content: RequestOutputLike#delta_reasoning_content.
  RequestOutputLike.delta_tool_calls: RequestOutputLike#delta_tool_calls.
  RequestOutputLike.reasoning_content: RequestOutputLike#reasoning_content.
  ResponsesStreamAccumulator.model: ResponsesStreamAccumulator#model.
  ResponsesStreamAccumulator.created_at: ResponsesStreamAccumulator#created_at.
  ResponsesStreamAccumulator.function_order: ResponsesStreamAccumulator#function_order.
  ResponsesStreamAccumulator.message_text_accum: ResponsesStreamAccumulator#message_text_accum.
  ResponsesStreamAccumulator._json_dump_arguments: ResponsesStreamAccumulator#_json_dump_arguments().
  CompletionOutputLike.finish_reason: CompletionOutputLike#finish_reason.
  RequestOutputLike.prompt_token_ids: RequestOutputLike#prompt_token_ids.
  RequestOutputLike.tokens_per_second: RequestOutputLike#tokens_per_second.
  RequestOutputLike.processing_time: RequestOutputLike#processing_time.
  RequestOutputLike.accumulated_text: RequestOutputLike#accumulated_text.
  ResponsesStreamAccumulator: ResponsesStreamAccumulator#
---
# Module: [`easydel/inference/stream_protocol.py`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py)

## Classes
### `CompletionOutputLike`  ·  implements/extends Protocol
- def: [`easydel/inference/stream_protocol.py:84`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L84)
- doc: Structural protocol describing a single completion sample from the engine.
- signature: `class CompletionOutputLike(Protocol):`
- members:
  - `finish_reason` — [`L93`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L93)
  - `reasoning_content` — [`L95`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L95)
  - `tool_calls` — [`L94`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L94)
- uses (calls/refs, reference-scoped): [`ToolCall`](openai_api_modules.md#ToolCall)
- used by: [`finalize`](stream_protocol.md#ResponsesStreamAccumulator.finalize), [`add_output`](stream_protocol.md#ResponsesStreamAccumulator.add_output), [`iter_chat_completion_stream_responses`](stream_protocol.md#iter_chat_completion_stream_responses), [`_primary_output`](stream_protocol.md#ResponsesStreamAccumulator._primary_output), [`outputs`](stream_protocol.md#RequestOutputLike.outputs)

### `RequestOutputLike`  ·  implements/extends Protocol
- def: [`easydel/inference/stream_protocol.py:98`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L98)
- doc: Structural protocol describing an engine output snapshot.
- signature: `class RequestOutputLike(Protocol):`
- members:
  - `accumulated_text` — [`L119`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L119)
  - `delta_reasoning_content` — [`L116`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L116)
  - `delta_text` — [`L115`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L115)
  - `delta_tool_calls` — [`L117`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L117)
  - `first_token_time` — [`L114`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L114)
  - `num_generated_tokens` — [`L111`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L111)
  - `outputs` — [`L110`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L110)
  - `processing_time` — [`L113`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L113)
  - `prompt_token_ids` — [`L109`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L109)
  - `reasoning_content` — [`L118`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L118)
  - `tokens_per_second` — [`L112`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L112)
  - `tool_calls` — [`L120`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L120)
- uses (calls/refs, reference-scoped): [`ToolCall`](openai_api_modules.md#ToolCall), [`CompletionOutputLike`](stream_protocol.md#CompletionOutputLike)
- used by: [`finalize`](stream_protocol.md#ResponsesStreamAccumulator.finalize), [`add_output`](stream_protocol.md#ResponsesStreamAccumulator.add_output), [`iter_chat_completion_stream_responses`](stream_protocol.md#iter_chat_completion_stream_responses), [`iter_responses_stream_frames`](stream_protocol.md#iter_responses_stream_frames), [`iter_responses_stream`](esurge/mixins/io.md#EngineIOMixin.iter_responses_stream), [`iter_chat_completion_stream`](esurge/mixins/io.md#EngineIOMixin.iter_chat_completion_stream), [`_primary_output`](stream_protocol.md#ResponsesStreamAccumulator._primary_output), [`prompt_token_count_from_output`](stream_protocol.md#prompt_token_count_from_output)

### `ResponsesStreamAccumulator`
- def: [`easydel/inference/stream_protocol.py:556`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L556)
- doc: Stateful builder that converts engine output snapshots into Responses API SSE frames.
- signature: `class ResponsesStreamAccumulator:`
- members:
  - `add_output(self, output: RequestOutputLike)` — [`L731`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L731) — Process one engine output snapshot and emit the corresponding SSE frames.
  - `finalize(self, *, last_output: RequestOutputLike, prompt_tokens: int)` — [`L835`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L835) — Close the stream by emitting all ``*.done`` events and ``response.completed``.
  - `initial_frames(self)` — [`L632`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L632) — Return the opening ``response.created`` event that starts the SSE stream.
  - `created_at` — [`L602`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L602)
  - `final_response_overrides` — [`L597`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L597)
  - `function_order` — [`L611`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L611)
  - `function_states` — [`L610`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L610)
  - `include_reasoning_summary` — [`L595`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L595)
  - `message_state` — [`L614`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L614)
  - `message_text_accum` — [`L615`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L615)
  - `model` — [`L594`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L594)
  - `next_output_index` — [`L605`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L605)
  - `output_items_stream` — [`L604`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L604)
  - `reasoning_state` — [`L607`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L607)
  - `reasoning_text_accum` — [`L608`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L608)
  - `response_id` — [`L593`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L593)
  - `saw_function_call_delta` — [`L612`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L612)
- protocol/private: `__init__`[`L584`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L584), `_ensure_message_item`[`L669`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L669), `_ensure_reasoning_item`[`L650`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L650), `_frame`[`L629`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L629), `_json_dump_arguments`[`L622`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L622), `_primary_output`[`L618`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L618), `_resolve_function_state`[`L696`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L696)
- uses (calls/refs, reference-scoped): [`index`](openai_api_modules.md#DeltaToolCall.index), [`arguments`](openai_api_modules.md#DeltaFunctionCall.arguments), [`arguments`](openai_api_modules.md#FunctionCall.arguments), [`name`](openai_api_modules.md#FunctionCall.name), [`build_responses_object`](stream_protocol.md#build_responses_object), [`id`](openai_api_modules.md#DeltaToolCall.id), [`as_update_dict`](typed_models.md#ResponsesFinalizationOptions.as_update_dict), [`name`](openai_api_modules.md#DeltaFunctionCall.name), [`ResponsesOutputItem`](typed_models.md#ResponsesOutputItem), [`StreamEventFrame`](typed_models.md#StreamEventFrame), [`id`](openai_api_modules.md#ToolCall.id), [`ResponseFunctionCallItem`](typed_models.md#ResponseFunctionCallItem), [`build_responses_reasoning_item`](stream_protocol.md#build_responses_reasoning_item), [`ResponseMessageItem`](typed_models.md#ResponseMessageItem), [`ResponsesResponse`](typed_models.md#ResponsesResponse), [`item`](typed_models.md#FunctionCallStreamState.item), [`ResponsesFinalizationOptions`](typed_models.md#ResponsesFinalizationOptions), [`RequestOutputLike`](stream_protocol.md#RequestOutputLike), [`compute_stream_delta_text`](stream_protocol.md#compute_stream_delta_text), [`ResponseOutputTextPart`](typed_models.md#ResponseOutputTextPart), [`item`](typed_models.md#ReasoningStreamState.item), [`arguments`](typed_models.md#ResponseFunctionCallItem.arguments), [`item_id`](typed_models.md#FunctionCallStreamState.item_id), [`item_id`](typed_models.md#MessageStreamState.item_id), [`ResponseCompletedEvent`](typed_models.md#ResponseCompletedEvent), [`ResponseOutputItemAddedEvent`](typed_models.md#ResponseOutputItemAddedEvent), [`ResponseOutputItemDoneEvent`](typed_models.md#ResponseOutputItemDoneEvent), [`item`](typed_models.md#MessageStreamState.item), [`item_id`](typed_models.md#ReasoningStreamState.item_id), [`ResponseFunctionCallArgumentsDeltaEvent`](typed_models.md#ResponseFunctionCallArgumentsDeltaEvent), [`content`](typed_models.md#ResponseMessageItem.content), [`normalize_tool_calls`](stream_protocol.md#normalize_tool_calls), [`outputs`](stream_protocol.md#RequestOutputLike.outputs), [`name`](typed_models.md#ResponseFunctionCallItem.name), [`output_index`](typed_models.md#FunctionCallStreamState.output_index), [`output_index`](typed_models.md#MessageStreamState.output_index), [`ResponseContentPartAddedEvent`](typed_models.md#ResponseContentPartAddedEvent), [`ResponseCreatedEvent`](typed_models.md#ResponseCreatedEvent), [`ResponseFunctionCallArgumentsDoneEvent`](typed_models.md#ResponseFunctionCallArgumentsDoneEvent), [`ResponseOutputTextDeltaEvent`](typed_models.md#ResponseOutputTextDeltaEvent)  (+70 more)
- used by: [`iter_responses_stream_frames`](stream_protocol.md#iter_responses_stream_frames)

## Functions
- `build_responses_function_call_items(tool_calls: list[tp.Any] | None)` — [`L307`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L307) — Convert a list of raw/normalized tool calls into ``ResponseFunctionCallItem`` instances.
- `build_responses_message_item(output_text: str)` — [`L335`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L335) — Create a ``ResponseMessageItem`` containing the given assistant text.
- `build_responses_object(*, response_id: str, model: str, output_text: str, prompt_tokens: int, completion_tokens: int, tool_calls: list[tp.Any] | None = None, reasoning_text: str | None = None, include_reasoning_summary: bool = False, output_items: list[ResponsesOutputItem] | None = None, created_at: int | None = None)` — [`L400`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L400) — Build a complete, non-streaming ``ResponsesResponse`` object.
- `build_responses_output_items(*, output_text: str, tool_calls: list[tp.Any] | None = None, reasoning_text: str | None = None, include_reasoning_summary: bool = False)` — [`L363`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L363) — Assemble the complete ``output`` list for a finished ``ResponsesResponse``.
- `build_responses_reasoning_item(reasoning_text: str)` — [`L297`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L297) — Create a ``ResponseReasoningItem`` wrapping the given reasoning text.
- `coerce_stream_delta_message(delta_message: tp.Any, *, fallback_text: str = "", default_role: str | None = None)` — [`L241`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L241) — Normalize an engine/parser streaming delta into a safe ``DeltaMessage``.
- `compute_stream_delta_text(current_text: str, previous_text: str, fallback_delta: str)` — [`L123`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L123) — Compute a safe streaming delta from two accumulated text snapshots.
- `iter_chat_completion_stream_responses(outputs: tp.Iterator[RequestOutputLike], *, model: str)` — [`L446`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L446) — Convert a stream of engine output snapshots into OpenAI Chat Completion SSE chunks.
- `iter_responses_stream_frames(outputs: tp.Iterator[RequestOutputLike], *, response_id: str, model: str, include_reasoning_summary: bool = False, final_response_overrides: ResponsesFinalizationOptions | dict[str, tp.Any] | None = None, created_at: int | None = None)` — [`L1023`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L1023) — Convert a stream of engine output snapshots into Responses API SSE event frames.
- `jsonify_tool_calls(tool_calls: tp.Any)` — [`L227`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L227) — Normalize tool calls and serialize each to a JSON-safe dict.
- `normalize_delta_tool_calls(tool_calls: tp.Any)` — [`L204`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L204) — Validate and coerce heterogeneous delta-tool-call data into ``DeltaToolCall`` instances.
- `normalize_tool_calls(tool_calls: tp.Any)` — [`L179`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L179) — Validate and coerce heterogeneous tool-call data into ``ToolCall`` instances.
- `prompt_token_count_from_output(output: RequestOutputLike)` — [`L163`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L163) — Extract the total prompt token count from a ``RequestOutputLike``.
- `responses_assistant_message_from_output_items(output_items: list[ResponsesOutputItem])` — [`L388`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L388) — Convert Responses API output items back into a ``ChatMessage``.
- `should_emit_responses_message_item(output_text: str, tool_calls: list[tp.Any] | None = None)` — [`L348`](../../../../../../raw/code/EasyDeL/easydel/inference/stream_protocol.py#L348) — Decide whether a message output item should be included in the response.

