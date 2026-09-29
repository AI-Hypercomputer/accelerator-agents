---
title: 'Module: easydel/inference/parsing/delegating_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/parsing/delegating_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.parsing.delegating_parser`/
symbols:
  DelegatingParser._process_tool_delta: DelegatingParser#_process_tool_delta().
  DelegatingParser._process_reasoning_delta: DelegatingParser#_process_reasoning_delta().
  DelegatingParser._process_tool_final: DelegatingParser#_process_tool_final().
  DelegatingParser.process_final: DelegatingParser#process_final().
  DelegatingParser.process_delta: DelegatingParser#process_delta().
  DelegatingParser._build_missing_final_tool_deltas: DelegatingParser#_build_missing_final_tool_deltas().
  DelegatingParser._merge_streamed_tool_call_state: DelegatingParser#_merge_streamed_tool_call_state().
  DelegatingParser.phase: DelegatingParser#phase.
  ParseResult.accumulated_content: ParseResult#accumulated_content.
  ParsePhase: ParsePhase#
  ParseResult.delta_content: ParseResult#delta_content.
  DelegatingParser._get_tool_request: DelegatingParser#_get_tool_request().
  ParseResult.to_dict: ParseResult#to_dict().
  DelegatingParser._raw_content_text: DelegatingParser#_raw_content_text.
  ParseResult.accumulated_reasoning: ParseResult#accumulated_reasoning.
  ParseResult.phase: ParseResult#phase.
  DelegatingParser._accumulated_reasoning: DelegatingParser#_accumulated_reasoning.
  ParseResult: ParseResult#
  DelegatingParser.reasoning_parser: DelegatingParser#reasoning_parser.
  DelegatingParser.__init__: DelegatingParser#__init__().
  DelegatingParser._compute_visible_content_delta: DelegatingParser#_compute_visible_content_delta().
  DelegatingParser.tool_parser: DelegatingParser#tool_parser.
  DelegatingParser._accumulated_content: DelegatingParser#_accumulated_content.
  ParsePhase.REASONING: ParsePhase#REASONING.
  ParsePhase.CONTENT: ParsePhase#CONTENT.
  DelegatingParser._canonicalize_reasoning_content: DelegatingParser#_canonicalize_reasoning_content().
  ParsePhase.TOOL_CALL: ParsePhase#TOOL_CALL.
  ParseResult.delta_tool_calls: ParseResult#delta_tool_calls.
  DelegatingParser.tool_request: DelegatingParser#tool_request.
  DelegatingParser._streamed_tool_call_state: DelegatingParser#_streamed_tool_call_state.
  DelegatingParser._is_tools_enabled: DelegatingParser#_is_tools_enabled().
  DelegatingParser._tokenize_for_tool_view: DelegatingParser#_tokenize_for_tool_view().
  logger: logger.
  ParsePhase.BUFFERING: ParsePhase#BUFFERING.
  ParseResult.delta_reasoning: ParseResult#delta_reasoning.
  TrackedToolCallState.function_arguments: TrackedToolCallState#function_arguments.
  DelegatingParser: DelegatingParser#
  DelegatingParser._tool_previous_text: DelegatingParser#_tool_previous_text.
  DelegatingParser._tool_previous_token_ids: DelegatingParser#_tool_previous_token_ids.
  DelegatingParser._content_committed: DelegatingParser#_content_committed.
  ParseResult.tool_calls: ParseResult#tool_calls.
  TrackedToolCallState: TrackedToolCallState#
  TrackedToolCallState.function_name: TrackedToolCallState#function_name.
  TrackedToolCallState.index: TrackedToolCallState#index.
  TrackedToolCallState.tool_call_id: TrackedToolCallState#tool_call_id.
  TrackedToolCallState.tool_type: TrackedToolCallState#tool_type.
---
# Module: [`easydel/inference/parsing/delegating_parser.py`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py)

## Classes
### `DelegatingParser`
- def: [`easydel/inference/parsing/delegating_parser.py:125`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L125)
- doc: Orchestrates reasoning and tool parsers with an explicit phase state machine.
- signature: `class DelegatingParser:`
- members:
  - `_build_missing_final_tool_deltas(self, tool_calls: list | None)` — [`L198`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L198) — Build final delta_tool_calls for any tool data not streamed earlier.
  - `_canonicalize_reasoning_content(self, *, accumulated_text: str, fallback_content: str)` — [`L246`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L246) — Return the canonical visible content after reasoning extraction.
  - `_compute_visible_content_delta(current_text: str, previous_text: str, fallback_delta: str = "")` — [`L160`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L160) — Compute a canonical visible-text delta from accumulated snapshots.
  - `_get_tool_request(self)` — [`L495`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L495) — Return the tool request, creating a dummy one if needed.
  - `_is_tools_enabled(self)` — [`L504`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L504) — Check whether tool calling is actually enabled for this request.
  - `_merge_streamed_tool_call_state(self, delta_tool_calls: list | None)` — [`L168`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L168) — Accumulate streamed tool-call metadata for final delta synthesis.
  - `_process_reasoning_delta(self, result: ParseResult, accumulated_text: str, delta_text: str, token_ids: list[int], prev_text: str, prev_token_ids: list[int])` — [`L382`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L382) — Run reasoning parser on streaming delta. Returns (content_text, content_delta).
  - `_process_tool_delta(self, result: ParseResult, content_text: str, content_delta: str, token_ids: list[int])` — [`L516`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L516) — Run tool parser on a streaming content delta. Mutates *result* in place.
  - `_process_tool_final(self, result: ParseResult, content_for_tools: str)` — [`L598`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L598) — Run tool parser batch extraction on the final content. Mutates *result*.
  - `_tokenize_for_tool_view(self, text: str)` — [`L629`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L629) — Tokenize text for the tool parser's independent view.
  - `process_delta(self, accumulated_text: str, delta_text: str, token_ids: list[int], prev_text: str, prev_token_ids: list[int])` — [`L262`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L262) — Process a streaming delta through reasoning then tool parsing.
  - `process_final(self, accumulated_text: str, token_ids: list[int])` — [`L321`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L321) — Process the final (finished) output through both parsers in batch mode.
  - `phase` — [`L147`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L147)
  - `reasoning_parser` — [`L143`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L143)
  - `tool_parser` — [`L144`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L144)
  - `tool_request` — [`L145`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L145)
- protocol/private: `__init__`[`L137`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L137), `_accumulated_content`[`L150`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L150), `_accumulated_reasoning`[`L149`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L149), `_content_committed`[`L157`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L157), `_raw_content_text`[`L151`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L151), `_streamed_tool_call_state`[`L155`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L155), `_tool_previous_text`[`L153`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L153), `_tool_previous_token_ids`[`L154`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L154)
- uses (calls/refs, reference-scoped): [`ChatCompletionRequest`](../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../tools/abstract_tool.md#ToolParser), [`DeltaToolCall`](../openai_api_modules.md#DeltaToolCall), [`content`](../openai_api_modules.md#DeltaMessage.content), [`DeltaFunctionCall`](../openai_api_modules.md#DeltaFunctionCall), [`tool_calls`](../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolCall`](../openai_api_modules.md#ToolCall), [`tool_calls`](../openai_api_modules.md#DeltaMessage.tool_calls), [`index`](../openai_api_modules.md#DeltaToolCall.index), [`arguments`](../openai_api_modules.md#DeltaFunctionCall.arguments), [`extract_tool_calls`](../tools/abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../tools/abstract_tool.md#ToolParser.extract_tool_calls_streaming), [`arguments`](../openai_api_modules.md#FunctionCall.arguments), [`ChatMessage`](../openai_api_modules.md#ChatMessage), [`name`](../openai_api_modules.md#FunctionCall.name), [`id`](../openai_api_modules.md#DeltaToolCall.id), [`name`](../openai_api_modules.md#DeltaFunctionCall.name), [`ReasoningParser`](../reasoning/abstract_reasoning.md#ReasoningParser), [`accumulated_content`](delegating_parser.md#ParseResult.accumulated_content), [`ParsePhase`](delegating_parser.md#ParsePhase), [`reasoning_content`](../openai_api_modules.md#DeltaMessage.reasoning_content), [`delta_content`](delegating_parser.md#ParseResult.delta_content), [`extract_reasoning`](../reasoning/abstract_reasoning.md#ReasoningParser.extract_reasoning), [`extract_reasoning_streaming`](../reasoning/abstract_reasoning.md#ReasoningParser.extract_reasoning_streaming), [`content`](../openai_api_modules.md#ChatMessage.content), [`role`](../openai_api_modules.md#ChatMessage.role), [`id`](../openai_api_modules.md#ToolCall.id), [`model`](../openai_api_modules.md#ChatCompletionRequest.model), [`accumulated_reasoning`](delegating_parser.md#ParseResult.accumulated_reasoning), [`compute_stream_delta_text`](../stream_protocol.md#compute_stream_delta_text), [`phase`](delegating_parser.md#ParseResult.phase), [`is_buffering_protocol`](../tools/abstract_tool.md#ToolParser.is_buffering_protocol), [`ParseResult`](delegating_parser.md#ParseResult), [`messages`](../openai_api_modules.md#ChatCompletionRequest.messages), [`CONTENT`](delegating_parser.md#ParsePhase.CONTENT), [`REASONING`](delegating_parser.md#ParsePhase.REASONING), [`TOOL_CALL`](delegating_parser.md#ParsePhase.TOOL_CALL), [`delta_tool_calls`](delegating_parser.md#ParseResult.delta_tool_calls)  (+10 more)
- used by: [`_add_request`](../esurge/mixins/requests.md#EngineRequestsMixin._add_request)

### `ParsePhase`  ·  implements/extends IntEnum
- def: [`easydel/inference/parsing/delegating_parser.py:38`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L38)
- doc: Explicit phase of the combined reasoning + tool parsing pipeline.
- signature: `class ParsePhase(enum.IntEnum):`
- members:
  - `BUFFERING` — [`L62`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L62)
  - `CONTENT` — [`L60`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L60)
  - `REASONING` — [`L59`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L59)
  - `TOOL_CALL` — [`L61`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L61)
- used by: [`_process_tool_delta`](delegating_parser.md#DelegatingParser._process_tool_delta), [`_process_reasoning_delta`](delegating_parser.md#DelegatingParser._process_reasoning_delta), [`_process_tool_final`](delegating_parser.md#DelegatingParser._process_tool_final), [`process_final`](delegating_parser.md#DelegatingParser.process_final), [`process_delta`](delegating_parser.md#DelegatingParser.process_delta), [`phase`](delegating_parser.md#DelegatingParser.phase), [`phase`](delegating_parser.md#ParseResult.phase)

### `ParseResult`
- def: [`easydel/inference/parsing/delegating_parser.py:66`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L66)
- doc: Structured output returned by `DelegatingParser.process_delta` / `process_final`.
- signature: `class ParseResult:`
- members:
  - `to_dict(self)` — [`L94`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L94) — Flatten into the dict format consumed by ``EngineParsingMixin._run_output_parsers``.
  - `accumulated_content` — [`L89`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L89)
  - `accumulated_reasoning` — [`L88`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L88)
  - `delta_content` — [`L87`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L87)
  - `delta_reasoning` — [`L86`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L86)
  - `delta_tool_calls` — [`L91`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L91)
  - `phase` — [`L92`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L92)
  - `tool_calls` — [`L90`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L90)
- uses (calls/refs, reference-scoped): [`ParsePhase`](delegating_parser.md#ParsePhase), [`CONTENT`](delegating_parser.md#ParsePhase.CONTENT)
- used by: [`_process_tool_delta`](delegating_parser.md#DelegatingParser._process_tool_delta), [`_process_reasoning_delta`](delegating_parser.md#DelegatingParser._process_reasoning_delta), [`_process_tool_final`](delegating_parser.md#DelegatingParser._process_tool_final), [`process_final`](delegating_parser.md#DelegatingParser.process_final), [`process_delta`](delegating_parser.md#DelegatingParser.process_delta)

### `TrackedToolCallState`
- def: [`easydel/inference/parsing/delegating_parser.py:107`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L107)
- doc: Per-tool-call accumulator that tracks what has already been streamed.
- signature: `class TrackedToolCallState:`
- members:
  - `function_arguments` — [`L122`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L122)
  - `function_name` — [`L121`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L121)
  - `index` — [`L118`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L118)
  - `tool_call_id` — [`L119`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L119)
  - `tool_type` — [`L120`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L120)
- used by: [`_build_missing_final_tool_deltas`](delegating_parser.md#DelegatingParser._build_missing_final_tool_deltas), [`_merge_streamed_tool_call_state`](delegating_parser.md#DelegatingParser._merge_streamed_tool_call_state), [`_streamed_tool_call_state`](delegating_parser.md#DelegatingParser._streamed_tool_call_state)

## Module values
- `logger` — [`L35`](../../../../../../../raw/code/EasyDeL/easydel/inference/parsing/delegating_parser.py#L35)

