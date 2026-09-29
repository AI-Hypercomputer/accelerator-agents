---
title: 'Module: python/sgl_jax/srt/function_call/glm4_moe_detector.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/function_call/glm4_moe_detector.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.function_call.glm4_moe_detector`/
symbols:
  Glm4MoeDetector._tool_indices: Glm4MoeDetector#_tool_indices.
  Glm4MoeDetector._process_xml_to_json_streaming: Glm4MoeDetector#_process_xml_to_json_streaming().
  Glm4MoeDetector.detect_and_parse: Glm4MoeDetector#detect_and_parse().
  Glm4MoeDetector._xml_tag_buffer: Glm4MoeDetector#_xml_tag_buffer.
  Glm4MoeDetector._stream_state: Glm4MoeDetector#_stream_state.
  Glm4MoeDetector._buffer: Glm4MoeDetector#_buffer.
  get_argument_type: get_argument_type().
  StreamState: StreamState#
  Glm4MoeDetector._last_arguments: Glm4MoeDetector#_last_arguments.
  Glm4MoeDetector.parse_streaming_increment: Glm4MoeDetector#parse_streaming_increment().
  Glm4MoeDetector._parse_argument_pairs: Glm4MoeDetector#_parse_argument_pairs().
  Glm4MoeDetector._current_value: Glm4MoeDetector#_current_value.
  Glm4MoeDetector._value_started: Glm4MoeDetector#_value_started.
  Glm4MoeDetector._streamed_raw_length: Glm4MoeDetector#_streamed_raw_length.
  Glm4MoeDetector._get_value_type: Glm4MoeDetector#_get_value_type().
  Glm4MoeDetector.bot_token: Glm4MoeDetector#bot_token.
  Glm4MoeDetector._format_value_complete: Glm4MoeDetector#_format_value_complete().
  Glm4MoeDetector: Glm4MoeDetector#
  logger: logger.
  Glm4MoeDetector._reset_streaming_state: Glm4MoeDetector#_reset_streaming_state().
  Glm4MoeDetector._current_key: Glm4MoeDetector#_current_key.
  Glm4MoeDetector._cached_value_type: Glm4MoeDetector#_cached_value_type.
  parse_arguments: parse_arguments().
  _convert_to_number: _convert_to_number().
  Glm4MoeDetector.eot_token: Glm4MoeDetector#eot_token.
  Glm4MoeDetector._is_first_param: Glm4MoeDetector#_is_first_param.
  Glm4MoeDetector.__init__: Glm4MoeDetector#__init__().
  Glm4MoeDetector.has_tool_call: Glm4MoeDetector#has_tool_call().
  Glm4MoeDetector.structure_info: Glm4MoeDetector#structure_info().
  Glm4MoeDetector.build_ebnf: Glm4MoeDetector#build_ebnf().
  StreamState.INIT: StreamState#INIT.
  StreamState.BETWEEN: StreamState#BETWEEN.
  StreamState.IN_KEY: StreamState#IN_KEY.
  StreamState.WAITING_VALUE: StreamState#WAITING_VALUE.
  StreamState.IN_VALUE: StreamState#IN_VALUE.
  Glm4MoeDetector.func_arg_regex: Glm4MoeDetector#func_arg_regex.
  Glm4MoeDetector.func_call_regex: Glm4MoeDetector#func_call_regex.
  Glm4MoeDetector.func_detail_regex: Glm4MoeDetector#func_detail_regex.
  Glm4MoeDetector.supports_structural_tag: Glm4MoeDetector#supports_structural_tag().
---
# Module: [`python/sgl_jax/srt/function_call/glm4_moe_detector.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py)

## Classes
### `Glm4MoeDetector`  ·  implements/extends BaseFormatDetector
- def: [`python/sgl_jax/srt/function_call/glm4_moe_detector.py:130`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L130)
- doc: Detector for GLM-4.5 and GLM-4.6 models.
- signature: `class Glm4MoeDetector(BaseFormatDetector):`
- members:
  - `_format_value_complete(self, value: str, value_type: str)` — [`L266`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L266) — Format complete value based on type.
  - `_get_value_type(self, func_name: str, key: str, tools: list[Tool])` — [`L215`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L215) — Get parameter type from tool definition, with fallback to auto-detection.
  - `_parse_argument_pairs(self, pairs: list[tuple[str, str]], func_name: str, tools: list[Tool])` — [`L572`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L572) — Parse argument key-value pairs with type coercion.
  - `_process_xml_to_json_streaming(self, raw_increment: str, func_name: str, tools: list[Tool])` — [`L291`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L291) — Convert XML increment to JSON streaming output using state machine.
  - `_reset_streaming_state(self)` — [`L165`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L165) — Reset the streaming state machine for a new tool call.
  - `build_ebnf(self, tools: list[Tool])` — [`L615`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L615)
  - `detect_and_parse(self, text: str, tools: list[Tool])` — [`L179`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L179) — One-time parsing: Detects and parses tool calls in the provided text.
  - `has_tool_call(self, text: str)` — [`L175`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L175) — Check if the text contains a glm-4.5 / glm-4.6 format tool call.
  - `parse_streaming_increment(self, new_text: str, tools: list[Tool])` — [`L403`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L403) — Streaming incremental parsing tool calls for GLM-4.5 and GLM-4.6 format.
  - `structure_info(self)` — [`L612`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L612)
  - `supports_structural_tag(self)` — [`L609`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L609)
  - `bot_token` — [`L149`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L149)
  - `eot_token` — [`L150`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L150)
  - `func_arg_regex` — [`L155`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L155)
  - `func_call_regex` — [`L151`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L151)
  - `func_detail_regex` — [`L152`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L152)
- protocol/private: `__init__`[`L147`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L147), `_buffer`[`L429`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L429), `_cached_value_type`[`L173`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L173), `_current_key`[`L168`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L168), `_current_value`[`L169`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L169), `_is_first_param`[`L171`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L171), `_last_arguments`[`L159`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L159), `_stream_state`[`L167`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L167), `_streamed_raw_length`[`L162`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L162), `_tool_indices`[`L438`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L438), `_value_started`[`L172`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L172), `_xml_tag_buffer`[`L170`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L170)
- uses (calls/refs, reference-scoped): [`current_tool_id`](base_format_detector.md#BaseFormatDetector.current_tool_id), [`StreamingParseResult`](core_types.md#StreamingParseResult), [`Tool`](../entrypoints/openai/protocol.md#Tool), [`ToolCallItem`](core_types.md#ToolCallItem), [`normal_text`](core_types.md#StreamingParseResult.normal_text), [`streamed_args_for_tool`](base_format_detector.md#BaseFormatDetector.streamed_args_for_tool), [`BaseFormatDetector`](base_format_detector.md#BaseFormatDetector), [`calls`](core_types.md#StreamingParseResult.calls), [`parse_base_json`](base_format_detector.md#BaseFormatDetector.parse_base_json), [`prev_tool_call_arr`](base_format_detector.md#BaseFormatDetector.prev_tool_call_arr), [`current_tool_name_sent`](base_format_detector.md#BaseFormatDetector.current_tool_name_sent), [`parameters`](core_types.md#ToolCallItem.parameters), [`tool_index`](core_types.md#ToolCallItem.tool_index), [`name`](core_types.md#ToolCallItem.name), [`_GetInfoFunc`](core_types.md#_GetInfoFunc), [`get_argument_type`](glm4_moe_detector.md#get_argument_type), [`_get_tool_indices`](base_format_detector.md#BaseFormatDetector._get_tool_indices), [`StreamState`](glm4_moe_detector.md#StreamState), [`__init__`](base_format_detector.md#BaseFormatDetector.__init__), [`logger`](glm4_moe_detector.md#logger), [`parse_arguments`](glm4_moe_detector.md#parse_arguments), [`_convert_to_number`](glm4_moe_detector.md#_convert_to_number), [`BETWEEN`](glm4_moe_detector.md#StreamState.BETWEEN), [`INIT`](glm4_moe_detector.md#StreamState.INIT), [`IN_KEY`](glm4_moe_detector.md#StreamState.IN_KEY), [`IN_VALUE`](glm4_moe_detector.md#StreamState.IN_VALUE), [`WAITING_VALUE`](glm4_moe_detector.md#StreamState.WAITING_VALUE)
- used by: [`parse_streaming_increment`](base_format_detector.md#BaseFormatDetector.parse_streaming_increment), [`BaseFormatDetector`](base_format_detector.md#BaseFormatDetector), [`detect_and_parse`](base_format_detector.md#BaseFormatDetector.detect_and_parse), [`ToolCallParserEnum`](function_call_parser.md#FunctionCallParser.ToolCallParserEnum), [`build_ebnf`](base_format_detector.md#BaseFormatDetector.build_ebnf), [`structure_info`](base_format_detector.md#BaseFormatDetector.structure_info), [`has_tool_call`](base_format_detector.md#BaseFormatDetector.has_tool_call), [`supports_structural_tag`](base_format_detector.md#BaseFormatDetector.supports_structural_tag)

### `StreamState`  ·  implements/extends Enum, str
- def: [`python/sgl_jax/srt/function_call/glm4_moe_detector.py:20`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L20)
- doc: State machine states for XML to JSON streaming conversion.
- signature: `class StreamState(str, Enum):`
- members:
  - `BETWEEN` — [`L24`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L24)
  - `INIT` — [`L23`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L23)
  - `IN_KEY` — [`L25`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L25)
  - `IN_VALUE` — [`L27`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L27)
  - `WAITING_VALUE` — [`L26`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L26)
- used by: [`_process_xml_to_json_streaming`](glm4_moe_detector.md#Glm4MoeDetector._process_xml_to_json_streaming), [`_stream_state`](glm4_moe_detector.md#Glm4MoeDetector._stream_state)

## Functions
- `_convert_to_number(value: str)` — [`L63`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L63) — Convert string to appropriate number type (int or float).
- `get_argument_type(func_name: str, arg_key: str, defined_tools: list[Tool])` — [`L30`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L30) — Get the expected type of a function argument from tool definitions.
- `parse_arguments(json_value: str, arg_type: str | None = None)` — [`L81`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L81) — Parse argument value with multiple fallback strategies.

## Module values
- `logger` — [`L17`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm4_moe_detector.py#L17)

