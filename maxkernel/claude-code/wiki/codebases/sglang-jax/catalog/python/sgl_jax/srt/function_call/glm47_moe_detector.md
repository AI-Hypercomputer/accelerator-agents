---
title: 'Module: python/sgl_jax/srt/function_call/glm47_moe_detector.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/function_call/glm47_moe_detector.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.function_call.glm47_moe_detector`/
symbols:
  Glm47MoeDetector._tool_indices: Glm47MoeDetector#_tool_indices.
  Glm47MoeDetector._process_xml_to_json_streaming: Glm47MoeDetector#_process_xml_to_json_streaming().
  Glm47MoeDetector._finalize_tool_call: Glm47MoeDetector#_finalize_tool_call().
  Glm47MoeDetector._send_tool_name_if_needed: Glm47MoeDetector#_send_tool_name_if_needed().
  Glm47MoeDetector.detect_and_parse: Glm47MoeDetector#detect_and_parse().
  Glm47MoeDetector._process_arguments_streaming: Glm47MoeDetector#_process_arguments_streaming().
  Glm47MoeDetector._buffer: Glm47MoeDetector#_buffer.
  Glm47MoeDetector.parse_streaming_increment: Glm47MoeDetector#parse_streaming_increment().
  Glm47MoeDetector._xml_tag_buffer: Glm47MoeDetector#_xml_tag_buffer.
  Glm47MoeDetector._stream_state: Glm47MoeDetector#_stream_state.
  StreamState: StreamState#
  Glm47MoeDetector._last_arguments: Glm47MoeDetector#_last_arguments.
  get_argument_type: get_argument_type().
  Glm47MoeDetector._parse_argument_pairs: Glm47MoeDetector#_parse_argument_pairs().
  Glm47MoeDetector._current_value: Glm47MoeDetector#_current_value.
  Glm47MoeDetector._value_started: Glm47MoeDetector#_value_started.
  Glm47MoeDetector._cached_value_type: Glm47MoeDetector#_cached_value_type.
  Glm47MoeDetector._get_value_type: Glm47MoeDetector#_get_value_type().
  Glm47MoeDetector._sent_empty_object: Glm47MoeDetector#_sent_empty_object.
  Glm47MoeDetector.bot_token: Glm47MoeDetector#bot_token.
  Glm47MoeDetector._streamed_raw_length: Glm47MoeDetector#_streamed_raw_length.
  Glm47MoeDetector._format_value_complete: Glm47MoeDetector#_format_value_complete().
  logger: logger.
  Glm47MoeDetector: Glm47MoeDetector#
  Glm47MoeDetector.eot_token: Glm47MoeDetector#eot_token.
  Glm47MoeDetector._reset_streaming_state: Glm47MoeDetector#_reset_streaming_state().
  Glm47MoeDetector._current_key: Glm47MoeDetector#_current_key.
  parse_arguments: parse_arguments().
  _convert_to_number: _convert_to_number().
  Glm47MoeDetector._tool_call_completed: Glm47MoeDetector#_tool_call_completed.
  Glm47MoeDetector._is_first_param: Glm47MoeDetector#_is_first_param.
  Glm47MoeDetector.__init__: Glm47MoeDetector#__init__().
  Glm47MoeDetector.has_tool_call: Glm47MoeDetector#has_tool_call().
  Glm47MoeDetector.structure_info: Glm47MoeDetector#structure_info().
  Glm47MoeDetector.build_ebnf: Glm47MoeDetector#build_ebnf().
  StreamState.INIT: StreamState#INIT.
  StreamState.BETWEEN: StreamState#BETWEEN.
  StreamState.IN_KEY: StreamState#IN_KEY.
  StreamState.WAITING_VALUE: StreamState#WAITING_VALUE.
  StreamState.IN_VALUE: StreamState#IN_VALUE.
  Glm47MoeDetector.func_call_regex: Glm47MoeDetector#func_call_regex.
  Glm47MoeDetector.func_arg_regex: Glm47MoeDetector#func_arg_regex.
  Glm47MoeDetector.func_detail_regex: Glm47MoeDetector#func_detail_regex.
  Glm47MoeDetector._extract_match_groups: Glm47MoeDetector#_extract_match_groups().
  Glm47MoeDetector.supports_structural_tag: Glm47MoeDetector#supports_structural_tag().
---
# Module: [`python/sgl_jax/srt/function_call/glm47_moe_detector.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py)

## Classes
### `Glm47MoeDetector`  ·  implements/extends BaseFormatDetector
- def: [`python/sgl_jax/srt/function_call/glm47_moe_detector.py:141`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L141)
- doc: Detector for GLM-4.7 and GLM-5 models.
- signature: `class Glm47MoeDetector(BaseFormatDetector):`
- members:
  - `_extract_match_groups(self, match: re.Match)` — [`L427`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L427) — Extract function name, arguments and end marker from regex match.
  - `_finalize_tool_call(self, func_name: str, func_args_raw: str, tools: list[Tool], match_end_pos: int, current_text: str)` — [`L525`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L525) — Complete tool call processing.
  - `_format_value_complete(self, value: str, value_type: str)` — [`L290`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L290) — Format complete value based on type.
  - `_get_value_type(self, func_name: str, key: str, tools: list[Tool])` — [`L239`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L239) — Get parameter type from tool definition, with fallback to auto-detection.
  - `_parse_argument_pairs(self, pairs: list[tuple[str, str]], func_name: str, tools: list[Tool])` — [`L714`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L714) — Parse argument key-value pairs with type coercion.
  - `_process_arguments_streaming(self, func_name: str, func_args_raw: str, tools: list[Tool])` — [`L484`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L484) — Process streaming arguments.
  - `_process_xml_to_json_streaming(self, raw_increment: str, func_name: str, tools: list[Tool])` — [`L315`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L315) — Convert XML increment to JSON streaming output using state machine.
  - `_reset_streaming_state(self)` — [`L168`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L168) — Reset the streaming state machine for a new tool call.
  - `_send_tool_name_if_needed(self, func_name: str, has_arg_key: bool, is_tool_end: str)` — [`L441`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L441) — Send tool name if needed.
  - `build_ebnf(self, tools: list[Tool])` — [`L757`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L757)
  - `detect_and_parse(self, text: str, tools: list[Tool])` — [`L184`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L184) — One-time parsing: Detects and parses tool calls in the provided text.
  - `has_tool_call(self, text: str)` — [`L180`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L180) — Check if the text contains a glm-4.5 / glm-4.6 format tool call.
  - `parse_streaming_increment(self, new_text: str, tools: list[Tool])` — [`L596`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L596) — Streaming incremental parsing tool calls for GLM-4.5 and GLM-4.6 format.
  - `structure_info(self)` — [`L754`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L754)
  - `supports_structural_tag(self)` — [`L751`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L751)
  - `bot_token` — [`L150`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L150)
  - `eot_token` — [`L151`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L151)
  - `func_arg_regex` — [`L156`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L156)
  - `func_call_regex` — [`L152`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L152)
  - `func_detail_regex` — [`L153`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L153)
- protocol/private: `__init__`[`L148`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L148), `_buffer`[`L584`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L584), `_cached_value_type`[`L176`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L176), `_current_key`[`L171`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L171), `_current_value`[`L172`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L172), `_is_first_param`[`L174`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L174), `_last_arguments`[`L160`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L160), `_sent_empty_object`[`L165`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L165), `_stream_state`[`L170`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L170), `_streamed_raw_length`[`L163`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L163), `_tool_call_completed`[`L164`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L164), `_tool_indices`[`L640`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L640), `_value_started`[`L175`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L175), `_xml_tag_buffer`[`L173`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L173)
- uses (calls/refs, reference-scoped): [`current_tool_id`](base_format_detector.md#BaseFormatDetector.current_tool_id), [`StreamingParseResult`](core_types.md#StreamingParseResult), [`Tool`](../entrypoints/openai/protocol.md#Tool), [`ToolCallItem`](core_types.md#ToolCallItem), [`normal_text`](core_types.md#StreamingParseResult.normal_text), [`streamed_args_for_tool`](base_format_detector.md#BaseFormatDetector.streamed_args_for_tool), [`BaseFormatDetector`](base_format_detector.md#BaseFormatDetector), [`calls`](core_types.md#StreamingParseResult.calls), [`parse_base_json`](base_format_detector.md#BaseFormatDetector.parse_base_json), [`prev_tool_call_arr`](base_format_detector.md#BaseFormatDetector.prev_tool_call_arr), [`current_tool_name_sent`](base_format_detector.md#BaseFormatDetector.current_tool_name_sent), [`parameters`](core_types.md#ToolCallItem.parameters), [`tool_index`](core_types.md#ToolCallItem.tool_index), [`name`](core_types.md#ToolCallItem.name), [`_GetInfoFunc`](core_types.md#_GetInfoFunc), [`_get_tool_indices`](base_format_detector.md#BaseFormatDetector._get_tool_indices), [`StreamState`](glm47_moe_detector.md#StreamState), [`get_argument_type`](glm47_moe_detector.md#get_argument_type), [`__init__`](base_format_detector.md#BaseFormatDetector.__init__), [`logger`](glm47_moe_detector.md#logger), [`parse_arguments`](glm47_moe_detector.md#parse_arguments), [`_convert_to_number`](glm47_moe_detector.md#_convert_to_number), [`BETWEEN`](glm47_moe_detector.md#StreamState.BETWEEN), [`INIT`](glm47_moe_detector.md#StreamState.INIT), [`IN_KEY`](glm47_moe_detector.md#StreamState.IN_KEY), [`IN_VALUE`](glm47_moe_detector.md#StreamState.IN_VALUE), [`WAITING_VALUE`](glm47_moe_detector.md#StreamState.WAITING_VALUE)
- used by: [`parse_streaming_increment`](base_format_detector.md#BaseFormatDetector.parse_streaming_increment), [`BaseFormatDetector`](base_format_detector.md#BaseFormatDetector), [`detect_and_parse`](base_format_detector.md#BaseFormatDetector.detect_and_parse), [`ToolCallParserEnum`](function_call_parser.md#FunctionCallParser.ToolCallParserEnum), [`build_ebnf`](base_format_detector.md#BaseFormatDetector.build_ebnf), [`structure_info`](base_format_detector.md#BaseFormatDetector.structure_info), [`has_tool_call`](base_format_detector.md#BaseFormatDetector.has_tool_call), [`supports_structural_tag`](base_format_detector.md#BaseFormatDetector.supports_structural_tag)

### `StreamState`  ·  implements/extends Enum, str
- def: [`python/sgl_jax/srt/function_call/glm47_moe_detector.py:20`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L20)
- doc: State machine states for XML to JSON streaming conversion.
- signature: `class StreamState(str, Enum):`
- members:
  - `BETWEEN` — [`L24`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L24)
  - `INIT` — [`L23`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L23)
  - `IN_KEY` — [`L25`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L25)
  - `IN_VALUE` — [`L27`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L27)
  - `WAITING_VALUE` — [`L26`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L26)
- used by: [`_process_xml_to_json_streaming`](glm47_moe_detector.md#Glm47MoeDetector._process_xml_to_json_streaming), [`_stream_state`](glm47_moe_detector.md#Glm47MoeDetector._stream_state)

## Functions
- `_convert_to_number(value: str)` — [`L74`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L74) — Convert string to appropriate number type (int or float).
- `get_argument_type(func_name: str, arg_key: str, defined_tools: list[Tool])` — [`L30`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L30) — Get the expected type of a function argument from tool definitions.
- `parse_arguments(json_value: str, arg_type: str | None = None)` — [`L92`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L92) — Parse argument value with multiple fallback strategies.

## Module values
- `logger` — [`L17`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/glm47_moe_detector.py#L17)

