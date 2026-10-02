---
title: 'Module: python/sgl_jax/srt/function_call/mimo_detector.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/function_call/mimo_detector.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.function_call.mimo_detector`/
symbols:
  MiMoDetector._buffer: MiMoDetector#_buffer.
  MiMoDetector.detect_and_parse: MiMoDetector#detect_and_parse().
  MiMoDetector.parse_streaming_increment: MiMoDetector#parse_streaming_increment().
  MiMoDetector._parse_tool_call: MiMoDetector#_parse_tool_call().
  _get_param_type: _get_param_type().
  _convert_param_value: _convert_param_value().
  logger: logger.
  MiMoDetector: MiMoDetector#
  MiMoDetector.bot_token: MiMoDetector#bot_token.
  MiMoDetector.eot_token: MiMoDetector#eot_token.
  MiMoDetector.__init__: MiMoDetector#__init__().
  MiMoDetector.has_tool_call: MiMoDetector#has_tool_call().
  MiMoDetector.structure_info: MiMoDetector#structure_info().
  MiMoDetector.build_ebnf: MiMoDetector#build_ebnf().
  MiMoDetector.tool_call_regex: MiMoDetector#tool_call_regex.
  MiMoDetector.func_regex: MiMoDetector#func_regex.
  MiMoDetector.param_regex: MiMoDetector#param_regex.
  MiMoDetector.supports_structural_tag: MiMoDetector#supports_structural_tag().
---
# Module: [`python/sgl_jax/srt/function_call/mimo_detector.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/mimo_detector.py)

## Classes
### `MiMoDetector`  ·  implements/extends BaseFormatDetector
- def: [`python/sgl_jax/srt/function_call/mimo_detector.py:121`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/mimo_detector.py#L121)
- doc: Detector for MiMo function call format.
- signature: `class MiMoDetector(BaseFormatDetector):`
- members:
  - `_parse_tool_call(self, tool_call_body: str, tools: list[Tool])` — [`L223`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/mimo_detector.py#L223) — Parse content inside <tool_call>...</tool_call>.
  - `build_ebnf(self, tools: list[Tool])` — [`L251`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/mimo_detector.py#L251)
  - `detect_and_parse(self, text: str, tools: list[Tool])` — [`L144`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/mimo_detector.py#L144) — Parse complete text for tool calls.
  - `has_tool_call(self, text: str)` — [`L141`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/mimo_detector.py#L141)
  - `parse_streaming_increment(self, new_text: str, tools: list[Tool])` — [`L177`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/mimo_detector.py#L177) — Streaming parsing: buffer until complete tool call block.
  - `structure_info(self)` — [`L248`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/mimo_detector.py#L248)
  - `supports_structural_tag(self)` — [`L245`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/mimo_detector.py#L245)
  - `bot_token` — [`L135`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/mimo_detector.py#L135)
  - `eot_token` — [`L136`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/mimo_detector.py#L136)
  - `func_regex` — [`L138`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/mimo_detector.py#L138)
  - `param_regex` — [`L139`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/mimo_detector.py#L139)
  - `tool_call_regex` — [`L137`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/mimo_detector.py#L137)
- protocol/private: `__init__`[`L133`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/mimo_detector.py#L133), `_buffer`[`L189`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/mimo_detector.py#L189)
- uses (calls/refs, reference-scoped): [`current_tool_id`](base_format_detector.md#BaseFormatDetector.current_tool_id), [`StreamingParseResult`](core_types.md#StreamingParseResult), [`Tool`](../entrypoints/openai/protocol.md#Tool), [`normal_text`](core_types.md#StreamingParseResult.normal_text), [`streamed_args_for_tool`](base_format_detector.md#BaseFormatDetector.streamed_args_for_tool), [`BaseFormatDetector`](base_format_detector.md#BaseFormatDetector), [`calls`](core_types.md#StreamingParseResult.calls), [`parse_base_json`](base_format_detector.md#BaseFormatDetector.parse_base_json), [`prev_tool_call_arr`](base_format_detector.md#BaseFormatDetector.prev_tool_call_arr), [`parameters`](core_types.md#ToolCallItem.parameters), [`tool_index`](core_types.md#ToolCallItem.tool_index), [`name`](core_types.md#ToolCallItem.name), [`_GetInfoFunc`](core_types.md#_GetInfoFunc), [`_get_tool_indices`](base_format_detector.md#BaseFormatDetector._get_tool_indices), [`_convert_param_value`](mimo_detector.md#_convert_param_value), [`logger`](mimo_detector.md#logger), [`__init__`](base_format_detector.md#BaseFormatDetector.__init__)
- used by: [`parse_streaming_increment`](base_format_detector.md#BaseFormatDetector.parse_streaming_increment), [`BaseFormatDetector`](base_format_detector.md#BaseFormatDetector), [`detect_and_parse`](base_format_detector.md#BaseFormatDetector.detect_and_parse), [`ToolCallParserEnum`](function_call_parser.md#FunctionCallParser.ToolCallParserEnum), [`build_ebnf`](base_format_detector.md#BaseFormatDetector.build_ebnf), [`structure_info`](base_format_detector.md#BaseFormatDetector.structure_info), [`has_tool_call`](base_format_detector.md#BaseFormatDetector.has_tool_call), [`supports_structural_tag`](base_format_detector.md#BaseFormatDetector.supports_structural_tag)

## Functions
- `_convert_param_value(param_value: str, param_name: str, func_name: str, tools: list[Tool])` — [`L25`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/mimo_detector.py#L25) — Convert parameter value based on its type in the schema.
- `_get_param_type(func_name: str, param_name: str, tools: list[Tool])` — [`L15`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/mimo_detector.py#L15) — Get parameter type from tool schema.

## Module values
- `logger` — [`L12`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/mimo_detector.py#L12)

