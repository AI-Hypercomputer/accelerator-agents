---
title: 'Module: python/sgl_jax/srt/function_call/qwen3_coder_detector.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/function_call/qwen3_coder_detector.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.function_call.qwen3_coder_detector`/
symbols:
  Qwen3CoderDetector._tool_indices: Qwen3CoderDetector#_tool_indices.
  Qwen3CoderDetector._parse_and_stream_parameters: Qwen3CoderDetector#_parse_and_stream_parameters().
  Qwen3CoderDetector._buf: Qwen3CoderDetector#_buf.
  Qwen3CoderDetector._parse_block: Qwen3CoderDetector#_parse_block().
  Qwen3CoderDetector._reset_streaming_state: Qwen3CoderDetector#_reset_streaming_state().
  Qwen3CoderDetector._extract: Qwen3CoderDetector#_extract().
  Qwen3CoderDetector.detect_and_parse: Qwen3CoderDetector#detect_and_parse().
  Qwen3CoderDetector.build_ebnf: Qwen3CoderDetector#build_ebnf().
  Qwen3CoderDetector.parse_streaming_increment: Qwen3CoderDetector#parse_streaming_increment().
  Qwen3CoderDetector.tool_call_end_token: Qwen3CoderDetector#tool_call_end_token.
  Qwen3CoderDetector.tool_call_start_token: Qwen3CoderDetector#tool_call_start_token.
  Qwen3CoderDetector._current_parameters: Qwen3CoderDetector#_current_parameters.
  Qwen3CoderDetector._function_name_sent: Qwen3CoderDetector#_function_name_sent.
  Qwen3CoderDetector: Qwen3CoderDetector#
  Qwen3CoderDetector._in_tool_call: Qwen3CoderDetector#_in_tool_call.
  Qwen3CoderDetector._current_function_name: Qwen3CoderDetector#_current_function_name.
  Qwen3CoderDetector.__init__: Qwen3CoderDetector#__init__().
  Qwen3CoderDetector.has_tool_call: Qwen3CoderDetector#has_tool_call().
  Qwen3CoderDetector.structure_info: Qwen3CoderDetector#structure_info().
  logger: logger.
  _safe_val: _safe_val().
  Qwen3CoderDetector._streamed_parameters: Qwen3CoderDetector#_streamed_parameters.
  Qwen3CoderDetector.tool_call_function_regex: Qwen3CoderDetector#tool_call_function_regex.
  Qwen3CoderDetector.tool_call_parameter_regex: Qwen3CoderDetector#tool_call_parameter_regex.
  Qwen3CoderDetector.tool_call_prefix: Qwen3CoderDetector#tool_call_prefix.
  Qwen3CoderDetector.tool_call_regex: Qwen3CoderDetector#tool_call_regex.
  Qwen3CoderDetector.supports_structural_tag: Qwen3CoderDetector#supports_structural_tag().
---
# Module: [`python/sgl_jax/srt/function_call/qwen3_coder_detector.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py)

## Classes
### `Qwen3CoderDetector`  ·  implements/extends BaseFormatDetector
- def: [`python/sgl_jax/srt/function_call/qwen3_coder_detector.py:31`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L31)
- doc: Detector for Qwen 3 models.
- signature: `class Qwen3CoderDetector(BaseFormatDetector):`
- members:
  - `_parse_and_stream_parameters(self, text_to_parse: str)` — [`L203`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L203) — Parse complete parameter blocks from text and return any tool call items to emit.
  - `_reset_streaming_state(self)` — [`L287`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L287) — Reset streaming state for the next tool call
  - `build_ebnf(self, tools: list[Tool])` — [`L348`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L348)
  - `detect_and_parse(self, text: str, tools: list[Tool])` — [`L72`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L72)
  - `has_tool_call(self, text: str)` — [`L69`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L69)
  - `parse_streaming_increment(self, new_text: str, tools: list[Tool])` — [`L76`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L76)
  - `structure_info(self)` — [`L345`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L345)
  - `supports_structural_tag(self)` — [`L342`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L342)
  - `tool_call_end_token` — [`L47`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L47)
  - `tool_call_function_regex` — [`L52`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L52)
  - `tool_call_parameter_regex` — [`L55`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L55)
  - `tool_call_prefix` — [`L48`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L48)
  - `tool_call_regex` — [`L49`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L49)
  - `tool_call_start_token` — [`L46`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L46)
- protocol/private: `__init__`[`L44`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L44), `_buf`[`L58`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L58), `_current_function_name`[`L61`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L61), `_current_parameters`[`L62`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L62), `_extract`[`L296`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L296), `_function_name_sent`[`L67`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L67), `_in_tool_call`[`L66`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L66), `_parse_block`[`L315`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L315), `_streamed_parameters`[`L63`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L63), `_tool_indices`[`L83`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L83)
- uses (calls/refs, reference-scoped): [`current_tool_id`](base_format_detector.md#BaseFormatDetector.current_tool_id), [`StreamingParseResult`](core_types.md#StreamingParseResult), [`Tool`](../entrypoints/openai/protocol.md#Tool), [`ToolCallItem`](core_types.md#ToolCallItem), [`normal_text`](core_types.md#StreamingParseResult.normal_text), [`streamed_args_for_tool`](base_format_detector.md#BaseFormatDetector.streamed_args_for_tool), [`BaseFormatDetector`](base_format_detector.md#BaseFormatDetector), [`calls`](core_types.md#StreamingParseResult.calls), [`parse_base_json`](base_format_detector.md#BaseFormatDetector.parse_base_json), [`prev_tool_call_arr`](base_format_detector.md#BaseFormatDetector.prev_tool_call_arr), [`build_ebnf`](ebnf_composer.md#EBNFComposer.build_ebnf), [`current_tool_name_sent`](base_format_detector.md#BaseFormatDetector.current_tool_name_sent), [`parameters`](core_types.md#ToolCallItem.parameters), [`EBNFComposer`](ebnf_composer.md#EBNFComposer), [`tool_index`](core_types.md#ToolCallItem.tool_index), [`name`](core_types.md#ToolCallItem.name), [`_GetInfoFunc`](core_types.md#_GetInfoFunc), [`_get_tool_indices`](base_format_detector.md#BaseFormatDetector._get_tool_indices), [`__init__`](base_format_detector.md#BaseFormatDetector.__init__), [`_safe_val`](qwen3_coder_detector.md#_safe_val), [`logger`](qwen3_coder_detector.md#logger)
- used by: [`parse_streaming_increment`](base_format_detector.md#BaseFormatDetector.parse_streaming_increment), [`BaseFormatDetector`](base_format_detector.md#BaseFormatDetector), [`detect_and_parse`](base_format_detector.md#BaseFormatDetector.detect_and_parse), [`ToolCallParserEnum`](function_call_parser.md#FunctionCallParser.ToolCallParserEnum), [`build_ebnf`](base_format_detector.md#BaseFormatDetector.build_ebnf), [`structure_info`](base_format_detector.md#BaseFormatDetector.structure_info), [`has_tool_call`](base_format_detector.md#BaseFormatDetector.has_tool_call), [`supports_structural_tag`](base_format_detector.md#BaseFormatDetector.supports_structural_tag)

## Functions
- `_safe_val(raw: str)` — [`L20`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L20)

## Module values
- `logger` — [`L17`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen3_coder_detector.py#L17)

