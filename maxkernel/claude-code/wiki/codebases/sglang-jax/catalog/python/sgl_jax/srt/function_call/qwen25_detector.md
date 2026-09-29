---
title: 'Module: python/sgl_jax/srt/function_call/qwen25_detector.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/function_call/qwen25_detector.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.function_call.qwen25_detector`/
symbols:
  Qwen25Detector.detect_and_parse: Qwen25Detector#detect_and_parse().
  Qwen25Detector.parse_streaming_increment: Qwen25Detector#parse_streaming_increment().
  Qwen25Detector.structure_info: Qwen25Detector#structure_info().
  Qwen25Detector.build_ebnf: Qwen25Detector#build_ebnf().
  Qwen25Detector._normal_text_buffer: Qwen25Detector#_normal_text_buffer.
  Qwen25Detector: Qwen25Detector#
  Qwen25Detector.bot_token: Qwen25Detector#bot_token.
  Qwen25Detector.eot_token: Qwen25Detector#eot_token.
  Qwen25Detector.__init__: Qwen25Detector#__init__().
  Qwen25Detector.has_tool_call: Qwen25Detector#has_tool_call().
  logger: logger.
  Qwen25Detector.tool_call_separator: Qwen25Detector#tool_call_separator.
---
# Module: [`python/sgl_jax/srt/function_call/qwen25_detector.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen25_detector.py)

## Classes
### `Qwen25Detector`  ·  implements/extends BaseFormatDetector
- def: [`python/sgl_jax/srt/function_call/qwen25_detector.py:17`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen25_detector.py#L17)
- doc: Detector for Qwen 2.5 / Qwen 3 / Ling-2.6 tool-call format.
- signature: `class Qwen25Detector(BaseFormatDetector):`
- members:
  - `build_ebnf(self, tools: list[Tool])` — [`L86`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen25_detector.py#L86)
  - `detect_and_parse(self, text: str, tools: list[Tool])` — [`L38`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen25_detector.py#L38)
  - `has_tool_call(self, text: str)` — [`L35`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen25_detector.py#L35)
  - `parse_streaming_increment(self, new_text: str, tools: list[Tool])` — [`L54`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen25_detector.py#L54)
  - `structure_info(self)` — [`L79`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen25_detector.py#L79)
  - `bot_token` — [`L29`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen25_detector.py#L29)
  - `eot_token` — [`L30`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen25_detector.py#L30)
  - `tool_call_separator` — [`L31`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen25_detector.py#L31)
- protocol/private: `__init__`[`L27`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen25_detector.py#L27), `_normal_text_buffer`[`L33`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen25_detector.py#L33)
- uses (calls/refs, reference-scoped): [`StreamingParseResult`](core_types.md#StreamingParseResult), [`Tool`](../entrypoints/openai/protocol.md#Tool), [`parse_streaming_increment`](base_format_detector.md#BaseFormatDetector.parse_streaming_increment), [`normal_text`](core_types.md#StreamingParseResult.normal_text), [`BaseFormatDetector`](base_format_detector.md#BaseFormatDetector), [`calls`](core_types.md#StreamingParseResult.calls), [`parse_base_json`](base_format_detector.md#BaseFormatDetector.parse_base_json), [`build_ebnf`](ebnf_composer.md#EBNFComposer.build_ebnf), [`EBNFComposer`](ebnf_composer.md#EBNFComposer), [`_GetInfoFunc`](core_types.md#_GetInfoFunc), [`__init__`](base_format_detector.md#BaseFormatDetector.__init__), [`StructureInfo`](core_types.md#StructureInfo), [`_ends_with_partial_token`](base_format_detector.md#BaseFormatDetector._ends_with_partial_token), [`begin`](core_types.md#StructureInfo.begin), [`end`](core_types.md#StructureInfo.end), [`logger`](qwen25_detector.md#logger), [`trigger`](core_types.md#StructureInfo.trigger)
- used by: [`parse_streaming_increment`](base_format_detector.md#BaseFormatDetector.parse_streaming_increment), [`BaseFormatDetector`](base_format_detector.md#BaseFormatDetector), [`detect_and_parse`](base_format_detector.md#BaseFormatDetector.detect_and_parse), [`ToolCallParserEnum`](function_call_parser.md#FunctionCallParser.ToolCallParserEnum), [`build_ebnf`](base_format_detector.md#BaseFormatDetector.build_ebnf), [`structure_info`](base_format_detector.md#BaseFormatDetector.structure_info), [`has_tool_call`](base_format_detector.md#BaseFormatDetector.has_tool_call)

## Module values
- `logger` — [`L14`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/qwen25_detector.py#L14)

