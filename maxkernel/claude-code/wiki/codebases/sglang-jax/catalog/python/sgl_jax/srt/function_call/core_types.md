---
title: 'Module: python/sgl_jax/srt/function_call/core_types.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/function_call/core_types.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.function_call.core_types`/
symbols:
  StreamingParseResult: StreamingParseResult#
  ToolCallItem: ToolCallItem#
  StreamingParseResult.normal_text: StreamingParseResult#normal_text.
  StreamingParseResult.calls: StreamingParseResult#calls.
  ToolCallItem.parameters: ToolCallItem#parameters.
  ToolCallItem.tool_index: ToolCallItem#tool_index.
  ToolCallItem.name: ToolCallItem#name.
  _GetInfoFunc: _GetInfoFunc.
  StructureInfo: StructureInfo#
  StructureInfo.begin: StructureInfo#begin.
  StructureInfo.end: StructureInfo#end.
  StructureInfo.trigger: StructureInfo#trigger.
---
# Module: [`python/sgl_jax/srt/function_call/core_types.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/core_types.py)

## Classes
### `StreamingParseResult`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/function_call/core_types.py:15`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/core_types.py#L15)
- doc: Result of streaming incremental parsing.
- signature: `class StreamingParseResult(BaseModel):`
- members:
  - `calls` — [`L19`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/core_types.py#L19)
  - `normal_text` — [`L18`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/core_types.py#L18)
- uses (calls/refs, reference-scoped): [`ToolCallItem`](core_types.md#ToolCallItem)
- used by: [`_tool_indices`](qwen3_coder_detector.md#Qwen3CoderDetector._tool_indices), [`_tool_indices`](glm4_moe_detector.md#Glm4MoeDetector._tool_indices), [`_tool_indices`](base_format_detector.md#BaseFormatDetector._tool_indices), [`_tool_indices`](glm47_moe_detector.md#Glm47MoeDetector._tool_indices), [`parse_streaming_increment`](base_format_detector.md#BaseFormatDetector.parse_streaming_increment), [`_buffer`](mimo_detector.md#MiMoDetector._buffer), [`detect_and_parse`](glm47_moe_detector.md#Glm47MoeDetector.detect_and_parse), [`detect_and_parse`](glm4_moe_detector.md#Glm4MoeDetector.detect_and_parse), [`detect_and_parse`](mimo_detector.md#MiMoDetector.detect_and_parse), [`detect_and_parse`](base_format_detector.md#BaseFormatDetector.detect_and_parse), [`detect_and_parse`](qwen25_detector.md#Qwen25Detector.detect_and_parse), [`parse_streaming_increment`](glm47_moe_detector.md#Glm47MoeDetector.parse_streaming_increment), [`parse_streaming_increment`](qwen25_detector.md#Qwen25Detector.parse_streaming_increment), [`parse_streaming_increment`](mimo_detector.md#MiMoDetector.parse_streaming_increment), [`_buffer`](glm4_moe_detector.md#Glm4MoeDetector._buffer), [`detect_and_parse`](qwen3_coder_detector.md#Qwen3CoderDetector.detect_and_parse), [`parse_streaming_increment`](glm4_moe_detector.md#Glm4MoeDetector.parse_streaming_increment), [`parse_streaming_increment`](qwen3_coder_detector.md#Qwen3CoderDetector.parse_streaming_increment)

### `StructureInfo`
- def: [`python/sgl_jax/srt/function_call/core_types.py:23`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/core_types.py#L23)
- signature: `class StructureInfo:`
- members:
  - `begin` — [`L24`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/core_types.py#L24)
  - `end` — [`L25`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/core_types.py#L25)
  - `trigger` — [`L26`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/core_types.py#L26)
- used by: [`_GetInfoFunc`](core_types.md#_GetInfoFunc), [`structure_info`](qwen25_detector.md#Qwen25Detector.structure_info)

### `ToolCallItem`  ·  implements/extends BaseModel
- def: [`python/sgl_jax/srt/function_call/core_types.py:7`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/core_types.py#L7)
- doc: Simple encapsulation of the parsed ToolCall result for easier usage in streaming contexts.
- signature: `class ToolCallItem(BaseModel):`
- members:
  - `name` — [`L11`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/core_types.py#L11)
  - `parameters` — [`L12`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/core_types.py#L12)
  - `tool_index` — [`L10`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/core_types.py#L10)
- used by: [`_tool_indices`](qwen3_coder_detector.md#Qwen3CoderDetector._tool_indices), [`_tool_indices`](glm4_moe_detector.md#Glm4MoeDetector._tool_indices), [`_tool_indices`](base_format_detector.md#BaseFormatDetector._tool_indices), [`_tool_indices`](glm47_moe_detector.md#Glm47MoeDetector._tool_indices), [`_finalize_tool_call`](glm47_moe_detector.md#Glm47MoeDetector._finalize_tool_call), [`_buffer`](mimo_detector.md#MiMoDetector._buffer), [`_send_tool_name_if_needed`](glm47_moe_detector.md#Glm47MoeDetector._send_tool_name_if_needed), [`_process_arguments_streaming`](glm47_moe_detector.md#Glm47MoeDetector._process_arguments_streaming), [`calls`](core_types.md#StreamingParseResult.calls), [`_parse_and_stream_parameters`](qwen3_coder_detector.md#Qwen3CoderDetector._parse_and_stream_parameters), [`parse_base_json`](base_format_detector.md#BaseFormatDetector.parse_base_json), [`_parse_block`](qwen3_coder_detector.md#Qwen3CoderDetector._parse_block), [`_extract`](qwen3_coder_detector.md#Qwen3CoderDetector._extract), [`parse_non_stream`](function_call_parser.md#FunctionCallParser.parse_non_stream), [`parse_stream_chunk`](function_call_parser.md#FunctionCallParser.parse_stream_chunk), [`parse_streaming_increment`](qwen3_coder_detector.md#Qwen3CoderDetector.parse_streaming_increment)

## Module values
- `_GetInfoFunc` — [`L34`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/core_types.py#L34)

