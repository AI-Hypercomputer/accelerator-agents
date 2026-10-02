---
title: 'Module: python/sgl_jax/srt/function_call/base_format_detector.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/function_call/base_format_detector.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.function_call.base_format_detector`/
symbols:
  BaseFormatDetector.current_tool_id: BaseFormatDetector#current_tool_id.
  BaseFormatDetector._tool_indices: BaseFormatDetector#_tool_indices.
  BaseFormatDetector.parse_streaming_increment: BaseFormatDetector#parse_streaming_increment().
  BaseFormatDetector.streamed_args_for_tool: BaseFormatDetector#streamed_args_for_tool.
  BaseFormatDetector: BaseFormatDetector#
  BaseFormatDetector.parse_base_json: BaseFormatDetector#parse_base_json().
  BaseFormatDetector.detect_and_parse: BaseFormatDetector#detect_and_parse().
  BaseFormatDetector.prev_tool_call_arr: BaseFormatDetector#prev_tool_call_arr.
  BaseFormatDetector.current_tool_name_sent: BaseFormatDetector#current_tool_name_sent.
  BaseFormatDetector.structure_info: BaseFormatDetector#structure_info().
  BaseFormatDetector.build_ebnf: BaseFormatDetector#build_ebnf().
  BaseFormatDetector.has_tool_call: BaseFormatDetector#has_tool_call().
  BaseFormatDetector._get_tool_indices: BaseFormatDetector#_get_tool_indices().
  BaseFormatDetector.supports_structural_tag: BaseFormatDetector#supports_structural_tag().
  BaseFormatDetector._buffer: BaseFormatDetector#_buffer.
  BaseFormatDetector.__init__: BaseFormatDetector#__init__().
  BaseFormatDetector.bot_token: BaseFormatDetector#bot_token.
  BaseFormatDetector.tool_call_separator: BaseFormatDetector#tool_call_separator.
  logger: logger.
  BaseFormatDetector.eot_token: BaseFormatDetector#eot_token.
  BaseFormatDetector._ends_with_partial_token: BaseFormatDetector#_ends_with_partial_token().
---
# Module: [`python/sgl_jax/srt/function_call/base_format_detector.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/base_format_detector.py)

## Classes
### `BaseFormatDetector`  ·  implements/extends ABC
- def: [`python/sgl_jax/srt/function_call/base_format_detector.py:25`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/base_format_detector.py#L25)
- doc: Base class providing two sets of interfaces: one-time and streaming incremental.
- signature: `class BaseFormatDetector(ABC):`
- members:
  - `_ends_with_partial_token(self, buffer: str, bot_token: str)` — [`L101`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/base_format_detector.py#L101) — Check if buffer ends with a partial bot_token.
  - `_get_tool_indices(self, tools: list[Tool])` — [`L52`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/base_format_detector.py#L52) — Get a mapping of tool names to their indices in the tools list.
  - `build_ebnf(self, tools: list[Tool])` — [`L324`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/base_format_detector.py#L324) — Build an EBNF grammar for constrained generation of function calls.
  - `detect_and_parse(self, text: str, tools: list[Tool])` — [`L93`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/base_format_detector.py#L93) — Parses the text in one go. Returns success=True if the format matches, otherwise False.
  - `has_tool_call(self, text: str)` — [`L299`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/base_format_detector.py#L299) — Check if the given text contains function call markers specific to this format.
  - `parse_base_json(self, action: Any, tools: list[Tool])` — [`L68`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/base_format_detector.py#L68)
  - `parse_streaming_increment(self, new_text: str, tools: list[Tool])` — [`L114`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/base_format_detector.py#L114) — Streaming incremental parsing with tool validation.
  - `structure_info(self)` — [`L310`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/base_format_detector.py#L310) — Return a function that creates StructureInfo for constrained generation.
  - `supports_structural_tag(self)` — [`L305`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/base_format_detector.py#L305) — Return True if this detector supports structural tag format.
  - `bot_token` — [`L48`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/base_format_detector.py#L48)
  - `current_tool_id` — [`L38`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/base_format_detector.py#L38)
  - `current_tool_name_sent` — [`L41`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/base_format_detector.py#L41)
  - `eot_token` — [`L49`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/base_format_detector.py#L49)
  - `prev_tool_call_arr` — [`L35`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/base_format_detector.py#L35)
  - `streamed_args_for_tool` — [`L45`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/base_format_detector.py#L45)
  - `tool_call_separator` — [`L50`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/base_format_detector.py#L50)
- protocol/private: `__init__`[`L28`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/base_format_detector.py#L28), `_buffer`[`L31`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/base_format_detector.py#L31), `_tool_indices`[`L153`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/base_format_detector.py#L153)
- uses (calls/refs, reference-scoped): [`StreamingParseResult`](core_types.md#StreamingParseResult), [`Tool`](../entrypoints/openai/protocol.md#Tool), [`ToolCallItem`](core_types.md#ToolCallItem), [`normal_text`](core_types.md#StreamingParseResult.normal_text), [`detect_and_parse`](glm47_moe_detector.md#Glm47MoeDetector.detect_and_parse), [`detect_and_parse`](glm4_moe_detector.md#Glm4MoeDetector.detect_and_parse), [`detect_and_parse`](mimo_detector.md#MiMoDetector.detect_and_parse), [`calls`](core_types.md#StreamingParseResult.calls), [`detect_and_parse`](qwen25_detector.md#Qwen25Detector.detect_and_parse), [`parameters`](core_types.md#ToolCallItem.parameters), [`parse_streaming_increment`](glm47_moe_detector.md#Glm47MoeDetector.parse_streaming_increment), [`parse_streaming_increment`](qwen25_detector.md#Qwen25Detector.parse_streaming_increment), [`tool_index`](core_types.md#ToolCallItem.tool_index), [`name`](core_types.md#ToolCallItem.name), [`parse_streaming_increment`](mimo_detector.md#MiMoDetector.parse_streaming_increment), [`_GetInfoFunc`](core_types.md#_GetInfoFunc), [`build_ebnf`](qwen25_detector.md#Qwen25Detector.build_ebnf), [`build_ebnf`](qwen3_coder_detector.md#Qwen3CoderDetector.build_ebnf), [`detect_and_parse`](qwen3_coder_detector.md#Qwen3CoderDetector.detect_and_parse), [`structure_info`](qwen25_detector.md#Qwen25Detector.structure_info), [`name`](../entrypoints/openai/protocol.md#Function.name), [`parse_streaming_increment`](glm4_moe_detector.md#Glm4MoeDetector.parse_streaming_increment), [`parse_streaming_increment`](qwen3_coder_detector.md#Qwen3CoderDetector.parse_streaming_increment), [`Glm47MoeDetector`](glm47_moe_detector.md#Glm47MoeDetector), [`Glm4MoeDetector`](glm4_moe_detector.md#Glm4MoeDetector), [`MiMoDetector`](mimo_detector.md#MiMoDetector), [`Qwen25Detector`](qwen25_detector.md#Qwen25Detector), [`Qwen3CoderDetector`](qwen3_coder_detector.md#Qwen3CoderDetector), [`build_ebnf`](glm47_moe_detector.md#Glm47MoeDetector.build_ebnf), [`build_ebnf`](glm4_moe_detector.md#Glm4MoeDetector.build_ebnf), [`build_ebnf`](mimo_detector.md#MiMoDetector.build_ebnf), [`has_tool_call`](glm47_moe_detector.md#Glm47MoeDetector.has_tool_call), [`has_tool_call`](glm4_moe_detector.md#Glm4MoeDetector.has_tool_call), [`has_tool_call`](mimo_detector.md#MiMoDetector.has_tool_call), [`has_tool_call`](qwen25_detector.md#Qwen25Detector.has_tool_call), [`has_tool_call`](qwen3_coder_detector.md#Qwen3CoderDetector.has_tool_call), [`structure_info`](glm47_moe_detector.md#Glm47MoeDetector.structure_info), [`structure_info`](glm4_moe_detector.md#Glm4MoeDetector.structure_info), [`structure_info`](mimo_detector.md#MiMoDetector.structure_info), [`structure_info`](qwen3_coder_detector.md#Qwen3CoderDetector.structure_info)  (+8 more)
- used by: [`_tool_indices`](qwen3_coder_detector.md#Qwen3CoderDetector._tool_indices), [`_tool_indices`](glm4_moe_detector.md#Glm4MoeDetector._tool_indices), [`_tool_indices`](glm47_moe_detector.md#Glm47MoeDetector._tool_indices), [`_finalize_tool_call`](glm47_moe_detector.md#Glm47MoeDetector._finalize_tool_call), [`_buffer`](mimo_detector.md#MiMoDetector._buffer), [`get_structure_tag`](function_call_parser.md#FunctionCallParser.get_structure_tag), [`_send_tool_name_if_needed`](glm47_moe_detector.md#Glm47MoeDetector._send_tool_name_if_needed), [`detect_and_parse`](glm47_moe_detector.md#Glm47MoeDetector.detect_and_parse), [`detect_and_parse`](glm4_moe_detector.md#Glm4MoeDetector.detect_and_parse), [`_process_arguments_streaming`](glm47_moe_detector.md#Glm47MoeDetector._process_arguments_streaming), [`detect_and_parse`](mimo_detector.md#MiMoDetector.detect_and_parse), [`_parse_and_stream_parameters`](qwen3_coder_detector.md#Qwen3CoderDetector._parse_and_stream_parameters), [`detect_and_parse`](qwen25_detector.md#Qwen25Detector.detect_and_parse), [`_buffer`](glm47_moe_detector.md#Glm47MoeDetector._buffer), [`_parse_block`](qwen3_coder_detector.md#Qwen3CoderDetector._parse_block), [`get_ebnf`](function_call_parser.md#FunctionCallParser.get_ebnf), [`get_structure_constraint`](function_call_parser.md#FunctionCallParser.get_structure_constraint), [`parse_streaming_increment`](qwen25_detector.md#Qwen25Detector.parse_streaming_increment), [`ToolCallParserEnum`](function_call_parser.md#FunctionCallParser.ToolCallParserEnum), [`_reset_streaming_state`](qwen3_coder_detector.md#Qwen3CoderDetector._reset_streaming_state), [`parse_streaming_increment`](mimo_detector.md#MiMoDetector.parse_streaming_increment), [`_last_arguments`](glm47_moe_detector.md#Glm47MoeDetector._last_arguments), [`_last_arguments`](glm4_moe_detector.md#Glm4MoeDetector._last_arguments), [`parse_non_stream`](function_call_parser.md#FunctionCallParser.parse_non_stream), [`parse_stream_chunk`](function_call_parser.md#FunctionCallParser.parse_stream_chunk), [`__init__`](function_call_parser.md#FunctionCallParser.__init__), [`has_tool_call`](function_call_parser.md#FunctionCallParser.has_tool_call), [`Glm47MoeDetector`](glm47_moe_detector.md#Glm47MoeDetector), [`Glm4MoeDetector`](glm4_moe_detector.md#Glm4MoeDetector), [`MiMoDetector`](mimo_detector.md#MiMoDetector), [`Qwen25Detector`](qwen25_detector.md#Qwen25Detector), [`Qwen3CoderDetector`](qwen3_coder_detector.md#Qwen3CoderDetector), [`__init__`](glm47_moe_detector.md#Glm47MoeDetector.__init__), [`__init__`](glm4_moe_detector.md#Glm4MoeDetector.__init__), [`__init__`](mimo_detector.md#MiMoDetector.__init__), [`__init__`](qwen25_detector.md#Qwen25Detector.__init__), [`__init__`](qwen3_coder_detector.md#Qwen3CoderDetector.__init__)

## Module values
- `logger` — [`L22`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/base_format_detector.py#L22)

