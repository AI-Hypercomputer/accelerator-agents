---
title: 'Module: python/sgl_jax/srt/function_call/function_call_parser.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/function_call/function_call_parser.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.function_call.function_call_parser`/
symbols:
  FunctionCallParser.get_structure_tag: FunctionCallParser#get_structure_tag().
  FunctionCallParser.get_structure_constraint: FunctionCallParser#get_structure_constraint().
  FunctionCallParser.get_ebnf: FunctionCallParser#get_ebnf().
  FunctionCallParser.ToolCallParserEnum: FunctionCallParser#ToolCallParserEnum.
  FunctionCallParser.tools: FunctionCallParser#tools.
  FunctionCallParser.parse_non_stream: FunctionCallParser#parse_non_stream().
  FunctionCallParser.parse_stream_chunk: FunctionCallParser#parse_stream_chunk().
  FunctionCallParser.__init__: FunctionCallParser#__init__().
  FunctionCallParser.has_tool_call: FunctionCallParser#has_tool_call().
  FunctionCallParser.detector: FunctionCallParser#detector.
  FunctionCallParser: FunctionCallParser#
  logger: logger.
---
# Module: [`python/sgl_jax/srt/function_call/function_call_parser.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/function_call_parser.py)

## Classes
### `FunctionCallParser`
- def: [`python/sgl_jax/srt/function_call/function_call_parser.py:22`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/function_call_parser.py#L22)
- doc: Parser for function/tool calls in model outputs.
- signature: `class FunctionCallParser:`
- members:
  - `get_ebnf(self, tool_choice: ToolChoice | Literal["required"])` — [`L175`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/function_call_parser.py#L175) — Get the EBNF grammar for the specified tool choice.
  - `get_structure_constraint(self, tool_choice: ToolChoice | Literal["auto", "required"])` — [`L146`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/function_call_parser.py#L146) — Returns the appropriate structure constraint for tool calls based on the tool_choice.
  - `get_structure_tag(self)` — [`L112`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/function_call_parser.py#L112) — Generate a structural tag response format for all available tools.
  - `has_tool_call(self, text: str)` — [`L50`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/function_call_parser.py#L50) — Check if the given text contains a tool call in the format supported by this parser.
  - `parse_non_stream(self, full_text: str)` — [`L65`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/function_call_parser.py#L65) — One-time parsing of the full text to extract tool calls.
  - `parse_stream_chunk(self, chunk_text: str)` — [`L86`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/function_call_parser.py#L86) — Streaming incremental parsing of chunks of text as they arrive.
  - `ToolCallParserEnum` — [`L31`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/function_call_parser.py#L31)
  - `detector` — [`L47`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/function_call_parser.py#L47) — ---
  - `tools` — [`L48`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/function_call_parser.py#L48)
- protocol/private: `__init__`[`L39`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/function_call_parser.py#L39)
- uses (calls/refs, reference-scoped): [`Tool`](../entrypoints/openai/protocol.md#Tool), [`parse_streaming_increment`](base_format_detector.md#BaseFormatDetector.parse_streaming_increment), [`ToolCallItem`](core_types.md#ToolCallItem), [`BaseFormatDetector`](base_format_detector.md#BaseFormatDetector), [`detect_and_parse`](base_format_detector.md#BaseFormatDetector.detect_and_parse), [`get_json_schema_constraint`](utils.md#get_json_schema_constraint), [`build_ebnf`](base_format_detector.md#BaseFormatDetector.build_ebnf), [`structure_info`](base_format_detector.md#BaseFormatDetector.structure_info), [`has_tool_call`](base_format_detector.md#BaseFormatDetector.has_tool_call), [`ToolChoice`](../entrypoints/openai/protocol.md#ToolChoice), [`supports_structural_tag`](base_format_detector.md#BaseFormatDetector.supports_structural_tag), [`name`](../entrypoints/openai/protocol.md#Function.name), [`parameters`](../entrypoints/openai/protocol.md#Function.parameters), [`Glm47MoeDetector`](glm47_moe_detector.md#Glm47MoeDetector), [`Glm4MoeDetector`](glm4_moe_detector.md#Glm4MoeDetector), [`MiMoDetector`](mimo_detector.md#MiMoDetector), [`Qwen25Detector`](qwen25_detector.md#Qwen25Detector), [`Qwen3CoderDetector`](qwen3_coder_detector.md#Qwen3CoderDetector), [`StructuralTagResponseFormat`](../entrypoints/openai/protocol.md#StructuralTagResponseFormat), [`StructuresResponseFormat`](../entrypoints/openai/protocol.md#StructuresResponseFormat), [`structures`](../entrypoints/openai/protocol.md#StructuralTagResponseFormat.structures), [`name`](../entrypoints/openai/protocol.md#ToolChoiceFuncName.name), [`strict`](../entrypoints/openai/protocol.md#Function.strict), [`begin`](../entrypoints/openai/protocol.md#StructuresResponseFormat.begin), [`end`](../entrypoints/openai/protocol.md#StructuresResponseFormat.end), [`logger`](function_call_parser.md#logger), [`triggers`](../entrypoints/openai/protocol.md#StructuralTagResponseFormat.triggers)
- used by: [`add_cli_args`](../server_args.md#ServerArgs.add_cli_args)

## Module values
- `logger` — [`L19`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/function_call/function_call_parser.py#L19)

