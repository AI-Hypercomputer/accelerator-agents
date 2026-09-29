---
title: 'Module: easydel/inference/tools/parsers/qwen3xml_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/qwen3xml_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.qwen3xml_tool_parser`/Qwen3XMLToolParser#
symbols:
  Qwen3XMLToolParser.extract_tool_calls_streaming: extract_tool_calls_streaming().
  Qwen3XMLToolParser.extract_tool_calls: extract_tool_calls().
  Qwen3XMLToolParser._parse_xml_function_call: _parse_xml_function_call().
  Qwen3XMLToolParser: ''
  Qwen3XMLToolParser.get_streaming_buffer_marker_pairs: get_streaming_buffer_marker_pairs().
  Qwen3XMLToolParser._iter_function_blocks: _iter_function_blocks().
  Qwen3XMLToolParser._iter_function_blocks_from_chunk: _iter_function_blocks_from_chunk().
  Qwen3XMLToolParser._find_first_tool_start: _find_first_tool_start().
  Qwen3XMLToolParser._extract_function_name_from_match: _extract_function_name_from_match().
  Qwen3XMLToolParser._extract_parameter_name_from_match: _extract_parameter_name_from_match().
  Qwen3XMLToolParser.__init__: __init__().
  Qwen3XMLToolParser.get_streaming_buffer_hints: get_streaming_buffer_hints().
  Qwen3XMLToolParser.function_start_regex: function_start_regex.
  Qwen3XMLToolParser._first_nonempty: _first_nonempty().
  Qwen3XMLToolParser.parameter_regex: parameter_regex.
---
# Module: [`easydel/inference/tools/parsers/qwen3xml_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3xml_tool_parser.py)

## Classes
### `Qwen3XMLToolParser`  ·  implements/extends Qwen3CoderToolParser
- def: [`easydel/inference/tools/parsers/qwen3xml_tool_parser.py:45`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3xml_tool_parser.py#L45)
- doc: Dedicated parser for Qwen3 XML outputs.
- signature: `class Qwen3XMLToolParser(Qwen3CoderToolParser):`
- members:
  - `__init__(self, tokenizer)` — [`L55`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3xml_tool_parser.py#L55) — Initialize the Qwen3 XML tool parser.
  - `_extract_function_name_from_match(self, match: re.Match[str])` — [`L124`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3xml_tool_parser.py#L124) — Extract the function name from a ``function_start_regex`` match.
  - `_extract_parameter_name_from_match(self, match: re.Match[str])` — [`L136`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3xml_tool_parser.py#L136) — Extract the parameter name from a ``parameter_regex`` match.
  - `_find_first_tool_start(self, text: str)` — [`L91`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3xml_tool_parser.py#L91) — Find the index of the first tool call or function tag in text.
  - `_first_nonempty(*values: str | None)` — [`L77`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3xml_tool_parser.py#L77) — Return the first non-empty string from the arguments, or None.
  - `_iter_function_blocks(self, text: str, *, complete_only: bool)` — [`L190`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3xml_tool_parser.py#L190) — Extract all function blocks from text, handling optional wrappers.
  - `_iter_function_blocks_from_chunk(self, chunk: str, *, complete_only: bool)` — [`L148`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3xml_tool_parser.py#L148) — Extract function block substrings from a single text chunk.
  - `_parse_xml_function_call(self, function_block: str, tools)` — [`L215`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3xml_tool_parser.py#L215) — Parse an XML function block into a ToolCall object.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L262`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3xml_tool_parser.py#L262) — Extract tool calls from a complete (non-streaming) model output.
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L313`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3xml_tool_parser.py#L313) — Extract tool calls from streaming model output.
  - `get_streaming_buffer_hints(self)` — [`L115`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3xml_tool_parser.py#L115) — Include bare XML prefixes used by the Qwen3 XML protocol.
  - `get_streaming_buffer_marker_pairs(self)` — [`L106`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3xml_tool_parser.py#L106) — Extend buffer detection to bare ``<function ...>`` and ``<parameter ...>`` tags.
  - `function_start_regex` — [`L67`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3xml_tool_parser.py#L67)
  - `parameter_regex` — [`L71`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3xml_tool_parser.py#L71)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`DeltaToolCall`](../../openai_api_modules.md#DeltaToolCall), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`DeltaFunctionCall`](../../openai_api_modules.md#DeltaFunctionCall), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolCall`](../../openai_api_modules.md#ToolCall), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`index`](../../openai_api_modules.md#DeltaToolCall.index), [`ToolParserManager`](../abstract_tool.md#ToolParserManager), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`FunctionCall`](../../openai_api_modules.md#FunctionCall), [`arguments`](../../openai_api_modules.md#FunctionCall.arguments), [`name`](../../openai_api_modules.md#FunctionCall.name), [`register_module`](../abstract_tool.md#ToolParserManager.register_module), [`id`](../../openai_api_modules.md#DeltaToolCall.id), [`name`](../../openai_api_modules.md#DeltaFunctionCall.name), [`tools`](../../openai_api_modules.md#ChatCompletionRequest.tools), [`current_tool_index`](qwen3coder_tool_parser.md#Qwen3CoderToolParser.current_tool_index), [`Qwen3CoderToolParser`](qwen3coder_tool_parser.md#Qwen3CoderToolParser), [`_get_arguments_config`](qwen3coder_tool_parser.md#Qwen3CoderToolParser._get_arguments_config), [`tool_call_start_token`](qwen3coder_tool_parser.md#Qwen3CoderToolParser.tool_call_start_token), [`prev_tool_call_arr`](qwen3coder_tool_parser.md#Qwen3CoderToolParser.prev_tool_call_arr), [`streamed_args_for_tool`](qwen3coder_tool_parser.md#Qwen3CoderToolParser.streamed_args_for_tool), [`streaming_request`](qwen3coder_tool_parser.md#Qwen3CoderToolParser.streaming_request), [`function_end_token`](qwen3coder_tool_parser.md#Qwen3CoderToolParser.function_end_token), [`get_streaming_buffer_hints`](../abstract_tool.md#ToolParser.get_streaming_buffer_hints), [`is_tool_call_started`](qwen3coder_tool_parser.md#Qwen3CoderToolParser.is_tool_call_started), [`_convert_param_value`](qwen3coder_tool_parser.md#Qwen3CoderToolParser._convert_param_value), [`get_streaming_buffer_marker_pairs`](../abstract_tool.md#ToolParser.get_streaming_buffer_marker_pairs), [`__init__`](qwen3coder_tool_parser.md#Qwen3CoderToolParser.__init__), [`_reset_streaming_state`](qwen3coder_tool_parser.md#Qwen3CoderToolParser._reset_streaming_state), [`_generate_tool_call_id`](qwen3coder_tool_parser.md#Qwen3CoderToolParser._generate_tool_call_id), [`parameter_end_token`](qwen3coder_tool_parser.md#Qwen3CoderToolParser.parameter_end_token), [`tool_call_regex`](qwen3coder_tool_parser.md#Qwen3CoderToolParser.tool_call_regex)
- used by: [`extract_tool_calls_streaming`](qwen3coder_tool_parser.md#Qwen3CoderToolParser.extract_tool_calls_streaming), [`extract_tool_calls`](qwen3coder_tool_parser.md#Qwen3CoderToolParser.extract_tool_calls), [`_parse_xml_function_call`](qwen3coder_tool_parser.md#Qwen3CoderToolParser._parse_xml_function_call), [`Qwen3CoderToolParser`](qwen3coder_tool_parser.md#Qwen3CoderToolParser), [`get_streaming_buffer_hints`](../abstract_tool.md#ToolParser.get_streaming_buffer_hints), [`get_streaming_buffer_marker_pairs`](../abstract_tool.md#ToolParser.get_streaming_buffer_marker_pairs)

