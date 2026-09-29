---
title: 'Module: easydel/inference/tools/parsers/qwen3coder_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/qwen3coder_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.qwen3coder_tool_parser`/
symbols:
  Qwen3CoderToolParser.extract_tool_calls_streaming: Qwen3CoderToolParser#extract_tool_calls_streaming().
  Qwen3CoderToolParser.extract_tool_calls: Qwen3CoderToolParser#extract_tool_calls().
  Qwen3CoderToolParser.current_tool_index: Qwen3CoderToolParser#current_tool_index.
  Qwen3CoderToolParser._parse_xml_function_call: Qwen3CoderToolParser#_parse_xml_function_call().
  logger: logger.
  Qwen3CoderToolParser.tool_call_end_token_id: Qwen3CoderToolParser#tool_call_end_token_id.
  Qwen3CoderToolParser: Qwen3CoderToolParser#
  Qwen3CoderToolParser._get_arguments_config: Qwen3CoderToolParser#_get_arguments_config().
  Qwen3CoderToolParser.tool_call_start_token: Qwen3CoderToolParser#tool_call_start_token.
  Qwen3CoderToolParser.prev_tool_call_arr: Qwen3CoderToolParser#prev_tool_call_arr.
  Qwen3CoderToolParser.streamed_args_for_tool: Qwen3CoderToolParser#streamed_args_for_tool.
  Qwen3CoderToolParser.streaming_request: Qwen3CoderToolParser#streaming_request.
  Qwen3CoderToolParser.function_end_token: Qwen3CoderToolParser#function_end_token.
  Qwen3CoderToolParser.tool_call_end_token: Qwen3CoderToolParser#tool_call_end_token.
  Qwen3CoderToolParser.tool_call_prefix: Qwen3CoderToolParser#tool_call_prefix.
  Qwen3CoderToolParser.is_tool_call_started: Qwen3CoderToolParser#is_tool_call_started.
  Qwen3CoderToolParser.tool_call_start_token_id: Qwen3CoderToolParser#tool_call_start_token_id.
  Qwen3CoderToolParser._get_function_calls: Qwen3CoderToolParser#_get_function_calls().
  Qwen3CoderToolParser.header_sent: Qwen3CoderToolParser#header_sent.
  Qwen3CoderToolParser._convert_param_value: Qwen3CoderToolParser#_convert_param_value().
  Qwen3CoderToolParser.current_function_name: Qwen3CoderToolParser#current_function_name.
  Qwen3CoderToolParser.param_count: Qwen3CoderToolParser#param_count.
  Qwen3CoderToolParser.json_closed: Qwen3CoderToolParser#json_closed.
  Qwen3CoderToolParser.parameter_prefix: Qwen3CoderToolParser#parameter_prefix.
  Qwen3CoderToolParser.in_function: Qwen3CoderToolParser#in_function.
  Qwen3CoderToolParser.__init__: Qwen3CoderToolParser#__init__().
  Qwen3CoderToolParser.tool_call_parameter_regex: Qwen3CoderToolParser#tool_call_parameter_regex.
  Qwen3CoderToolParser.current_tool_id: Qwen3CoderToolParser#current_tool_id.
  Qwen3CoderToolParser._reset_streaming_state: Qwen3CoderToolParser#_reset_streaming_state().
  Qwen3CoderToolParser.json_started: Qwen3CoderToolParser#json_started.
  Qwen3CoderToolParser.accumulated_params: Qwen3CoderToolParser#accumulated_params.
  Qwen3CoderToolParser.failed_count: Qwen3CoderToolParser#failed_count.
  Qwen3CoderToolParser.parameter_end_token: Qwen3CoderToolParser#parameter_end_token.
  Qwen3CoderToolParser.tool_call_regex: Qwen3CoderToolParser#tool_call_regex.
  Qwen3CoderToolParser._generate_tool_call_id: Qwen3CoderToolParser#_generate_tool_call_id().
  Qwen3CoderToolParser.tool_call_complete_regex: Qwen3CoderToolParser#tool_call_complete_regex.
  Qwen3CoderToolParser.tool_call_function_regex: Qwen3CoderToolParser#tool_call_function_regex.
  Qwen3CoderToolParser.current_param_name: Qwen3CoderToolParser#current_param_name.
  Qwen3CoderToolParser.in_param: Qwen3CoderToolParser#in_param.
  Qwen3CoderToolParser.accumulated_text: Qwen3CoderToolParser#accumulated_text.
  Qwen3CoderToolParser.current_tool_name_sent: Qwen3CoderToolParser#current_tool_name_sent.
  Qwen3CoderToolParser.current_param_value: Qwen3CoderToolParser#current_param_value.
---
# Module: [`easydel/inference/tools/parsers/qwen3coder_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py)

## Classes
### `Qwen3CoderToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/qwen3coder_tool_parser.py:45`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L45)
- doc: Tool parser for Qwen3 Coder XML tool calls.
- signature: `class Qwen3CoderToolParser(ToolParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L62`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L62) — Initialize the Qwen3 Coder tool parser.
  - `_convert_param_value(self, param_value: str, param_name: str, param_config: dict, func_name: str)` — [`L175`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L175) — Convert a raw string parameter value to its schema-defined type.
  - `_generate_tool_call_id(self)` — [`L115`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L115) — Generate a unique tool call ID.
  - `_get_arguments_config(self, func_name: str, tools: list[ToolDefinition] | None)` — [`L142`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L142) — Extract the parameter properties schema for a function.
  - `_get_function_calls(self, model_output: str)` — [`L317`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L317) — Extract raw function-call body strings from model output.
  - `_parse_xml_function_call(self, function_call_str: str, tools: list[ToolDefinition] | None)` — [`L280`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L280) — Parse a single XML function call string into a ToolCall object.
  - `_reset_streaming_state(self)` — [`L119`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L119) — Reset all streaming state for a new message.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L343`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L343) — Extract tool calls from a complete (non-streaming) model output.
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L398`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L398) — Extract tool calls incrementally from streaming model output.
  - `accumulated_params` — [`L139`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L139)
  - `accumulated_text` — [`L136`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L136)
  - `current_function_name` — [`L130`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L130)
  - `current_param_name` — [`L131`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L131)
  - `current_param_value` — [`L132`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L132)
  - `current_tool_id` — [`L82`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L82)
  - `current_tool_index` — [`L126`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L126)
  - `current_tool_name_sent` — [`L79`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L79)
  - `failed_count` — [`L92`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L92)
  - `function_end_token` — [`L88`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L88)
  - `header_sent` — [`L128`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L128)
  - `in_function` — [`L135`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L135)
  - `in_param` — [`L134`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L134)
  - `is_tool_call_started` — [`L91`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L91)
  - `json_closed` — [`L138`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L138)
  - `json_started` — [`L137`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L137)
  - `param_count` — [`L133`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L133)
  - `parameter_end_token` — [`L90`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L90)
  - `parameter_prefix` — [`L89`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L89)
  - `prev_tool_call_arr` — [`L80`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L80)
  - `streamed_args_for_tool` — [`L83`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L83)
  - `streaming_request` — [`L140`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L140)
  - `tool_call_complete_regex` — [`L96`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L96)
  - `tool_call_end_token` — [`L86`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L86)
  - `tool_call_end_token_id` — [`L108`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L108)
  - `tool_call_function_regex` — [`L98`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L98)
  - `tool_call_parameter_regex` — [`L99`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L99)
  - `tool_call_prefix` — [`L87`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L87)
  - `tool_call_regex` — [`L97`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L97)
  - `tool_call_start_token` — [`L85`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L85)
  - `tool_call_start_token_id` — [`L107`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L107)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`DeltaToolCall`](../../openai_api_modules.md#DeltaToolCall), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`DeltaFunctionCall`](../../openai_api_modules.md#DeltaFunctionCall), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolCall`](../../openai_api_modules.md#ToolCall), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`index`](../../openai_api_modules.md#DeltaToolCall.index), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`FunctionCall`](../../openai_api_modules.md#FunctionCall), [`extract_tool_calls_streaming`](qwen3xml_tool_parser.md#Qwen3XMLToolParser.extract_tool_calls_streaming), [`arguments`](../../openai_api_modules.md#FunctionCall.arguments), [`name`](../../openai_api_modules.md#FunctionCall.name), [`id`](../../openai_api_modules.md#DeltaToolCall.id), [`name`](../../openai_api_modules.md#DeltaFunctionCall.name), [`vocab`](../abstract_tool.md#ToolParser.vocab), [`tools`](../../openai_api_modules.md#ChatCompletionRequest.tools), [`extract_tool_calls`](qwen3xml_tool_parser.md#Qwen3XMLToolParser.extract_tool_calls), [`model_tokenizer`](../abstract_tool.md#ToolParser.model_tokenizer), [`_parse_xml_function_call`](qwen3xml_tool_parser.md#Qwen3XMLToolParser._parse_xml_function_call), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`ToolDefinition`](../../openai_api_modules.md#ToolDefinition), [`logger`](qwen3coder_tool_parser.md#logger), [`Qwen3XMLToolParser`](qwen3xml_tool_parser.md#Qwen3XMLToolParser), [`Step3p5ToolParser`](step3p5_tool_parser.md#Step3p5ToolParser), [`name`](../../openai_api_modules.md#FunctionDefinition.name), [`parameters`](../../openai_api_modules.md#FunctionDefinition.parameters)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming), [`extract_tool_calls_streaming`](qwen3xml_tool_parser.md#Qwen3XMLToolParser.extract_tool_calls_streaming), [`extract_tool_calls`](qwen3xml_tool_parser.md#Qwen3XMLToolParser.extract_tool_calls), [`_parse_xml_function_call`](qwen3xml_tool_parser.md#Qwen3XMLToolParser._parse_xml_function_call), [`Qwen3XMLToolParser`](qwen3xml_tool_parser.md#Qwen3XMLToolParser), [`Step3p5ToolParser`](step3p5_tool_parser.md#Step3p5ToolParser), [`get_streaming_buffer_marker_pairs`](qwen3xml_tool_parser.md#Qwen3XMLToolParser.get_streaming_buffer_marker_pairs), [`_iter_function_blocks`](qwen3xml_tool_parser.md#Qwen3XMLToolParser._iter_function_blocks), [`_iter_function_blocks_from_chunk`](qwen3xml_tool_parser.md#Qwen3XMLToolParser._iter_function_blocks_from_chunk), [`_find_first_tool_start`](qwen3xml_tool_parser.md#Qwen3XMLToolParser._find_first_tool_start), [`__init__`](qwen3xml_tool_parser.md#Qwen3XMLToolParser.__init__)

## Module values
- `logger` — [`L41`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/qwen3coder_tool_parser.py#L41)

