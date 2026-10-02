---
title: 'Module: easydel/inference/tools/parsers/deepseekv32_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/deepseekv32_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.deepseekv32_tool_parser`/
symbols:
  DeepSeekV32ToolParser._extract_delta_tool_calls: DeepSeekV32ToolParser#_extract_delta_tool_calls().
  DeepSeekV32ToolParser.extract_tool_calls: DeepSeekV32ToolParser#extract_tool_calls().
  DeepSeekV32ToolParser.extract_tool_calls_streaming: DeepSeekV32ToolParser#extract_tool_calls_streaming().
  DeepSeekV32ToolParser._convert_params_with_schema: DeepSeekV32ToolParser#_convert_params_with_schema().
  DeepSeekV32ToolParser._reset_streaming_state: DeepSeekV32ToolParser#_reset_streaming_state().
  DeepSeekV32ToolParser.adjust_request: DeepSeekV32ToolParser#adjust_request().
  logger: logger.
  DeepSeekV32ToolParser.current_tool_index: DeepSeekV32ToolParser#current_tool_index.
  DeepSeekV32ToolParser.streaming_request: DeepSeekV32ToolParser#streaming_request.
  DeepSeekV32ToolParser: DeepSeekV32ToolParser#
  DeepSeekV32ToolParser._parse_invoke_params: DeepSeekV32ToolParser#_parse_invoke_params().
  DeepSeekV32ToolParser.tool_call_start_token: DeepSeekV32ToolParser#tool_call_start_token.
  DeepSeekV32ToolParser.parameter_complete_regex: DeepSeekV32ToolParser#parameter_complete_regex.
  DeepSeekV32ToolParser.prev_tool_call_arr: DeepSeekV32ToolParser#prev_tool_call_arr.
  DeepSeekV32ToolParser.is_tool_call_started: DeepSeekV32ToolParser#is_tool_call_started.
  DeepSeekV32ToolParser.__init__: DeepSeekV32ToolParser#__init__().
  DeepSeekV32ToolParser.dsml_start_check: DeepSeekV32ToolParser#dsml_start_check.
  DeepSeekV32ToolParser.streamed_args_for_tool: DeepSeekV32ToolParser#streamed_args_for_tool.
  DeepSeekV32ToolParser.invoke_complete_regex: DeepSeekV32ToolParser#invoke_complete_regex.
  DeepSeekV32ToolParser.dsml_token: DeepSeekV32ToolParser#dsml_token.
  DeepSeekV32ToolParser.tool_call_complete_regex: DeepSeekV32ToolParser#tool_call_complete_regex.
  DeepSeekV32ToolParser._generate_tool_call_id: DeepSeekV32ToolParser#_generate_tool_call_id().
  DeepSeekV32ToolParser._convert_param_value: DeepSeekV32ToolParser#_convert_param_value().
  DeepSeekV32ToolParser.tool_call_end_token: DeepSeekV32ToolParser#tool_call_end_token.
  DeepSeekV32ToolParser.invoke_start_prefix: DeepSeekV32ToolParser#invoke_start_prefix.
  DeepSeekV32ToolParser.invoke_end_token: DeepSeekV32ToolParser#invoke_end_token.
  DeepSeekV32ToolParser.parameter_prefix: DeepSeekV32ToolParser#parameter_prefix.
  DeepSeekV32ToolParser.parameter_end_token: DeepSeekV32ToolParser#parameter_end_token.
  DeepSeekV32ToolParser.current_tool_name_sent: DeepSeekV32ToolParser#current_tool_name_sent.
  DeepSeekV32ToolParser.current_tool_id: DeepSeekV32ToolParser#current_tool_id.
  DeepSeekV32ToolParser.failed_count: DeepSeekV32ToolParser#failed_count.
  DeepSeekV32ToolParser.invoke_index: DeepSeekV32ToolParser#invoke_index.
  DeepSeekV32ToolParser.header_sent: DeepSeekV32ToolParser#header_sent.
  DeepSeekV32ToolParser.current_function_name: DeepSeekV32ToolParser#current_function_name.
  DeepSeekV32ToolParser.current_param_name: DeepSeekV32ToolParser#current_param_name.
  DeepSeekV32ToolParser.current_param_value: DeepSeekV32ToolParser#current_param_value.
  DeepSeekV32ToolParser.param_count: DeepSeekV32ToolParser#param_count.
  DeepSeekV32ToolParser.in_param: DeepSeekV32ToolParser#in_param.
  DeepSeekV32ToolParser.in_function: DeepSeekV32ToolParser#in_function.
  DeepSeekV32ToolParser.json_started: DeepSeekV32ToolParser#json_started.
  DeepSeekV32ToolParser.json_closed: DeepSeekV32ToolParser#json_closed.
  DeepSeekV32ToolParser.accumulated_params: DeepSeekV32ToolParser#accumulated_params.
  DeepSeekV32ToolParser._extract_name: DeepSeekV32ToolParser#_extract_name().
  DeepSeekV32ToolParser._extract_param_name: DeepSeekV32ToolParser#_extract_param_name().
---
# Module: [`easydel/inference/tools/parsers/deepseekv32_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py)

## Classes
### `DeepSeekV32ToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/deepseekv32_tool_parser.py:56`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L56)
- doc: Tool parser for DeepSeek V3.2 models using DSML format.
- signature: `class DeepSeekV32ToolParser(ToolParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L108`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L108) — Initialize the DeepSeek V3.2 tool parser.
  - `_convert_param_value(self, value: str, param_type: str)` — [`L325`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L325) — Convert parameter value to the correct Python type.
  - `_convert_params_with_schema(self, function_name: str, param_dict: dict[str, str], request: ChatCompletionRequest | None)` — [`L370`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L370) — Convert raw string parameter values using the tool schema types.
  - `_extract_delta_tool_calls(self, current_text: str, request: ChatCompletionRequest | None)` — [`L412`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L412) — Extract DeltaToolCalls from newly completed invoke blocks.
  - `_extract_name(self, name_str: str)` — [`L288`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L288) — Extract name from a potentially quoted string.
  - `_extract_param_name(self, input_str: str)` — [`L308`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L308) — Extract parameter name from an attribute string.
  - `_generate_tool_call_id(self)` — [`L190`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L190) — Generate a unique tool call ID.
  - `_parse_invoke_params(self, invoke_str: str)` — [`L213`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L213) — Parse parameters from an invoke block string.
  - `_reset_streaming_state(self)` — [`L201`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L201) — Reset all streaming state for a new message.
  - `adjust_request(self, request: ChatCompletionRequest)` — [`L172`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L172) — Adjust the chat completion request for DSML tool call parsing.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L232`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L232) — Extract tool calls from complete model output (non-streaming).
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L456`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L456) — Extract tool calls from streaming model output.
  - `accumulated_params` — [`L155`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L155)
  - `current_function_name` — [`L147`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L147)
  - `current_param_name` — [`L148`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L148)
  - `current_param_value` — [`L149`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L149)
  - `current_tool_id` — [`L138`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L138)
  - `current_tool_index` — [`L144`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L144)
  - `current_tool_name_sent` — [`L136`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L136)
  - `dsml_start_check` — [`L127`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L127)
  - `dsml_token` — [`L126`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L126)
  - `failed_count` — [`L141`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L141)
  - `header_sent` — [`L146`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L146)
  - `in_function` — [`L152`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L152)
  - `in_param` — [`L151`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L151)
  - `invoke_complete_regex` — [`L163`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L163)
  - `invoke_end_token` — [`L131`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L131)
  - `invoke_index` — [`L145`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L145)
  - `invoke_start_prefix` — [`L130`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L130)
  - `is_tool_call_started` — [`L140`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L140)
  - `json_closed` — [`L154`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L154)
  - `json_started` — [`L153`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L153)
  - `param_count` — [`L150`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L150)
  - `parameter_complete_regex` — [`L164`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L164)
  - `parameter_end_token` — [`L133`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L133)
  - `parameter_prefix` — [`L132`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L132)
  - `prev_tool_call_arr` — [`L123`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L123)
  - `streamed_args_for_tool` — [`L139`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L139)
  - `streaming_request` — [`L156`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L156)
  - `tool_call_complete_regex` — [`L162`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L162)
  - `tool_call_end_token` — [`L129`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L129)
  - `tool_call_start_token` — [`L128`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L128)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`DeltaToolCall`](../../openai_api_modules.md#DeltaToolCall), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`DeltaFunctionCall`](../../openai_api_modules.md#DeltaFunctionCall), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolCall`](../../openai_api_modules.md#ToolCall), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`index`](../../openai_api_modules.md#DeltaToolCall.index), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`FunctionCall`](../../openai_api_modules.md#FunctionCall), [`arguments`](../../openai_api_modules.md#FunctionCall.arguments), [`name`](../../openai_api_modules.md#FunctionCall.name), [`id`](../../openai_api_modules.md#DeltaToolCall.id), [`name`](../../openai_api_modules.md#DeltaFunctionCall.name), [`tools`](../../openai_api_modules.md#ChatCompletionRequest.tools), [`model_tokenizer`](../abstract_tool.md#ToolParser.model_tokenizer), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`adjust_request`](../abstract_tool.md#ToolParser.adjust_request), [`tool_choice`](../../openai_api_modules.md#ChatCompletionRequest.tool_choice), [`name`](../../openai_api_modules.md#FunctionDefinition.name), [`parameters`](../../openai_api_modules.md#FunctionDefinition.parameters), [`logger`](deepseekv32_tool_parser.md#logger)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming), [`adjust_request`](../abstract_tool.md#ToolParser.adjust_request)

## Module values
- `logger` — [`L52`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv32_tool_parser.py#L52)

