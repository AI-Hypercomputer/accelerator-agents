---
title: 'Module: easydel/inference/tools/parsers/minimax_m2_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/minimax_m2_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.minimax_m2_tool_parser`/
symbols:
  MinimaxM2ToolParser.extract_tool_calls_streaming: MinimaxM2ToolParser#extract_tool_calls_streaming().
  MinimaxM2ToolParser._reset_streaming_state: MinimaxM2ToolParser#_reset_streaming_state().
  MinimaxM2ToolParser.extract_tool_calls: MinimaxM2ToolParser#extract_tool_calls().
  MinimaxM2ToolParser._parse_single_invoke: MinimaxM2ToolParser#_parse_single_invoke().
  MinimaxM2ToolParser.current_tool_index: MinimaxM2ToolParser#current_tool_index.
  MinimaxM2ToolParser.streaming_request: MinimaxM2ToolParser#streaming_request.
  MinimaxM2ToolParser.tool_call_end_token_id: MinimaxM2ToolParser#tool_call_end_token_id.
  MinimaxM2ToolParser.prev_tool_call_arr: MinimaxM2ToolParser#prev_tool_call_arr.
  MinimaxM2ToolParser.tool_call_start_token: MinimaxM2ToolParser#tool_call_start_token.
  MinimaxM2ToolParser.invoke_start_prefix: MinimaxM2ToolParser#invoke_start_prefix.
  MinimaxM2ToolParser.param_count: MinimaxM2ToolParser#param_count.
  MinimaxM2ToolParser.invoke_end_token: MinimaxM2ToolParser#invoke_end_token.
  MinimaxM2ToolParser.in_function: MinimaxM2ToolParser#in_function.
  MinimaxM2ToolParser.tool_call_start_token_id: MinimaxM2ToolParser#tool_call_start_token_id.
  logger: logger.
  MinimaxM2ToolParser.current_function_name: MinimaxM2ToolParser#current_function_name.
  MinimaxM2ToolParser.current_param_name: MinimaxM2ToolParser#current_param_name.
  MinimaxM2ToolParser.json_started: MinimaxM2ToolParser#json_started.
  MinimaxM2ToolParser.json_closed: MinimaxM2ToolParser#json_closed.
  MinimaxM2ToolParser._get_param_types_from_config: MinimaxM2ToolParser#_get_param_types_from_config().
  MinimaxM2ToolParser.parameter_prefix: MinimaxM2ToolParser#parameter_prefix.
  MinimaxM2ToolParser._extract_name: MinimaxM2ToolParser#_extract_name().
  MinimaxM2ToolParser: MinimaxM2ToolParser#
  MinimaxM2ToolParser.parameter_complete_regex: MinimaxM2ToolParser#parameter_complete_regex.
  MinimaxM2ToolParser._extract_types_from_schema: MinimaxM2ToolParser#_extract_types_from_schema().
  MinimaxM2ToolParser.is_tool_call_started: MinimaxM2ToolParser#is_tool_call_started.
  MinimaxM2ToolParser.header_sent: MinimaxM2ToolParser#header_sent.
  MinimaxM2ToolParser.accumulated_params: MinimaxM2ToolParser#accumulated_params.
  MinimaxM2ToolParser._convert_param_value_with_types: MinimaxM2ToolParser#_convert_param_value_with_types().
  MinimaxM2ToolParser.tool_call_end_token: MinimaxM2ToolParser#tool_call_end_token.
  MinimaxM2ToolParser.current_tool_id: MinimaxM2ToolParser#current_tool_id.
  MinimaxM2ToolParser.streamed_args_for_tool: MinimaxM2ToolParser#streamed_args_for_tool.
  MinimaxM2ToolParser.__init__: MinimaxM2ToolParser#__init__().
  MinimaxM2ToolParser._convert_param_value: MinimaxM2ToolParser#_convert_param_value().
  MinimaxM2ToolParser.in_param: MinimaxM2ToolParser#in_param.
  MinimaxM2ToolParser.accumulated_text: MinimaxM2ToolParser#accumulated_text.
  MinimaxM2ToolParser.tool_call_complete_regex: MinimaxM2ToolParser#tool_call_complete_regex.
  MinimaxM2ToolParser.parameter_end_token: MinimaxM2ToolParser#parameter_end_token.
  MinimaxM2ToolParser.invoke_index: MinimaxM2ToolParser#invoke_index.
  MinimaxM2ToolParser.current_param_value: MinimaxM2ToolParser#current_param_value.
  MinimaxM2ToolParser.invoke_complete_regex: MinimaxM2ToolParser#invoke_complete_regex.
  MinimaxM2ToolParser._generate_tool_call_id: MinimaxM2ToolParser#_generate_tool_call_id().
  MinimaxM2ToolParser.current_tool_name_sent: MinimaxM2ToolParser#current_tool_name_sent.
  MinimaxM2ToolParser.failed_count: MinimaxM2ToolParser#failed_count.
---
# Module: [`easydel/inference/tools/parsers/minimax_m2_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py)

## Classes
### `MinimaxM2ToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/minimax_m2_tool_parser.py:63`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L63)
- doc: Tool parser for Minimax M2 models with XML-style tool calls.
- signature: `class MinimaxM2ToolParser(ToolParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L97`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L97) — Initialize the MinimaxM2ToolParser.
  - `_convert_param_value(self, value: str, param_type: str)` — [`L212`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L212) — Convert a parameter value based on a single type.
  - `_convert_param_value_with_types(self, value: str, param_types: list[str])` — [`L281`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L281) — Convert a parameter value based on possible types.
  - `_extract_name(self, name_str: str)` — [`L194`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L194) — Extract a name from a quoted or unquoted string.
  - `_extract_types_from_schema(self, schema: Any)` — [`L227`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L227) — Extract possible types from a JSON schema definition.
  - `_generate_tool_call_id(self)` — [`L159`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L159) — Generate a unique tool call ID.
  - `_get_param_types_from_config(self, param_name: str, param_config: dict)` — [`L352`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L352) — Get possible types for a parameter from tool configuration.
  - `_parse_single_invoke(self, invoke_str: str, tools: list | None)` — [`L373`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L373) — Parse a single invoke element into a ToolCall.
  - `_reset_streaming_state(self)` — [`L169`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L169) — Reset all streaming state to initial values.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L426`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L426) — Extract tool calls from a complete model response.
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L471`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L471) — Extract tool calls incrementally during streaming.
  - `accumulated_params` — [`L141`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L141)
  - `accumulated_text` — [`L138`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L138)
  - `current_function_name` — [`L132`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L132)
  - `current_param_name` — [`L133`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L133)
  - `current_param_value` — [`L134`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L134)
  - `current_tool_id` — [`L124`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L124)
  - `current_tool_index` — [`L129`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L129)
  - `current_tool_name_sent` — [`L123`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L123)
  - `failed_count` — [`L127`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L127)
  - `header_sent` — [`L131`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L131)
  - `in_function` — [`L137`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L137)
  - `in_param` — [`L136`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L136)
  - `invoke_complete_regex` — [`L147`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L147)
  - `invoke_end_token` — [`L119`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L119)
  - `invoke_index` — [`L130`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L130)
  - `invoke_start_prefix` — [`L118`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L118)
  - `is_tool_call_started` — [`L126`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L126)
  - `json_closed` — [`L140`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L140)
  - `json_started` — [`L139`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L139)
  - `param_count` — [`L135`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L135)
  - `parameter_complete_regex` — [`L148`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L148)
  - `parameter_end_token` — [`L121`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L121)
  - `parameter_prefix` — [`L120`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L120)
  - `prev_tool_call_arr` — [`L114`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L114)
  - `streamed_args_for_tool` — [`L125`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L125)
  - `streaming_request` — [`L142`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L142)
  - `tool_call_complete_regex` — [`L146`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L146)
  - `tool_call_end_token` — [`L117`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L117)
  - `tool_call_end_token_id` — [`L154`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L154)
  - `tool_call_start_token` — [`L116`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L116)
  - `tool_call_start_token_id` — [`L153`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L153)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`DeltaToolCall`](../../openai_api_modules.md#DeltaToolCall), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`DeltaFunctionCall`](../../openai_api_modules.md#DeltaFunctionCall), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolCall`](../../openai_api_modules.md#ToolCall), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`index`](../../openai_api_modules.md#DeltaToolCall.index), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`FunctionCall`](../../openai_api_modules.md#FunctionCall), [`arguments`](../../openai_api_modules.md#FunctionCall.arguments), [`name`](../../openai_api_modules.md#FunctionCall.name), [`id`](../../openai_api_modules.md#DeltaToolCall.id), [`name`](../../openai_api_modules.md#DeltaFunctionCall.name), [`vocab`](../abstract_tool.md#ToolParser.vocab), [`tools`](../../openai_api_modules.md#ChatCompletionRequest.tools), [`model_tokenizer`](../abstract_tool.md#ToolParser.model_tokenizer), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`name`](../../openai_api_modules.md#FunctionDefinition.name), [`parameters`](../../openai_api_modules.md#FunctionDefinition.parameters), [`logger`](minimax_m2_tool_parser.md#logger)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming)

## Module values
- `logger` — [`L59`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_m2_tool_parser.py#L59)

