---
title: 'Module: easydel/inference/tools/parsers/seed_oss_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/seed_oss_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.seed_oss_tool_parser`/
symbols:
  SeedOssToolParser.extract_tool_calls_streaming: SeedOssToolParser#extract_tool_calls_streaming().
  SeedOssToolParser.extract_tool_calls: SeedOssToolParser#extract_tool_calls().
  SeedOssToolParser._parse_xml_function_call: SeedOssToolParser#_parse_xml_function_call().
  SeedOssToolParser.think_end_token_id: SeedOssToolParser#think_end_token_id.
  SeedOssToolParser.current_tool_index: SeedOssToolParser#current_tool_index.
  logger: logger.
  SeedOssToolParser.tool_call_start_token: SeedOssToolParser#tool_call_start_token.
  SeedOssToolParser.current_param_value: SeedOssToolParser#current_param_value.
  SeedOssToolParser.tool_call_end_token: SeedOssToolParser#tool_call_end_token.
  SeedOssToolParser.get_arguments_config: SeedOssToolParser#get_arguments_config().
  SeedOssToolParser.think_end_token: SeedOssToolParser#think_end_token.
  SeedOssToolParser.prev_tool_call_arr: SeedOssToolParser#prev_tool_call_arr.
  SeedOssToolParser.tool_call_prefix: SeedOssToolParser#tool_call_prefix.
  SeedOssToolParser.tool_call_start_token_id: SeedOssToolParser#tool_call_start_token_id.
  SeedOssToolParser.tool_call_end_token_id: SeedOssToolParser#tool_call_end_token_id.
  SeedOssToolParser.param_count: SeedOssToolParser#param_count.
  SeedOssToolParser._get_function_calls: SeedOssToolParser#_get_function_calls().
  SeedOssToolParser.header_sent: SeedOssToolParser#header_sent.
  SeedOssToolParser.__init__: SeedOssToolParser#__init__().
  SeedOssToolParser.is_tool_call_started: SeedOssToolParser#is_tool_call_started.
  SeedOssToolParser.json_started: SeedOssToolParser#json_started.
  SeedOssToolParser.json_closed: SeedOssToolParser#json_closed.
  SeedOssToolParser: SeedOssToolParser#
  SeedOssToolParser.parameter_prefix: SeedOssToolParser#parameter_prefix.
  SeedOssToolParser.parameter_end_token: SeedOssToolParser#parameter_end_token.
  SeedOssToolParser.current_function_name: SeedOssToolParser#current_function_name.
  SeedOssToolParser.in_function: SeedOssToolParser#in_function.
  SeedOssToolParser.convert_param_value: SeedOssToolParser#convert_param_value().
  SeedOssToolParser.is_thinking_end: SeedOssToolParser#is_thinking_end.
  SeedOssToolParser._reset_streaming_state: SeedOssToolParser#_reset_streaming_state().
  SeedOssToolParser.current_param_name: SeedOssToolParser#current_param_name.
  SeedOssToolParser.in_param: SeedOssToolParser#in_param.
  SeedOssToolParser.failed_count: SeedOssToolParser#failed_count.
  SeedOssToolParser.function_end_token: SeedOssToolParser#function_end_token.
  SeedOssToolParser.TOOL_CALL_START: SeedOssToolParser#TOOL_CALL_START.
  SeedOssToolParser.TOOL_CALL_END: SeedOssToolParser#TOOL_CALL_END.
  SeedOssToolParser.think_start_token: SeedOssToolParser#think_start_token.
  SeedOssToolParser.tool_call_complete_regex: SeedOssToolParser#tool_call_complete_regex.
  SeedOssToolParser.tool_call_regex: SeedOssToolParser#tool_call_regex.
  SeedOssToolParser.tool_call_function_regex: SeedOssToolParser#tool_call_function_regex.
  SeedOssToolParser.tool_call_parameter_regex: SeedOssToolParser#tool_call_parameter_regex.
  SeedOssToolParser._generate_tool_call_id: SeedOssToolParser#_generate_tool_call_id().
  SeedOssToolParser.accumulated_text: SeedOssToolParser#accumulated_text.
---
# Module: [`easydel/inference/tools/parsers/seed_oss_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py)

## Classes
### `SeedOssToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/seed_oss_tool_parser.py:61`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L61)
- doc: Tool parser for Seed OSS model outputs.
- signature: `class SeedOssToolParser(ToolParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L111`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L111) — Initialize the Seed OSS tool parser.
  - `_generate_tool_call_id(self)` — [`L161`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L161) — Generate a unique tool call ID.
  - `_get_function_calls(self, model_output: str)` — [`L370`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L370) — Extract raw function call strings from model output.
  - `_parse_xml_function_call(self, function_call_str: str, tools: list[ToolDefinition] | None)` — [`L192`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L192) — Parse an XML-formatted function call string into a ToolCall object.
  - `_reset_streaming_state(self)` — [`L172`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L172) — Reset all streaming state variables.
  - `convert_param_value(param_value: str, param_name: str, param_config: dict, func_name: str)` — [`L250`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L250) — Convert a parameter value to its declared type.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L396`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L396) — Extract tool calls from a complete model output.
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L463`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L463) — Extract tool calls incrementally during streaming generation.
  - `get_arguments_config(func_name: str)` — [`L223`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L223) — Get parameter configuration for a function from tool definitions.
  - `TOOL_CALL_END` — [`L109`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L109)
  - `TOOL_CALL_START` — [`L108`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L108)
  - `accumulated_text` — [`L188`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L188)
  - `current_function_name` — [`L182`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L182)
  - `current_param_name` — [`L183`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L183)
  - `current_param_value` — [`L184`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L184)
  - `current_tool_index` — [`L178`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L178)
  - `failed_count` — [`L142`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L142)
  - `function_end_token` — [`L135`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L135)
  - `header_sent` — [`L180`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L180)
  - `in_function` — [`L187`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L187)
  - `in_param` — [`L186`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L186)
  - `is_thinking_end` — [`L141`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L141)
  - `is_tool_call_started` — [`L140`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L140)
  - `json_closed` — [`L190`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L190)
  - `json_started` — [`L189`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L189)
  - `param_count` — [`L185`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L185)
  - `parameter_end_token` — [`L137`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L137)
  - `parameter_prefix` — [`L136`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L136)
  - `prev_tool_call_arr` — [`L130`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L130)
  - `think_end_token` — [`L139`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L139)
  - `think_end_token_id` — [`L147`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L147)
  - `think_start_token` — [`L138`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L138)
  - `tool_call_complete_regex` — [`L155`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L155)
  - `tool_call_end_token` — [`L133`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L133)
  - `tool_call_end_token_id` — [`L146`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L146)
  - `tool_call_function_regex` — [`L158`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L158)
  - `tool_call_parameter_regex` — [`L159`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L159)
  - `tool_call_prefix` — [`L134`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L134)
  - `tool_call_regex` — [`L156`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L156)
  - `tool_call_start_token` — [`L132`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L132)
  - `tool_call_start_token_id` — [`L145`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L145)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`DeltaToolCall`](../../openai_api_modules.md#DeltaToolCall), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`DeltaFunctionCall`](../../openai_api_modules.md#DeltaFunctionCall), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolCall`](../../openai_api_modules.md#ToolCall), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`current_tool_id`](../abstract_tool.md#ToolParser.current_tool_id), [`index`](../../openai_api_modules.md#DeltaToolCall.index), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`FunctionCall`](../../openai_api_modules.md#FunctionCall), [`arguments`](../../openai_api_modules.md#FunctionCall.arguments), [`name`](../../openai_api_modules.md#FunctionCall.name), [`id`](../../openai_api_modules.md#DeltaToolCall.id), [`name`](../../openai_api_modules.md#DeltaFunctionCall.name), [`vocab`](../abstract_tool.md#ToolParser.vocab), [`tools`](../../openai_api_modules.md#ChatCompletionRequest.tools), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`ToolDefinition`](../../openai_api_modules.md#ToolDefinition), [`logger`](seed_oss_tool_parser.md#logger), [`name`](../../openai_api_modules.md#FunctionDefinition.name), [`parameters`](../../openai_api_modules.md#FunctionDefinition.parameters)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming)

## Module values
- `logger` — [`L57`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/seed_oss_tool_parser.py#L57)

