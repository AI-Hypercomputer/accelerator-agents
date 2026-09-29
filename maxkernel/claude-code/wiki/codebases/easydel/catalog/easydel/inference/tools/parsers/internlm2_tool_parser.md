---
title: 'Module: easydel/inference/tools/parsers/internlm2_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/internlm2_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.internlm2_tool_parser`/
symbols:
  Internlm2ToolParser.extract_tool_calls_streaming: Internlm2ToolParser#extract_tool_calls_streaming().
  Internlm2ToolParser.extract_tool_calls: Internlm2ToolParser#extract_tool_calls().
  logger: logger.
  Internlm2ToolParser.adjust_request: Internlm2ToolParser#adjust_request().
  Internlm2ToolParser: Internlm2ToolParser#
  Internlm2ToolParser.position: Internlm2ToolParser#position.
  Internlm2ToolParser.get_arguments: Internlm2ToolParser#get_arguments().
  Internlm2ToolParser.__init__: Internlm2ToolParser#__init__().
  Internlm2ToolParser.tool_call_start_token: Internlm2ToolParser#tool_call_start_token.
  Internlm2ToolParser.tool_call_end_token: Internlm2ToolParser#tool_call_end_token.
---
# Module: [`easydel/inference/tools/parsers/internlm2_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/internlm2_tool_parser.py)

## Classes
### `Internlm2ToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/internlm2_tool_parser.py:60`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/internlm2_tool_parser.py#L60)
- doc: Tool parser for InternLM2 models.
- signature: `class Internlm2ToolParser(ToolParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L79`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/internlm2_tool_parser.py#L79) — Initialize the Internlm2ToolParser.
  - `adjust_request(self, request: ChatCompletionRequest)` — [`L91`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/internlm2_tool_parser.py#L91) — Adjust the request settings for InternLM2 tool parsing.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L255`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/internlm2_tool_parser.py#L255) — Extract tool calls from a complete model response.
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L128`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/internlm2_tool_parser.py#L128) — Extract tool calls incrementally during streaming.
  - `get_arguments(self, obj: dict)` — [`L109`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/internlm2_tool_parser.py#L109) — Extract arguments from a tool call object.
  - `position` — [`L87`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/internlm2_tool_parser.py#L87)
  - `tool_call_end_token` — [`L89`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/internlm2_tool_parser.py#L89)
  - `tool_call_start_token` — [`L88`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/internlm2_tool_parser.py#L88)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`DeltaToolCall`](../../openai_api_modules.md#DeltaToolCall), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`DeltaFunctionCall`](../../openai_api_modules.md#DeltaFunctionCall), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolCall`](../../openai_api_modules.md#ToolCall), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`current_tool_id`](../abstract_tool.md#ToolParser.current_tool_id), [`index`](../../openai_api_modules.md#DeltaToolCall.index), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`FunctionCall`](../../openai_api_modules.md#FunctionCall), [`arguments`](../../openai_api_modules.md#FunctionCall.arguments), [`name`](../../openai_api_modules.md#FunctionCall.name), [`id`](../../openai_api_modules.md#DeltaToolCall.id), [`name`](../../openai_api_modules.md#DeltaFunctionCall.name), [`tools`](../../openai_api_modules.md#ChatCompletionRequest.tools), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`streamed_args_for_tool`](../abstract_tool.md#ToolParser.streamed_args_for_tool), [`prev_tool_call_arr`](../abstract_tool.md#ToolParser.prev_tool_call_arr), [`current_tool_name_sent`](../abstract_tool.md#ToolParser.current_tool_name_sent), [`tool_choice`](../../openai_api_modules.md#ChatCompletionRequest.tool_choice), [`extract_intermediate_diff`](../utils.md#extract_intermediate_diff), [`logger`](internlm2_tool_parser.md#logger), [`name`](../../openai_api_modules.md#FunctionDefinition.name)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming), [`adjust_request`](../abstract_tool.md#ToolParser.adjust_request)

## Module values
- `logger` — [`L56`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/internlm2_tool_parser.py#L56)

