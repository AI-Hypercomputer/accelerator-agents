---
title: 'Module: easydel/inference/tools/parsers/ernie45_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/ernie45_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.ernie45_tool_parser`/
symbols:
  Ernie45ToolParser.extract_tool_calls_streaming: Ernie45ToolParser#extract_tool_calls_streaming().
  Ernie45ToolParser.extract_tool_calls: Ernie45ToolParser#extract_tool_calls().
  Ernie45ToolParser.newline_token_id: Ernie45ToolParser#newline_token_id.
  Ernie45ToolParser.parser_token_ids: Ernie45ToolParser#parser_token_ids.
  logger: logger.
  Ernie45ToolParser.response_start_token_id: Ernie45ToolParser#response_start_token_id.
  Ernie45ToolParser.response_end_token_id: Ernie45ToolParser#response_end_token_id.
  Ernie45ToolParser.prev_tool_call_arr: Ernie45ToolParser#prev_tool_call_arr.
  Ernie45ToolParser.think_end_token_id: Ernie45ToolParser#think_end_token_id.
  Ernie45ToolParser.__init__: Ernie45ToolParser#__init__().
  Ernie45ToolParser.tool_call_start_token_id: Ernie45ToolParser#tool_call_start_token_id.
  Ernie45ToolParser.tool_call_end_token_id: Ernie45ToolParser#tool_call_end_token_id.
  Ernie45ToolParser._buffer: Ernie45ToolParser#_buffer.
  Ernie45ToolParser: Ernie45ToolParser#
  Ernie45ToolParser.tool_calls_start_token: Ernie45ToolParser#tool_calls_start_token.
  Ernie45ToolParser.streamed_args_for_tool: Ernie45ToolParser#streamed_args_for_tool.
  Ernie45ToolParser.tool_call_end_token: Ernie45ToolParser#tool_call_end_token.
  Ernie45ToolParser.tool_call_regex: Ernie45ToolParser#tool_call_regex.
  Ernie45ToolParser.response_start_token: Ernie45ToolParser#response_start_token.
  Ernie45ToolParser.response_end_token: Ernie45ToolParser#response_end_token.
  Ernie45ToolParser.tool_call_start_token: Ernie45ToolParser#tool_call_start_token.
  Ernie45ToolParser.think_end_token: Ernie45ToolParser#think_end_token.
  Ernie45ToolParser.newline_token: Ernie45ToolParser#newline_token.
---
# Module: [`easydel/inference/tools/parsers/ernie45_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/ernie45_tool_parser.py)

## Classes
### `Ernie45ToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/ernie45_tool_parser.py:56`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/ernie45_tool_parser.py#L56)
- doc: Tool parser for ERNIE 4.5 (Baidu) model outputs.
- signature: `class Ernie45ToolParser(ToolParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L101`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/ernie45_tool_parser.py#L101) — Initialize the ERNIE 4.5 tool parser.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L156`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/ernie45_tool_parser.py#L156) — Extract tool calls from a complete model output.
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L209`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/ernie45_tool_parser.py#L209) — Extract tool calls incrementally during streaming generation.
  - `newline_token` — [`L135`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/ernie45_tool_parser.py#L135)
  - `newline_token_id` — [`L147`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/ernie45_tool_parser.py#L147)
  - `parser_token_ids` — [`L148`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/ernie45_tool_parser.py#L148)
  - `prev_tool_call_arr` — [`L125`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/ernie45_tool_parser.py#L125)
  - `response_end_token` — [`L131`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/ernie45_tool_parser.py#L131)
  - `response_end_token_id` — [`L144`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/ernie45_tool_parser.py#L144)
  - `response_start_token` — [`L130`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/ernie45_tool_parser.py#L130)
  - `response_start_token_id` — [`L143`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/ernie45_tool_parser.py#L143)
  - `streamed_args_for_tool` — [`L127`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/ernie45_tool_parser.py#L127)
  - `think_end_token` — [`L129`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/ernie45_tool_parser.py#L129)
  - `think_end_token_id` — [`L142`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/ernie45_tool_parser.py#L142)
  - `tool_call_end_token` — [`L133`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/ernie45_tool_parser.py#L133)
  - `tool_call_end_token_id` — [`L146`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/ernie45_tool_parser.py#L146)
  - `tool_call_regex` — [`L137`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/ernie45_tool_parser.py#L137)
  - `tool_call_start_token` — [`L132`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/ernie45_tool_parser.py#L132)
  - `tool_call_start_token_id` — [`L145`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/ernie45_tool_parser.py#L145)
  - `tool_calls_start_token` — [`L134`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/ernie45_tool_parser.py#L134)
- protocol/private: `_buffer`[`L154`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/ernie45_tool_parser.py#L154)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`DeltaToolCall`](../../openai_api_modules.md#DeltaToolCall), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`DeltaFunctionCall`](../../openai_api_modules.md#DeltaFunctionCall), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolCall`](../../openai_api_modules.md#ToolCall), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`current_tool_id`](../abstract_tool.md#ToolParser.current_tool_id), [`index`](../../openai_api_modules.md#DeltaToolCall.index), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`FunctionCall`](../../openai_api_modules.md#FunctionCall), [`arguments`](../../openai_api_modules.md#FunctionCall.arguments), [`name`](../../openai_api_modules.md#FunctionCall.name), [`id`](../../openai_api_modules.md#DeltaToolCall.id), [`name`](../../openai_api_modules.md#DeltaFunctionCall.name), [`vocab`](../abstract_tool.md#ToolParser.vocab), [`model_tokenizer`](../abstract_tool.md#ToolParser.model_tokenizer), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`current_tool_name_sent`](../abstract_tool.md#ToolParser.current_tool_name_sent), [`id`](../../openai_api_modules.md#ToolCall.id), [`logger`](ernie45_tool_parser.md#logger)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming)

## Module values
- `logger` — [`L52`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/ernie45_tool_parser.py#L52)

