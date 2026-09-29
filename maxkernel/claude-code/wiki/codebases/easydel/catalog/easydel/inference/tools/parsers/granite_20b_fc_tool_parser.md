---
title: 'Module: easydel/inference/tools/parsers/granite_20b_fc_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/granite_20b_fc_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.granite_20b_fc_tool_parser`/
symbols:
  Granite20bFCToolParser.extract_tool_calls_streaming: Granite20bFCToolParser#extract_tool_calls_streaming().
  Granite20bFCToolParser.extract_tool_calls: Granite20bFCToolParser#extract_tool_calls().
  logger: logger.
  Granite20bFCToolParser.bot_token: Granite20bFCToolParser#bot_token.
  Granite20bFCToolParser: Granite20bFCToolParser#
  Granite20bFCToolParser.tool_start_token: Granite20bFCToolParser#tool_start_token.
  Granite20bFCToolParser.__init__: Granite20bFCToolParser#__init__().
  Granite20bFCToolParser.tool_call_regex: Granite20bFCToolParser#tool_call_regex.
---
# Module: [`easydel/inference/tools/parsers/granite_20b_fc_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/granite_20b_fc_tool_parser.py)

## Classes
### `Granite20bFCToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/granite_20b_fc_tool_parser.py:55`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/granite_20b_fc_tool_parser.py#L55)
- doc: Tool call parser for Granite 20B function calling model.
- signature: `class Granite20bFCToolParser(ToolParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L93`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/granite_20b_fc_tool_parser.py#L93) — Initialize the Granite 20B FC tool parser.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L108`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/granite_20b_fc_tool_parser.py#L108) — Extract tool calls from complete model output.
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L173`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/granite_20b_fc_tool_parser.py#L173) — Extract tool calls from streaming model output.
  - `bot_token` — [`L104`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/granite_20b_fc_tool_parser.py#L104)
  - `tool_call_regex` — [`L106`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/granite_20b_fc_tool_parser.py#L106)
  - `tool_start_token` — [`L105`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/granite_20b_fc_tool_parser.py#L105)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`DeltaToolCall`](../../openai_api_modules.md#DeltaToolCall), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`DeltaFunctionCall`](../../openai_api_modules.md#DeltaFunctionCall), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolCall`](../../openai_api_modules.md#ToolCall), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`current_tool_id`](../abstract_tool.md#ToolParser.current_tool_id), [`index`](../../openai_api_modules.md#DeltaToolCall.index), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`FunctionCall`](../../openai_api_modules.md#FunctionCall), [`arguments`](../../openai_api_modules.md#FunctionCall.arguments), [`name`](../../openai_api_modules.md#FunctionCall.name), [`id`](../../openai_api_modules.md#DeltaToolCall.id), [`name`](../../openai_api_modules.md#DeltaFunctionCall.name), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`streamed_args_for_tool`](../abstract_tool.md#ToolParser.streamed_args_for_tool), [`prev_tool_call_arr`](../abstract_tool.md#ToolParser.prev_tool_call_arr), [`current_tool_name_sent`](../abstract_tool.md#ToolParser.current_tool_name_sent), [`logger`](granite_20b_fc_tool_parser.md#logger), [`consume_space`](../utils.md#consume_space), [`find_common_prefix`](../utils.md#find_common_prefix), [`is_complete_json`](../utils.md#is_complete_json), [`partial_json_loads`](../utils.md#partial_json_loads)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming)

## Module values
- `logger` — [`L51`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/granite_20b_fc_tool_parser.py#L51)

