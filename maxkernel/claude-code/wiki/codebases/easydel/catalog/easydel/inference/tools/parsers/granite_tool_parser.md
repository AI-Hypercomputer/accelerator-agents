---
title: 'Module: easydel/inference/tools/parsers/granite_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/granite_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.granite_tool_parser`/
symbols:
  GraniteToolParser.extract_tool_calls_streaming: GraniteToolParser#extract_tool_calls_streaming().
  GraniteToolParser.extract_tool_calls: GraniteToolParser#extract_tool_calls().
  logger: logger.
  GraniteToolParser: GraniteToolParser#
  GraniteToolParser.bot_token: GraniteToolParser#bot_token.
  GraniteToolParser.bot_string: GraniteToolParser#bot_string.
  GraniteToolParser.__init__: GraniteToolParser#__init__().
---
# Module: [`easydel/inference/tools/parsers/granite_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/granite_tool_parser.py)

## Classes
### `GraniteToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/granite_tool_parser.py:55`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/granite_tool_parser.py#L55)
- doc: Tool call parser for Granite 3.0 models.
- signature: `class GraniteToolParser(ToolParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L87`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/granite_tool_parser.py#L87) — Initialize the Granite tool parser.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L99`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/granite_tool_parser.py#L99) — Extract tool calls from complete model output.
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L155`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/granite_tool_parser.py#L155) — Extract tool calls from streaming model output.
  - `bot_string` — [`L97`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/granite_tool_parser.py#L97)
  - `bot_token` — [`L96`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/granite_tool_parser.py#L96)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`DeltaToolCall`](../../openai_api_modules.md#DeltaToolCall), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`DeltaFunctionCall`](../../openai_api_modules.md#DeltaFunctionCall), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolCall`](../../openai_api_modules.md#ToolCall), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`current_tool_id`](../abstract_tool.md#ToolParser.current_tool_id), [`index`](../../openai_api_modules.md#DeltaToolCall.index), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`FunctionCall`](../../openai_api_modules.md#FunctionCall), [`arguments`](../../openai_api_modules.md#FunctionCall.arguments), [`name`](../../openai_api_modules.md#FunctionCall.name), [`id`](../../openai_api_modules.md#DeltaToolCall.id), [`name`](../../openai_api_modules.md#DeltaFunctionCall.name), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`streamed_args_for_tool`](../abstract_tool.md#ToolParser.streamed_args_for_tool), [`prev_tool_call_arr`](../abstract_tool.md#ToolParser.prev_tool_call_arr), [`current_tool_name_sent`](../abstract_tool.md#ToolParser.current_tool_name_sent), [`consume_space`](../utils.md#consume_space), [`find_common_prefix`](../utils.md#find_common_prefix), [`is_complete_json`](../utils.md#is_complete_json), [`partial_json_loads`](../utils.md#partial_json_loads), [`logger`](granite_tool_parser.md#logger)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming)

## Module values
- `logger` — [`L51`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/granite_tool_parser.py#L51)

