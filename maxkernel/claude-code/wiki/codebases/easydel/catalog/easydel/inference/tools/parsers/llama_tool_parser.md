---
title: 'Module: easydel/inference/tools/parsers/llama_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/llama_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.llama_tool_parser`/
symbols:
  Llama3JsonToolParser.extract_tool_calls_streaming: Llama3JsonToolParser#extract_tool_calls_streaming().
  Llama3JsonToolParser.extract_tool_calls: Llama3JsonToolParser#extract_tool_calls().
  Llama3JsonToolParser.current_tool_id: Llama3JsonToolParser#current_tool_id.
  logger: logger.
  Llama3JsonToolParser.streamed_args_for_tool: Llama3JsonToolParser#streamed_args_for_tool.
  Llama3JsonToolParser.bot_token: Llama3JsonToolParser#bot_token.
  Llama3JsonToolParser: Llama3JsonToolParser#
  Llama3JsonToolParser.current_tool_name_sent: Llama3JsonToolParser#current_tool_name_sent.
  Llama3JsonToolParser.__init__: Llama3JsonToolParser#__init__().
  Llama3JsonToolParser.bot_token_id: Llama3JsonToolParser#bot_token_id.
  Llama3JsonToolParser.prev_tool_call_arr: Llama3JsonToolParser#prev_tool_call_arr.
  Llama3JsonToolParser.tool_call_regex: Llama3JsonToolParser#tool_call_regex.
---
# Module: [`easydel/inference/tools/parsers/llama_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/llama_tool_parser.py)

## Classes
### `Llama3JsonToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/llama_tool_parser.py:69`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/llama_tool_parser.py#L69)
- doc: Tool call parser for Llama 3.x and 4 models with JSON format.
- signature: `class Llama3JsonToolParser(ToolParser):`
- members:
  - `__init__(self, tokenizer: PreTrainedTokenizerBase)` — [`L121`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/llama_tool_parser.py#L121) — Initialize the Llama JSON tool parser.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L141`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/llama_tool_parser.py#L141) — Extract tool calls from a complete Llama model response.
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L212`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/llama_tool_parser.py#L212) — Extract tool calls incrementally during streaming generation.
  - `bot_token` — [`L135`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/llama_tool_parser.py#L135)
  - `bot_token_id` — [`L136`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/llama_tool_parser.py#L136)
  - `current_tool_id` — [`L132`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/llama_tool_parser.py#L132)
  - `current_tool_name_sent` — [`L133`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/llama_tool_parser.py#L133)
  - `prev_tool_call_arr` — [`L131`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/llama_tool_parser.py#L131)
  - `streamed_args_for_tool` — [`L134`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/llama_tool_parser.py#L134)
  - `tool_call_regex` — [`L137`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/llama_tool_parser.py#L137)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`DeltaToolCall`](../../openai_api_modules.md#DeltaToolCall), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`DeltaFunctionCall`](../../openai_api_modules.md#DeltaFunctionCall), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolCall`](../../openai_api_modules.md#ToolCall), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`index`](../../openai_api_modules.md#DeltaToolCall.index), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`FunctionCall`](../../openai_api_modules.md#FunctionCall), [`arguments`](../../openai_api_modules.md#FunctionCall.arguments), [`name`](../../openai_api_modules.md#FunctionCall.name), [`id`](../../openai_api_modules.md#DeltaToolCall.id), [`name`](../../openai_api_modules.md#DeltaFunctionCall.name), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`find_common_prefix`](../utils.md#find_common_prefix), [`is_complete_json`](../utils.md#is_complete_json), [`partial_json_loads`](../utils.md#partial_json_loads)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming)

## Module values
- `logger` — [`L64`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/llama_tool_parser.py#L64)

