---
title: 'Module: easydel/inference/tools/parsers/kimi_k2_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/kimi_k2_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.kimi_k2_tool_parser`/
symbols:
  KimiK2ToolParser.extract_tool_calls_streaming: KimiK2ToolParser#extract_tool_calls_streaming().
  KimiK2ToolParser.extract_tool_calls: KimiK2ToolParser#extract_tool_calls().
  logger: logger.
  KimiK2ToolParser.current_tool_id: KimiK2ToolParser#current_tool_id.
  KimiK2ToolParser.tool_call_end_token_id: KimiK2ToolParser#tool_call_end_token_id.
  KimiK2ToolParser.prev_tool_call_arr: KimiK2ToolParser#prev_tool_call_arr.
  KimiK2ToolParser.tool_call_end_token: KimiK2ToolParser#tool_call_end_token.
  KimiK2ToolParser.tool_calls_start_token_id: KimiK2ToolParser#tool_calls_start_token_id.
  KimiK2ToolParser.tool_call_start_token_id: KimiK2ToolParser#tool_call_start_token_id.
  KimiK2ToolParser.tool_calls_end_token_id: KimiK2ToolParser#tool_calls_end_token_id.
  KimiK2ToolParser.tool_call_start_token: KimiK2ToolParser#tool_call_start_token.
  KimiK2ToolParser: KimiK2ToolParser#
  KimiK2ToolParser.streamed_args_for_tool: KimiK2ToolParser#streamed_args_for_tool.
  KimiK2ToolParser.tool_calls_start_token: KimiK2ToolParser#tool_calls_start_token.
  KimiK2ToolParser.stream_tool_call_name_regex: KimiK2ToolParser#stream_tool_call_name_regex.
  KimiK2ToolParser.current_tool_name_sent: KimiK2ToolParser#current_tool_name_sent.
  KimiK2ToolParser.__init__: KimiK2ToolParser#__init__().
  KimiK2ToolParser.tool_calls_end_token: KimiK2ToolParser#tool_calls_end_token.
  KimiK2ToolParser.tool_call_regex: KimiK2ToolParser#tool_call_regex.
  KimiK2ToolParser.stream_tool_call_portion_regex: KimiK2ToolParser#stream_tool_call_portion_regex.
---
# Module: [`easydel/inference/tools/parsers/kimi_k2_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/kimi_k2_tool_parser.py)

## Classes
### `KimiK2ToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/kimi_k2_tool_parser.py:53`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/kimi_k2_tool_parser.py#L53)
- doc: Tool parser for Kimi K2 model outputs.
- signature: `class KimiK2ToolParser(ToolParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L96`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/kimi_k2_tool_parser.py#L96) — Initialize the Kimi K2 tool parser.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L146`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/kimi_k2_tool_parser.py#L146) — Extract tool calls from a complete model output.
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L204`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/kimi_k2_tool_parser.py#L204) — Extract tool calls incrementally during streaming generation.
  - `current_tool_id` — [`L116`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/kimi_k2_tool_parser.py#L116)
  - `current_tool_name_sent` — [`L114`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/kimi_k2_tool_parser.py#L114)
  - `prev_tool_call_arr` — [`L115`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/kimi_k2_tool_parser.py#L115)
  - `stream_tool_call_name_regex` — [`L133`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/kimi_k2_tool_parser.py#L133)
  - `stream_tool_call_portion_regex` — [`L129`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/kimi_k2_tool_parser.py#L129)
  - `streamed_args_for_tool` — [`L117`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/kimi_k2_tool_parser.py#L117)
  - `tool_call_end_token` — [`L123`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/kimi_k2_tool_parser.py#L123)
  - `tool_call_end_token_id` — [`L141`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/kimi_k2_tool_parser.py#L141)
  - `tool_call_regex` — [`L125`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/kimi_k2_tool_parser.py#L125)
  - `tool_call_start_token` — [`L122`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/kimi_k2_tool_parser.py#L122)
  - `tool_call_start_token_id` — [`L140`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/kimi_k2_tool_parser.py#L140)
  - `tool_calls_end_token` — [`L120`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/kimi_k2_tool_parser.py#L120)
  - `tool_calls_end_token_id` — [`L138`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/kimi_k2_tool_parser.py#L138)
  - `tool_calls_start_token` — [`L119`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/kimi_k2_tool_parser.py#L119)
  - `tool_calls_start_token_id` — [`L137`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/kimi_k2_tool_parser.py#L137)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`DeltaToolCall`](../../openai_api_modules.md#DeltaToolCall), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`DeltaFunctionCall`](../../openai_api_modules.md#DeltaFunctionCall), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolCall`](../../openai_api_modules.md#ToolCall), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`index`](../../openai_api_modules.md#DeltaToolCall.index), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`FunctionCall`](../../openai_api_modules.md#FunctionCall), [`arguments`](../../openai_api_modules.md#FunctionCall.arguments), [`name`](../../openai_api_modules.md#FunctionCall.name), [`id`](../../openai_api_modules.md#DeltaToolCall.id), [`name`](../../openai_api_modules.md#DeltaFunctionCall.name), [`vocab`](../abstract_tool.md#ToolParser.vocab), [`model_tokenizer`](../abstract_tool.md#ToolParser.model_tokenizer), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`logger`](kimi_k2_tool_parser.md#logger), [`id`](../../openai_api_modules.md#ToolCall.id)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming)

## Module values
- `logger` — [`L49`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/kimi_k2_tool_parser.py#L49)

