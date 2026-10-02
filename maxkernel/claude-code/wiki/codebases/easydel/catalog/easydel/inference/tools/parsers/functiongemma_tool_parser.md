---
title: 'Module: easydel/inference/tools/parsers/functiongemma_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/functiongemma_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.functiongemma_tool_parser`/
symbols:
  FunctionGemmaToolParser.extract_tool_calls_streaming: FunctionGemmaToolParser#extract_tool_calls_streaming().
  FunctionGemmaToolParser.extract_tool_calls: FunctionGemmaToolParser#extract_tool_calls().
  FunctionGemmaToolParser.current_tool_id: FunctionGemmaToolParser#current_tool_id.
  FunctionGemmaToolParser.tool_call_end_token_ids: FunctionGemmaToolParser#tool_call_end_token_ids.
  FunctionGemmaToolParser._parse_arguments: FunctionGemmaToolParser#_parse_arguments().
  FunctionGemmaToolParser.tool_call_start_token: FunctionGemmaToolParser#tool_call_start_token.
  FunctionGemmaToolParser.streamed_args_for_tool: FunctionGemmaToolParser#streamed_args_for_tool.
  FunctionGemmaToolParser.adjust_request: FunctionGemmaToolParser#adjust_request().
  FunctionGemmaToolParser._buffer_delta_text: FunctionGemmaToolParser#_buffer_delta_text().
  logger: logger.
  FunctionGemmaToolParser.tool_call_start_token_ids: FunctionGemmaToolParser#tool_call_start_token_ids.
  FunctionGemmaToolParser.tool_call_end_token: FunctionGemmaToolParser#tool_call_end_token.
  FunctionGemmaToolParser.prev_tool_call_arr: FunctionGemmaToolParser#prev_tool_call_arr.
  FunctionGemmaToolParser: FunctionGemmaToolParser#
  FunctionGemmaToolParser.tool_call_regex: FunctionGemmaToolParser#tool_call_regex.
  FunctionGemmaToolParser._scan_escape_token: FunctionGemmaToolParser#_scan_escape_token().
  FunctionGemmaToolParser.current_tool_name_sent: FunctionGemmaToolParser#current_tool_name_sent.
  FunctionGemmaToolParser.buffered_delta_text: FunctionGemmaToolParser#buffered_delta_text.
  FunctionGemmaToolParser._split_top_level: FunctionGemmaToolParser#_split_top_level().
  FunctionGemmaToolParser._split_key_value: FunctionGemmaToolParser#_split_key_value().
  FunctionGemmaToolParser.__init__: FunctionGemmaToolParser#__init__().
  FunctionGemmaToolParser.escape_tokens: FunctionGemmaToolParser#escape_tokens.
  FunctionGemmaToolParser._normalize_argument_value: FunctionGemmaToolParser#_normalize_argument_value().
---
# Module: [`easydel/inference/tools/parsers/functiongemma_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/functiongemma_tool_parser.py)

## Classes
### `FunctionGemmaToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/functiongemma_tool_parser.py:52`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/functiongemma_tool_parser.py#L52)
- doc: Tool parser for Google's FunctionGemma model.
- signature: `class FunctionGemmaToolParser(ToolParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L86`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/functiongemma_tool_parser.py#L86) — Initialize the FunctionGemma tool parser.
  - `_buffer_delta_text(self, delta_text: str)` — [`L329`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/functiongemma_tool_parser.py#L329) — Buffer delta text to handle partial special tokens.
  - `_normalize_argument_value(self, value: str)` — [`L198`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/functiongemma_tool_parser.py#L198) — Convert Gemma/FunctionGemma escaped argument syntax into Python values.
  - `_parse_arguments(self, args_str: str)` — [`L222`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/functiongemma_tool_parser.py#L222) — Parse the argument string into a dictionary.
  - `_scan_escape_token(self, text: str, index: int)` — [`L124`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/functiongemma_tool_parser.py#L124) — Advance across a configured escape token if present at *index*.
  - `_split_key_value(self, text: str)` — [`L167`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/functiongemma_tool_parser.py#L167) — Split a single top-level ``key:value`` segment.
  - `_split_top_level(self, text: str, separator: str)` — [`L132`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/functiongemma_tool_parser.py#L132) — Split *text* on *separator* while respecting nested JSON-like syntax.
  - `adjust_request(self, request: ChatCompletionRequest)` — [`L253`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/functiongemma_tool_parser.py#L253) — Adjust the request to ensure special tokens are not skipped.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L271`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/functiongemma_tool_parser.py#L271) — Extract tool calls from a complete model output.
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L360`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/functiongemma_tool_parser.py#L360) — Extract tool calls from streaming model output.
  - `buffered_delta_text` — [`L122`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/functiongemma_tool_parser.py#L122)
  - `current_tool_id` — [`L99`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/functiongemma_tool_parser.py#L99)
  - `current_tool_name_sent` — [`L97`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/functiongemma_tool_parser.py#L97)
  - `escape_tokens` — [`L84`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/functiongemma_tool_parser.py#L84)
  - `prev_tool_call_arr` — [`L98`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/functiongemma_tool_parser.py#L98)
  - `streamed_args_for_tool` — [`L100`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/functiongemma_tool_parser.py#L100)
  - `tool_call_end_token` — [`L83`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/functiongemma_tool_parser.py#L83)
  - `tool_call_end_token_ids` — [`L114`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/functiongemma_tool_parser.py#L114)
  - `tool_call_regex` — [`L104`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/functiongemma_tool_parser.py#L104)
  - `tool_call_start_token` — [`L82`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/functiongemma_tool_parser.py#L82)
  - `tool_call_start_token_ids` — [`L110`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/functiongemma_tool_parser.py#L110)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`DeltaToolCall`](../../openai_api_modules.md#DeltaToolCall), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`DeltaFunctionCall`](../../openai_api_modules.md#DeltaFunctionCall), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolCall`](../../openai_api_modules.md#ToolCall), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`index`](../../openai_api_modules.md#DeltaToolCall.index), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`FunctionCall`](../../openai_api_modules.md#FunctionCall), [`arguments`](../../openai_api_modules.md#FunctionCall.arguments), [`name`](../../openai_api_modules.md#FunctionCall.name), [`id`](../../openai_api_modules.md#DeltaToolCall.id), [`name`](../../openai_api_modules.md#DeltaFunctionCall.name), [`tools`](../../openai_api_modules.md#ChatCompletionRequest.tools), [`model_tokenizer`](../abstract_tool.md#ToolParser.model_tokenizer), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`adjust_request`](../abstract_tool.md#ToolParser.adjust_request), [`tool_choice`](../../openai_api_modules.md#ChatCompletionRequest.tool_choice), [`logger`](functiongemma_tool_parser.md#logger)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming), [`adjust_request`](../abstract_tool.md#ToolParser.adjust_request)

## Module values
- `logger` — [`L48`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/functiongemma_tool_parser.py#L48)

