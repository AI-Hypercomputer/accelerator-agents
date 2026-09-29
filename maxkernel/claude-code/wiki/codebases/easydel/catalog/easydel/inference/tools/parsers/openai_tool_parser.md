---
title: 'Module: easydel/inference/tools/parsers/openai_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/openai_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.openai_tool_parser`/
symbols:
  OpenAIToolParser.extract_tool_calls_streaming: OpenAIToolParser#extract_tool_calls_streaming().
  OpenAIToolParser.extract_tool_calls: OpenAIToolParser#extract_tool_calls().
  OpenAIToolParser.current_tool_id: OpenAIToolParser#current_tool_id.
  OpenAIToolParser._extract_json_candidate: OpenAIToolParser#_extract_json_candidate().
  logger: logger.
  OpenAIToolParser: OpenAIToolParser#
  OpenAIToolParser.prev_tool_call_arr: OpenAIToolParser#prev_tool_call_arr.
  OpenAIToolParser.streamed_args_for_tool: OpenAIToolParser#streamed_args_for_tool.
  OpenAIToolParser.current_tool_name_sent: OpenAIToolParser#current_tool_name_sent.
  OpenAIToolParser.__init__: OpenAIToolParser#__init__().
  OpenAIToolParser._normalize_tool_call_objects: OpenAIToolParser#_normalize_tool_call_objects().
  OpenAIToolParser._json_block_re: OpenAIToolParser#_json_block_re.
  OpenAIToolParser.tool_call_start_token: OpenAIToolParser#tool_call_start_token.
  OpenAIToolParser.tool_call_end_token: OpenAIToolParser#tool_call_end_token.
---
# Module: [`easydel/inference/tools/parsers/openai_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/openai_tool_parser.py)

## Classes
### `OpenAIToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/openai_tool_parser.py:65`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/openai_tool_parser.py#L65)
- doc: Best-effort OpenAI-style tool call parser for local model outputs.
- signature: `class OpenAIToolParser(ToolParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L101`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/openai_tool_parser.py#L101) — Initialize the OpenAI tool parser.
  - `_extract_json_candidate(text: str)` — [`L118`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/openai_tool_parser.py#L118) — Extract JSON content from model output text.
  - `_normalize_tool_call_objects(obj: object)` — [`L156`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/openai_tool_parser.py#L156) — Normalize various JSON structures into a standard tool call format.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L204`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/openai_tool_parser.py#L204) — Extract tool calls from a complete model response.
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L255`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/openai_tool_parser.py#L255) — Extract tool calls incrementally during streaming generation.
  - `current_tool_id` — [`L111`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/openai_tool_parser.py#L111)
  - `current_tool_name_sent` — [`L112`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/openai_tool_parser.py#L112)
  - `prev_tool_call_arr` — [`L110`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/openai_tool_parser.py#L110)
  - `streamed_args_for_tool` — [`L113`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/openai_tool_parser.py#L113)
  - `tool_call_end_token` — [`L115`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/openai_tool_parser.py#L115)
  - `tool_call_start_token` — [`L114`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/openai_tool_parser.py#L114)
- protocol/private: `_json_block_re`[`L99`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/openai_tool_parser.py#L99)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`DeltaToolCall`](../../openai_api_modules.md#DeltaToolCall), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`DeltaFunctionCall`](../../openai_api_modules.md#DeltaFunctionCall), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolCall`](../../openai_api_modules.md#ToolCall), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`index`](../../openai_api_modules.md#DeltaToolCall.index), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`FunctionCall`](../../openai_api_modules.md#FunctionCall), [`arguments`](../../openai_api_modules.md#FunctionCall.arguments), [`name`](../../openai_api_modules.md#FunctionCall.name), [`id`](../../openai_api_modules.md#DeltaToolCall.id), [`name`](../../openai_api_modules.md#DeltaFunctionCall.name), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`find_common_prefix`](../utils.md#find_common_prefix), [`is_complete_json`](../utils.md#is_complete_json), [`partial_json_loads`](../utils.md#partial_json_loads), [`logger`](openai_tool_parser.md#logger)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming)

## Module values
- `logger` — [`L61`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/openai_tool_parser.py#L61)

