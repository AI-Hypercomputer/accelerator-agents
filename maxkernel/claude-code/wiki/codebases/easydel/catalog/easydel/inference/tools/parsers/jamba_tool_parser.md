---
title: 'Module: easydel/inference/tools/parsers/jamba_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/jamba_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.jamba_tool_parser`/
symbols:
  JambaToolParser.extract_tool_calls_streaming: JambaToolParser#extract_tool_calls_streaming().
  JambaToolParser.extract_tool_calls: JambaToolParser#extract_tool_calls().
  logger: logger.
  JambaToolParser.current_tool_id: JambaToolParser#current_tool_id.
  JambaToolParser.tool_calls_end_token_id: JambaToolParser#tool_calls_end_token_id.
  JambaToolParser.tool_calls_regex: JambaToolParser#tool_calls_regex.
  JambaToolParser.adjust_request: JambaToolParser#adjust_request().
  JambaToolParser.tool_calls_start_token_id: JambaToolParser#tool_calls_start_token_id.
  JambaToolParser.tool_calls_start_token: JambaToolParser#tool_calls_start_token.
  JambaToolParser.__init__: JambaToolParser#__init__().
  JambaToolParser.streamed_args_for_tool: JambaToolParser#streamed_args_for_tool.
  JambaToolParser: JambaToolParser#
  JambaToolParser.current_tool_name_sent: JambaToolParser#current_tool_name_sent.
  JambaToolParser.tool_calls_end_token: JambaToolParser#tool_calls_end_token.
  JambaToolParser.prev_tool_call_arr: JambaToolParser#prev_tool_call_arr.
---
# Module: [`easydel/inference/tools/parsers/jamba_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/jamba_tool_parser.py)

## Classes
### `JambaToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/jamba_tool_parser.py:61`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/jamba_tool_parser.py#L61)
- doc: Tool parser for Jamba models.
- signature: `class JambaToolParser(ToolParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L89`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/jamba_tool_parser.py#L89) — Initialize the JambaToolParser.
  - `adjust_request(self, request: ChatCompletionRequest)` — [`L132`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/jamba_tool_parser.py#L132) — Adjust the request settings for Jamba tool parsing.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L150`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/jamba_tool_parser.py#L150) — Extract tool calls from a complete model response.
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L198`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/jamba_tool_parser.py#L198) — Extract tool calls incrementally during streaming.
  - `current_tool_id` — [`L117`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/jamba_tool_parser.py#L117)
  - `current_tool_name_sent` — [`L115`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/jamba_tool_parser.py#L115)
  - `prev_tool_call_arr` — [`L116`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/jamba_tool_parser.py#L116)
  - `streamed_args_for_tool` — [`L118`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/jamba_tool_parser.py#L118)
  - `tool_calls_end_token` — [`L121`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/jamba_tool_parser.py#L121)
  - `tool_calls_end_token_id` — [`L128`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/jamba_tool_parser.py#L128)
  - `tool_calls_regex` — [`L123`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/jamba_tool_parser.py#L123)
  - `tool_calls_start_token` — [`L120`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/jamba_tool_parser.py#L120)
  - `tool_calls_start_token_id` — [`L127`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/jamba_tool_parser.py#L127)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`DeltaToolCall`](../../openai_api_modules.md#DeltaToolCall), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`DeltaFunctionCall`](../../openai_api_modules.md#DeltaFunctionCall), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolCall`](../../openai_api_modules.md#ToolCall), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`index`](../../openai_api_modules.md#DeltaToolCall.index), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`FunctionCall`](../../openai_api_modules.md#FunctionCall), [`arguments`](../../openai_api_modules.md#FunctionCall.arguments), [`name`](../../openai_api_modules.md#FunctionCall.name), [`id`](../../openai_api_modules.md#DeltaToolCall.id), [`name`](../../openai_api_modules.md#DeltaFunctionCall.name), [`vocab`](../abstract_tool.md#ToolParser.vocab), [`tools`](../../openai_api_modules.md#ChatCompletionRequest.tools), [`model_tokenizer`](../abstract_tool.md#ToolParser.model_tokenizer), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`logger`](jamba_tool_parser.md#logger), [`tool_choice`](../../openai_api_modules.md#ChatCompletionRequest.tool_choice), [`extract_intermediate_diff`](../utils.md#extract_intermediate_diff)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming), [`adjust_request`](../abstract_tool.md#ToolParser.adjust_request)

## Module values
- `logger` — [`L57`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/jamba_tool_parser.py#L57)

