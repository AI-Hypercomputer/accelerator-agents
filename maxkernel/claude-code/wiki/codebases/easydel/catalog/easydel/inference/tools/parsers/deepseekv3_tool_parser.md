---
title: 'Module: easydel/inference/tools/parsers/deepseekv3_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/deepseekv3_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.deepseekv3_tool_parser`/
symbols:
  DeepSeekV3ToolParser.extract_tool_calls_streaming: DeepSeekV3ToolParser#extract_tool_calls_streaming().
  DeepSeekV3ToolParser.extract_tool_calls: DeepSeekV3ToolParser#extract_tool_calls().
  logger: logger.
  DeepSeekV3ToolParser.current_tool_id: DeepSeekV3ToolParser#current_tool_id.
  DeepSeekV3ToolParser.tool_call_end_token_id: DeepSeekV3ToolParser#tool_call_end_token_id.
  DeepSeekV3ToolParser.prev_tool_call_arr: DeepSeekV3ToolParser#prev_tool_call_arr.
  DeepSeekV3ToolParser.tool_call_end_token: DeepSeekV3ToolParser#tool_call_end_token.
  DeepSeekV3ToolParser.tool_calls_start_token_id: DeepSeekV3ToolParser#tool_calls_start_token_id.
  DeepSeekV3ToolParser.tool_call_start_token_id: DeepSeekV3ToolParser#tool_call_start_token_id.
  DeepSeekV3ToolParser.tool_calls_end_token_id: DeepSeekV3ToolParser#tool_calls_end_token_id.
  DeepSeekV3ToolParser.tool_call_start_token: DeepSeekV3ToolParser#tool_call_start_token.
  DeepSeekV3ToolParser: DeepSeekV3ToolParser#
  DeepSeekV3ToolParser.streamed_args_for_tool: DeepSeekV3ToolParser#streamed_args_for_tool.
  DeepSeekV3ToolParser.tool_calls_start_token: DeepSeekV3ToolParser#tool_calls_start_token.
  DeepSeekV3ToolParser.stream_tool_call_name_regex: DeepSeekV3ToolParser#stream_tool_call_name_regex.
  DeepSeekV3ToolParser.current_tool_name_sent: DeepSeekV3ToolParser#current_tool_name_sent.
  DeepSeekV3ToolParser.__init__: DeepSeekV3ToolParser#__init__().
  DeepSeekV3ToolParser.tool_calls_end_token: DeepSeekV3ToolParser#tool_calls_end_token.
  DeepSeekV3ToolParser.tool_call_regex: DeepSeekV3ToolParser#tool_call_regex.
  DeepSeekV3ToolParser.stream_tool_call_portion_regex: DeepSeekV3ToolParser#stream_tool_call_portion_regex.
---
# Module: [`easydel/inference/tools/parsers/deepseekv3_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv3_tool_parser.py)

## Classes
### `DeepSeekV3ToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/deepseekv3_tool_parser.py:54`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv3_tool_parser.py#L54)
- doc: Tool parser for DeepSeek V3 models.
- signature: `class DeepSeekV3ToolParser(ToolParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L87`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv3_tool_parser.py#L87) — Initialize the DeepSeek V3 tool parser.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L136`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv3_tool_parser.py#L136) — Extract tool calls from complete model output.
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L197`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv3_tool_parser.py#L197) — Extract tool calls from streaming model output.
  - `current_tool_id` — [`L106`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv3_tool_parser.py#L106)
  - `current_tool_name_sent` — [`L104`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv3_tool_parser.py#L104)
  - `prev_tool_call_arr` — [`L105`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv3_tool_parser.py#L105)
  - `stream_tool_call_name_regex` — [`L123`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv3_tool_parser.py#L123)
  - `stream_tool_call_portion_regex` — [`L119`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv3_tool_parser.py#L119)
  - `streamed_args_for_tool` — [`L107`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv3_tool_parser.py#L107)
  - `tool_call_end_token` — [`L113`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv3_tool_parser.py#L113)
  - `tool_call_end_token_id` — [`L131`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv3_tool_parser.py#L131)
  - `tool_call_regex` — [`L115`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv3_tool_parser.py#L115)
  - `tool_call_start_token` — [`L112`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv3_tool_parser.py#L112)
  - `tool_call_start_token_id` — [`L130`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv3_tool_parser.py#L130)
  - `tool_calls_end_token` — [`L110`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv3_tool_parser.py#L110)
  - `tool_calls_end_token_id` — [`L128`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv3_tool_parser.py#L128)
  - `tool_calls_start_token` — [`L109`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv3_tool_parser.py#L109)
  - `tool_calls_start_token_id` — [`L127`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv3_tool_parser.py#L127)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`DeltaToolCall`](../../openai_api_modules.md#DeltaToolCall), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`DeltaFunctionCall`](../../openai_api_modules.md#DeltaFunctionCall), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolCall`](../../openai_api_modules.md#ToolCall), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`index`](../../openai_api_modules.md#DeltaToolCall.index), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`FunctionCall`](../../openai_api_modules.md#FunctionCall), [`arguments`](../../openai_api_modules.md#FunctionCall.arguments), [`name`](../../openai_api_modules.md#FunctionCall.name), [`id`](../../openai_api_modules.md#DeltaToolCall.id), [`name`](../../openai_api_modules.md#DeltaFunctionCall.name), [`vocab`](../abstract_tool.md#ToolParser.vocab), [`model_tokenizer`](../abstract_tool.md#ToolParser.model_tokenizer), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`logger`](deepseekv3_tool_parser.md#logger)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming)

## Module values
- `logger` — [`L50`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv3_tool_parser.py#L50)

