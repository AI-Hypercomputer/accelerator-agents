---
title: 'Module: easydel/inference/tools/parsers/deepseekv31_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/deepseekv31_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.deepseekv31_tool_parser`/
symbols:
  DeepSeekV31ToolParser.extract_tool_calls_streaming: DeepSeekV31ToolParser#extract_tool_calls_streaming().
  DeepSeekV31ToolParser.extract_tool_calls: DeepSeekV31ToolParser#extract_tool_calls().
  logger: logger.
  DeepSeekV31ToolParser.current_tool_id: DeepSeekV31ToolParser#current_tool_id.
  DeepSeekV31ToolParser.tool_call_end_token_id: DeepSeekV31ToolParser#tool_call_end_token_id.
  DeepSeekV31ToolParser.prev_tool_call_arr: DeepSeekV31ToolParser#prev_tool_call_arr.
  DeepSeekV31ToolParser.tool_call_end_token: DeepSeekV31ToolParser#tool_call_end_token.
  DeepSeekV31ToolParser.tool_calls_start_token_id: DeepSeekV31ToolParser#tool_calls_start_token_id.
  DeepSeekV31ToolParser.tool_call_start_token_id: DeepSeekV31ToolParser#tool_call_start_token_id.
  DeepSeekV31ToolParser.tool_calls_end_token_id: DeepSeekV31ToolParser#tool_calls_end_token_id.
  DeepSeekV31ToolParser.tool_call_start_token: DeepSeekV31ToolParser#tool_call_start_token.
  DeepSeekV31ToolParser: DeepSeekV31ToolParser#
  DeepSeekV31ToolParser.streamed_args_for_tool: DeepSeekV31ToolParser#streamed_args_for_tool.
  DeepSeekV31ToolParser.tool_calls_start_token: DeepSeekV31ToolParser#tool_calls_start_token.
  DeepSeekV31ToolParser.stream_tool_call_name_regex: DeepSeekV31ToolParser#stream_tool_call_name_regex.
  DeepSeekV31ToolParser.current_tool_name_sent: DeepSeekV31ToolParser#current_tool_name_sent.
  DeepSeekV31ToolParser.__init__: DeepSeekV31ToolParser#__init__().
  DeepSeekV31ToolParser.tool_calls_end_token: DeepSeekV31ToolParser#tool_calls_end_token.
  DeepSeekV31ToolParser.tool_call_regex: DeepSeekV31ToolParser#tool_call_regex.
  DeepSeekV31ToolParser.stream_tool_call_portion_regex: DeepSeekV31ToolParser#stream_tool_call_portion_regex.
---
# Module: [`easydel/inference/tools/parsers/deepseekv31_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv31_tool_parser.py)

## Classes
### `DeepSeekV31ToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/deepseekv31_tool_parser.py:51`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv31_tool_parser.py#L51)
- doc: Tool parser for DeepSeek V3.1 models.
- signature: `class DeepSeekV31ToolParser(ToolParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L86`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv31_tool_parser.py#L86) — Initialize the DeepSeek V3.1 tool parser.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L137`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv31_tool_parser.py#L137) — Extract tool calls from complete model output.
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L191`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv31_tool_parser.py#L191) — Extract tool calls from streaming model output.
  - `current_tool_id` — [`L105`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv31_tool_parser.py#L105)
  - `current_tool_name_sent` — [`L103`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv31_tool_parser.py#L103)
  - `prev_tool_call_arr` — [`L104`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv31_tool_parser.py#L104)
  - `stream_tool_call_name_regex` — [`L124`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv31_tool_parser.py#L124)
  - `stream_tool_call_portion_regex` — [`L119`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv31_tool_parser.py#L119)
  - `streamed_args_for_tool` — [`L106`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv31_tool_parser.py#L106)
  - `tool_call_end_token` — [`L112`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv31_tool_parser.py#L112)
  - `tool_call_end_token_id` — [`L132`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv31_tool_parser.py#L132)
  - `tool_call_regex` — [`L114`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv31_tool_parser.py#L114)
  - `tool_call_start_token` — [`L111`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv31_tool_parser.py#L111)
  - `tool_call_start_token_id` — [`L131`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv31_tool_parser.py#L131)
  - `tool_calls_end_token` — [`L109`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv31_tool_parser.py#L109)
  - `tool_calls_end_token_id` — [`L129`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv31_tool_parser.py#L129)
  - `tool_calls_start_token` — [`L108`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv31_tool_parser.py#L108)
  - `tool_calls_start_token_id` — [`L128`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv31_tool_parser.py#L128)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`DeltaToolCall`](../../openai_api_modules.md#DeltaToolCall), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`DeltaFunctionCall`](../../openai_api_modules.md#DeltaFunctionCall), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolCall`](../../openai_api_modules.md#ToolCall), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`index`](../../openai_api_modules.md#DeltaToolCall.index), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`FunctionCall`](../../openai_api_modules.md#FunctionCall), [`arguments`](../../openai_api_modules.md#FunctionCall.arguments), [`name`](../../openai_api_modules.md#FunctionCall.name), [`id`](../../openai_api_modules.md#DeltaToolCall.id), [`name`](../../openai_api_modules.md#DeltaFunctionCall.name), [`vocab`](../abstract_tool.md#ToolParser.vocab), [`model_tokenizer`](../abstract_tool.md#ToolParser.model_tokenizer), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`logger`](deepseekv31_tool_parser.md#logger)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming)

## Module values
- `logger` — [`L47`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/deepseekv31_tool_parser.py#L47)

