---
title: 'Module: easydel/inference/tools/parsers/hermes_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/hermes_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.hermes_tool_parser`/HermesToolParser#
symbols:
  HermesToolParser.extract_tool_calls_streaming: extract_tool_calls_streaming().
  HermesToolParser.extract_tool_calls: extract_tool_calls().
  HermesToolParser: ''
  HermesToolParser.current_tool_id: current_tool_id.
  HermesToolParser.tool_call_start_token: tool_call_start_token.
  HermesToolParser.prev_tool_call_arr: prev_tool_call_arr.
  HermesToolParser.tool_call_end_token: tool_call_end_token.
  HermesToolParser.buffered_delta_text: buffered_delta_text.
  HermesToolParser.tool_call_delta_buffer: tool_call_delta_buffer().
  HermesToolParser.tool_call_start_token_array: tool_call_start_token_array.
  HermesToolParser.tool_call_end_token_array: tool_call_end_token_array.
  HermesToolParser.tool_call_start_token_ids: tool_call_start_token_ids.
  HermesToolParser.tool_call_end_token_ids: tool_call_end_token_ids.
  HermesToolParser.current_tool_name_sent: current_tool_name_sent.
  HermesToolParser.streamed_args_for_tool: streamed_args_for_tool.
  HermesToolParser.__init__: __init__().
  HermesToolParser.scratch_pad_regex: scratch_pad_regex.
  HermesToolParser.tool_call_regex: tool_call_regex.
---
# Module: [`easydel/inference/tools/parsers/hermes_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hermes_tool_parser.py)

## Classes
### `HermesToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/hermes_tool_parser.py:67`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hermes_tool_parser.py#L67)
- doc: Tool call parser for Hermes models.
- signature: `class HermesToolParser(ToolParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L95`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hermes_tool_parser.py#L95) — Initialize the Hermes tool parser.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L165`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hermes_tool_parser.py#L165) — Extract tool calls from complete model response.
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L217`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hermes_tool_parser.py#L217) — Extract tool calls from streaming model output.
  - `tool_call_delta_buffer(self, delta_text: str)` — [`L135`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hermes_tool_parser.py#L135) — Buffer delta text to handle multi-token delimiters.
  - `buffered_delta_text` — [`L133`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hermes_tool_parser.py#L133)
  - `current_tool_id` — [`L109`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hermes_tool_parser.py#L109)
  - `current_tool_name_sent` — [`L107`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hermes_tool_parser.py#L107)
  - `prev_tool_call_arr` — [`L108`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hermes_tool_parser.py#L108)
  - `scratch_pad_regex` — [`L116`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hermes_tool_parser.py#L116)
  - `streamed_args_for_tool` — [`L110`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hermes_tool_parser.py#L110)
  - `tool_call_end_token` — [`L113`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hermes_tool_parser.py#L113)
  - `tool_call_end_token_array` — [`L129`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hermes_tool_parser.py#L129)
  - `tool_call_end_token_ids` — [`L123`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hermes_tool_parser.py#L123)
  - `tool_call_regex` — [`L115`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hermes_tool_parser.py#L115)
  - `tool_call_start_token` — [`L112`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hermes_tool_parser.py#L112)
  - `tool_call_start_token_array` — [`L125`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hermes_tool_parser.py#L125)
  - `tool_call_start_token_ids` — [`L120`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hermes_tool_parser.py#L120)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`DeltaToolCall`](../../openai_api_modules.md#DeltaToolCall), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`DeltaFunctionCall`](../../openai_api_modules.md#DeltaFunctionCall), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolCall`](../../openai_api_modules.md#ToolCall), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`index`](../../openai_api_modules.md#DeltaToolCall.index), [`ToolParserManager`](../abstract_tool.md#ToolParserManager), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`FunctionCall`](../../openai_api_modules.md#FunctionCall), [`arguments`](../../openai_api_modules.md#FunctionCall.arguments), [`name`](../../openai_api_modules.md#FunctionCall.name), [`register_module`](../abstract_tool.md#ToolParserManager.register_module), [`id`](../../openai_api_modules.md#DeltaToolCall.id), [`name`](../../openai_api_modules.md#DeltaFunctionCall.name), [`model_tokenizer`](../abstract_tool.md#ToolParser.model_tokenizer), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`LongcatFlashToolParser`](longcat_tool_parser.md#LongcatFlashToolParser)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming), [`__init__`](longcat_tool_parser.md#LongcatFlashToolParser.__init__), [`LongcatFlashToolParser`](longcat_tool_parser.md#LongcatFlashToolParser), [`tool_call_end_token_ids`](longcat_tool_parser.md#LongcatFlashToolParser.tool_call_end_token_ids), [`tool_call_start_token_ids`](longcat_tool_parser.md#LongcatFlashToolParser.tool_call_start_token_ids)

