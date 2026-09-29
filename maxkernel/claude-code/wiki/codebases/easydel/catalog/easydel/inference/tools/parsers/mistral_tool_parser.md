---
title: 'Module: easydel/inference/tools/parsers/mistral_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/mistral_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.mistral_tool_parser`/
symbols:
  MistralToolParser.extract_tool_calls_streaming: MistralToolParser#extract_tool_calls_streaming().
  MistralToolParser.extract_tool_calls: MistralToolParser#extract_tool_calls().
  MistralToolParser.current_tool_id: MistralToolParser#current_tool_id.
  logger: logger.
  MistralToolParser.adjust_request: MistralToolParser#adjust_request().
  MistralToolParser: MistralToolParser#
  MistralToolParser.fn_name_regex: MistralToolParser#fn_name_regex.
  MistralToolParser.bot_token_id: MistralToolParser#bot_token_id.
  MistralToolCall: MistralToolCall#
  MistralToolParser.bot_token: MistralToolParser#bot_token.
  MistralToolParser.tool_call_regex: MistralToolParser#tool_call_regex.
  MistralToolCall.id: MistralToolCall#id.
  MistralToolParser.__init__: MistralToolParser#__init__().
  MistralToolParser.streamed_args_for_tool: MistralToolParser#streamed_args_for_tool.
  MistralToolCall.generate_random_id: MistralToolCall#generate_random_id().
  MistralToolParser.current_tool_name_sent: MistralToolParser#current_tool_name_sent.
  MistralToolParser.prev_tool_call_arr: MistralToolParser#prev_tool_call_arr.
  ALPHANUMERIC: ALPHANUMERIC.
  _is_fn_name_regex_support: _is_fn_name_regex_support().
  MistralToolCall.is_valid_id: MistralToolCall#is_valid_id().
---
# Module: [`easydel/inference/tools/parsers/mistral_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/mistral_tool_parser.py)

## Classes
### `MistralToolCall`  ·  implements/extends ToolCall
- def: [`easydel/inference/tools/parsers/mistral_tool_parser.py:75`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/mistral_tool_parser.py#L75)
- doc: A specialized ToolCall class for Mistral models with auto-generated IDs.
- signature: `class MistralToolCall(ToolCall):`
- members:
  - `generate_random_id()` — [`L99`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/mistral_tool_parser.py#L99) — Generate a random 9-character alphanumeric ID.
  - `is_valid_id(id: str)` — [`L119`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/mistral_tool_parser.py#L119) — Validate whether a string is a valid Mistral tool call ID.
  - `id` — [`L96`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/mistral_tool_parser.py#L96)
- uses (calls/refs, reference-scoped): [`ToolCall`](../../openai_api_modules.md#ToolCall), [`ALPHANUMERIC`](mistral_tool_parser.md#ALPHANUMERIC)
- used by: [`ToolCall`](../../openai_api_modules.md#ToolCall), [`extract_tool_calls_streaming`](mistral_tool_parser.md#MistralToolParser.extract_tool_calls_streaming), [`extract_tool_calls`](mistral_tool_parser.md#MistralToolParser.extract_tool_calls)

### `MistralToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/mistral_tool_parser.py:163`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/mistral_tool_parser.py#L163)
- doc: Tool call parser for Mistral models (7B Instruct v0.3+).
- signature: `class MistralToolParser(ToolParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L217`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/mistral_tool_parser.py#L217) — Initialize the Mistral tool parser.
  - `adjust_request(self, request: ChatCompletionRequest)` — [`L249`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/mistral_tool_parser.py#L249) — Adjust the chat completion request for Mistral tool parsing compatibility.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L283`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/mistral_tool_parser.py#L283) — Extract tool calls from a complete Mistral model response.
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L369`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/mistral_tool_parser.py#L369) — Extract tool calls incrementally during streaming generation.
  - `bot_token` — [`L238`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/mistral_tool_parser.py#L238)
  - `bot_token_id` — [`L239`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/mistral_tool_parser.py#L239)
  - `current_tool_id` — [`L235`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/mistral_tool_parser.py#L235)
  - `current_tool_name_sent` — [`L236`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/mistral_tool_parser.py#L236)
  - `fn_name_regex` — [`L242`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/mistral_tool_parser.py#L242)
  - `prev_tool_call_arr` — [`L234`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/mistral_tool_parser.py#L234)
  - `streamed_args_for_tool` — [`L237`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/mistral_tool_parser.py#L237)
  - `tool_call_regex` — [`L240`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/mistral_tool_parser.py#L240)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`DeltaToolCall`](../../openai_api_modules.md#DeltaToolCall), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`DeltaFunctionCall`](../../openai_api_modules.md#DeltaFunctionCall), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`index`](../../openai_api_modules.md#DeltaToolCall.index), [`ToolParserManager`](../abstract_tool.md#ToolParserManager), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`FunctionCall`](../../openai_api_modules.md#FunctionCall), [`arguments`](../../openai_api_modules.md#FunctionCall.arguments), [`name`](../../openai_api_modules.md#FunctionCall.name), [`register_module`](../abstract_tool.md#ToolParserManager.register_module), [`id`](../../openai_api_modules.md#DeltaToolCall.id), [`name`](../../openai_api_modules.md#DeltaFunctionCall.name), [`vocab`](../abstract_tool.md#ToolParser.vocab), [`tools`](../../openai_api_modules.md#ChatCompletionRequest.tools), [`model_tokenizer`](../abstract_tool.md#ToolParser.model_tokenizer), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`tool_choice`](../../openai_api_modules.md#ChatCompletionRequest.tool_choice), [`extract_intermediate_diff`](../utils.md#extract_intermediate_diff), [`logger`](mistral_tool_parser.md#logger), [`MistralToolCall`](mistral_tool_parser.md#MistralToolCall), [`generate_random_id`](mistral_tool_parser.md#MistralToolCall.generate_random_id), [`_is_fn_name_regex_support`](mistral_tool_parser.md#_is_fn_name_regex_support)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming), [`adjust_request`](../abstract_tool.md#ToolParser.adjust_request)

## Functions
- `_is_fn_name_regex_support(model_tokenizer: AnyTokenizer)` — [`L143`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/mistral_tool_parser.py#L143) — Check if the tokenizer supports regex-based function name extraction.

## Module values
- `ALPHANUMERIC` — [`L72`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/mistral_tool_parser.py#L72)
- `logger` — [`L70`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/mistral_tool_parser.py#L70)

