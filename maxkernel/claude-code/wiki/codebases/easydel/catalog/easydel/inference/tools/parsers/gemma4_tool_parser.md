---
title: 'Module: easydel/inference/tools/parsers/gemma4_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/gemma4_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.gemma4_tool_parser`/
symbols:
  Gemma4ToolParser._handle_tool_call_middle: Gemma4ToolParser#_handle_tool_call_middle().
  Gemma4ToolParser.extract_tool_calls: Gemma4ToolParser#extract_tool_calls().
  Gemma4ToolParser._handle_tool_call_end: Gemma4ToolParser#_handle_tool_call_end().
  Gemma4ToolParser._extract_streaming: Gemma4ToolParser#_extract_streaming().
  Gemma4ToolParser._emit_argument_diff: Gemma4ToolParser#_emit_argument_diff().
  STRING_DELIM: STRING_DELIM.
  Gemma4ToolParser.extract_tool_calls_streaming: Gemma4ToolParser#extract_tool_calls_streaming().
  _parse_gemma4_args: _parse_gemma4_args().
  Gemma4ToolParser.current_tool_id: Gemma4ToolParser#current_tool_id.
  TOOL_CALL_START: TOOL_CALL_START.
  _parse_gemma4_array: _parse_gemma4_array().
  TOOL_CALL_END: TOOL_CALL_END.
  Gemma4ToolParser.streamed_args_for_tool: Gemma4ToolParser#streamed_args_for_tool.
  Gemma4ToolParser.tool_call_end_token_ids: Gemma4ToolParser#tool_call_end_token_ids.
  Gemma4ToolParser.adjust_request: Gemma4ToolParser#adjust_request().
  Gemma4ToolParser: Gemma4ToolParser#
  Gemma4ToolParser._buffer_delta_text: Gemma4ToolParser#_buffer_delta_text().
  Gemma4ToolParser.tool_call_start_token_ids: Gemma4ToolParser#tool_call_start_token_ids.
  Gemma4ToolParser._extract_partial_call: Gemma4ToolParser#_extract_partial_call().
  Gemma4ToolParser.prev_tool_call_arr: Gemma4ToolParser#prev_tool_call_arr.
  Gemma4ToolParser.tool_call_regex: Gemma4ToolParser#tool_call_regex.
  Gemma4ToolParser.current_tool_name_sent: Gemma4ToolParser#current_tool_name_sent.
  Gemma4ToolParser.buffered_delta_text: Gemma4ToolParser#buffered_delta_text.
  Gemma4ToolParser.tool_call_start_token: Gemma4ToolParser#tool_call_start_token.
  Gemma4ToolParser.tool_call_end_token: Gemma4ToolParser#tool_call_end_token.
  Gemma4ToolParser.__init__: Gemma4ToolParser#__init__().
  logger: logger.
  _parse_gemma4_value: _parse_gemma4_value().
---
# Module: [`easydel/inference/tools/parsers/gemma4_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py)

## Classes
### `Gemma4ToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/gemma4_tool_parser.py:257`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L257)
- doc: Parse Gemma4 `<|tool_call>call:...<tool_call|>` tool invocations.
- signature: `class Gemma4ToolParser(ToolParser):`
- members:
  - `_emit_argument_diff(self, raw_args_str: str)` — [`L489`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L489) — Parse raw args, convert to JSON, withhold trailing closers, diff and emit.
  - `adjust_request(self, request: ChatCompletionRequest)` — [`L311`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L311)
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L317`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L317)
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L355`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L355)
  - `buffered_delta_text` — [`L293`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L293)
  - `current_tool_id` — [`L276`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L276)
  - `current_tool_name_sent` — [`L274`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L274)
  - `prev_tool_call_arr` — [`L275`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L275)
  - `streamed_args_for_tool` — [`L277`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L277)
  - `tool_call_end_token` — [`L269`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L269)
  - `tool_call_end_token_ids` — [`L288`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L288)
  - `tool_call_regex` — [`L281`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L281)
  - `tool_call_start_token` — [`L268`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L268)
  - `tool_call_start_token_ids` — [`L287`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L287)
- protocol/private: `__init__`[`L271`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L271), `_buffer_delta_text`[`L295`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L295), `_extract_partial_call`[`L416`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L416), `_extract_streaming`[`L376`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L376), `_handle_tool_call_end`[`L464`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L464), `_handle_tool_call_middle`[`L440`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L440)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`DeltaToolCall`](../../openai_api_modules.md#DeltaToolCall), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`DeltaFunctionCall`](../../openai_api_modules.md#DeltaFunctionCall), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolCall`](../../openai_api_modules.md#ToolCall), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`index`](../../openai_api_modules.md#DeltaToolCall.index), [`ToolParserManager`](../abstract_tool.md#ToolParserManager), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`FunctionCall`](../../openai_api_modules.md#FunctionCall), [`arguments`](../../openai_api_modules.md#FunctionCall.arguments), [`name`](../../openai_api_modules.md#FunctionCall.name), [`register_module`](../abstract_tool.md#ToolParserManager.register_module), [`id`](../../openai_api_modules.md#DeltaToolCall.id), [`name`](../../openai_api_modules.md#DeltaFunctionCall.name), [`tools`](../../openai_api_modules.md#ChatCompletionRequest.tools), [`model_tokenizer`](../abstract_tool.md#ToolParser.model_tokenizer), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`adjust_request`](../abstract_tool.md#ToolParser.adjust_request), [`_parse_gemma4_args`](gemma4_tool_parser.md#_parse_gemma4_args), [`TOOL_CALL_START`](gemma4_tool_parser.md#TOOL_CALL_START), [`tool_choice`](../../openai_api_modules.md#ChatCompletionRequest.tool_choice), [`TOOL_CALL_END`](gemma4_tool_parser.md#TOOL_CALL_END), [`logger`](gemma4_tool_parser.md#logger)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming), [`adjust_request`](../abstract_tool.md#ToolParser.adjust_request)

## Functions
- `_parse_gemma4_args(args_str: str, *, partial: bool = False)` — [`L72`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L72) — Parse Gemma4's custom key:value format into a Python dict.
- `_parse_gemma4_array(arr_str: str, *, partial: bool = False)` — [`L189`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L189) — Parse a Gemma4 array content string into a Python list.
- `_parse_gemma4_value(value_str: str)` — [`L54`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L54) — Parse a single Gemma4 value (after key:) into a Python object.

## Module values
- `STRING_DELIM` — [`L51`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L51)
- `TOOL_CALL_END` — [`L50`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L50)
- `TOOL_CALL_START` — [`L49`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L49)
- `logger` — [`L46`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gemma4_tool_parser.py#L46)

