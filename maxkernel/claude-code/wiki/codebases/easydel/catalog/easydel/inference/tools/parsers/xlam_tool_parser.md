---
title: 'Module: easydel/inference/tools/parsers/xlam_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/xlam_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.xlam_tool_parser`/
symbols:
  xLAMToolParser.extract_tool_calls_streaming: xLAMToolParser#extract_tool_calls_streaming().
  xLAMToolParser.extract_tool_calls: xLAMToolParser#extract_tool_calls().
  xLAMToolParser.streaming_state: xLAMToolParser#streaming_state.
  xLAMToolParser.streamed_args: xLAMToolParser#streamed_args.
  xLAMToolParser: xLAMToolParser#
  xLAMToolParser.preprocess_model_output: xLAMToolParser#preprocess_model_output().
  xLAMToolParser.prev_tool_calls: xLAMToolParser#prev_tool_calls.
  xLAMToolParser.current_tools_sent: xLAMToolParser#current_tools_sent.
  xLAMToolParser.__init__: xLAMToolParser#__init__().
  logger: logger.
  _random_uuid: _random_uuid().
  xLAMToolParser.json_code_block_patterns: xLAMToolParser#json_code_block_patterns.
  _make_tool_call_id: _make_tool_call_id().
  xLAMToolParser.prev_tool_call_arr: xLAMToolParser#prev_tool_call_arr.
  xLAMToolParser.thinking_tag_pattern: xLAMToolParser#thinking_tag_pattern.
  xLAMToolParser.tool_call_start_token: xLAMToolParser#tool_call_start_token.
  xLAMToolParser.tool_call_end_token: xLAMToolParser#tool_call_end_token.
---
# Module: [`easydel/inference/tools/parsers/xlam_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/xlam_tool_parser.py)

## Classes
### `xLAMToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/xlam_tool_parser.py:74`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/xlam_tool_parser.py#L74)
- doc: Tool parser for xLAM and similar models with flexible JSON formats.
- signature: `class xLAMToolParser(ToolParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L106`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/xlam_tool_parser.py#L106) — Initialize the xLAM tool parser.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L195`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/xlam_tool_parser.py#L195) — Extract tool calls from a complete model output.
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L244`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/xlam_tool_parser.py#L244) — Extract tool calls from streaming model output.
  - `preprocess_model_output(self, model_output: str)` — [`L141`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/xlam_tool_parser.py#L141) — Preprocess model output to extract potential tool calls.
  - `current_tools_sent` — [`L123`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/xlam_tool_parser.py#L123)
  - `json_code_block_patterns` — [`L128`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/xlam_tool_parser.py#L128)
  - `prev_tool_call_arr` — [`L124`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/xlam_tool_parser.py#L124)
  - `prev_tool_calls` — [`L117`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/xlam_tool_parser.py#L117)
  - `streamed_args` — [`L120`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/xlam_tool_parser.py#L120)
  - `streaming_state` — [`L135`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/xlam_tool_parser.py#L135)
  - `thinking_tag_pattern` — [`L133`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/xlam_tool_parser.py#L133)
  - `tool_call_end_token` — [`L126`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/xlam_tool_parser.py#L126)
  - `tool_call_start_token` — [`L125`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/xlam_tool_parser.py#L125)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`DeltaToolCall`](../../openai_api_modules.md#DeltaToolCall), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`DeltaFunctionCall`](../../openai_api_modules.md#DeltaFunctionCall), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolCall`](../../openai_api_modules.md#ToolCall), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`current_tool_id`](../abstract_tool.md#ToolParser.current_tool_id), [`index`](../../openai_api_modules.md#DeltaToolCall.index), [`ToolParserManager`](../abstract_tool.md#ToolParserManager), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`FunctionCall`](../../openai_api_modules.md#FunctionCall), [`arguments`](../../openai_api_modules.md#FunctionCall.arguments), [`name`](../../openai_api_modules.md#FunctionCall.name), [`register_module`](../abstract_tool.md#ToolParserManager.register_module), [`id`](../../openai_api_modules.md#DeltaToolCall.id), [`name`](../../openai_api_modules.md#DeltaFunctionCall.name), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`current_tool_name_sent`](../abstract_tool.md#ToolParser.current_tool_name_sent), [`id`](../../openai_api_modules.md#ToolCall.id), [`_random_uuid`](xlam_tool_parser.md#_random_uuid), [`logger`](xlam_tool_parser.md#logger), [`_make_tool_call_id`](xlam_tool_parser.md#_make_tool_call_id)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming)

## Functions
- `_make_tool_call_id()` — [`L63`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/xlam_tool_parser.py#L63) — Generate a unique tool call ID.
- `_random_uuid()` — [`L54`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/xlam_tool_parser.py#L54) — Generate a random UUID hex string.

## Module values
- `logger` — [`L51`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/xlam_tool_parser.py#L51)

