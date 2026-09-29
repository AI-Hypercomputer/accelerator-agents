---
title: 'Module: easydel/inference/tools/parsers/olmo3_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/olmo3_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.olmo3_tool_parser`/
symbols:
  Olmo3PythonicToolParser.extract_tool_calls_streaming: Olmo3PythonicToolParser#extract_tool_calls_streaming().
  _compute_tool_delta: _compute_tool_delta().
  Olmo3PythonicToolParser.extract_tool_calls: Olmo3PythonicToolParser#extract_tool_calls().
  _handle_single_tool: _handle_single_tool().
  Olmo3PythonicToolParser: Olmo3PythonicToolParser#
  _UnexpectedAstError: _UnexpectedAstError#
  _get_parameter_value: _get_parameter_value().
  Olmo3PythonicToolParser.current_tool_index: Olmo3PythonicToolParser#current_tool_index().
  _make_valid_python: _make_valid_python().
  Olmo3PythonicToolParser.__init__: Olmo3PythonicToolParser#__init__().
  logger: logger.
  Olmo3PythonicToolParser.TOOL_CALL_REGEX: Olmo3PythonicToolParser#TOOL_CALL_REGEX.
  Olmo3PythonicToolParser.tool_call_start_token: Olmo3PythonicToolParser#tool_call_start_token.
  Olmo3PythonicToolParser.tool_call_end_token: Olmo3PythonicToolParser#tool_call_end_token.
---
# Module: [`easydel/inference/tools/parsers/olmo3_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/olmo3_tool_parser.py)

## Classes
### `Olmo3PythonicToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/olmo3_tool_parser.py:68`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/olmo3_tool_parser.py#L68)
- doc: Parser for OLMo-3 models that emit newline-separated pythonic tool calls.
- signature: `class Olmo3PythonicToolParser(ToolParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L98`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/olmo3_tool_parser.py#L98) — Initialize the OLMo-3 Pythonic tool parser.
  - `current_tool_index(self)` — [`L109`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/olmo3_tool_parser.py#L109) — Get the current tool index being processed.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L126`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/olmo3_tool_parser.py#L126) — Extract tool calls from a complete model output.
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L168`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/olmo3_tool_parser.py#L168) — Extract tool calls from streaming model output.
  - `TOOL_CALL_REGEX` — [`L93`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/olmo3_tool_parser.py#L93)
  - `tool_call_end_token` — [`L106`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/olmo3_tool_parser.py#L106)
  - `tool_call_start_token` — [`L105`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/olmo3_tool_parser.py#L105)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`DeltaToolCall`](../../openai_api_modules.md#DeltaToolCall), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`current_tool_id`](../abstract_tool.md#ToolParser.current_tool_id), [`ToolParserManager`](../abstract_tool.md#ToolParserManager), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`register_module`](../abstract_tool.md#ToolParserManager.register_module), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`_compute_tool_delta`](olmo3_tool_parser.md#_compute_tool_delta), [`streamed_args_for_tool`](../abstract_tool.md#ToolParser.streamed_args_for_tool), [`prev_tool_call_arr`](../abstract_tool.md#ToolParser.prev_tool_call_arr), [`_handle_single_tool`](olmo3_tool_parser.md#_handle_single_tool), [`_UnexpectedAstError`](olmo3_tool_parser.md#_UnexpectedAstError), [`_make_valid_python`](olmo3_tool_parser.md#_make_valid_python), [`logger`](olmo3_tool_parser.md#logger)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming)

### `_UnexpectedAstError`  ·  implements/extends Exception
- def: [`easydel/inference/tools/parsers/olmo3_tool_parser.py:56`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/olmo3_tool_parser.py#L56)
- doc: Exception raised when AST parsing encounters unexpected structure.
- signature: `class _UnexpectedAstError(Exception):`
- used by: [`extract_tool_calls_streaming`](olmo3_tool_parser.md#Olmo3PythonicToolParser.extract_tool_calls_streaming), [`extract_tool_calls`](olmo3_tool_parser.md#Olmo3PythonicToolParser.extract_tool_calls), [`_handle_single_tool`](olmo3_tool_parser.md#_handle_single_tool), [`_get_parameter_value`](olmo3_tool_parser.md#_get_parameter_value), [`_make_valid_python`](olmo3_tool_parser.md#_make_valid_python)

## Functions
- `_compute_tool_delta(previously_sent_args: str, new_call: ToolCall, index: int, withheld_suffix: str)` — [`L402`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/olmo3_tool_parser.py#L402) — Compute the delta between previously sent and new tool call arguments.
- `_get_parameter_value(val: ast.expr)` — [`L263`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/olmo3_tool_parser.py#L263) — Extract a Python value from an AST expression node.
- `_handle_single_tool(call: ast.Call)` — [`L297`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/olmo3_tool_parser.py#L297) — Convert an AST Call node to a ToolCall object.
- `_make_valid_python(text: str)` — [`L323`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/olmo3_tool_parser.py#L323) — Complete partial Python syntax to make it parseable.

## Module values
- `logger` — [`L53`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/olmo3_tool_parser.py#L53)

