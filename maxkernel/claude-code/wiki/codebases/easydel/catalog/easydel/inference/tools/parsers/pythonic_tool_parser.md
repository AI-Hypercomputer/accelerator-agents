---
title: 'Module: easydel/inference/tools/parsers/pythonic_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/pythonic_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.pythonic_tool_parser`/
symbols:
  PythonicToolParser.extract_tool_calls_streaming: PythonicToolParser#extract_tool_calls_streaming().
  _compute_tool_delta: _compute_tool_delta().
  PythonicToolParser.extract_tool_calls: PythonicToolParser#extract_tool_calls().
  _handle_single_tool: _handle_single_tool().
  PythonicToolParser: PythonicToolParser#
  _UnexpectedAstError: _UnexpectedAstError#
  _get_parameter_value: _get_parameter_value().
  PythonicToolParser.current_tool_index: PythonicToolParser#current_tool_index().
  logger: logger.
  _make_valid_python: _make_valid_python().
  PythonicToolParser.__init__: PythonicToolParser#__init__().
  PythonicToolParser.TOOL_CALL_REGEX: PythonicToolParser#TOOL_CALL_REGEX.
  PythonicToolParser.tool_call_start_token: PythonicToolParser#tool_call_start_token.
  PythonicToolParser.tool_call_end_token: PythonicToolParser#tool_call_end_token.
---
# Module: [`easydel/inference/tools/parsers/pythonic_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/pythonic_tool_parser.py)

## Classes
### `PythonicToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/pythonic_tool_parser.py:70`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/pythonic_tool_parser.py#L70)
- doc: Tool call parser for models using Python-style function call syntax.
- signature: `class PythonicToolParser(ToolParser):`
- members:
  - `__init__(self, tokenizer: PreTrainedTokenizerBase)` — [`L101`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/pythonic_tool_parser.py#L101) — Initialize the PythonicToolParser.
  - `current_tool_index(self)` — [`L113`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/pythonic_tool_parser.py#L113) — Get the current tool index being processed.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L130`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/pythonic_tool_parser.py#L130) — Extract tool calls from a complete model response.
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L171`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/pythonic_tool_parser.py#L171) — Extract tool calls incrementally during streaming.
  - `TOOL_CALL_REGEX` — [`L96`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/pythonic_tool_parser.py#L96)
  - `tool_call_end_token` — [`L110`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/pythonic_tool_parser.py#L110)
  - `tool_call_start_token` — [`L109`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/pythonic_tool_parser.py#L109)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`current_tool_id`](../abstract_tool.md#ToolParser.current_tool_id), [`ToolParserManager`](../abstract_tool.md#ToolParserManager), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`register_module`](../abstract_tool.md#ToolParserManager.register_module), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`_compute_tool_delta`](pythonic_tool_parser.md#_compute_tool_delta), [`streamed_args_for_tool`](../abstract_tool.md#ToolParser.streamed_args_for_tool), [`prev_tool_call_arr`](../abstract_tool.md#ToolParser.prev_tool_call_arr), [`_handle_single_tool`](pythonic_tool_parser.md#_handle_single_tool), [`_UnexpectedAstError`](pythonic_tool_parser.md#_UnexpectedAstError), [`logger`](pythonic_tool_parser.md#logger), [`_make_valid_python`](pythonic_tool_parser.md#_make_valid_python)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming)

### `_UnexpectedAstError`  ·  implements/extends Exception
- def: [`easydel/inference/tools/parsers/pythonic_tool_parser.py:58`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/pythonic_tool_parser.py#L58)
- doc: Exception raised when AST parsing encounters unexpected structure.
- signature: `class _UnexpectedAstError(Exception):`
- used by: [`extract_tool_calls_streaming`](pythonic_tool_parser.md#PythonicToolParser.extract_tool_calls_streaming), [`extract_tool_calls`](pythonic_tool_parser.md#PythonicToolParser.extract_tool_calls), [`_handle_single_tool`](pythonic_tool_parser.md#_handle_single_tool), [`_get_parameter_value`](pythonic_tool_parser.md#_get_parameter_value), [`_make_valid_python`](pythonic_tool_parser.md#_make_valid_python)

## Functions
- `_compute_tool_delta(previously_sent_args: str, new_call: ToolCall, index: int, withheld_suffix: str)` — [`L396`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/pythonic_tool_parser.py#L396) — Compute the delta between previously sent and current tool call arguments.
- `_get_parameter_value(val: ast.expr)` — [`L255`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/pythonic_tool_parser.py#L255) — Extract parameter value from AST expression.
- `_handle_single_tool(call: ast.Call)` — [`L286`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/pythonic_tool_parser.py#L286) — Convert AST Call node to ToolCall object.
- `_make_valid_python(text: str)` — [`L316`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/pythonic_tool_parser.py#L316) — Complete partial Python code by adding closing brackets.

## Module values
- `logger` — [`L55`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/pythonic_tool_parser.py#L55)

