---
title: 'Module: easydel/inference/tools/parsers/step3_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/step3_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.step3_tool_parser`/
symbols:
  Step3ToolParser.extract_tool_calls_streaming: Step3ToolParser#extract_tool_calls_streaming().
  Step3ToolParser.extract_tool_calls: Step3ToolParser#extract_tool_calls().
  Step3ToolParser._cast_arguments: Step3ToolParser#_cast_arguments().
  Step3ToolParser.SPECIAL_TOKENS: Step3ToolParser#SPECIAL_TOKENS.
  Step3ToolParser.position: Step3ToolParser#position.
  Step3ToolParser.adjust_request: Step3ToolParser#adjust_request().
  Step3ToolParser.TOOL_CALLS_BEGIN: Step3ToolParser#TOOL_CALLS_BEGIN.
  Step3ToolParser.TOOL_CALL_END: Step3ToolParser#TOOL_CALL_END.
  logger: logger.
  Step3ToolParser.TOOL_CALLS_END: Step3ToolParser#TOOL_CALLS_END.
  Step3ToolParser.TOOL_CALL_BEGIN: Step3ToolParser#TOOL_CALL_BEGIN.
  Step3ToolParser: Step3ToolParser#
  Step3ToolParser.__init__: Step3ToolParser#__init__().
  Step3ToolParser.TOOL_SEP: Step3ToolParser#TOOL_SEP.
  Step3ToolParser.tool_block_started: Step3ToolParser#tool_block_started.
  Step3ToolParser.tool_block_finished: Step3ToolParser#tool_block_finished.
  Step3ToolParser._parse_steptml_invoke: Step3ToolParser#_parse_steptml_invoke().
---
# Module: [`easydel/inference/tools/parsers/step3_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/step3_tool_parser.py)

## Classes
### `Step3ToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/step3_tool_parser.py:62`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/step3_tool_parser.py#L62)
- doc: Tool parser for Step3 models with XML-like format.
- signature: `class Step3ToolParser(ToolParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L108`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/step3_tool_parser.py#L108) — Initialize the Step3 tool parser.
  - `_cast_arguments(self, func_name: str, params: dict[str, Any], request: ChatCompletionRequest)` — [`L166`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/step3_tool_parser.py#L166) — Cast argument values to their appropriate types based on schema.
  - `_parse_steptml_invoke(action_text: str)` — [`L140`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/step3_tool_parser.py#L140) — Parse a steptml:invoke element to extract function name and parameters.
  - `adjust_request(self, request: ChatCompletionRequest)` — [`L122`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/step3_tool_parser.py#L122) — Adjust the request to ensure special tokens are not skipped.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L350`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/step3_tool_parser.py#L350) — Extract tool calls from a complete model output.
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L211`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/step3_tool_parser.py#L211) — Extract tool calls from streaming model output.
  - `SPECIAL_TOKENS` — [`L106`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/step3_tool_parser.py#L106)
  - `TOOL_CALLS_BEGIN` — [`L101`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/step3_tool_parser.py#L101)
  - `TOOL_CALLS_END` — [`L102`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/step3_tool_parser.py#L102)
  - `TOOL_CALL_BEGIN` — [`L103`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/step3_tool_parser.py#L103)
  - `TOOL_CALL_END` — [`L104`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/step3_tool_parser.py#L104)
  - `TOOL_SEP` — [`L105`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/step3_tool_parser.py#L105)
  - `position` — [`L118`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/step3_tool_parser.py#L118)
  - `tool_block_finished` — [`L120`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/step3_tool_parser.py#L120)
  - `tool_block_started` — [`L119`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/step3_tool_parser.py#L119)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`DeltaToolCall`](../../openai_api_modules.md#DeltaToolCall), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`DeltaFunctionCall`](../../openai_api_modules.md#DeltaFunctionCall), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolCall`](../../openai_api_modules.md#ToolCall), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`current_tool_id`](../abstract_tool.md#ToolParser.current_tool_id), [`index`](../../openai_api_modules.md#DeltaToolCall.index), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`FunctionCall`](../../openai_api_modules.md#FunctionCall), [`arguments`](../../openai_api_modules.md#FunctionCall.arguments), [`name`](../../openai_api_modules.md#FunctionCall.name), [`id`](../../openai_api_modules.md#DeltaToolCall.id), [`name`](../../openai_api_modules.md#DeltaFunctionCall.name), [`tools`](../../openai_api_modules.md#ChatCompletionRequest.tools), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`prev_tool_call_arr`](../abstract_tool.md#ToolParser.prev_tool_call_arr), [`current_tool_name_sent`](../abstract_tool.md#ToolParser.current_tool_name_sent), [`tool_choice`](../../openai_api_modules.md#ChatCompletionRequest.tool_choice), [`name`](../../openai_api_modules.md#FunctionDefinition.name), [`parameters`](../../openai_api_modules.md#FunctionDefinition.parameters)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming), [`adjust_request`](../abstract_tool.md#ToolParser.adjust_request)

## Module values
- `logger` — [`L58`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/step3_tool_parser.py#L58)

