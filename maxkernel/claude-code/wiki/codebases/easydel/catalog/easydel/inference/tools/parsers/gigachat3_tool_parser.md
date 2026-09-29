---
title: 'Module: easydel/inference/tools/parsers/gigachat3_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/gigachat3_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.gigachat3_tool_parser`/
symbols:
  GigaChat3ToolParser.extract_tool_calls_streaming: GigaChat3ToolParser#extract_tool_calls_streaming().
  GigaChat3ToolParser.extract_tool_calls: GigaChat3ToolParser#extract_tool_calls().
  ARGS_REGEX: ARGS_REGEX.
  GigaChat3ToolParser.prev_tool_call_arr: GigaChat3ToolParser#prev_tool_call_arr.
  GigaChat3ToolParser.content_buffer: GigaChat3ToolParser#content_buffer.
  GigaChat3ToolParser: GigaChat3ToolParser#
  REGEX_FUNCTION_CALL: REGEX_FUNCTION_CALL.
  GigaChat3ToolParser.__init__: GigaChat3ToolParser#__init__().
  GigaChat3ToolParser.tool_started: GigaChat3ToolParser#tool_started.
  GigaChat3ToolParser.tool_name_sent: GigaChat3ToolParser#tool_name_sent.
  GigaChat3ToolParser.tool_id: GigaChat3ToolParser#tool_id.
  GigaChat3ToolParser.trigger_start: GigaChat3ToolParser#trigger_start.
  NAME_REGEX: NAME_REGEX.
  logger: logger.
---
# Module: [`easydel/inference/tools/parsers/gigachat3_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gigachat3_tool_parser.py)

## Classes
### `GigaChat3ToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/gigachat3_tool_parser.py:58`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gigachat3_tool_parser.py#L58)
- doc: Tool parser for GigaChat3 model outputs.
- signature: `class GigaChat3ToolParser(ToolParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L84`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gigachat3_tool_parser.py#L84) — Initialize the GigaChat3 tool parser.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L99`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gigachat3_tool_parser.py#L99) — Extract tool calls from a complete model output.
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L153`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gigachat3_tool_parser.py#L153) — Extract tool calls incrementally during streaming generation.
  - `content_buffer` — [`L96`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gigachat3_tool_parser.py#L96)
  - `prev_tool_call_arr` — [`L95`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gigachat3_tool_parser.py#L95)
  - `tool_id` — [`L94`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gigachat3_tool_parser.py#L94)
  - `tool_name_sent` — [`L93`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gigachat3_tool_parser.py#L93)
  - `tool_started` — [`L92`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gigachat3_tool_parser.py#L92)
  - `trigger_start` — [`L97`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gigachat3_tool_parser.py#L97)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`DeltaToolCall`](../../openai_api_modules.md#DeltaToolCall), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`DeltaFunctionCall`](../../openai_api_modules.md#DeltaFunctionCall), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolCall`](../../openai_api_modules.md#ToolCall), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`index`](../../openai_api_modules.md#DeltaToolCall.index), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`FunctionCall`](../../openai_api_modules.md#FunctionCall), [`arguments`](../../openai_api_modules.md#FunctionCall.arguments), [`name`](../../openai_api_modules.md#FunctionCall.name), [`id`](../../openai_api_modules.md#DeltaToolCall.id), [`name`](../../openai_api_modules.md#DeltaFunctionCall.name), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`ARGS_REGEX`](gigachat3_tool_parser.md#ARGS_REGEX), [`REGEX_FUNCTION_CALL`](gigachat3_tool_parser.md#REGEX_FUNCTION_CALL), [`NAME_REGEX`](gigachat3_tool_parser.md#NAME_REGEX)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming)

## Module values
- `ARGS_REGEX` — [`L54`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gigachat3_tool_parser.py#L54)
- `NAME_REGEX` — [`L53`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gigachat3_tool_parser.py#L53)
- `REGEX_FUNCTION_CALL` — [`L52`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gigachat3_tool_parser.py#L52)
- `logger` — [`L50`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/gigachat3_tool_parser.py#L50)

