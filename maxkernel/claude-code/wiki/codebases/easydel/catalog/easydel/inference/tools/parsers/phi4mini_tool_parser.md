---
title: 'Module: easydel/inference/tools/parsers/phi4mini_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/phi4mini_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.phi4mini_tool_parser`/
symbols:
  Phi4MiniJsonToolParser.extract_tool_calls: Phi4MiniJsonToolParser#extract_tool_calls().
  logger: logger.
  Phi4MiniJsonToolParser.extract_tool_calls_streaming: Phi4MiniJsonToolParser#extract_tool_calls_streaming().
  Phi4MiniJsonToolParser: Phi4MiniJsonToolParser#
  Phi4MiniJsonToolParser.__init__: Phi4MiniJsonToolParser#__init__().
  Phi4MiniJsonToolParser.prev_tool_call_arr: Phi4MiniJsonToolParser#prev_tool_call_arr.
  Phi4MiniJsonToolParser.current_tool_id: Phi4MiniJsonToolParser#current_tool_id.
  Phi4MiniJsonToolParser.current_tool_name_sent: Phi4MiniJsonToolParser#current_tool_name_sent.
  Phi4MiniJsonToolParser.streamed_args_for_tool: Phi4MiniJsonToolParser#streamed_args_for_tool.
  Phi4MiniJsonToolParser.bot_token: Phi4MiniJsonToolParser#bot_token.
---
# Module: [`easydel/inference/tools/parsers/phi4mini_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/phi4mini_tool_parser.py)

## Classes
### `Phi4MiniJsonToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/phi4mini_tool_parser.py:47`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/phi4mini_tool_parser.py#L47)
- doc: Tool call parser for Phi-4-mini models.
- signature: `class Phi4MiniJsonToolParser(ToolParser):`
- members:
  - `__init__(self, tokenizer: PreTrainedTokenizerBase)` — [`L77`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/phi4mini_tool_parser.py#L77) — Initialize the Phi-4 Mini JSON tool parser.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L94`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/phi4mini_tool_parser.py#L94) — Extract tool calls from a complete model response.
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L152`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/phi4mini_tool_parser.py#L152) — Extract tool calls from streaming model output.
  - `bot_token` — [`L92`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/phi4mini_tool_parser.py#L92)
  - `current_tool_id` — [`L89`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/phi4mini_tool_parser.py#L89)
  - `current_tool_name_sent` — [`L90`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/phi4mini_tool_parser.py#L90)
  - `prev_tool_call_arr` — [`L88`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/phi4mini_tool_parser.py#L88)
  - `streamed_args_for_tool` — [`L91`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/phi4mini_tool_parser.py#L91)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolCall`](../../openai_api_modules.md#ToolCall), [`FunctionCall`](../../openai_api_modules.md#FunctionCall), [`arguments`](../../openai_api_modules.md#FunctionCall.arguments), [`name`](../../openai_api_modules.md#FunctionCall.name), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`id`](../../openai_api_modules.md#ToolCall.id), [`logger`](phi4mini_tool_parser.md#logger)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming)

## Module values
- `logger` — [`L43`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/phi4mini_tool_parser.py#L43)

