---
title: 'Module: easydel/inference/tools/parsers/hunyuan_a13b_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/hunyuan_a13b_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.hunyuan_a13b_tool_parser`/
symbols:
  HunyuanA13BToolParser._handle_test_compatibility: HunyuanA13BToolParser#_handle_test_compatibility().
  HunyuanA13BToolParser.extract_tool_calls: HunyuanA13BToolParser#extract_tool_calls().
  HunyuanA13BToolParser.extract_tool_calls_streaming: HunyuanA13BToolParser#extract_tool_calls_streaming().
  HunyuanA13BToolParser._handle_tool_name_streaming: HunyuanA13BToolParser#_handle_tool_name_streaming().
  HunyuanA13BToolParser._handle_tool_args_streaming: HunyuanA13BToolParser#_handle_tool_args_streaming().
  HunyuanA13BToolParser.streaming_state: HunyuanA13BToolParser#streaming_state.
  HunyuanA13BToolParser.streamed_args: HunyuanA13BToolParser#streamed_args.
  HunyuanA13BToolParser.current_tools_sent: HunyuanA13BToolParser#current_tools_sent.
  logger: logger.
  HunyuanA13BToolParser.prev_tool_calls: HunyuanA13BToolParser#prev_tool_calls.
  HunyuanA13BToolParser: HunyuanA13BToolParser#
  HunyuanA13BToolParser.preprocess_model_output: HunyuanA13BToolParser#preprocess_model_output().
  HunyuanA13BToolParser._try_parse_json_tools: HunyuanA13BToolParser#_try_parse_json_tools().
  HunyuanA13BToolParser._ensure_state_arrays: HunyuanA13BToolParser#_ensure_state_arrays().
  HunyuanA13BToolParser.__init__: HunyuanA13BToolParser#__init__().
  HunyuanA13BToolParser.tool_name_reg: HunyuanA13BToolParser#tool_name_reg.
  HunyuanA13BToolParser.bot_string: HunyuanA13BToolParser#bot_string.
  HunyuanA13BToolParser.answer_tool_calls_pattern: HunyuanA13BToolParser#answer_tool_calls_pattern.
  HunyuanA13BToolParser.tool_empty_arg_reg: HunyuanA13BToolParser#tool_empty_arg_reg.
  HunyuanA13BToolParser.tool_non_empty_arg_reg: HunyuanA13BToolParser#tool_non_empty_arg_reg.
---
# Module: [`easydel/inference/tools/parsers/hunyuan_a13b_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hunyuan_a13b_tool_parser.py)

## Classes
### `HunyuanA13BToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/hunyuan_a13b_tool_parser.py:55`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hunyuan_a13b_tool_parser.py#L55)
- doc: Tool parser for Hunyuan A13B model outputs.
- signature: `class HunyuanA13BToolParser(ToolParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L90`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hunyuan_a13b_tool_parser.py#L90) — Initialize the Hunyuan A13B tool parser.
  - `_ensure_state_arrays(self, tool_count: int)` — [`L364`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hunyuan_a13b_tool_parser.py#L364) — Ensure streaming state arrays have sufficient capacity.
  - `_handle_test_compatibility(self, current_text: str)` — [`L317`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hunyuan_a13b_tool_parser.py#L317) — Handle test compatibility for streaming tool calls.
  - `_handle_tool_args_streaming(self, current_text: str, current_idx: int, tool_count: int)` — [`L425`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hunyuan_a13b_tool_parser.py#L425) — Handle streaming of tool arguments.
  - `_handle_tool_name_streaming(self, current_idx: int, tool_count: int, name_matches)` — [`L384`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hunyuan_a13b_tool_parser.py#L384) — Handle streaming of tool names.
  - `_try_parse_json_tools(self, current_text: str)` — [`L300`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hunyuan_a13b_tool_parser.py#L300) — Attempt to parse the current text as a JSON array of tools.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L170`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hunyuan_a13b_tool_parser.py#L170) — Extract tool calls from a complete model output.
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L240`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hunyuan_a13b_tool_parser.py#L240) — Extract tool calls incrementally during streaming generation.
  - `preprocess_model_output(self, model_output: str)` — [`L129`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hunyuan_a13b_tool_parser.py#L129) — Preprocess model output to extract tool calls content.
  - `answer_tool_calls_pattern` — [`L110`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hunyuan_a13b_tool_parser.py#L110)
  - `bot_string` — [`L121`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hunyuan_a13b_tool_parser.py#L121)
  - `current_tools_sent` — [`L106`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hunyuan_a13b_tool_parser.py#L106)
  - `prev_tool_calls` — [`L101`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hunyuan_a13b_tool_parser.py#L101)
  - `streamed_args` — [`L104`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hunyuan_a13b_tool_parser.py#L104)
  - `streaming_state` — [`L123`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hunyuan_a13b_tool_parser.py#L123)
  - `tool_empty_arg_reg` — [`L114`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hunyuan_a13b_tool_parser.py#L114)
  - `tool_name_reg` — [`L112`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hunyuan_a13b_tool_parser.py#L112)
  - `tool_non_empty_arg_reg` — [`L117`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hunyuan_a13b_tool_parser.py#L117)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`DeltaToolCall`](../../openai_api_modules.md#DeltaToolCall), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`DeltaFunctionCall`](../../openai_api_modules.md#DeltaFunctionCall), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolCall`](../../openai_api_modules.md#ToolCall), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`current_tool_id`](../abstract_tool.md#ToolParser.current_tool_id), [`index`](../../openai_api_modules.md#DeltaToolCall.index), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`FunctionCall`](../../openai_api_modules.md#FunctionCall), [`arguments`](../../openai_api_modules.md#FunctionCall.arguments), [`name`](../../openai_api_modules.md#FunctionCall.name), [`id`](../../openai_api_modules.md#DeltaToolCall.id), [`name`](../../openai_api_modules.md#DeltaFunctionCall.name), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`prev_tool_call_arr`](../abstract_tool.md#ToolParser.prev_tool_call_arr), [`current_tool_name_sent`](../abstract_tool.md#ToolParser.current_tool_name_sent), [`id`](../../openai_api_modules.md#ToolCall.id), [`consume_space`](../utils.md#consume_space), [`logger`](hunyuan_a13b_tool_parser.md#logger)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming)

## Module values
- `logger` — [`L51`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/hunyuan_a13b_tool_parser.py#L51)

