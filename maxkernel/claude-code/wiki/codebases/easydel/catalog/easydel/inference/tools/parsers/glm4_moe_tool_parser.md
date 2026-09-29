---
title: 'Module: easydel/inference/tools/parsers/glm4_moe_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/glm4_moe_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.glm4_moe_tool_parser`/
symbols:
  Glm4MoeModelToolParser.extract_tool_calls_streaming: Glm4MoeModelToolParser#extract_tool_calls_streaming().
  Glm4MoeModelToolParser.extract_tool_calls: Glm4MoeModelToolParser#extract_tool_calls().
  Glm4MoeModelToolParser: Glm4MoeModelToolParser#
  Glm4MoeModelToolParser._is_string_type: Glm4MoeModelToolParser#_is_string_type().
  Glm4MoeModelToolParser.prev_tool_call_arr: Glm4MoeModelToolParser#prev_tool_call_arr.
  logger: logger.
  Glm4MoeModelToolParser.__init__: Glm4MoeModelToolParser#__init__().
  Glm4MoeModelToolParser.tool_call_start_token_id: Glm4MoeModelToolParser#tool_call_start_token_id.
  Glm4MoeModelToolParser.tool_call_end_token_id: Glm4MoeModelToolParser#tool_call_end_token_id.
  Glm4MoeModelToolParser._buffer: Glm4MoeModelToolParser#_buffer.
  Glm4MoeModelToolParser.streamed_args_for_tool: Glm4MoeModelToolParser#streamed_args_for_tool.
  Glm4MoeModelToolParser.tool_call_end_token: Glm4MoeModelToolParser#tool_call_end_token.
  Glm4MoeModelToolParser.tool_calls_start_token: Glm4MoeModelToolParser#tool_calls_start_token.
  Glm4MoeModelToolParser.func_arg_regex: Glm4MoeModelToolParser#func_arg_regex.
  Glm4MoeModelToolParser.tool_call_start_token: Glm4MoeModelToolParser#tool_call_start_token.
  Glm4MoeModelToolParser.func_call_regex: Glm4MoeModelToolParser#func_call_regex.
  Glm4MoeModelToolParser.func_detail_regex: Glm4MoeModelToolParser#func_detail_regex.
  Glm4MoeModelToolParser._deserialize: Glm4MoeModelToolParser#_deserialize().
---
# Module: [`easydel/inference/tools/parsers/glm4_moe_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/glm4_moe_tool_parser.py)

## Classes
### `Glm4MoeModelToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/glm4_moe_tool_parser.py:55`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/glm4_moe_tool_parser.py#L55)
- doc: Tool parser for GLM-4 MoE (Mixture of Experts) models.
- signature: `class Glm4MoeModelToolParser(ToolParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L87`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/glm4_moe_tool_parser.py#L87) — Initialize the GLM-4 MoE tool parser.
  - `_deserialize(value: str)` — [`L173`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/glm4_moe_tool_parser.py#L173) — Deserialize a string value to its Python type.
  - `_is_string_type(tool_name: str, arg_name: str, tools: list[ToolDefinition] | None)` — [`L152`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/glm4_moe_tool_parser.py#L152) — Check if a parameter is defined as string type.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L120`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/glm4_moe_tool_parser.py#L120) — Extract tool calls from complete GLM-4 model output.
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L222`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/glm4_moe_tool_parser.py#L222) — Handle streaming extraction of GLM-4 tool calls.
  - `func_arg_regex` — [`L112`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/glm4_moe_tool_parser.py#L112)
  - `func_call_regex` — [`L110`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/glm4_moe_tool_parser.py#L110)
  - `func_detail_regex` — [`L111`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/glm4_moe_tool_parser.py#L111)
  - `prev_tool_call_arr` — [`L102`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/glm4_moe_tool_parser.py#L102)
  - `streamed_args_for_tool` — [`L104`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/glm4_moe_tool_parser.py#L104)
  - `tool_call_end_token` — [`L106`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/glm4_moe_tool_parser.py#L106)
  - `tool_call_end_token_id` — [`L117`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/glm4_moe_tool_parser.py#L117)
  - `tool_call_start_token` — [`L105`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/glm4_moe_tool_parser.py#L105)
  - `tool_call_start_token_id` — [`L116`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/glm4_moe_tool_parser.py#L116)
  - `tool_calls_start_token` — [`L108`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/glm4_moe_tool_parser.py#L108)
- protocol/private: `_buffer`[`L118`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/glm4_moe_tool_parser.py#L118)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`DeltaToolCall`](../../openai_api_modules.md#DeltaToolCall), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`DeltaFunctionCall`](../../openai_api_modules.md#DeltaFunctionCall), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolCall`](../../openai_api_modules.md#ToolCall), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`current_tool_id`](../abstract_tool.md#ToolParser.current_tool_id), [`index`](../../openai_api_modules.md#DeltaToolCall.index), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`FunctionCall`](../../openai_api_modules.md#FunctionCall), [`arguments`](../../openai_api_modules.md#FunctionCall.arguments), [`name`](../../openai_api_modules.md#FunctionCall.name), [`id`](../../openai_api_modules.md#DeltaToolCall.id), [`name`](../../openai_api_modules.md#DeltaFunctionCall.name), [`vocab`](../abstract_tool.md#ToolParser.vocab), [`tools`](../../openai_api_modules.md#ChatCompletionRequest.tools), [`model_tokenizer`](../abstract_tool.md#ToolParser.model_tokenizer), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`current_tool_name_sent`](../abstract_tool.md#ToolParser.current_tool_name_sent), [`ToolDefinition`](../../openai_api_modules.md#ToolDefinition), [`id`](../../openai_api_modules.md#ToolCall.id), [`Glm47MoeModelToolParser`](glm47_moe_tool_parser.md#Glm47MoeModelToolParser), [`name`](../../openai_api_modules.md#FunctionDefinition.name), [`parameters`](../../openai_api_modules.md#FunctionDefinition.parameters), [`logger`](glm4_moe_tool_parser.md#logger)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming), [`Glm47MoeModelToolParser`](glm47_moe_tool_parser.md#Glm47MoeModelToolParser), [`__init__`](glm47_moe_tool_parser.md#Glm47MoeModelToolParser.__init__)

## Module values
- `logger` — [`L51`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/glm4_moe_tool_parser.py#L51)

