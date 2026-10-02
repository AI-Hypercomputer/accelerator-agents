---
title: 'Module: easydel/inference/tools/tool_calling_mixin.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/tool_calling_mixin.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.tool_calling_mixin`/
symbols:
  ToolCallingMixin.extract_tool_calls_batch: ToolCallingMixin#extract_tool_calls_batch().
  ToolCallingMixin.extract_tool_calls_streaming: ToolCallingMixin#extract_tool_calls_streaming().
  ToolCallingMixin.initialize_tool_parsers: ToolCallingMixin#initialize_tool_parsers().
  ToolCallingMixin.tool_parsers: ToolCallingMixin#tool_parsers.
  ToolCallingMixin.create_tools_response: ToolCallingMixin#create_tools_response().
  logger: logger.
  ToolCallingMixin.get_tool_parser_for_model: ToolCallingMixin#get_tool_parser_for_model().
  ToolCallingMixin: ToolCallingMixin#
  ToolCallingMixin.create_tool_execution_placeholder: ToolCallingMixin#create_tool_execution_placeholder().
---
# Module: [`easydel/inference/tools/tool_calling_mixin.py`](../../../../../../../raw/code/EasyDeL/easydel/inference/tools/tool_calling_mixin.py)

## Classes
### `ToolCallingMixin`
- def: [`easydel/inference/tools/tool_calling_mixin.py:58`](../../../../../../../raw/code/EasyDeL/easydel/inference/tools/tool_calling_mixin.py#L58)
- doc: Mixin class providing tool calling functionality for inference API servers.
- signature: `class ToolCallingMixin:`
- members:
  - `create_tool_execution_placeholder(self)` — [`L409`](../../../../../../../raw/code/EasyDeL/easydel/inference/tools/tool_calling_mixin.py#L409) — Create a placeholder response for tool execution endpoints.
  - `create_tools_response(self, model_names: list[str])` — [`L344`](../../../../../../../raw/code/EasyDeL/easydel/inference/tools/tool_calling_mixin.py#L344) — Create a standardized tools response for API listing endpoints.
  - `extract_tool_calls_batch(self, response_text: str, request: ChatCompletionRequest, model_name: str)` — [`L160`](../../../../../../../raw/code/EasyDeL/easydel/inference/tools/tool_calling_mixin.py#L160) — Extract tool calls from a complete (non-streaming) response.
  - `extract_tool_calls_streaming(self, model_name: str, previous_text: str, current_text: str, delta_text: str, previous_token_ids: list[int] | None = None, current_token_ids: list[int] | None = None, delta_token_ids: list[int] | None = None, request: ChatCompletionRequest | None = None)` — [`L223`](../../../../../../../raw/code/EasyDeL/easydel/inference/tools/tool_calling_mixin.py#L223) — Extract tool calls from streaming response chunks.
  - `get_tool_parser_for_model(self, model_name: str)` — [`L321`](../../../../../../../raw/code/EasyDeL/easydel/inference/tools/tool_calling_mixin.py#L321) — Get the tool parser instance for a specific model.
  - `initialize_tool_parsers(self, model_processors: dict[str, tp.Any], tool_parser_name: str | dict[str, str | None], enable_function_calling: bool)` — [`L96`](../../../../../../../raw/code/EasyDeL/easydel/inference/tools/tool_calling_mixin.py#L96) — Initialize tool parsers for all registered models.
  - `tool_parsers` — [`L94`](../../../../../../../raw/code/EasyDeL/easydel/inference/tools/tool_calling_mixin.py#L94)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../openai_api_modules.md#DeltaMessage), [`ChatCompletionRequest`](../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](abstract_tool.md#ToolParser), [`content`](../openai_api_modules.md#DeltaMessage.content), [`tool_calls`](../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolParserManager`](abstract_tool.md#ToolParserManager), [`extract_tool_calls`](abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](abstract_tool.md#ToolParser.extract_tool_calls_streaming), [`ChatMessage`](../openai_api_modules.md#ChatMessage), [`content`](../openai_api_modules.md#ChatMessage.content), [`role`](../openai_api_modules.md#ChatMessage.role), [`tool_parsers`](abstract_tool.md#ToolParserManager.tool_parsers), [`get_tool_parser`](abstract_tool.md#ToolParserManager.get_tool_parser), [`logger`](tool_calling_mixin.md#logger), [`tool_calls`](../openai_api_modules.md#ChatMessage.tool_calls)

## Module values
- `logger` — [`L55`](../../../../../../../raw/code/EasyDeL/easydel/inference/tools/tool_calling_mixin.py#L55)

