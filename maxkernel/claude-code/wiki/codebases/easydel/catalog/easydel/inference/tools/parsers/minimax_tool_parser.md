---
title: 'Module: easydel/inference/tools/parsers/minimax_tool_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/tools/parsers/minimax_tool_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.tools.parsers.minimax_tool_parser`/
symbols:
  MinimaxToolParser.extract_tool_calls_streaming: MinimaxToolParser#extract_tool_calls_streaming().
  MinimaxToolParser._handle_tool_args_streaming: MinimaxToolParser#_handle_tool_args_streaming().
  MinimaxToolParser.extract_tool_calls: MinimaxToolParser#extract_tool_calls().
  MinimaxToolParser._handle_tool_name_streaming: MinimaxToolParser#_handle_tool_name_streaming().
  MinimaxToolParser.streaming_state: MinimaxToolParser#streaming_state.
  MinimaxToolParser.pending_buffer: MinimaxToolParser#pending_buffer.
  MinimaxToolParser.tool_call_end_token_id: MinimaxToolParser#tool_call_end_token_id.
  MinimaxToolParser._process_buffer: MinimaxToolParser#_process_buffer().
  MinimaxToolParser.tool_call_start_token: MinimaxToolParser#tool_call_start_token.
  MinimaxToolParser._get_current_tool_content: MinimaxToolParser#_get_current_tool_content().
  MinimaxToolParser.tool_call_end_token: MinimaxToolParser#tool_call_end_token.
  MinimaxToolParser._should_buffer_content: MinimaxToolParser#_should_buffer_content().
  MinimaxToolParser._split_content_for_buffering: MinimaxToolParser#_split_content_for_buffering().
  logger: logger.
  MinimaxToolParser.tool_call_start_token_id: MinimaxToolParser#tool_call_start_token_id.
  MinimaxToolParser.in_thinking_tag: MinimaxToolParser#in_thinking_tag.
  MinimaxToolParser.preprocess_model_output: MinimaxToolParser#preprocess_model_output().
  MinimaxToolParser._is_potential_tag_start: MinimaxToolParser#_is_potential_tag_start().
  MinimaxToolParser._is_end_tool_calls: MinimaxToolParser#_is_end_tool_calls().
  MinimaxToolParser._extract_tool_content: MinimaxToolParser#_extract_tool_content().
  MinimaxToolParser._get_current_tool_index: MinimaxToolParser#_get_current_tool_index().
  MinimaxToolParser: MinimaxToolParser#
  MinimaxToolParser._reset_streaming_state: MinimaxToolParser#_reset_streaming_state().
  MinimaxToolParser._advance_to_next_tool: MinimaxToolParser#_advance_to_next_tool().
  MinimaxToolParser._update_thinking_state: MinimaxToolParser#_update_thinking_state().
  MinimaxToolParser._set_current_tool_index: MinimaxToolParser#_set_current_tool_index().
  MinimaxToolParser._get_next_unsent_tool_index: MinimaxToolParser#_get_next_unsent_tool_index().
  MinimaxToolParser._ensure_state_arrays: MinimaxToolParser#_ensure_state_arrays().
  MinimaxToolParser._detect_tools_in_text: MinimaxToolParser#_detect_tools_in_text().
  MinimaxToolParser._find_tool_start_outside_thinking: MinimaxToolParser#_find_tool_start_outside_thinking().
  MinimaxToolParser.__init__: MinimaxToolParser#__init__().
  MinimaxToolParser.thinking_tag_pattern: MinimaxToolParser#thinking_tag_pattern.
  MinimaxToolParser.tool_name_pattern: MinimaxToolParser#tool_name_pattern.
  MinimaxToolParser._clean_delta_braces: MinimaxToolParser#_clean_delta_braces().
  MinimaxToolParser._find_tool_boundaries: MinimaxToolParser#_find_tool_boundaries().
  MinimaxToolParser.tool_call_regex: MinimaxToolParser#tool_call_regex.
  MinimaxToolParser.tool_args_pattern: MinimaxToolParser#tool_args_pattern.
  MinimaxToolParser.remove_tool_calls_from_think: MinimaxToolParser#remove_tool_calls_from_think().
  MinimaxToolParser._clean_duplicate_braces: MinimaxToolParser#_clean_duplicate_braces().
  MinimaxToolParser._extract_tool_args: MinimaxToolParser#_extract_tool_args().
  MinimaxToolParser._extract_content_before_tools: MinimaxToolParser#_extract_content_before_tools().
---
# Module: [`easydel/inference/tools/parsers/minimax_tool_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py)

## Classes
### `MinimaxToolParser`  ·  implements/extends ToolParser
- def: [`easydel/inference/tools/parsers/minimax_tool_parser.py:62`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L62)
- doc: Tool parser for Minimax models.
- signature: `class MinimaxToolParser(ToolParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L97`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L97) — Initialize the MinimaxToolParser.
  - `_advance_to_next_tool(self)` — [`L430`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L430) — Advance to the next tool in the streaming sequence.
  - `_clean_delta_braces(self, delta_text: str)` — [`L195`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L195) — Clean delta text by removing excessive closing braces.
  - `_clean_duplicate_braces(self, args_text: str)` — [`L163`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L163) — Clean duplicate closing braces from arguments text.
  - `_detect_tools_in_text(self, text: str)` — [`L500`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L500) — Detect the number of tools in text by counting name patterns.
  - `_ensure_state_arrays(self, tool_count: int)` — [`L473`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L473) — Ensure state arrays have sufficient capacity for tool_count tools.
  - `_extract_content_before_tools(self, current_text: str, delta_text: str, tool_start: int)` — [`L895`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L895) — Extract content that appears before tool calls.
  - `_extract_tool_args(self, tool_content: str, args_match)` — [`L558`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L558) — Extract tool arguments from tool content.
  - `_extract_tool_content(self, current_text: str, tool_start: int)` — [`L918`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L918) — Extract tool content from current text starting at tool_start.
  - `_find_tool_boundaries(self, text: str)` — [`L512`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L512) — Find the boundaries of tool calls in text.
  - `_find_tool_start_outside_thinking(self, current_text: str)` — [`L867`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L867) — Find the start position of tool calls outside of thinking tags.
  - `_get_current_tool_content(self, text: str, tool_index: int)` — [`L590`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L590) — Get the content of a specific tool by index.
  - `_get_current_tool_index(self)` — [`L445`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L445) — Get the current tool index.
  - `_get_next_unsent_tool_index(self, tool_count: int)` — [`L453`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L453) — Get the index of the next unsent tool.
  - `_handle_tool_args_streaming(self, tool_content: str, tool_count: int)` — [`L672`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L672) — Handle streaming of tool arguments.
  - `_handle_tool_name_streaming(self, tool_content: str, tool_count: int)` — [`L625`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L625) — Handle streaming of tool names.
  - `_is_end_tool_calls(self, current_text: str)` — [`L739`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L739) — Check if tool calls section has ended.
  - `_is_potential_tag_start(self, text: str)` — [`L317`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L317) — Check if text might be the start of a tool call tag.
  - `_process_buffer(self, new_content: str)` — [`L379`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L379) — Process buffered content and return output content.
  - `_reset_streaming_state(self)` — [`L419`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L419) — Reset the streaming state to initial values.
  - `_set_current_tool_index(self, index: int)` — [`L437`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L437) — Set the current tool index.
  - `_should_buffer_content(self, delta_text: str)` — [`L334`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L334) — Determine if content should be buffered for later processing.
  - `_split_content_for_buffering(self, delta_text: str)` — [`L355`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L355) — Split delta text into safe content and potential tag content.
  - `_update_thinking_state(self, text: str)` — [`L304`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L304) — Update the thinking tag state based on text content.
  - `extract_tool_calls(self, model_output: str, request: ChatCompletionRequest)` — [`L219`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L219) — Extract tool calls from model output for non-streaming mode.
  - `extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request: ChatCompletionRequest)` — [`L774`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L774) — Extract tool calls incrementally during streaming.
  - `preprocess_model_output(self, model_output: str)` — [`L140`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L140) — Preprocess model output by removing tool calls from thinking tags.
  - `remove_tool_calls_from_think(match)` — [`L155`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L155) — Strip <tool_calls> blocks from inside a <think> match group.
  - `in_thinking_tag` — [`L126`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L126)
  - `pending_buffer` — [`L125`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L125)
  - `streaming_state` — [`L112`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L112)
  - `thinking_tag_pattern` — [`L121`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L121)
  - `tool_args_pattern` — [`L123`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L123)
  - `tool_call_end_token` — [`L119`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L119)
  - `tool_call_end_token_id` — [`L132`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L132)
  - `tool_call_regex` — [`L120`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L120)
  - `tool_call_start_token` — [`L118`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L118)
  - `tool_call_start_token_id` — [`L131`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L131)
  - `tool_name_pattern` — [`L122`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L122)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ExtractedToolCallInformation`](../../openai_api_modules.md#ExtractedToolCallInformation), [`ChatCompletionRequest`](../../openai_api_modules.md#ChatCompletionRequest), [`ToolParser`](../abstract_tool.md#ToolParser), [`DeltaToolCall`](../../openai_api_modules.md#DeltaToolCall), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`DeltaFunctionCall`](../../openai_api_modules.md#DeltaFunctionCall), [`tool_calls`](../../openai_api_modules.md#ExtractedToolCallInformation.tool_calls), [`content`](../../openai_api_modules.md#ExtractedToolCallInformation.content), [`tools_called`](../../openai_api_modules.md#ExtractedToolCallInformation.tools_called), [`ToolCall`](../../openai_api_modules.md#ToolCall), [`tool_calls`](../../openai_api_modules.md#DeltaMessage.tool_calls), [`index`](../../openai_api_modules.md#DeltaToolCall.index), [`arguments`](../../openai_api_modules.md#DeltaFunctionCall.arguments), [`FunctionCall`](../../openai_api_modules.md#FunctionCall), [`arguments`](../../openai_api_modules.md#FunctionCall.arguments), [`name`](../../openai_api_modules.md#FunctionCall.name), [`id`](../../openai_api_modules.md#DeltaToolCall.id), [`name`](../../openai_api_modules.md#DeltaFunctionCall.name), [`vocab`](../abstract_tool.md#ToolParser.vocab), [`model_tokenizer`](../abstract_tool.md#ToolParser.model_tokenizer), [`__init__`](../abstract_tool.md#ToolParser.__init__), [`extract_intermediate_diff`](../utils.md#extract_intermediate_diff), [`logger`](minimax_tool_parser.md#logger)
- used by: [`ToolParser`](../abstract_tool.md#ToolParser), [`extract_tool_calls`](../abstract_tool.md#ToolParser.extract_tool_calls), [`extract_tool_calls_streaming`](../abstract_tool.md#ToolParser.extract_tool_calls_streaming)

## Module values
- `logger` — [`L58`](../../../../../../../../raw/code/EasyDeL/easydel/inference/tools/parsers/minimax_tool_parser.py#L58)

