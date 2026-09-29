---
title: 'Module: easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.reasoning.parsers.gemma4_reasoning_parser`/
symbols:
  Gemma4ReasoningParser.extract_reasoning_streaming: Gemma4ReasoningParser#extract_reasoning_streaming().
  Gemma4ReasoningParser.is_reasoning_end: Gemma4ReasoningParser#is_reasoning_end().
  Gemma4ReasoningParser.extract_reasoning: Gemma4ReasoningParser#extract_reasoning().
  CHANNEL_END: CHANNEL_END.
  Gemma4ReasoningParser._parse_channels: Gemma4ReasoningParser#_parse_channels().
  Gemma4ReasoningParser._reasoning_text: Gemma4ReasoningParser#_reasoning_text.
  Gemma4ReasoningParser._channel_pattern: Gemma4ReasoningParser#_channel_pattern.
  Gemma4ReasoningParser.start_token_id: Gemma4ReasoningParser#start_token_id().
  Gemma4ReasoningParser.end_token_id: Gemma4ReasoningParser#end_token_id().
  _THOUGHT_PREFIX: _THOUGHT_PREFIX.
  Gemma4ReasoningParser: Gemma4ReasoningParser#
  CHANNEL_START: CHANNEL_START.
  Gemma4ReasoningParser._prefix_stripped: Gemma4ReasoningParser#_prefix_stripped.
  Gemma4ReasoningParser._tool_call_token_id: Gemma4ReasoningParser#_tool_call_token_id().
  Gemma4ReasoningParser._new_turn_token_id: Gemma4ReasoningParser#_new_turn_token_id().
  Gemma4ReasoningParser._tool_response_token_id: Gemma4ReasoningParser#_tool_response_token_id().
  _strip_thought_label: _strip_thought_label().
  Gemma4ReasoningParser.start_token: Gemma4ReasoningParser#start_token.
  Gemma4ReasoningParser.end_token: Gemma4ReasoningParser#end_token.
  Gemma4ReasoningParser.__init__: Gemma4ReasoningParser#__init__().
  _suffix: _suffix().
  Gemma4ReasoningParser._REASONING_CHANNELS: Gemma4ReasoningParser#_REASONING_CHANNELS.
  Gemma4ReasoningParser.adjust_request: Gemma4ReasoningParser#adjust_request().
  Gemma4ReasoningParser.extract_content_ids: Gemma4ReasoningParser#extract_content_ids().
---
# Module: [`easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py)

## Classes
### `Gemma4ReasoningParser`  ·  implements/extends ReasoningParser
- def: [`easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py:45`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py#L45)
- doc: Extract reasoning content from Gemma4 model outputs.
- signature: `class Gemma4ReasoningParser(ReasoningParser):`
- members:
  - `_parse_channels(self, text: str)` — [`L193`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py#L193) — Split text into reasoning and content by channel markers.
  - `adjust_request(self, request)` — [`L89`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py#L89) — Disable special-token stripping to preserve channel markers.
  - `end_token_id(self)` — [`L74`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py#L74)
  - `extract_content_ids(self, input_ids: list[int])` — [`L121`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py#L121)
  - `extract_reasoning(self, model_output: str, request=None)` — [`L124`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py#L124) — Extract reasoning and content from complete model output.
  - `extract_reasoning_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request=None)` — [`L140`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py#L140) — Extract streaming reasoning, stripping ``thought\n`` prefix.
  - `is_reasoning_end(self, input_ids: Sequence[int])` — [`L94`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py#L94) — Check whether reasoning has ended by scanning token IDs backwards.
  - `start_token_id(self)` — [`L70`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py#L70)
  - `end_token` — [`L58`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py#L58)
  - `start_token` — [`L57`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py#L57)
- protocol/private: `_REASONING_CHANNELS`[`L191`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py#L191), `__init__`[`L60`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py#L60), `_channel_pattern`[`L64`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py#L64), `_new_turn_token_id`[`L82`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py#L82), `_prefix_stripped`[`L63`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py#L63), `_reasoning_text`[`L62`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py#L62), `_tool_call_token_id`[`L78`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py#L78), `_tool_response_token_id`[`L86`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py#L86)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`ReasoningParser`](../abstract_reasoning.md#ReasoningParser), [`reasoning_content`](../../openai_api_modules.md#DeltaMessage.reasoning_content), [`vocab`](../abstract_reasoning.md#ReasoningParser.vocab), [`CHANNEL_END`](gemma4_reasoning_parser.md#CHANNEL_END), [`_THOUGHT_PREFIX`](gemma4_reasoning_parser.md#_THOUGHT_PREFIX), [`__init__`](../abstract_reasoning.md#ReasoningParser.__init__), [`CHANNEL_START`](gemma4_reasoning_parser.md#CHANNEL_START), [`_strip_thought_label`](gemma4_reasoning_parser.md#_strip_thought_label), [`_suffix`](gemma4_reasoning_parser.md#_suffix)
- used by: [`ReasoningParser`](../abstract_reasoning.md#ReasoningParser), [`extract_reasoning`](../abstract_reasoning.md#ReasoningParser.extract_reasoning), [`extract_reasoning_streaming`](../abstract_reasoning.md#ReasoningParser.extract_reasoning_streaming), [`extract_content_ids`](../abstract_reasoning.md#ReasoningParser.extract_content_ids), [`is_reasoning_end`](../abstract_reasoning.md#ReasoningParser.is_reasoning_end)

## Functions
- `_strip_thought_label(text: str)` — [`L219`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py#L219) — Remove the ``thought\n`` role label from the beginning of text.
- `_suffix(previous: str | None, current: str | None)` — [`L226`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py#L226) — Compute the incremental text delta between two cumulative strings.

## Module values
- `CHANNEL_END` — [`L41`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py#L41)
- `CHANNEL_START` — [`L40`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py#L40)
- `_THOUGHT_PREFIX` — [`L38`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gemma4_reasoning_parser.py#L38)

