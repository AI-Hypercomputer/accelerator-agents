---
title: 'Module: easydel/inference/reasoning/parsers/granite_reasoning_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/reasoning/parsers/granite_reasoning_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.reasoning.parsers.granite_reasoning_parser`/
symbols:
  GraniteReasoningParser.extract_reasoning_streaming: GraniteReasoningParser#extract_reasoning_streaming().
  GraniteReasoningParser.__init__: GraniteReasoningParser#__init__().
  _RESPONSE_STARTERS: _RESPONSE_STARTERS.
  GraniteReasoningParser.is_reasoning_end: GraniteReasoningParser#is_reasoning_end().
  GraniteReasoningParser: GraniteReasoningParser#
  GraniteReasoningParser._response_starters: GraniteReasoningParser#_response_starters.
  GraniteReasoningParser._in_reasoning: GraniteReasoningParser#_in_reasoning.
  GraniteReasoningParser._thought_starters: GraniteReasoningParser#_thought_starters.
  GraniteReasoningParser.extract_reasoning: GraniteReasoningParser#extract_reasoning().
  _THOUGHT_STARTERS: _THOUGHT_STARTERS.
  GraniteReasoningParser._reasoning_done: GraniteReasoningParser#_reasoning_done.
  GraniteReasoningParser._regex: GraniteReasoningParser#_regex.
  GraniteReasoningParser.extract_content_ids: GraniteReasoningParser#extract_content_ids().
---
# Module: [`easydel/inference/reasoning/parsers/granite_reasoning_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/granite_reasoning_parser.py)

## Classes
### `GraniteReasoningParser`  ·  implements/extends ReasoningParser
- def: [`easydel/inference/reasoning/parsers/granite_reasoning_parser.py:42`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/granite_reasoning_parser.py#L42)
- doc: Reasoning parser for Granite models using text delimiters.
- signature: `class GraniteReasoningParser(ReasoningParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L45`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/granite_reasoning_parser.py#L45) — Initialize with tokenizer and compile regex for thought/response delimiters.
  - `extract_content_ids(self, input_ids: list[int])` — [`L69`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/granite_reasoning_parser.py#L69) — Return all token IDs (Granite uses text-level delimiter splitting).
  - `extract_reasoning(self, model_output: str, request=None)` — [`L73`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/granite_reasoning_parser.py#L73) — Extract reasoning and content using thought/response text delimiters.
  - `extract_reasoning_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request=None)` — [`L82`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/granite_reasoning_parser.py#L82) — Stream reasoning/content by tracking thought/response text delimiters.
  - `is_reasoning_end(self, input_ids: Sequence[int])` — [`L64`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/granite_reasoning_parser.py#L64) — Check if a response starter phrase appears in decoded text.
- protocol/private: `_in_reasoning`[`L61`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/granite_reasoning_parser.py#L61), `_reasoning_done`[`L62`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/granite_reasoning_parser.py#L62), `_regex`[`L55`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/granite_reasoning_parser.py#L55), `_response_starters`[`L60`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/granite_reasoning_parser.py#L60), `_thought_starters`[`L59`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/granite_reasoning_parser.py#L59)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`ReasoningParser`](../abstract_reasoning.md#ReasoningParser), [`reasoning_content`](../../openai_api_modules.md#DeltaMessage.reasoning_content), [`_RESPONSE_STARTERS`](granite_reasoning_parser.md#_RESPONSE_STARTERS), [`__init__`](../abstract_reasoning.md#ReasoningParser.__init__), [`model_tokenizer`](../abstract_reasoning.md#ReasoningParser.model_tokenizer), [`_THOUGHT_STARTERS`](granite_reasoning_parser.md#_THOUGHT_STARTERS)
- used by: [`ReasoningParser`](../abstract_reasoning.md#ReasoningParser), [`extract_reasoning`](../abstract_reasoning.md#ReasoningParser.extract_reasoning), [`extract_reasoning_streaming`](../abstract_reasoning.md#ReasoningParser.extract_reasoning_streaming), [`extract_content_ids`](../abstract_reasoning.md#ReasoningParser.extract_content_ids), [`is_reasoning_end`](../abstract_reasoning.md#ReasoningParser.is_reasoning_end)

## Module values
- `_RESPONSE_STARTERS` — [`L35`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/granite_reasoning_parser.py#L35)
- `_THOUGHT_STARTERS` — [`L31`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/granite_reasoning_parser.py#L31)

