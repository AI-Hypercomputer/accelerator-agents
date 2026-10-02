---
title: 'Module: easydel/inference/reasoning/parsers/identity_reasoning_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/reasoning/parsers/identity_reasoning_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.reasoning.parsers.identity_reasoning_parser`/IdentityReasoningParser#
symbols:
  IdentityReasoningParser: ''
  IdentityReasoningParser.extract_reasoning_streaming: extract_reasoning_streaming().
  IdentityReasoningParser.is_reasoning_end: is_reasoning_end().
  IdentityReasoningParser.extract_content_ids: extract_content_ids().
  IdentityReasoningParser.extract_reasoning: extract_reasoning().
---
# Module: [`easydel/inference/reasoning/parsers/identity_reasoning_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/identity_reasoning_parser.py)

## Classes
### `IdentityReasoningParser`  ·  implements/extends ReasoningParser
- def: [`easydel/inference/reasoning/parsers/identity_reasoning_parser.py:23`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/identity_reasoning_parser.py#L23)
- doc: Pass-through parser: no reasoning extraction, all text is content.
- signature: `class IdentityReasoningParser(ReasoningParser):`
- members:
  - `extract_content_ids(self, input_ids: list[int])` — [`L30`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/identity_reasoning_parser.py#L30) — Return all token IDs unchanged (no reasoning tokens to strip).
  - `extract_reasoning(self, model_output: str, request=None)` — [`L34`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/identity_reasoning_parser.py#L34) — Return (None, full_output) -- all text is content.
  - `extract_reasoning_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request=None)` — [`L38`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/identity_reasoning_parser.py#L38) — Return delta as content (no reasoning extraction).
  - `is_reasoning_end(self, input_ids: Sequence[int])` — [`L26`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/identity_reasoning_parser.py#L26) — Always returns True (no reasoning to end).
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`ReasoningParserManager`](../abstract_reasoning.md#ReasoningParserManager), [`ReasoningParser`](../abstract_reasoning.md#ReasoningParser), [`register_module`](../abstract_reasoning.md#ReasoningParserManager.register_module)
- used by: [`ReasoningParser`](../abstract_reasoning.md#ReasoningParser), [`configure_prompt_context`](deepseek_v3_reasoning_parser.md#DeepSeekV3ReasoningParser.configure_prompt_context), [`extract_reasoning`](../abstract_reasoning.md#ReasoningParser.extract_reasoning), [`extract_reasoning_streaming`](../abstract_reasoning.md#ReasoningParser.extract_reasoning_streaming), [`_delegate`](deepseek_v3_reasoning_parser.md#DeepSeekV3ReasoningParser._delegate), [`extract_content_ids`](../abstract_reasoning.md#ReasoningParser.extract_content_ids), [`is_reasoning_end`](../abstract_reasoning.md#ReasoningParser.is_reasoning_end)

