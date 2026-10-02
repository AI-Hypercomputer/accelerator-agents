---
title: 'Module: easydel/inference/reasoning/parsers/deepseek_v3_reasoning_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/reasoning/parsers/deepseek_v3_reasoning_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.reasoning.parsers.deepseek_v3_reasoning_parser`/DeepSeekV3ReasoningParser#
symbols:
  DeepSeekV3ReasoningParser.configure_prompt_context: configure_prompt_context().
  DeepSeekV3ReasoningParser._delegate: _delegate.
  DeepSeekV3ReasoningParser._sync_delegate_state: _sync_delegate_state().
  DeepSeekV3ReasoningParser.extract_reasoning_streaming: extract_reasoning_streaming().
  DeepSeekV3ReasoningParser: ''
  DeepSeekV3ReasoningParser.__init__: __init__().
  DeepSeekV3ReasoningParser.is_reasoning_end: is_reasoning_end().
  DeepSeekV3ReasoningParser.extract_content_ids: extract_content_ids().
  DeepSeekV3ReasoningParser.extract_reasoning: extract_reasoning().
---
# Module: [`easydel/inference/reasoning/parsers/deepseek_v3_reasoning_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/deepseek_v3_reasoning_parser.py)

## Classes
### `DeepSeekV3ReasoningParser`  ·  implements/extends ReasoningParser
- def: [`easydel/inference/reasoning/parsers/deepseek_v3_reasoning_parser.py:30`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/deepseek_v3_reasoning_parser.py#L30)
- doc: Conditional reasoning parser: delegates to R1 or Identity based on tokenizer config.
- signature: `class DeepSeekV3ReasoningParser(ReasoningParser):`
- members:
  - `__init__(self, tokenizer)` — [`L40`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/deepseek_v3_reasoning_parser.py#L40) — Initialize and select delegate parser based on tokenizer chat template.
  - `configure_prompt_context(self, prompt_text: str, prompt_token_ids: Sequence[int])` — [`L67`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/deepseek_v3_reasoning_parser.py#L67) — Configure prompt context; may upgrade Identity delegate to R1 if prompt starts reasoning.
  - `extract_content_ids(self, input_ids: list[int])` — [`L87`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/deepseek_v3_reasoning_parser.py#L87) — Delegate content ID extraction to the selected parser.
  - `extract_reasoning(self, model_output: str, request=None)` — [`L92`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/deepseek_v3_reasoning_parser.py#L92) — Delegate batch reasoning extraction to the selected parser.
  - `extract_reasoning_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request=None)` — [`L97`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/deepseek_v3_reasoning_parser.py#L97) — Delegate streaming reasoning extraction to the selected parser.
  - `is_reasoning_end(self, input_ids: Sequence[int])` — [`L82`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/deepseek_v3_reasoning_parser.py#L82) — Delegate reasoning-end check to the selected parser.
- protocol/private: `_delegate`[`L59`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/deepseek_v3_reasoning_parser.py#L59), `_sync_delegate_state`[`L63`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/deepseek_v3_reasoning_parser.py#L63)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`ReasoningParserManager`](../abstract_reasoning.md#ReasoningParserManager), [`ReasoningParser`](../abstract_reasoning.md#ReasoningParser), [`extract_reasoning_streaming`](../basic_parsers.md#BaseThinkingReasoningParser.extract_reasoning_streaming), [`start_token`](../basic_parsers.md#BaseThinkingReasoningParser.start_token), [`register_module`](../abstract_reasoning.md#ReasoningParserManager.register_module), [`extract_reasoning`](../basic_parsers.md#BaseThinkingReasoningParser.extract_reasoning), [`DeepSeekR1ReasoningParser`](deepseek_r1_reasoning_parser.md#DeepSeekR1ReasoningParser), [`IdentityReasoningParser`](identity_reasoning_parser.md#IdentityReasoningParser), [`vocab`](../abstract_reasoning.md#ReasoningParser.vocab), [`configure_prompt_context`](../basic_parsers.md#BaseThinkingReasoningParser.configure_prompt_context), [`configure_prompt_context`](../abstract_reasoning.md#ReasoningParser.configure_prompt_context), [`__init__`](../abstract_reasoning.md#ReasoningParser.__init__), [`extract_content_ids`](../basic_parsers.md#BaseThinkingReasoningParser.extract_content_ids), [`is_reasoning_end`](../basic_parsers.md#BaseThinkingReasoningParser.is_reasoning_end), [`assume_reasoning`](../abstract_reasoning.md#ReasoningParser.assume_reasoning), [`model_tokenizer`](../abstract_reasoning.md#ReasoningParser.model_tokenizer)
- used by: [`ReasoningParser`](../abstract_reasoning.md#ReasoningParser), [`extract_reasoning`](../abstract_reasoning.md#ReasoningParser.extract_reasoning), [`extract_reasoning_streaming`](../abstract_reasoning.md#ReasoningParser.extract_reasoning_streaming), [`extract_content_ids`](../abstract_reasoning.md#ReasoningParser.extract_content_ids), [`is_reasoning_end`](../abstract_reasoning.md#ReasoningParser.is_reasoning_end), [`configure_prompt_context`](../abstract_reasoning.md#ReasoningParser.configure_prompt_context)

