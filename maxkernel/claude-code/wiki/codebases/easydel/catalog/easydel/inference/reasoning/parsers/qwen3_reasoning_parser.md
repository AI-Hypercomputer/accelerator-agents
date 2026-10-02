---
title: 'Module: easydel/inference/reasoning/parsers/qwen3_reasoning_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/reasoning/parsers/qwen3_reasoning_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.reasoning.parsers.qwen3_reasoning_parser`/Qwen3ReasoningParser#
symbols:
  Qwen3ReasoningParser.extract_reasoning_streaming: extract_reasoning_streaming().
  Qwen3ReasoningParser.extract_reasoning: extract_reasoning().
  Qwen3ReasoningParser: ''
  Qwen3ReasoningParser.start_token: start_token.
  Qwen3ReasoningParser.end_token: end_token.
---
# Module: [`easydel/inference/reasoning/parsers/qwen3_reasoning_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/qwen3_reasoning_parser.py)

## Classes
### `Qwen3ReasoningParser`  ·  implements/extends BaseThinkingReasoningParser
- def: [`easydel/inference/reasoning/parsers/qwen3_reasoning_parser.py:30`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/qwen3_reasoning_parser.py#L30)
- doc: Reasoning parser for Qwen3 models using &lt;think&gt;...&lt;/think&gt;.
- signature: `class Qwen3ReasoningParser(BaseThinkingReasoningParser):`
- members:
  - `extract_reasoning(self, model_output: str, request=None)` — [`L41`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/qwen3_reasoning_parser.py#L41) — Extract reasoning with strict tag requirements (both tags needed unless prompt-gated).
  - `extract_reasoning_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request=None)` — [`L60`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/qwen3_reasoning_parser.py#L60) — Stream with strict mode: treat as content if no start tag seen early enough.
  - `end_token` — [`L39`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/qwen3_reasoning_parser.py#L39)
  - `start_token` — [`L38`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/qwen3_reasoning_parser.py#L38)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`BaseThinkingReasoningParser`](../basic_parsers.md#BaseThinkingReasoningParser), [`ReasoningParserManager`](../abstract_reasoning.md#ReasoningParserManager), [`extract_reasoning_streaming`](../basic_parsers.md#BaseThinkingReasoningParser.extract_reasoning_streaming), [`start_token`](../basic_parsers.md#BaseThinkingReasoningParser.start_token), [`register_module`](../abstract_reasoning.md#ReasoningParserManager.register_module), [`extract_reasoning`](../basic_parsers.md#BaseThinkingReasoningParser.extract_reasoning), [`end_token`](../basic_parsers.md#BaseThinkingReasoningParser.end_token), [`_start_token_id`](../basic_parsers.md#BaseThinkingReasoningParser._start_token_id), [`_is_prompt_reasoning_active`](../basic_parsers.md#BaseThinkingReasoningParser._is_prompt_reasoning_active)
- used by: [`BaseThinkingReasoningParser`](../basic_parsers.md#BaseThinkingReasoningParser), [`extract_reasoning_streaming`](../basic_parsers.md#BaseThinkingReasoningParser.extract_reasoning_streaming), [`extract_reasoning`](../basic_parsers.md#BaseThinkingReasoningParser.extract_reasoning)

