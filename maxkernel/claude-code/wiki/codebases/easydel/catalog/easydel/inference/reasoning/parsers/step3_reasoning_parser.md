---
title: 'Module: easydel/inference/reasoning/parsers/step3_reasoning_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/reasoning/parsers/step3_reasoning_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.reasoning.parsers.step3_reasoning_parser`/Step3ReasoningParser#
symbols:
  Step3ReasoningParser: ''
  Step3ReasoningParser.extract_reasoning_streaming: extract_reasoning_streaming().
  Step3ReasoningParser.extract_reasoning: extract_reasoning().
  Step3ReasoningParser.start_token: start_token.
  Step3ReasoningParser.end_token: end_token.
---
# Module: [`easydel/inference/reasoning/parsers/step3_reasoning_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/step3_reasoning_parser.py)

## Classes
### `Step3ReasoningParser`  ·  implements/extends BaseThinkingReasoningParser
- def: [`easydel/inference/reasoning/parsers/step3_reasoning_parser.py:28`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/step3_reasoning_parser.py#L28)
- doc: Reasoning parser for Step3 models. Uses only &lt;/think&gt; end token.
- signature: `class Step3ReasoningParser(BaseThinkingReasoningParser):`
- members:
  - `extract_reasoning(self, model_output: str, request=None)` — [`L34`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/step3_reasoning_parser.py#L34) — Extract reasoning by splitting at </think> (everything before is reasoning).
  - `extract_reasoning_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request=None)` — [`L47`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/step3_reasoning_parser.py#L47) — Stream all text as reasoning until </think> appears, then switch to content.
  - `end_token` — [`L32`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/step3_reasoning_parser.py#L32)
  - `start_token` — [`L31`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/step3_reasoning_parser.py#L31)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`BaseThinkingReasoningParser`](../basic_parsers.md#BaseThinkingReasoningParser), [`ReasoningParserManager`](../abstract_reasoning.md#ReasoningParserManager), [`start_token`](../basic_parsers.md#BaseThinkingReasoningParser.start_token), [`register_module`](../abstract_reasoning.md#ReasoningParserManager.register_module), [`reasoning_content`](../../openai_api_modules.md#DeltaMessage.reasoning_content), [`end_token`](../basic_parsers.md#BaseThinkingReasoningParser.end_token), [`Step3p5ReasoningParser`](step3p5_reasoning_parser.md#Step3p5ReasoningParser)
- used by: [`BaseThinkingReasoningParser`](../basic_parsers.md#BaseThinkingReasoningParser), [`extract_reasoning_streaming`](../basic_parsers.md#BaseThinkingReasoningParser.extract_reasoning_streaming), [`extract_reasoning`](../basic_parsers.md#BaseThinkingReasoningParser.extract_reasoning), [`Step3p5ReasoningParser`](step3p5_reasoning_parser.md#Step3p5ReasoningParser)

