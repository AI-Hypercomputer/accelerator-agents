---
title: 'Module: easydel/inference/reasoning/parsers/minimax_m2_reasoning_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/reasoning/parsers/minimax_m2_reasoning_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.reasoning.parsers.minimax_m2_reasoning_parser`/MiniMaxM2
symbols:
  MiniMaxM2ReasoningParser.extract_reasoning_streaming: ReasoningParser#extract_reasoning_streaming().
  MiniMaxM2ReasoningParser: ReasoningParser#
  MiniMaxM2AppendThinkReasoningParser: AppendThinkReasoningParser#
  MiniMaxM2ReasoningParser.extract_reasoning: ReasoningParser#extract_reasoning().
  MiniMaxM2AppendThinkReasoningParser.extract_reasoning: AppendThinkReasoningParser#extract_reasoning().
  MiniMaxM2ReasoningParser.start_token: ReasoningParser#start_token.
  MiniMaxM2ReasoningParser.end_token: ReasoningParser#end_token.
  MiniMaxM2AppendThinkReasoningParser.start_token: AppendThinkReasoningParser#start_token.
  MiniMaxM2AppendThinkReasoningParser.end_token: AppendThinkReasoningParser#end_token.
---
# Module: [`easydel/inference/reasoning/parsers/minimax_m2_reasoning_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/minimax_m2_reasoning_parser.py)

## Classes
### `MiniMaxM2AppendThinkReasoningParser`  ·  implements/extends BaseThinkingReasoningParser
- def: [`easydel/inference/reasoning/parsers/minimax_m2_reasoning_parser.py:87`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/minimax_m2_reasoning_parser.py#L87)
- doc: Reasoning parser that synthetically prepends &lt;think&gt; to model output.
- signature: `class MiniMaxM2AppendThinkReasoningParser(BaseThinkingReasoningParser):`
- members:
  - `extract_reasoning(self, model_output: str, request=None)` — [`L97`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/minimax_m2_reasoning_parser.py#L97) — Prepend <think> if missing, then delegate to base parser.
  - `end_token` — [`L95`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/minimax_m2_reasoning_parser.py#L95)
  - `start_token` — [`L94`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/minimax_m2_reasoning_parser.py#L94)
- uses (calls/refs, reference-scoped): [`BaseThinkingReasoningParser`](../basic_parsers.md#BaseThinkingReasoningParser), [`ReasoningParserManager`](../abstract_reasoning.md#ReasoningParserManager), [`start_token`](../basic_parsers.md#BaseThinkingReasoningParser.start_token), [`register_module`](../abstract_reasoning.md#ReasoningParserManager.register_module), [`extract_reasoning`](../basic_parsers.md#BaseThinkingReasoningParser.extract_reasoning)
- used by: [`BaseThinkingReasoningParser`](../basic_parsers.md#BaseThinkingReasoningParser), [`extract_reasoning`](../basic_parsers.md#BaseThinkingReasoningParser.extract_reasoning)

### `MiniMaxM2ReasoningParser`  ·  implements/extends BaseThinkingReasoningParser
- def: [`easydel/inference/reasoning/parsers/minimax_m2_reasoning_parser.py:30`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/minimax_m2_reasoning_parser.py#L30)
- doc: Asymmetric reasoning parser: no start token, only &lt;/think&gt; end token.
- signature: `class MiniMaxM2ReasoningParser(BaseThinkingReasoningParser):`
- members:
  - `extract_reasoning(self, model_output: str, request=None)` — [`L39`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/minimax_m2_reasoning_parser.py#L39) — Extract reasoning by splitting at </think> (no start token required).
  - `extract_reasoning_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request=None)` — [`L52`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/minimax_m2_reasoning_parser.py#L52) — Stream reasoning (all text before </think>) and content (after).
  - `end_token` — [`L37`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/minimax_m2_reasoning_parser.py#L37)
  - `start_token` — [`L36`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/minimax_m2_reasoning_parser.py#L36)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`BaseThinkingReasoningParser`](../basic_parsers.md#BaseThinkingReasoningParser), [`ReasoningParserManager`](../abstract_reasoning.md#ReasoningParserManager), [`start_token`](../basic_parsers.md#BaseThinkingReasoningParser.start_token), [`register_module`](../abstract_reasoning.md#ReasoningParserManager.register_module), [`reasoning_content`](../../openai_api_modules.md#DeltaMessage.reasoning_content), [`end_token`](../basic_parsers.md#BaseThinkingReasoningParser.end_token)
- used by: [`BaseThinkingReasoningParser`](../basic_parsers.md#BaseThinkingReasoningParser), [`extract_reasoning_streaming`](../basic_parsers.md#BaseThinkingReasoningParser.extract_reasoning_streaming), [`extract_reasoning`](../basic_parsers.md#BaseThinkingReasoningParser.extract_reasoning)

