---
title: 'Module: easydel/inference/reasoning/parsers/mistral_reasoning_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/reasoning/parsers/mistral_reasoning_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.reasoning.parsers.mistral_reasoning_parser`/MistralReasoningParser#
symbols:
  MistralReasoningParser: ''
  MistralReasoningParser.start_token: start_token.
  MistralReasoningParser.end_token: end_token.
---
# Module: [`easydel/inference/reasoning/parsers/mistral_reasoning_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/mistral_reasoning_parser.py)

## Classes
### `MistralReasoningParser`  ·  implements/extends BaseThinkingReasoningParser
- def: [`easydel/inference/reasoning/parsers/mistral_reasoning_parser.py:23`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/mistral_reasoning_parser.py#L23)
- doc: Reasoning parser for Mistral models using \[THINK\]...\[/THINK\] tokens.
- signature: `class MistralReasoningParser(BaseThinkingReasoningParser):`
- members:
  - `end_token` — [`L27`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/mistral_reasoning_parser.py#L27)
  - `start_token` — [`L26`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/mistral_reasoning_parser.py#L26)
- uses (calls/refs, reference-scoped): [`BaseThinkingReasoningParser`](../basic_parsers.md#BaseThinkingReasoningParser), [`ReasoningParserManager`](../abstract_reasoning.md#ReasoningParserManager), [`register_module`](../abstract_reasoning.md#ReasoningParserManager.register_module)
- used by: [`BaseThinkingReasoningParser`](../basic_parsers.md#BaseThinkingReasoningParser)

