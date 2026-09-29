---
title: 'Module: easydel/inference/reasoning/parsers/seedoss_reasoning_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/reasoning/parsers/seedoss_reasoning_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.reasoning.parsers.seedoss_reasoning_parser`/SeedOSSReasoningParser#
symbols:
  SeedOSSReasoningParser: ''
  SeedOSSReasoningParser.start_token: start_token.
  SeedOSSReasoningParser.end_token: end_token.
---
# Module: [`easydel/inference/reasoning/parsers/seedoss_reasoning_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/seedoss_reasoning_parser.py)

## Classes
### `SeedOSSReasoningParser`  ·  implements/extends BaseThinkingReasoningParser
- def: [`easydel/inference/reasoning/parsers/seedoss_reasoning_parser.py:22`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/seedoss_reasoning_parser.py#L22)
- doc: Reasoning parser for Seed OSS models using &lt;think&gt;...&lt;/think&gt; tags.
- signature: `class SeedOSSReasoningParser(BaseThinkingReasoningParser):`
- members:
  - `end_token` — [`L26`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/seedoss_reasoning_parser.py#L26)
  - `start_token` — [`L25`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/seedoss_reasoning_parser.py#L25)
- uses (calls/refs, reference-scoped): [`BaseThinkingReasoningParser`](../basic_parsers.md#BaseThinkingReasoningParser), [`ReasoningParserManager`](../abstract_reasoning.md#ReasoningParserManager), [`register_module`](../abstract_reasoning.md#ReasoningParserManager.register_module)
- used by: [`BaseThinkingReasoningParser`](../basic_parsers.md#BaseThinkingReasoningParser)

