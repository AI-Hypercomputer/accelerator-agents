---
title: 'Module: easydel/inference/reasoning/reasoning_mixin.py'
type: catalog
provenance: extracted
module: easydel/inference/reasoning/reasoning_mixin.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.reasoning.reasoning_mixin`/
symbols:
  ReasoningMixin.extract_reasoning_streaming: ReasoningMixin#extract_reasoning_streaming().
  ReasoningMixin.initialize_reasoning_parsers: ReasoningMixin#initialize_reasoning_parsers().
  ReasoningMixin.reasoning_parsers: ReasoningMixin#reasoning_parsers.
  ReasoningMixin.extract_reasoning_batch: ReasoningMixin#extract_reasoning_batch().
  ReasoningMixin.get_reasoning_parser_for_model: ReasoningMixin#get_reasoning_parser_for_model().
  logger: logger.
  ReasoningMixin: ReasoningMixin#
---
# Module: [`easydel/inference/reasoning/reasoning_mixin.py`](../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/reasoning_mixin.py)

## Classes
### `ReasoningMixin`
- def: [`easydel/inference/reasoning/reasoning_mixin.py:26`](../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/reasoning_mixin.py#L26)
- doc: Mixin class providing reasoning extraction for inference API servers.
- signature: `class ReasoningMixin:`
- members:
  - `extract_reasoning_batch(self, response_text: str, model_name: str)` — [`L59`](../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/reasoning_mixin.py#L59) — Extract reasoning from a complete response.
  - `extract_reasoning_streaming(self, model_name: str, previous_text: str, current_text: str, delta_text: str, previous_token_ids: list[int] | None = None, current_token_ids: list[int] | None = None, delta_token_ids: list[int] | None = None, request=None)` — [`L75`](../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/reasoning_mixin.py#L75) — Extract reasoning from streaming response chunks.
  - `get_reasoning_parser_for_model(self, model_name: str)` — [`L121`](../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/reasoning_mixin.py#L121) — Get the reasoning parser instance for a specific model.
  - `initialize_reasoning_parsers(self, model_processors: dict[str, tp.Any], reasoning_parser_name: str, enable_reasoning: bool)` — [`L34`](../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/reasoning_mixin.py#L34) — Initialize reasoning parsers for all registered models.
  - `reasoning_parsers` — [`L32`](../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/reasoning_mixin.py#L32)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../openai_api_modules.md#DeltaMessage), [`content`](../openai_api_modules.md#DeltaMessage.content), [`ReasoningParserManager`](abstract_reasoning.md#ReasoningParserManager), [`ReasoningParser`](abstract_reasoning.md#ReasoningParser), [`reasoning_content`](../openai_api_modules.md#DeltaMessage.reasoning_content), [`extract_reasoning`](abstract_reasoning.md#ReasoningParser.extract_reasoning), [`extract_reasoning_streaming`](abstract_reasoning.md#ReasoningParser.extract_reasoning_streaming), [`get_reasoning_parser`](abstract_reasoning.md#ReasoningParserManager.get_reasoning_parser), [`logger`](reasoning_mixin.md#logger)

## Module values
- `logger` — [`L23`](../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/reasoning_mixin.py#L23)

