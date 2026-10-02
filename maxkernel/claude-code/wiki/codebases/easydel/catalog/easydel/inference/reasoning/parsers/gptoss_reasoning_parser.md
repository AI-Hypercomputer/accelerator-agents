---
title: 'Module: easydel/inference/reasoning/parsers/gptoss_reasoning_parser.py'
type: catalog
provenance: extracted
module: easydel/inference/reasoning/parsers/gptoss_reasoning_parser.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.reasoning.parsers.gptoss_reasoning_parser`/GptOssReasoningParser#
symbols:
  GptOssReasoningParser.extract_reasoning_streaming: extract_reasoning_streaming().
  GptOssReasoningParser._message_token_id: _message_token_id.
  GptOssReasoningParser: ''
  GptOssReasoningParser.CHANNEL_TAG: CHANNEL_TAG.
  GptOssReasoningParser.MESSAGE_TAG: MESSAGE_TAG.
  GptOssReasoningParser._channel_token_id: _channel_token_id.
  GptOssReasoningParser.extract_reasoning: extract_reasoning().
  GptOssReasoningParser._in_reasoning: _in_reasoning.
  GptOssReasoningParser.__init__: __init__().
  GptOssReasoningParser.is_reasoning_end: is_reasoning_end().
  GptOssReasoningParser.extract_content_ids: extract_content_ids().
  GptOssReasoningParser._reasoning_done: _reasoning_done.
---
# Module: [`easydel/inference/reasoning/parsers/gptoss_reasoning_parser.py`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gptoss_reasoning_parser.py)

## Classes
### `GptOssReasoningParser`  ·  implements/extends ReasoningParser
- def: [`easydel/inference/reasoning/parsers/gptoss_reasoning_parser.py:28`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gptoss_reasoning_parser.py#L28)
- doc: Reasoning parser for GptOss models using channel tags.
- signature: `class GptOssReasoningParser(ReasoningParser):`
- members:
  - `__init__(self, tokenizer: AnyTokenizer)` — [`L37`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gptoss_reasoning_parser.py#L37) — Initialize and resolve channel/message tag token IDs from vocabulary.
  - `extract_content_ids(self, input_ids: list[int])` — [`L55`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gptoss_reasoning_parser.py#L55) — Extract token IDs after the <|message|> delimiter.
  - `extract_reasoning(self, model_output: str, request=None)` — [`L62`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gptoss_reasoning_parser.py#L62) — Split output at <|channel|> and <|message|> tags into reasoning and content.
  - `extract_reasoning_streaming(self, previous_text: str, current_text: str, delta_text: str, previous_token_ids: Sequence[int], current_token_ids: Sequence[int], delta_token_ids: Sequence[int], request=None)` — [`L75`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gptoss_reasoning_parser.py#L75) — Stream reasoning/content by tracking <|channel|> and <|message|> state.
  - `is_reasoning_end(self, input_ids: Sequence[int])` — [`L49`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gptoss_reasoning_parser.py#L49) — Check if the <|message|> token is present, signaling end of reasoning.
  - `CHANNEL_TAG` — [`L34`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gptoss_reasoning_parser.py#L34)
  - `MESSAGE_TAG` — [`L35`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gptoss_reasoning_parser.py#L35)
- protocol/private: `_channel_token_id`[`L44`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gptoss_reasoning_parser.py#L44), `_in_reasoning`[`L46`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gptoss_reasoning_parser.py#L46), `_message_token_id`[`L45`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gptoss_reasoning_parser.py#L45), `_reasoning_done`[`L47`](../../../../../../../../raw/code/EasyDeL/easydel/inference/reasoning/parsers/gptoss_reasoning_parser.py#L47)
- uses (calls/refs, reference-scoped): [`DeltaMessage`](../../openai_api_modules.md#DeltaMessage), [`content`](../../openai_api_modules.md#DeltaMessage.content), [`ReasoningParserManager`](../abstract_reasoning.md#ReasoningParserManager), [`ReasoningParser`](../abstract_reasoning.md#ReasoningParser), [`register_module`](../abstract_reasoning.md#ReasoningParserManager.register_module), [`reasoning_content`](../../openai_api_modules.md#DeltaMessage.reasoning_content), [`vocab`](../abstract_reasoning.md#ReasoningParser.vocab), [`__init__`](../abstract_reasoning.md#ReasoningParser.__init__)
- used by: [`ReasoningParser`](../abstract_reasoning.md#ReasoningParser), [`extract_reasoning`](../abstract_reasoning.md#ReasoningParser.extract_reasoning), [`extract_reasoning_streaming`](../abstract_reasoning.md#ReasoningParser.extract_reasoning_streaming), [`extract_content_ids`](../abstract_reasoning.md#ReasoningParser.extract_content_ids), [`is_reasoning_end`](../abstract_reasoning.md#ReasoningParser.is_reasoning_end)

