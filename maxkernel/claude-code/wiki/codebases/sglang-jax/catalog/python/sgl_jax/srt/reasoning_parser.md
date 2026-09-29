---
title: 'Module: python/sgl_jax/srt/reasoning_parser.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/reasoning_parser.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.reasoning_parser`/
symbols:
  StreamingParseResult: StreamingParseResult#
  BaseReasoningFormatDetector.parse_streaming_increment: BaseReasoningFormatDetector#parse_streaming_increment().
  BaseReasoningFormatDetector: BaseReasoningFormatDetector#
  ReasoningParser.DetectorMap: ReasoningParser#DetectorMap.
  Gemma4Detector._in_reasoning: Gemma4Detector#_in_reasoning.
  Gemma4Detector._buffer: Gemma4Detector#_buffer.
  BaseReasoningFormatDetector.detect_and_parse: BaseReasoningFormatDetector#detect_and_parse().
  Gemma4Detector.parse_streaming_increment: Gemma4Detector#parse_streaming_increment().
  BaseReasoningFormatDetector.think_end_token: BaseReasoningFormatDetector#think_end_token.
  ReasoningParser.parse_non_stream: ReasoningParser#parse_non_stream().
  ReasoningParser.parse_stream_chunk: ReasoningParser#parse_stream_chunk().
  Gemma4Detector.stripped_think_start: Gemma4Detector#stripped_think_start.
  BaseReasoningFormatDetector.tool_start_token: BaseReasoningFormatDetector#tool_start_token.
  BaseReasoningFormatDetector._in_reasoning: BaseReasoningFormatDetector#_in_reasoning.
  BaseReasoningFormatDetector.think_start_token: BaseReasoningFormatDetector#think_start_token.
  BaseReasoningFormatDetector._buffer: BaseReasoningFormatDetector#_buffer.
  BaseReasoningFormatDetector.__init__: BaseReasoningFormatDetector#__init__().
  Qwen3Detector: Qwen3Detector#
  DeepSeekR1Detector: DeepSeekR1Detector#
  KimiDetector: KimiDetector#
  Glm45Detector: Glm45Detector#
  Gemma4Detector: Gemma4Detector#
  ReasoningParser: ReasoningParser#
  DeepSeekR1Detector.__init__: DeepSeekR1Detector#__init__().
  Qwen3Detector.__init__: Qwen3Detector#__init__().
  KimiDetector.__init__: KimiDetector#__init__().
  Glm45Detector.__init__: Glm45Detector#__init__().
  Gemma4Detector.__init__: Gemma4Detector#__init__().
  Gemma4Detector.detect_and_parse: Gemma4Detector#detect_and_parse().
  ReasoningParser.__init__: ReasoningParser#__init__().
  StreamingParseResult.normal_text: StreamingParseResult#normal_text.
  StreamingParseResult.reasoning_text: StreamingParseResult#reasoning_text.
  BaseReasoningFormatDetector.stream_reasoning: BaseReasoningFormatDetector#stream_reasoning.
  BaseReasoningFormatDetector.stripped_think_start: BaseReasoningFormatDetector#stripped_think_start.
  ReasoningParser.detector: ReasoningParser#detector.
  StreamingParseResult.__init__: StreamingParseResult#__init__().
---
# Module: [`python/sgl_jax/srt/reasoning_parser.py`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py)

## Classes
### `BaseReasoningFormatDetector`
- def: [`python/sgl_jax/srt/reasoning_parser.py:9`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L9)
- doc: Base class providing two sets of interfaces: one-time and streaming incremental.
- signature: `class BaseReasoningFormatDetector:`
- members:
  - `detect_and_parse(self, text: str)` — [`L29`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L29) — One-time parsing: Detects and parses reasoning sections in the provided text.
  - `parse_streaming_increment(self, new_text: str)` — [`L59`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L59) — Streaming incremental parsing for reasoning content.
  - `stream_reasoning` — [`L23`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L23)
  - `stripped_think_start` — [`L27`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L27)
  - `think_end_token` — [`L21`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L21)
  - `think_start_token` — [`L20`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L20)
  - `tool_start_token` — [`L24`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L24)
- protocol/private: `__init__`[`L12`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L12), `_buffer`[`L26`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L26), `_in_reasoning`[`L22`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L22)
- uses (calls/refs, reference-scoped): [`StreamingParseResult`](reasoning_parser.md#StreamingParseResult), [`parse_streaming_increment`](reasoning_parser.md#Gemma4Detector.parse_streaming_increment), [`Qwen3Detector`](reasoning_parser.md#Qwen3Detector), [`DeepSeekR1Detector`](reasoning_parser.md#DeepSeekR1Detector), [`Gemma4Detector`](reasoning_parser.md#Gemma4Detector), [`Glm45Detector`](reasoning_parser.md#Glm45Detector), [`KimiDetector`](reasoning_parser.md#KimiDetector), [`detect_and_parse`](reasoning_parser.md#Gemma4Detector.detect_and_parse)
- used by: [`DetectorMap`](reasoning_parser.md#ReasoningParser.DetectorMap), [`_in_reasoning`](reasoning_parser.md#Gemma4Detector._in_reasoning), [`_buffer`](reasoning_parser.md#Gemma4Detector._buffer), [`parse_streaming_increment`](reasoning_parser.md#Gemma4Detector.parse_streaming_increment), [`parse_non_stream`](reasoning_parser.md#ReasoningParser.parse_non_stream), [`parse_stream_chunk`](reasoning_parser.md#ReasoningParser.parse_stream_chunk), [`stripped_think_start`](reasoning_parser.md#Gemma4Detector.stripped_think_start), [`Qwen3Detector`](reasoning_parser.md#Qwen3Detector), [`DeepSeekR1Detector`](reasoning_parser.md#DeepSeekR1Detector), [`Gemma4Detector`](reasoning_parser.md#Gemma4Detector), [`Glm45Detector`](reasoning_parser.md#Glm45Detector), [`KimiDetector`](reasoning_parser.md#KimiDetector), [`__init__`](reasoning_parser.md#DeepSeekR1Detector.__init__), [`__init__`](reasoning_parser.md#Gemma4Detector.__init__), [`__init__`](reasoning_parser.md#Glm45Detector.__init__), [`__init__`](reasoning_parser.md#KimiDetector.__init__), [`__init__`](reasoning_parser.md#Qwen3Detector.__init__)

### `DeepSeekR1Detector`  ·  implements/extends BaseReasoningFormatDetector
- def: [`python/sgl_jax/srt/reasoning_parser.py:126`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L126)
- doc: Detector for DeepSeek-R1 model.
- signature: `class DeepSeekR1Detector(BaseReasoningFormatDetector):`
- protocol/private: `__init__`[`L139`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L139)
- uses (calls/refs, reference-scoped): [`BaseReasoningFormatDetector`](reasoning_parser.md#BaseReasoningFormatDetector), [`__init__`](reasoning_parser.md#BaseReasoningFormatDetector.__init__)
- used by: [`BaseReasoningFormatDetector`](reasoning_parser.md#BaseReasoningFormatDetector), [`DetectorMap`](reasoning_parser.md#ReasoningParser.DetectorMap)

### `Gemma4Detector`  ·  implements/extends BaseReasoningFormatDetector
- def: [`python/sgl_jax/srt/reasoning_parser.py:218`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L218)
- doc: Detector for Gemma 4 model.
- signature: `class Gemma4Detector(BaseReasoningFormatDetector):`
- members:
  - `detect_and_parse(self, text: str)` — [`L233`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L233)
  - `parse_streaming_increment(self, new_text: str)` — [`L258`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L258)
  - `stripped_think_start` — [`L271`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L271)
- protocol/private: `__init__`[`L225`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L225), `_buffer`[`L281`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L281), `_in_reasoning`[`L236`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L236)
- uses (calls/refs, reference-scoped): [`StreamingParseResult`](reasoning_parser.md#StreamingParseResult), [`BaseReasoningFormatDetector`](reasoning_parser.md#BaseReasoningFormatDetector), [`think_end_token`](reasoning_parser.md#BaseReasoningFormatDetector.think_end_token), [`think_start_token`](reasoning_parser.md#BaseReasoningFormatDetector.think_start_token), [`__init__`](reasoning_parser.md#BaseReasoningFormatDetector.__init__), [`stream_reasoning`](reasoning_parser.md#BaseReasoningFormatDetector.stream_reasoning)
- used by: [`parse_streaming_increment`](reasoning_parser.md#BaseReasoningFormatDetector.parse_streaming_increment), [`BaseReasoningFormatDetector`](reasoning_parser.md#BaseReasoningFormatDetector), [`DetectorMap`](reasoning_parser.md#ReasoningParser.DetectorMap), [`detect_and_parse`](reasoning_parser.md#BaseReasoningFormatDetector.detect_and_parse)

### `Glm45Detector`  ·  implements/extends BaseReasoningFormatDetector
- def: [`python/sgl_jax/srt/reasoning_parser.py:195`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L195)
- doc: Detector for GLM-4.5 / 4.6 / 4.7 models.
- signature: `class Glm45Detector(BaseReasoningFormatDetector):`
- protocol/private: `__init__`[`L208`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L208)
- uses (calls/refs, reference-scoped): [`BaseReasoningFormatDetector`](reasoning_parser.md#BaseReasoningFormatDetector), [`__init__`](reasoning_parser.md#BaseReasoningFormatDetector.__init__)
- used by: [`BaseReasoningFormatDetector`](reasoning_parser.md#BaseReasoningFormatDetector), [`DetectorMap`](reasoning_parser.md#ReasoningParser.DetectorMap)

### `KimiDetector`  ·  implements/extends BaseReasoningFormatDetector
- def: [`python/sgl_jax/srt/reasoning_parser.py:177`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L177)
- doc: Detector for Kimi Thinking model.
- signature: `class KimiDetector(BaseReasoningFormatDetector):`
- protocol/private: `__init__`[`L186`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L186)
- uses (calls/refs, reference-scoped): [`BaseReasoningFormatDetector`](reasoning_parser.md#BaseReasoningFormatDetector), [`__init__`](reasoning_parser.md#BaseReasoningFormatDetector.__init__)
- used by: [`BaseReasoningFormatDetector`](reasoning_parser.md#BaseReasoningFormatDetector), [`DetectorMap`](reasoning_parser.md#ReasoningParser.DetectorMap)

### `Qwen3Detector`  ·  implements/extends BaseReasoningFormatDetector
- def: [`python/sgl_jax/srt/reasoning_parser.py:150`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L150)
- doc: Detector for Qwen3 model.
- signature: `class Qwen3Detector(BaseReasoningFormatDetector):`
- protocol/private: `__init__`[`L163`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L163)
- uses (calls/refs, reference-scoped): [`BaseReasoningFormatDetector`](reasoning_parser.md#BaseReasoningFormatDetector), [`__init__`](reasoning_parser.md#BaseReasoningFormatDetector.__init__)
- used by: [`BaseReasoningFormatDetector`](reasoning_parser.md#BaseReasoningFormatDetector), [`DetectorMap`](reasoning_parser.md#ReasoningParser.DetectorMap)

### `ReasoningParser`
- def: [`python/sgl_jax/srt/reasoning_parser.py:302`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L302)
- doc: Parser that handles both streaming and non-streaming scenarios for extracting
- signature: `class ReasoningParser:`
- members:
  - `parse_non_stream(self, full_text: str)` — [`L332`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L332) — Non-streaming call: one-time parsing
  - `parse_stream_chunk(self, chunk_text: str)` — [`L337`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L337) — Streaming call: incremental parsing
  - `DetectorMap` — [`L313`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L313)
  - `detector` — [`L330`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L330)
- protocol/private: `__init__`[`L322`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L322)
- uses (calls/refs, reference-scoped): [`parse_streaming_increment`](reasoning_parser.md#BaseReasoningFormatDetector.parse_streaming_increment), [`BaseReasoningFormatDetector`](reasoning_parser.md#BaseReasoningFormatDetector), [`detect_and_parse`](reasoning_parser.md#BaseReasoningFormatDetector.detect_and_parse), [`Qwen3Detector`](reasoning_parser.md#Qwen3Detector), [`DeepSeekR1Detector`](reasoning_parser.md#DeepSeekR1Detector), [`Gemma4Detector`](reasoning_parser.md#Gemma4Detector), [`Glm45Detector`](reasoning_parser.md#Glm45Detector), [`KimiDetector`](reasoning_parser.md#KimiDetector), [`normal_text`](reasoning_parser.md#StreamingParseResult.normal_text), [`reasoning_text`](reasoning_parser.md#StreamingParseResult.reasoning_text)
- used by: [`add_cli_args`](server_args.md#ServerArgs.add_cli_args)

### `StreamingParseResult`
- def: [`python/sgl_jax/srt/reasoning_parser.py:1`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L1)
- doc: Result of streaming incremental parsing.
- signature: `class StreamingParseResult:`
- members:
  - `normal_text` — [`L5`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L5)
  - `reasoning_text` — [`L6`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L6)
- protocol/private: `__init__`[`L4`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/reasoning_parser.py#L4)
- used by: [`parse_streaming_increment`](reasoning_parser.md#BaseReasoningFormatDetector.parse_streaming_increment), [`_in_reasoning`](reasoning_parser.md#Gemma4Detector._in_reasoning), [`_buffer`](reasoning_parser.md#Gemma4Detector._buffer), [`detect_and_parse`](reasoning_parser.md#BaseReasoningFormatDetector.detect_and_parse), [`parse_streaming_increment`](reasoning_parser.md#Gemma4Detector.parse_streaming_increment), [`parse_non_stream`](reasoning_parser.md#ReasoningParser.parse_non_stream), [`parse_stream_chunk`](reasoning_parser.md#ReasoningParser.parse_stream_chunk), [`detect_and_parse`](reasoning_parser.md#Gemma4Detector.detect_and_parse)

