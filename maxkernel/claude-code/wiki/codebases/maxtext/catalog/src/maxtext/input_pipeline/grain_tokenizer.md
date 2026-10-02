---
title: 'Module: src/maxtext/input_pipeline/grain_tokenizer.py'
type: catalog
provenance: extracted
module: src/maxtext/input_pipeline/grain_tokenizer.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.input_pipeline.grain_tokenizer`/Tokenize
symbols:
  TokenizerTransformBase._get_processor: rTransformBase#_get_processor().
  TokenizerTransformBase.sequence_length: rTransformBase#sequence_length.
  TokenizeAndTrim.map: AndTrim#map().
  TokenizeAndChunk.__post_init__: AndChunk#__post_init__().
  TokenizeAndChunk.flat_map: AndChunk#flat_map().
  TokenizerTransformBase: rTransformBase#
  TokenizerTransformBase.feature_names: rTransformBase#feature_names.
  TokenizerTransformBase._initialize_processor_lock: rTransformBase#_initialize_processor_lock.
  TokenizeAndChunk.feature_name: AndChunk#feature_name.
  TokenizerTransformBase._processor: rTransformBase#_processor.
  TokenizerTransformBase.__setstate__: rTransformBase#__setstate__().
  TokenizerTransformBase._encode: rTransformBase#_encode().
  TokenizeAndTrim: AndTrim#
  TokenizeAndChunk: AndChunk#
  TokenizerTransformBase.tokenizer: rTransformBase#tokenizer.
  TokenizerTransformBase.__post_init__: rTransformBase#__post_init__().
  TokenizerTransformBase.__getstate__: rTransformBase#__getstate__().
  TokenizeAndChunk.max_fan_out: AndChunk#max_fan_out.
---
# Module: [`src/maxtext/input_pipeline/grain_tokenizer.py`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_tokenizer.py)

## Classes
### `TokenizeAndChunk`  ·  implements/extends FlatMap, TokenizerTransformBase
- def: [`src/maxtext/input_pipeline/grain_tokenizer.py:82`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_tokenizer.py#L82)
- doc: Tokenize and chunk features into multiple examples of sequence length.
- signature: `class TokenizeAndChunk(TokenizerTransformBase, grain.experimental.FlatMapTransform):`
- members:
  - `flat_map(self, element: dict[str, Any])` — [`L95`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_tokenizer.py#L95) — Tokenize and chunk text into multiple examples of sequence length.
  - `feature_name` — [`L92`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_tokenizer.py#L92)
  - `max_fan_out` — [`L85`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_tokenizer.py#L85)
- protocol/private: `__post_init__`[`L87`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_tokenizer.py#L87)
- uses (calls/refs, reference-scoped): [`sequence_length`](grain_tokenizer.md#TokenizerTransformBase.sequence_length), [`TokenizerTransformBase`](grain_tokenizer.md#TokenizerTransformBase), [`feature_names`](grain_tokenizer.md#TokenizerTransformBase.feature_names), [`_encode`](grain_tokenizer.md#TokenizerTransformBase._encode), [`__post_init__`](grain_tokenizer.md#TokenizerTransformBase.__post_init__)
- used by: [`TokenizerTransformBase`](grain_tokenizer.md#TokenizerTransformBase)

### `TokenizeAndTrim`  ·  implements/extends Map, TokenizerTransformBase
- def: [`src/maxtext/input_pipeline/grain_tokenizer.py:69`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_tokenizer.py#L69)
- doc: Tokenize and trim features to sequence length.
- signature: `class TokenizeAndTrim(TokenizerTransformBase, grain.MapTransform):`
- members:
  - `map(self, element: dict[str, Any])` — [`L72`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_tokenizer.py#L72) — Maps to each element.
- uses (calls/refs, reference-scoped): [`sequence_length`](grain_tokenizer.md#TokenizerTransformBase.sequence_length), [`TokenizerTransformBase`](grain_tokenizer.md#TokenizerTransformBase), [`feature_names`](grain_tokenizer.md#TokenizerTransformBase.feature_names), [`_encode`](grain_tokenizer.md#TokenizerTransformBase._encode)
- used by: [`TokenizerTransformBase`](grain_tokenizer.md#TokenizerTransformBase)

### `TokenizerTransformBase`
- def: [`src/maxtext/input_pipeline/grain_tokenizer.py:27`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_tokenizer.py#L27)
- doc: Base class for tokenizer transforms with common functionality.
- signature: `class TokenizerTransformBase:`
- members:
  - `_encode(self, text: str)` — [`L51`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_tokenizer.py#L51) — Common method to encode text using the tokenizer.
  - `feature_names` — [`L31`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_tokenizer.py#L31)
  - `sequence_length` — [`L32`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_tokenizer.py#L32)
  - `tokenizer` — [`L33`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_tokenizer.py#L33)
- protocol/private: `__getstate__`[`L56`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_tokenizer.py#L56), `__post_init__`[`L35`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_tokenizer.py#L35), `__setstate__`[`L62`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_tokenizer.py#L62), `_get_processor`[`L44`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_tokenizer.py#L44), `_initialize_processor_lock`[`L37`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_tokenizer.py#L37), `_processor`[`L36`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/grain_tokenizer.py#L36)
- uses (calls/refs, reference-scoped): [`TokenizeAndChunk`](grain_tokenizer.md#TokenizeAndChunk), [`TokenizeAndTrim`](grain_tokenizer.md#TokenizeAndTrim)
- used by: [`__post_init__`](grain_tokenizer.md#TokenizeAndChunk.__post_init__), [`flat_map`](grain_tokenizer.md#TokenizeAndChunk.flat_map), [`map`](grain_tokenizer.md#TokenizeAndTrim.map), [`feature_name`](grain_tokenizer.md#TokenizeAndChunk.feature_name), [`TokenizeAndChunk`](grain_tokenizer.md#TokenizeAndChunk), [`TokenizeAndTrim`](grain_tokenizer.md#TokenizeAndTrim)

