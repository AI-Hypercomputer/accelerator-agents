---
title: 'Module: src/maxdiffusion/models/ltx2/text_encoders/feature_extractor_ltx2.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx2/text_encoders/feature_extractor_ltx2.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx2.text_encoders.feature_extractor_ltx2`/
symbols:
  LTX2GemmaFeatureExtractor.__call__: LTX2GemmaFeatureExtractor#__call__().
  Array: Array.
  _norm_and_concat_padded_batch: _norm_and_concat_padded_batch().
  LTX2GemmaFeatureExtractor.__init__: LTX2GemmaFeatureExtractor#__init__().
  LTX2GemmaFeatureExtractor: LTX2GemmaFeatureExtractor#
  DType: DType.
  LTX2GemmaFeatureExtractor.per_modality_projections: LTX2GemmaFeatureExtractor#per_modality_projections.
  LTX2GemmaFeatureExtractor.video_linear: LTX2GemmaFeatureExtractor#video_linear.
  LTX2GemmaFeatureExtractor.audio_linear: LTX2GemmaFeatureExtractor#audio_linear.
  LTX2GemmaFeatureExtractor.linear: LTX2GemmaFeatureExtractor#linear.
---
# Module: [`src/maxdiffusion/models/ltx2/text_encoders/feature_extractor_ltx2.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/feature_extractor_ltx2.py)

## Classes
### `LTX2GemmaFeatureExtractor`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx2/text_encoders/feature_extractor_ltx2.py:93`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/feature_extractor_ltx2.py#L93)
- doc: Feature extractor module for Gemma models in LTX-2.
- signature: `class LTX2GemmaFeatureExtractor(nnx.Module):`
- members:
  - `__call__(self, hidden_states: Union[Tuple[Array, ...], Array], attention_mask: Array)` — [`L157`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/feature_extractor_ltx2.py#L157) — Args:
  - `__init__(self, input_dim: int, output_dim: int, dtype: DType = jnp.float32, rngs: nnx.Rngs = None, per_modality_projections: bool = False, use_bias: bool = False, video_output_dim: Optional[int] = None, audio_output_dim: Optional[int] = None, sharding_specs: Optional[Any] = None)` — [`L99`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/feature_extractor_ltx2.py#L99) — Args:
  - `audio_linear` — [`L137`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/feature_extractor_ltx2.py#L137)
  - `linear` — [`L147`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/feature_extractor_ltx2.py#L147)
  - `per_modality_projections` — [`L116`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/feature_extractor_ltx2.py#L116)
  - `video_linear` — [`L128`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/feature_extractor_ltx2.py#L128)
- uses (calls/refs, reference-scoped): [`Array`](feature_extractor_ltx2.md#Array), [`_norm_and_concat_padded_batch`](feature_extractor_ltx2.md#_norm_and_concat_padded_batch), [`DType`](feature_extractor_ltx2.md#DType)
- used by: [`feature_extractor`](text_encoders_ltx2.md#LTX2AudioVideoGemmaTextEncoder.feature_extractor)

## Functions
- `_norm_and_concat_padded_batch(encoded_text: Array, attention_mask: Array)` — [`L26`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/feature_extractor_ltx2.py#L26) — Normalize and flatten multi-layer hidden states, respecting padding.

## Module values
- `Array` — [`L22`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/feature_extractor_ltx2.py#L22)
- `DType` — [`L23`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/feature_extractor_ltx2.py#L23)

