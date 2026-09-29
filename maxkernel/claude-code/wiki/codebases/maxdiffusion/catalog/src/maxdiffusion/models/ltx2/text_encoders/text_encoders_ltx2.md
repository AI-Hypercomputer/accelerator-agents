---
title: 'Module: src/maxdiffusion/models/ltx2/text_encoders/text_encoders_ltx2.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx2/text_encoders/text_encoders_ltx2.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx2.text_encoders.text_encoders_ltx2`/
symbols:
  LTX2AudioVideoGemmaTextEncoder.__call__: LTX2AudioVideoGemmaTextEncoder#__call__().
  Array: Array.
  LTX2AudioVideoGemmaTextEncoder.video_embeddings_connector: LTX2AudioVideoGemmaTextEncoder#video_embeddings_connector.
  LTX2AudioVideoGemmaTextEncoder.audio_embeddings_connector: LTX2AudioVideoGemmaTextEncoder#audio_embeddings_connector.
  LTX2AudioVideoGemmaTextEncoder.feature_extractor: LTX2AudioVideoGemmaTextEncoder#feature_extractor.
  LTX2AudioVideoGemmaTextEncoder.__init__: LTX2AudioVideoGemmaTextEncoder#__init__().
  DType: DType.
  LTX2AudioVideoGemmaTextEncoder.per_modality_projections: LTX2AudioVideoGemmaTextEncoder#per_modality_projections.
  LTX2AudioVideoGemmaTextEncoder.video_text_proj_in: LTX2AudioVideoGemmaTextEncoder#video_text_proj_in.
  LTX2AudioVideoGemmaTextEncoder.audio_text_proj_in: LTX2AudioVideoGemmaTextEncoder#audio_text_proj_in.
  LTX2AudioVideoGemmaTextEncoder: LTX2AudioVideoGemmaTextEncoder#
  LTX2AudioVideoGemmaTextEncoder.caption_channels: LTX2AudioVideoGemmaTextEncoder#caption_channels.
---
# Module: [`src/maxdiffusion/models/ltx2/text_encoders/text_encoders_ltx2.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/text_encoders_ltx2.py)

## Classes
### `LTX2AudioVideoGemmaTextEncoder`  ·  implements/extends ConfigMixin, FlaxModelMixin, Module
- def: [`src/maxdiffusion/models/ltx2/text_encoders/text_encoders_ltx2.py:32`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/text_encoders_ltx2.py#L32)
- doc: Encoder for Audio-Video tasks.
- signature: `class LTX2AudioVideoGemmaTextEncoder(nnx.Module, FlaxModelMixin, ConfigMixin):`
- members:
  - `__call__(self, hidden_states: Union[Tuple[Array, ...], List[Array]], attention_mask: Array)` — [`L158`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/text_encoders_ltx2.py#L158) — Returns:
  - `audio_embeddings_connector` — [`L141`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/text_encoders_ltx2.py#L141)
  - `audio_text_proj_in` — [`L103`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/text_encoders_ltx2.py#L103)
  - `caption_channels` — [`L85`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/text_encoders_ltx2.py#L85)
  - `feature_extractor` — [`L112`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/text_encoders_ltx2.py#L112)
  - `per_modality_projections` — [`L86`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/text_encoders_ltx2.py#L86)
  - `video_embeddings_connector` — [`L125`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/text_encoders_ltx2.py#L125)
  - `video_text_proj_in` — [`L95`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/text_encoders_ltx2.py#L95)
- protocol/private: `__init__`[`L39`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/text_encoders_ltx2.py#L39)
- uses (calls/refs, reference-scoped): [`dim`](embeddings_connector_ltx2.md#Embeddings1DConnector.dim), [`Array`](text_encoders_ltx2.md#Array), [`Embeddings1DConnector`](embeddings_connector_ltx2.md#Embeddings1DConnector), [`LTX2GemmaFeatureExtractor`](feature_extractor_ltx2.md#LTX2GemmaFeatureExtractor), [`DType`](text_encoders_ltx2.md#DType)

## Module values
- `Array` — [`L28`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/text_encoders_ltx2.py#L28)
- `DType` — [`L29`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/text_encoders_ltx2.py#L29)

