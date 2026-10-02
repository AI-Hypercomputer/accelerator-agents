---
title: 'Module: src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx2.text_encoders.embeddings_connector_ltx2`/
symbols:
  Embeddings1DConnector._compute_1d_rope: Embeddings1DConnector#_compute_1d_rope().
  Embeddings1DConnector.__call__: Embeddings1DConnector#__call__().
  Array: Array.
  _BasicTransformerBlock1D.__call__: _BasicTransformerBlock1D#__call__().
  Embeddings1DConnector._replace_padded_with_learnable_registers: Embeddings1DConnector#_replace_padded_with_learnable_registers().
  Embeddings1DConnector.dim: Embeddings1DConnector#dim.
  Embeddings1DConnector.num_learnable_registers: Embeddings1DConnector#num_learnable_registers.
  Embeddings1DConnector.create_block: Embeddings1DConnector#create_block().
  Embeddings1DConnector.stacked_blocks: Embeddings1DConnector#stacked_blocks.
  Embeddings1DConnector.learnable_registers: Embeddings1DConnector#learnable_registers.
  Embeddings1DConnector.final_norm: Embeddings1DConnector#final_norm.
  Embeddings1DConnector: Embeddings1DConnector#
  Embeddings1DConnector.rope_type: Embeddings1DConnector#rope_type.
  DType: DType.
  _BasicTransformerBlock1D: _BasicTransformerBlock1D#
  _BasicTransformerBlock1D.attn1: _BasicTransformerBlock1D#attn1.
  _BasicTransformerBlock1D.ff: _BasicTransformerBlock1D#ff.
  _BasicTransformerBlock1D.norm1: _BasicTransformerBlock1D#norm1.
  _BasicTransformerBlock1D.norm2: _BasicTransformerBlock1D#norm2.
  Embeddings1DConnector.heads: Embeddings1DConnector#heads.
  Embeddings1DConnector.theta: Embeddings1DConnector#theta.
  Embeddings1DConnector.num_layers: Embeddings1DConnector#num_layers.
  Embeddings1DConnector.base_seq_len: Embeddings1DConnector#base_seq_len.
  Embeddings1DConnector.double_precision: Embeddings1DConnector#double_precision.
  Embeddings1DConnector.block_scan_fn: Embeddings1DConnector#block_scan_fn().
  _BasicTransformerBlock1D.__init__: _BasicTransformerBlock1D#__init__().
  Embeddings1DConnector.__init__: Embeddings1DConnector#__init__().
  Embeddings1DConnector.head_dim: Embeddings1DConnector#head_dim.
---
# Module: [`src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py)

## Classes
### `Embeddings1DConnector`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py:79`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L79)
- doc: Applies 1D transformer processing with Thinking Tokens (Learnable Registers).
- signature: `class Embeddings1DConnector(nnx.Module):`
- members:
  - `block_scan_fn(carry, block_module)` — [`L250`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L250)
  - `create_block(rngs)` — [`L117`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L117)
  - `base_seq_len` — [`L109`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L109)
  - `dim` — [`L102`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L102)
  - `double_precision` — [`L110`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L110)
  - `final_norm` — [`L140`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L140)
  - `head_dim` — [`L104`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L104)
  - `heads` — [`L103`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L103)
  - `learnable_registers` — [`L136`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L136)
  - `num_layers` — [`L107`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L107)
  - `num_learnable_registers` — [`L106`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L106)
  - `rope_type` — [`L108`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L108)
  - `stacked_blocks` — [`L131`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L131)
  - `theta` — [`L105`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L105)
- protocol/private: `__call__`[`L233`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L233), `__init__`[`L85`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L85), `_compute_1d_rope`[`L184`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L184), `_replace_padded_with_learnable_registers`[`L142`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L142)
- uses (calls/refs, reference-scoped): [`Array`](embeddings_connector_ltx2.md#Array), [`_BasicTransformerBlock1D`](embeddings_connector_ltx2.md#_BasicTransformerBlock1D), [`DType`](embeddings_connector_ltx2.md#DType)
- used by: [`__call__`](text_encoders_ltx2.md#LTX2AudioVideoGemmaTextEncoder.__call__), [`audio_embeddings_connector`](text_encoders_ltx2.md#LTX2AudioVideoGemmaTextEncoder.audio_embeddings_connector), [`video_embeddings_connector`](text_encoders_ltx2.md#LTX2AudioVideoGemmaTextEncoder.video_embeddings_connector)

### `_BasicTransformerBlock1D`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py:29`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L29)
- signature: `class _BasicTransformerBlock1D(nnx.Module):`
- members:
  - `attn1` — [`L43`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L43)
  - `ff` — [`L56`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L56)
  - `norm1` — [`L57`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L57)
  - `norm2` — [`L58`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L58)
- protocol/private: `__call__`[`L60`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L60), `__init__`[`L31`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L31)
- uses (calls/refs, reference-scoped): [`Array`](embeddings_connector_ltx2.md#Array)
- used by: [`create_block`](embeddings_connector_ltx2.md#Embeddings1DConnector.create_block)

## Module values
- `Array` — [`L25`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L25)
- `DType` — [`L26`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx2/text_encoders/embeddings_connector_ltx2.py#L26)

