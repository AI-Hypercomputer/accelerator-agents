---
title: 'Module: src/maxtext/layers/decoders.py'
type: catalog
provenance: extracted
module: src/maxtext/layers/decoders.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.layers.decoders`/
symbols:
  Decoder.config: Decoder#config.
  Decoder.__call__: Decoder#__call__().
  Decoder.get_pipeline_stage_module: Decoder#get_pipeline_stage_module().
  Decoder.mesh: Decoder#mesh.
  Decoder._apply_gemma3_scanned_blocks: Decoder#_apply_gemma3_scanned_blocks().
  Decoder.quant: Decoder#quant.
  Decoder._apply_deepseek4_scanned_blocks: Decoder#_apply_deepseek4_scanned_blocks().
  Decoder._apply_gemma4_scanned_blocks: Decoder#_apply_gemma4_scanned_blocks().
  Decoder.model_mode: Decoder#model_mode.
  SequentialBlockDecoderLayers.__call__: SequentialBlockDecoderLayers#__call__().
  Decoder.scan_decoder_layers: Decoder#scan_decoder_layers().
  Decoder.get_remat_policy: Decoder#get_remat_policy().
  Decoder._build_nnx_pipeline_stage: Decoder#_build_nnx_pipeline_stage().
  Decoder._apply_gemma4_small_layers: Decoder#_apply_gemma4_small_layers().
  Decoder._apply_single_engram_layer: Decoder#_apply_single_engram_layer().
  Decoder.pipeline_module: Decoder#pipeline_module.
  DecoderLayer.__call__: DecoderLayer#__call__().
  Decoder.norm_layer: Decoder#norm_layer.
  Decoder.set_remat_policy: Decoder#set_remat_policy().
  Decoder.apply_output_head: Decoder#apply_output_head().
  Decoder._apply_interleaved_scanned_layers: Decoder#_apply_interleaved_scanned_layers().
  Decoder._apply_scanned_chunk: Decoder#_apply_scanned_chunk().
  Decoder.get_decoder_layers: Decoder#get_decoder_layers().
  Decoder._get_nnx_decoder_block_classes: Decoder#_get_nnx_decoder_block_classes().
  Decoder.get_scannable: Decoder#get_scannable().
  Decoder._apply_embedding: Decoder#_apply_embedding().
  Decoder.get_norm_layer: Decoder#get_norm_layer().
  SequentialBlockDecoderLayers.config: SequentialBlockDecoderLayers#config.
  Decoder.minimal_policy: Decoder#minimal_policy().
  Decoder.decoder_layer: Decoder#decoder_layer.
  Decoder.build_pipeline_stage_layers: Decoder#build_pipeline_stage_layers().
  Decoder.move_to_device: Decoder#move_to_device().
  DecoderLayer.config: DecoderLayer#config.
  DecoderLayer.mesh: DecoderLayer#mesh.
  DecoderLayer.quant: DecoderLayer#quant.
  SequentialBlockDecoderLayers.decoder_layer: SequentialBlockDecoderLayers#decoder_layer.
  SequentialBlockDecoderLayers.num_decoder_layers: SequentialBlockDecoderLayers#num_decoder_layers.
  SequentialBlockDecoderLayers.mesh: SequentialBlockDecoderLayers#mesh.
  SequentialBlockDecoderLayers.quant: SequentialBlockDecoderLayers#quant.
  DecoderLayer: DecoderLayer#
  DecoderLayer.model_mode: DecoderLayer#model_mode.
  SequentialBlockDecoderLayers: SequentialBlockDecoderLayers#
  SequentialBlockDecoderLayers.model_mode: SequentialBlockDecoderLayers#model_mode.
  deepstack_process: deepstack_process().
  Decoder.map_fn: Decoder#map_fn().
  Decoder.get_layer_to_pipeline: Decoder#get_layer_to_pipeline().
  Decoder._find_next_boundary: Decoder#_find_next_boundary().
  Decoder: Decoder#
  Decoder.setup: Decoder#setup().
---
# Module: [`src/maxtext/layers/decoders.py`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py)

## Classes
### `Decoder`  ·  implements/extends Module
- def: [`src/maxtext/layers/decoders.py:297`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L297)
- doc: A stack of decoder layers as a part of an encoder-decoder architecture.
- signature: `class Decoder(nn.Module):`
- members:
  - `_apply_deepseek4_scanned_blocks(self, y, decoder_segment_ids, decoder_positions, deterministic, model_mode, previous_chunk, slot, decoder_input_tokens)` — [`L1524`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L1524) — Applies DeepSeek V4 scanned decoder blocks. — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `_apply_embedding(self, shared_embedding: nn.Module | nnx.Module, decoder_input_tokens, decoder_positions, deterministic, model_mode, multimodal_input=None)` — [`L694`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L694) — Applies token and positional embeddings to the input tokens. — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `_apply_gemma3_scanned_blocks(self, y, decoder_segment_ids, decoder_positions, deterministic, model_mode, bidirectional_mask, previous_chunk, slot, kv_caches=None, attention_metadata=None)` — [`L1300`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L1300) — Applies Gemma3 scanned decoder blocks, handling main scan and remainders. — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `_apply_gemma4_scanned_blocks(self, y, decoder_segment_ids, decoder_positions, deterministic, model_mode, bidirectional_mask, previous_chunk, slot, kv_caches=None, attention_metadata=None)` — [`L1404`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L1404) — Applies Gemma4 scanned decoder blocks, handling main scan and remainders. — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `_apply_gemma4_small_layers(self, y, decoder_input_tokens, decoder_segment_ids, decoder_positions, deterministic, model_mode, multimodal_input=None, kv_caches=None, attention_metadata=None, previous_chunk=None, slot=None)` — [`L1620`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L1620) — Apply Gemma 4 small (E2B / E4B) decoder layers. — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `_apply_interleaved_scanned_layers(self, y, layer_type, start_idx, end_idx, engram_indices, **kwargs)` — [`L1781`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L1781) — Applies a mix of scanned standard layers and unscanned Engram layers. — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `_apply_scanned_chunk(self, y, current_idx, next_boundary, layer_type, **kwargs)` — [`L1762`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L1762) — Applies a contiguous chunk of layers using the scan operation. — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `_apply_single_engram_layer(self, y, current_idx, layer_type, **kwargs)` — [`L1735`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L1735) — Applies a single, unscanned Engram layer. — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `_build_nnx_pipeline_stage(self, decoder_blocks, rngs)` — [`L580`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L580) — Creates a single NNX pipeline stage module. — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `_find_next_boundary(self, current_idx, end_idx, engram_indices)` — [`L1728`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L1728) — Finds the next index boundary, either the next Engram layer index or the overall end index. — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `_get_nnx_decoder_block_classes(self)` — [`L516`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L516) — Returns NNX decoder block classes for pipeline stage creation. — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `apply_output_head(self, shared_embedding: nn.Module | nnx.Module, y, deterministic, model_mode)` — [`L787`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L787) — Applies final normalization and projects hidden states to logits. — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `build_pipeline_stage_layers(rngs)` — [`L317`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L317) — Builds one pipeline stage module from the selected NNX decoder block classes. — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `get_decoder_layers(self)` — [`L450`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L450) — Retrieves a list of decoder layer classes based on the `decoder_block` config. — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `get_layer_to_pipeline(blocks, cfg)` — [`L598`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L598) — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `get_norm_layer(self, num_features: int)` — [`L632`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L632) — get normalization layer (return type inherits from nn.Module) — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `get_pipeline_stage_module(self, decoder_blocks)` — [`L595`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L595) — get pipeline stage module — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `get_remat_policy(self)` — [`L344`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L344) — Get remat policy — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `get_scannable(normal_cls, scannable_cls)` — [`L520`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L520)
  - `map_fn(path, value)` — [`L561`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L561) — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `minimal_policy(self, with_context=False, with_quantization=False)` — [`L325`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L325) — Helper for creating minimal checkpoint policies. — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `move_to_device(variables)` — [`L558`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L558) — Move parameters to device with proper sharding. — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `scan_decoder_layers(self, cfg, decoder_layer, length, metadata_axis_name, mesh, in_axes_tuple, **kwargs)` — [`L666`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L666) — scan decoder layers, calls `flax.linen.transforms.scan` — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `set_remat_policy(self, block_layers, policy)` — [`L552`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L552) — Set remat policy — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `setup(self)` — [`L305`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L305) — Initialize decoder layer.
  - `config` — [`L300`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L300) — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `decoder_layer` — [`L307`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L307) — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `mesh` — [`L301`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L301) — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `model_mode` — [`L303`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L303) — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `norm_layer` — [`L308`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L308) — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `pipeline_module` — [`L321`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L321) — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `quant` — [`L302`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L302) — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
- protocol/private: `__call__`[`L853`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L853)
- uses (calls/refs, reference-scoped): [`config`](decoders.md#SequentialBlockDecoderLayers.config), [`decoder_layer`](decoders.md#SequentialBlockDecoderLayers.decoder_layer), [`mesh`](decoders.md#SequentialBlockDecoderLayers.mesh), [`num_decoder_layers`](decoders.md#SequentialBlockDecoderLayers.num_decoder_layers), [`quant`](decoders.md#SequentialBlockDecoderLayers.quant), [`DecoderLayer`](decoders.md#DecoderLayer), [`SequentialBlockDecoderLayers`](decoders.md#SequentialBlockDecoderLayers), [`deepstack_process`](decoders.md#deepstack_process), [`model_mode`](decoders.md#SequentialBlockDecoderLayers.model_mode)

### `DecoderLayer`  ·  implements/extends Module
- def: [`src/maxtext/layers/decoders.py:77`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L77) — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
- doc: Transformer decoder layer that attends to the encoder.
- signature: `class DecoderLayer(nn.Module):`
- members:
  - `config` — [`L84`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L84)
  - `mesh` — [`L85`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L85)
  - `model_mode` — [`L86`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L86)
  - `quant` — [`L87`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L87)
- protocol/private: `__call__`[`L90`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L90)
- used by: [`get_decoder_layers`](decoders.md#Decoder.get_decoder_layers)

### `SequentialBlockDecoderLayers`  ·  implements/extends Module
- def: [`src/maxtext/layers/decoders.py:233`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L233) — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
- doc: Sequential unscanned series of decoder layers.
- signature: `class SequentialBlockDecoderLayers(nn.Module):`
- members:
  - `config` — [`L238`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L238)
  - `decoder_layer` — [`L236`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L236)
  - `mesh` — [`L239`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L239)
  - `model_mode` — [`L241`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L241)
  - `num_decoder_layers` — [`L237`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L237) — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)
  - `quant` — [`L240`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L240)
- protocol/private: `__call__`[`L244`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L244)
- used by: [`get_pipeline_stage_module`](decoders.md#Decoder.get_pipeline_stage_module)

## Functions
- `deepstack_process(hidden_states, bidirectional_mask, visual_embeds)` — [`L272`](../../../../../../../raw/code/maxtext/src/maxtext/layers/decoders.py#L272) — Process deepstack visual embeddings by adding them to hidden states at visual token positions. — documented in [maxtext-layers-decoders](../../../../concepts/maxtext-layers-decoders.md)

