---
title: 'Module: src/maxtext/layers/encoders.py'
type: catalog
provenance: extracted
module: src/maxtext/layers/encoders.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.layers.encoders`/
symbols:
  VisionEncoder.config: VisionEncoder#config.
  VisionEncoder.rngs: VisionEncoder#rngs.
  VisionEncoder.mesh: VisionEncoder#mesh.
  VisionEncoder._setup_vision_encoder_layers: VisionEncoder#_setup_vision_encoder_layers().
  AudioEncoder._setup_audio_encoder_layers: AudioEncoder#_setup_audio_encoder_layers().
  VisionEncoder.__call__: VisionEncoder#__call__().
  AudioEncoder.__call__: AudioEncoder#__call__().
  AudioEncoder.config: AudioEncoder#config.
  VisionEncoder.projector_name: VisionEncoder#projector_name.
  AudioEncoder.projector_name: AudioEncoder#projector_name.
  vision_encoder_as_linen: vision_encoder_as_linen().
  audio_encoder_as_linen: audio_encoder_as_linen().
  AudioEncoder.rngs: AudioEncoder#rngs.
  VisionEncoder: VisionEncoder#
  VisionEncoder.encoder_name: VisionEncoder#encoder_name.
  AudioEncoder: AudioEncoder#
  AudioEncoder.mesh: AudioEncoder#mesh.
  AudioEncoder.encoder_name: AudioEncoder#encoder_name.
  VisionEncoder.__init__: VisionEncoder#__init__().
  AudioEncoder.__init__: AudioEncoder#__init__().
---
# Module: [`src/maxtext/layers/encoders.py`](../../../../../../../raw/code/maxtext/src/maxtext/layers/encoders.py)

## Classes
### `AudioEncoder`  ·  implements/extends Module
- def: [`src/maxtext/layers/encoders.py:119`](../../../../../../../raw/code/maxtext/src/maxtext/layers/encoders.py#L119)
- doc: Audio encoder to encode audio features into soft tokens.
- signature: `class AudioEncoder(nnx.Module):`
- members:
  - `_setup_audio_encoder_layers(self)` — [`L128`](../../../../../../../raw/code/maxtext/src/maxtext/layers/encoders.py#L128) — Setup audio encoder layers specific to the model, instantiate NNX modules.
  - `config` — [`L123`](../../../../../../../raw/code/maxtext/src/maxtext/layers/encoders.py#L123)
  - `encoder_name` — [`L126`](../../../../../../../raw/code/maxtext/src/maxtext/layers/encoders.py#L126)
  - `mesh` — [`L124`](../../../../../../../raw/code/maxtext/src/maxtext/layers/encoders.py#L124)
  - `projector_name` — [`L126`](../../../../../../../raw/code/maxtext/src/maxtext/layers/encoders.py#L126)
  - `rngs` — [`L125`](../../../../../../../raw/code/maxtext/src/maxtext/layers/encoders.py#L125)
- protocol/private: `__call__`[`L141`](../../../../../../../raw/code/maxtext/src/maxtext/layers/encoders.py#L141), `__init__`[`L122`](../../../../../../../raw/code/maxtext/src/maxtext/layers/encoders.py#L122)
- used by: [`audio_encoder_as_linen`](encoders.md#audio_encoder_as_linen)

### `VisionEncoder`  ·  implements/extends Module
- def: [`src/maxtext/layers/encoders.py:26`](../../../../../../../raw/code/maxtext/src/maxtext/layers/encoders.py#L26)
- doc: Vision encoder to encode images into soft tokens.
- signature: `class VisionEncoder(nnx.Module):`
- members:
  - `_setup_vision_encoder_layers(self)` — [`L35`](../../../../../../../raw/code/maxtext/src/maxtext/layers/encoders.py#L35) — Setup vision encoder layers specific to the model, instantiate NNX modules.
  - `config` — [`L30`](../../../../../../../raw/code/maxtext/src/maxtext/layers/encoders.py#L30)
  - `encoder_name` — [`L33`](../../../../../../../raw/code/maxtext/src/maxtext/layers/encoders.py#L33)
  - `mesh` — [`L31`](../../../../../../../raw/code/maxtext/src/maxtext/layers/encoders.py#L31)
  - `projector_name` — [`L33`](../../../../../../../raw/code/maxtext/src/maxtext/layers/encoders.py#L33)
  - `rngs` — [`L32`](../../../../../../../raw/code/maxtext/src/maxtext/layers/encoders.py#L32)
- protocol/private: `__call__`[`L96`](../../../../../../../raw/code/maxtext/src/maxtext/layers/encoders.py#L96), `__init__`[`L29`](../../../../../../../raw/code/maxtext/src/maxtext/layers/encoders.py#L29)
- used by: [`vision_encoder_as_linen`](encoders.md#vision_encoder_as_linen)

## Functions
- `audio_encoder_as_linen(config: Config, mesh: Mesh)` — [`L172`](../../../../../../../raw/code/maxtext/src/maxtext/layers/encoders.py#L172) — Creates an AudioEncoder module.
- `vision_encoder_as_linen(config: Config, mesh: Mesh)` — [`L156`](../../../../../../../raw/code/maxtext/src/maxtext/layers/encoders.py#L156) — Creates a VisionEncoder module.

