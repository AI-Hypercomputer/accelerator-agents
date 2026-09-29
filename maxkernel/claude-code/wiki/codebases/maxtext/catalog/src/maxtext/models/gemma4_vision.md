---
title: 'Module: src/maxtext/models/gemma4_vision.py'
type: catalog
provenance: extracted
module: src/maxtext/models/gemma4_vision.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.models.gemma4_vision`/
symbols:
  VisionEntry.__call__: VisionEntry#__call__().
  VisionEntry.input_projection: VisionEntry#input_projection.
  Gemma4EncoderBlock.__call__: Gemma4EncoderBlock#__call__().
  Gemma4VisionEncoderLayer.__call__: Gemma4VisionEncoderLayer#__call__().
  Gemma4VisionRotaryEmbedding.__call__: Gemma4VisionRotaryEmbedding#__call__().
  Gemma4VisionEncoderLayer.vision_entry: Gemma4VisionEncoderLayer#vision_entry.
  VisionEntry.pos_emb_param: VisionEntry#pos_emb_param.
  VisionExit._maybe_downsample: VisionExit#_maybe_downsample().
  VisionExit._single_call: VisionExit#_single_call().
  Gemma4EncoderBlock.rngs: Gemma4EncoderBlock#rngs.
  Gemma4EncoderBlock.attention: Gemma4EncoderBlock#attention.
  Gemma4VisionEncoderLayer.vision_exit: Gemma4VisionEncoderLayer#vision_exit.
  VisionExit.__call__: VisionExit#__call__().
  Gemma4VisionEncoderLayer.rngs: Gemma4VisionEncoderLayer#rngs.
  Gemma4VisionProjector.__call__: Gemma4VisionProjector#__call__().
  VisionEntry.pos_emb_shape_yx: VisionEntry#pos_emb_shape_yx.
  apply_multidimensional_rope._apply_rope: apply_multidimensional_rope()._apply_rope().
  apply_multidimensional_rope: apply_multidimensional_rope().
  Gemma4EncoderBlock.pre_attention_norm: Gemma4EncoderBlock#pre_attention_norm.
  Gemma4EncoderBlock.post_attention_norm: Gemma4EncoderBlock#post_attention_norm.
  Gemma4EncoderBlock.pre_ffw_norm: Gemma4EncoderBlock#pre_ffw_norm.
  Gemma4EncoderBlock.post_ffw_norm: Gemma4EncoderBlock#post_ffw_norm.
  Gemma4EncoderBlock.mlp: Gemma4EncoderBlock#mlp.
  Gemma4VisionEncoderLayer.std_bias: Gemma4VisionEncoderLayer#std_bias.
  Gemma4VisionEncoderLayer.std_scale: Gemma4VisionEncoderLayer#std_scale.
  Gemma4VisionProjector.projection: Gemma4VisionProjector#projection.
  Gemma4VisionProjector.norm: Gemma4VisionProjector#norm.
  VisionEntry.patch_size: VisionEntry#patch_size.
  VisionExit.output_length: VisionExit#output_length.
  Gemma4Attention.init_rotary_embedding: Gemma4Attention#init_rotary_embedding().
  gemma4_vision_encoder_as_linen: gemma4_vision_encoder_as_linen().
  VisionEntry.d_model: VisionEntry#d_model.
  VisionEntry.matmul_precision: VisionEntry#matmul_precision.
  Gemma4VisionRotaryEmbedding: Gemma4VisionRotaryEmbedding#
  Gemma4VisionProjector.rngs: Gemma4VisionProjector#rngs.
  factorized_posemb: factorized_posemb().
  patchify: patchify().
  VisionEntry: VisionEntry#
  VisionEntry.normalize_input_range: VisionEntry#normalize_input_range.
  VisionEntry.dtype: VisionEntry#dtype.
  VisionEntry.weight_dtype: VisionEntry#weight_dtype.
  avg_pool_by_positions: avg_pool_by_positions().
  VisionExit: VisionExit#
  VisionExit.d_model: VisionExit#d_model.
  VisionExit.precision: VisionExit#precision.
  Gemma4VisionRotaryEmbedding.base_frequency: Gemma4VisionRotaryEmbedding#base_frequency.
  Gemma4VisionRotaryEmbedding.rotary_fraction: Gemma4VisionRotaryEmbedding#rotary_fraction.
  Gemma4VisionRotaryEmbedding.scale_factor: Gemma4VisionRotaryEmbedding#scale_factor.
  Gemma4Attention: Gemma4Attention#
  Gemma4EncoderBlock: Gemma4EncoderBlock#
  Gemma4VisionEncoderLayer: Gemma4VisionEncoderLayer#
  Gemma4VisionEncoderLayer.config: Gemma4VisionEncoderLayer#config.
  VisionEntry.__init__: VisionEntry#__init__().
  VisionExit.__init__: VisionExit#__init__().
  Gemma4VisionRotaryEmbedding.__init__: Gemma4VisionRotaryEmbedding#__init__().
  Gemma4EncoderBlock.__init__: Gemma4EncoderBlock#__init__().
  Gemma4EncoderBlock.config: Gemma4EncoderBlock#config.
  Gemma4EncoderBlock.mesh: Gemma4EncoderBlock#mesh.
  Gemma4VisionEncoderLayer.__init__: Gemma4VisionEncoderLayer#__init__().
  Gemma4VisionEncoderLayer.mesh: Gemma4VisionEncoderLayer#mesh.
  Gemma4VisionProjector: Gemma4VisionProjector#
  Gemma4VisionProjector.__init__: Gemma4VisionProjector#__init__().
  Gemma4VisionProjector.config: Gemma4VisionProjector#config.
  Gemma4VisionProjector.mesh: Gemma4VisionProjector#mesh.
---
# Module: [`src/maxtext/models/gemma4_vision.py`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py)

## Classes
### `Gemma4Attention`  ·  implements/extends Attention
- def: [`src/maxtext/models/gemma4_vision.py:411`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L411)
- doc: Gemma 4 specific Attention module.
- signature: `class Gemma4Attention(attentions.Attention):`
- members:
  - `init_rotary_embedding(self)` — [`L414`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L414) — Initializes the rotary position embedding module for Gemma 4 vision.
- uses (calls/refs, reference-scoped): [`Gemma4VisionRotaryEmbedding`](gemma4_vision.md#Gemma4VisionRotaryEmbedding)
- used by: [`attention`](gemma4_vision.md#Gemma4EncoderBlock.attention)

### `Gemma4EncoderBlock`  ·  implements/extends Module
- def: [`src/maxtext/models/gemma4_vision.py:422`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L422)
- doc: Single transformer encoder block (MHSA + MLP).
- signature: `class Gemma4EncoderBlock(nnx.Module):`
- members:
  - `__call__(self, x: jax.Array, positions: jax.Array | None = None, deterministic: bool = False)` — [`L521`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L521) — Applies the encoder block (MHSA + MLP) to the inputs.
  - `attention` — [`L468`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L468)
  - `config` — [`L432`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L432)
  - `mesh` — [`L433`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L433)
  - `mlp` — [`L509`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L509)
  - `post_attention_norm` — [`L446`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L446)
  - `post_ffw_norm` — [`L500`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L500)
  - `pre_attention_norm` — [`L437`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L437)
  - `pre_ffw_norm` — [`L491`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L491)
  - `rngs` — [`L434`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L434)
- protocol/private: `__init__`[`L425`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L425)
- uses (calls/refs, reference-scoped): [`Gemma4Attention`](gemma4_vision.md#Gemma4Attention)
- used by: [`vision_entry`](gemma4_vision.md#Gemma4VisionEncoderLayer.vision_entry)

### `Gemma4VisionEncoderLayer`  ·  implements/extends Module
- def: [`src/maxtext/models/gemma4_vision.py:536`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L536)
- doc: Gemma 4 Vision Encoder Layer.
- signature: `class Gemma4VisionEncoderLayer(nnx.Module):`
- members:
  - `__call__(self, inputs: jax.Array, deterministic: bool = False)` — [`L576`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L576) — Applies the vision encoder layer.
  - `config` — [`L540`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L540)
  - `mesh` — [`L541`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L541)
  - `rngs` — [`L542`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L542)
  - `std_bias` — [`L569`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L569)
  - `std_scale` — [`L572`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L572)
  - `vision_entry` — [`L545`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L545)
  - `vision_exit` — [`L563`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L563)
- protocol/private: `__init__`[`L539`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L539)
- uses (calls/refs, reference-scoped): [`Gemma4EncoderBlock`](gemma4_vision.md#Gemma4EncoderBlock), [`VisionEntry`](gemma4_vision.md#VisionEntry), [`VisionExit`](gemma4_vision.md#VisionExit)
- used by: [`gemma4_vision_encoder_as_linen`](gemma4_vision.md#gemma4_vision_encoder_as_linen)

### `Gemma4VisionProjector`  ·  implements/extends Module
- def: [`src/maxtext/models/gemma4_vision.py:606`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L606)
- doc: A layer that projects image embeddings to the embedding space of the text encoder.
- signature: `class Gemma4VisionProjector(nnx.Module):`
- members:
  - `__call__(self, x: jax.Array)` — [`L634`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L634) — Projects vision embeddings into the text embedding space.
  - `config` — [`L610`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L610)
  - `mesh` — [`L611`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L611)
  - `norm` — [`L624`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L624)
  - `projection` — [`L614`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L614)
  - `rngs` — [`L612`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L612)
- protocol/private: `__init__`[`L609`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L609)

### `Gemma4VisionRotaryEmbedding`  ·  implements/extends Module
- def: [`src/maxtext/models/gemma4_vision.py:387`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L387)
- doc: Rotary position embedding for Gemma 4 vision.
- signature: `class Gemma4VisionRotaryEmbedding(nnx.Module):`
- members:
  - `__call__(self, inputs: jax.Array, positions: jax.Array)` — [`L400`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L400) — Applies rotary position embeddings to the inputs.
  - `base_frequency` — [`L396`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L396)
  - `rotary_fraction` — [`L397`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L397)
  - `scale_factor` — [`L398`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L398)
- protocol/private: `__init__`[`L390`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L390)
- uses (calls/refs, reference-scoped): [`apply_multidimensional_rope`](gemma4_vision.md#apply_multidimensional_rope)
- used by: [`init_rotary_embedding`](gemma4_vision.md#Gemma4Attention.init_rotary_embedding)

### `VisionEntry`  ·  implements/extends Module
- def: [`src/maxtext/models/gemma4_vision.py:88`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L88)
- doc: The vision entry layer.
- signature: `class VisionEntry(nnx.Module):`
- members:
  - `__call__(self, images_or_patches: jax.Array, positions_xy: jax.Array | None = None)` — [`L133`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L133) — Processes input images or patches and applies projection and position embeddings.
  - `d_model` — [`L103`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L103)
  - `dtype` — [`L107`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L107)
  - `input_projection` — [`L111`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L111)
  - `matmul_precision` — [`L109`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L109)
  - `normalize_input_range` — [`L106`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L106)
  - `patch_size` — [`L104`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L104)
  - `pos_emb_param` — [`L125`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L125)
  - `pos_emb_shape_yx` — [`L105`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L105)
  - `weight_dtype` — [`L108`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L108)
- protocol/private: `__init__`[`L91`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L91)
- uses (calls/refs, reference-scoped): [`factorized_posemb`](gemma4_vision.md#factorized_posemb), [`patchify`](gemma4_vision.md#patchify)
- used by: [`vision_entry`](gemma4_vision.md#Gemma4VisionEncoderLayer.vision_entry)

### `VisionExit`  ·  implements/extends Module
- def: [`src/maxtext/models/gemma4_vision.py:306`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L306)
- doc: Vision exit layer with scaling and optional spatial pooling.
- signature: `class VisionExit(nnx.Module):`
- members:
  - `__call__(self, x: jax.Array, *, positions_xy: jax.Array | None = None, output_length_overrides: tuple[int, ...] | None = None)` — [`L372`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L372) — Applies vision exit processing, optionally downsampling to requested output lengths.
  - `_maybe_downsample(self, x: jax.Array, *, positions_xy: jax.Array | None = None, length: int)` — [`L314`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L314) — Downsamples the vision features if required by the output length.
  - `_single_call(self, x: jax.Array, *, positions_xy: jax.Array | None = None, length: int)` — [`L358`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L358) — Processes the features for a single target length.
  - `d_model` — [`L310`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L310)
  - `output_length` — [`L311`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L311)
  - `precision` — [`L312`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L312)
- protocol/private: `__init__`[`L309`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L309)
- uses (calls/refs, reference-scoped): [`avg_pool_by_positions`](gemma4_vision.md#avg_pool_by_positions)
- used by: [`vision_exit`](gemma4_vision.md#Gemma4VisionEncoderLayer.vision_exit)

## Functions
- `_apply_rope(x_in: jax.Array, pos: jax.Array, base_freq: int, scale: float)` — [`L191`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L191)
- `apply_multidimensional_rope(inputs: jax.Array, positions: jax.Array, *, base_frequency: int, rotary_fraction: float | None = None, scale_factor: float = 1)` — [`L168`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L168) — Applies multidimensional RoPE. Based on Gemma 4 implementation.
- `avg_pool_by_positions(x: jax.Array, *, positions_xy: jax.Array, length: int, precision)` — [`L274`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L274) — Performs 2D spatial pooling according to patch positions.
- `factorized_posemb(posemb: jax.Array, positions_xy: jax.Array, precision)` — [`L33`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L33) — Computes factorized position embedding from (x, y) coordinates.
- `gemma4_vision_encoder_as_linen(config: Config, mesh: Mesh)` — [`L641`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L641) — Wraps the Gemma 4 Vision Encoder as a Linen module.
- `patchify(images: jax.Array, patch_size: int)` — [`L52`](../../../../../../../raw/code/maxtext/src/maxtext/models/gemma4_vision.py#L52) — Patchifies images and returns patches and (x, y) coordinates.

