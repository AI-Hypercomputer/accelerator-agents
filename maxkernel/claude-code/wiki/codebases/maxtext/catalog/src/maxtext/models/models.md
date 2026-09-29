---
title: 'Module: src/maxtext/models/models.py'
type: catalog
provenance: extracted
module: src/maxtext/models/models.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.models.models`/
symbols:
  Transformer.decoder: Transformer#decoder.
  Transformer.config: Transformer#config.
  TransformerLinenPure.__call__: TransformerLinenPure#__call__().
  Transformer.__call__: Transformer#__call__().
  TransformerLinenPure.decoder: TransformerLinenPure#decoder.
  TransformerLinenPure.config: TransformerLinenPure#config.
  transformer_as_linen: transformer_as_linen().
  TransformerLinenPure.mtp_block: TransformerLinenPure#mtp_block.
  Transformer.mtp_block: Transformer#mtp_block.
  Transformer.token_embedder: Transformer#token_embedder.
  Transformer.model_mode: Transformer#model_mode.
  TransformerLinenPure.shared_embedding: TransformerLinenPure#shared_embedding.
  TransformerLinenPure.setup: TransformerLinenPure#setup().
  TransformerLinenPure.logits_from_hidden_states_for_vocab_tiling: TransformerLinenPure#logits_from_hidden_states_for_vocab_tiling().
  Transformer.logits_from_hidden_states_for_vocab_tiling: Transformer#logits_from_hidden_states_for_vocab_tiling().
  TransformerLinenPure.mesh: TransformerLinenPure#mesh.
  Transformer.mesh: Transformer#mesh.
  TransformerLinenPure: TransformerLinenPure#
  TransformerLinenPure.model_mode: TransformerLinenPure#model_mode.
  TransformerLinenPure.vision_encoder: TransformerLinenPure#vision_encoder.
  TransformerLinenPure.audio_encoder: TransformerLinenPure#audio_encoder.
  Transformer.quant: Transformer#quant.
  Transformer.vision_encoder: Transformer#vision_encoder.
  Transformer.audio_encoder: Transformer#audio_encoder.
  TransformerLinenPure.quant: TransformerLinenPure#quant.
  TransformerLinen: TransformerLinen#
  Transformer: Transformer#
  TransformerLinenPure.init: TransformerLinenPure#init().
  TransformerLinenPure.apply: TransformerLinenPure#apply().
  TransformerLinen.init: TransformerLinen#init().
  TransformerLinen.apply: TransformerLinen#apply().
  Transformer.__init__: Transformer#__init__().
  Transformer.no_op: Transformer#no_op().
  Transformer.init_cache: Transformer#init_cache().
---
# Module: [`src/maxtext/models/models.py`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py)

## Classes
### `Transformer`
- def: [`src/maxtext/models/models.py:323`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L323)
- members:
  - `__call__(self, decoder_input_tokens: jnp.ndarray, decoder_positions: jnp.ndarray, decoder_segment_ids=None, cache=None, encoder_images: jax.Array | None = None, encoder_image_masks: jax.Array | None = None, encoder_videos: jax.Array | None = None, encoder_video_masks: jax.Array | None = None, encoder_audios: jax.Array | None = None, enable_dropout=True, model_mode=MODEL_MODE_TRAIN, previous_chunk=None, true_length: int | None = None, slot: int | None = None, decoder_target_tokens: jax.Array | None = None, decoder_target_mask: jax.Array | None = None, kv_caches: list[jax.Array] | None = None, attention_metadata: dict[str, Any] | None = None)` — [`L435`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L435) — Applies the Zero-1 FSDP wrapped Transformer model.
  - `__init__(self, config: Config, mesh: Mesh, quant: Quant, *, model_mode: str = MODEL_MODE_TRAIN, rngs: nnx.Rngs)` — [`L329`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L329) — Initialize shared_embedding & decoder layers.
  - `init_cache(self, cache_size: int, batch_size: int, dtype=jnp.float32)` — [`L422`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L422) — Initializes the KV cache for the Transformer.
  - `logits_from_hidden_states_for_vocab_tiling(self, hidden_states, deterministic, model_mode)` — [`L413`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L413) — Computes logits from hidden states; used by vocabulary tiling.
  - `no_op(self, *args, **kwargs)` — [`L409`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L409) — A no-op method to allow the model to be used in a lazy context.
  - `audio_encoder` — [`L357`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L357)
  - `config` — [`L339`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L339)
  - `decoder` — [`L359`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L359)
  - `mesh` — [`L340`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L340)
  - `model_mode` — [`L342`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L342)
  - `mtp_block` — [`L401`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L401)
  - `quant` — [`L341`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L341)
  - `token_embedder` — [`L346`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L346)
  - `vision_encoder` — [`L356`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L356)
- used by: [`transformer_as_linen`](models.md#transformer_as_linen)

### `TransformerLinen`
- def: [`src/maxtext/models/models.py:305`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L305)
- members:
  - `apply(self, *args, model_mode: str = MODEL_MODE_TRAIN, **kwargs)` — [`L315`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L315) — Applies the model.
  - `init(self, *args, model_mode: str = MODEL_MODE_TRAIN, **kwargs)` — [`L308`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L308) — Initializes the model.
- used by: [`transformer_as_linen`](models.md#transformer_as_linen)

### `TransformerLinenPure`  ·  implements/extends Module
- def: [`src/maxtext/models/models.py:45`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L45)
- doc: An autoregressive transformer model.
- signature: `class TransformerLinenPure(nn.Module):`
- members:
  - `__call__(self, decoder_input_tokens: jnp.ndarray, decoder_positions: jnp.ndarray, decoder_segment_ids=None, encoder_images: None | jnp.ndarray = None, encoder_image_masks: None | jnp.ndarray = None, encoder_videos: None | jnp.ndarray = None, encoder_video_masks: None | jnp.ndarray = None, encoder_audios: None | jnp.ndarray = None, enable_dropout=True, model_mode=MODEL_MODE_TRAIN, previous_chunk=None, true_length: None | int = None, slot: None | int = None, decoder_target_tokens: None | jnp.ndarray = None, decoder_target_mask: None | jnp.ndarray = None, nnx_method=None, kv_caches: list[jax.Array] | None = None, attention_metadata: dict[str, Any] | None = None)` — [`L123`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L123) — Applies Transformer decoder-branch on encoded-input and target.
  - `apply(self, *args, model_mode: str = MODEL_MODE_TRAIN, **kwargs)` — [`L67`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L67) — Applies the model.
  - `init(self, *args, model_mode: str = MODEL_MODE_TRAIN, **kwargs)` — [`L61`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L61) — Initializes the model.
  - `logits_from_hidden_states_for_vocab_tiling(self, hidden_states, deterministic, model_mode)` — [`L110`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L110) — Compute logits from hidden states (wrapping decoder.apply_output_head).
  - `setup(self)` — [`L73`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L73) — Initialize shared_embedding & decoder layers.
  - `audio_encoder` — [`L89`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L89)
  - `config` — [`L51`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L51)
  - `decoder` — [`L90`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L90)
  - `mesh` — [`L52`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L52)
  - `model_mode` — [`L58`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L58)
  - `mtp_block` — [`L102`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L102)
  - `quant` — [`L53`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L53)
  - `shared_embedding` — [`L78`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L78)
  - `vision_encoder` — [`L88`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L88)
- used by: [`transformer_as_linen`](models.md#transformer_as_linen)

## Functions
- `transformer_as_linen(config: Config, mesh: Mesh, quant: Quant, model_mode: str = MODEL_MODE_TRAIN, *, name: str | None = None)` — [`L256`](../../../../../../../raw/code/maxtext/src/maxtext/models/models.py#L256) — Constructs a Transformer model as a Linen or NNX module.

