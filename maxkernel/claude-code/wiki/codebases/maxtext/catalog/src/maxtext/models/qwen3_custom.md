---
title: 'Module: src/maxtext/models/qwen3_custom.py'
type: catalog
provenance: extracted
module: src/maxtext/models/qwen3_custom.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.models.qwen3_custom`/Qwen3Custom
symbols:
  Qwen3CustomMoeDecoderLayer.__call__: MoeDecoderLayer#__call__().
  Qwen3CustomMoeDecoderLayer.layer_up_projection: MoeDecoderLayer#layer_up_projection.
  Qwen3CustomMoeDecoderLayer.self_attention: MoeDecoderLayer#self_attention.
  Qwen3CustomMoeDecoderLayer.apply_attention_with_norm: MoeDecoderLayer#apply_attention_with_norm().
  Qwen3CustomMoeDecoderLayerToLinen: MoeDecoderLayerToLinen.
  Qwen3CustomAttention: Attention#
  Qwen3CustomMoeDecoderLayer: MoeDecoderLayer#
  Qwen3CustomMoeDecoderLayer.latent_norm: MoeDecoderLayer#latent_norm.
  Qwen3CustomMoeDecoderLayer.moe_block: MoeDecoderLayer#moe_block.
  Qwen3CustomMoeDecoderLayer.out_sharding: MoeDecoderLayer#out_sharding.
  Qwen3CustomAttention.init_out_w: Attention#init_out_w().
  Qwen3CustomMoeDecoderLayer.__init__: MoeDecoderLayer#__init__().
---
# Module: [`src/maxtext/models/qwen3_custom.py`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_custom.py)

## Classes
### `Qwen3CustomAttention`  ·  implements/extends Attention
- def: [`src/maxtext/models/qwen3_custom.py:37`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_custom.py#L37)
- doc: Custom GQA attention that supports sub-dimensional output.
- signature: `class Qwen3CustomAttention(Attention):`
- members:
  - `init_out_w(self, output_dim: int)` — [`L40`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_custom.py#L40) — Initializes the output projection.
- used by: [`self_attention`](qwen3_custom.md#Qwen3CustomMoeDecoderLayer.self_attention)

### `Qwen3CustomMoeDecoderLayer`  ·  implements/extends AttentionWithNorm
- def: [`src/maxtext/models/qwen3_custom.py:67`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_custom.py#L67)
- doc: Qwen3 Transformer decoder layer (Custom MoE).
- signature: `class Qwen3CustomMoeDecoderLayer(AttentionWithNorm):`
- members:
  - `__call__(self, inputs: jnp.ndarray, decoder_segment_ids: None | jnp.ndarray, decoder_positions: None | jnp.ndarray, deterministic: bool, model_mode: str, previous_chunk=None, slot: None | int = None, kv_cache: None | jnp.ndarray = None, attention_metadata: None | dict[str, Any] = None)` — [`L198`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_custom.py#L198) — Applies the Qwen3CustomMoeDecoderLayer to the inputs.
  - `__init__(self, config: Config, mesh: Mesh, model_mode: str, quant: None | Quant, rngs: nnx.Rngs)` — [`L70`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_custom.py#L70) — Initializes the instance.
  - `apply_attention_with_norm(self, inputs: jnp.ndarray, decoder_segment_ids: None | jnp.ndarray, decoder_positions: None | jnp.ndarray, deterministic: bool, model_mode: str, kv_cache: None | jnp.ndarray = None, attention_metadata: None | dict[str, Any] = None)` — [`L171`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_custom.py#L171)
  - `latent_norm` — [`L127`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_custom.py#L127)
  - `layer_up_projection` — [`L152`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_custom.py#L152)
  - `moe_block` — [`L136`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_custom.py#L136)
  - `out_sharding` — [`L169`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_custom.py#L169)
  - `self_attention` — [`L94`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_custom.py#L94)
- uses (calls/refs, reference-scoped): [`Qwen3CustomAttention`](qwen3_custom.md#Qwen3CustomAttention)
- used by: [`Qwen3CustomMoeDecoderLayerToLinen`](qwen3_custom.md#Qwen3CustomMoeDecoderLayerToLinen)

## Module values
- `Qwen3CustomMoeDecoderLayerToLinen` — [`L261`](../../../../../../../raw/code/maxtext/src/maxtext/models/qwen3_custom.py#L261)

