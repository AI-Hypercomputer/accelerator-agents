---
title: 'Module: src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.wan.transformers.transformer_wan_vace`/
symbols:
  WanVACEModel.__call__: WanVACEModel#__call__().
  WanVACETransformerBlock.__call__: WanVACETransformerBlock#__call__().
  WanVACETransformerBlock.conditional_named_scope: WanVACETransformerBlock#conditional_named_scope().
  WanVACEModel.compute_kv_cache: WanVACEModel#compute_kv_cache().
  WanVACEModel.gradient_checkpoint: WanVACEModel#gradient_checkpoint.
  WanVACEModel.blocks: WanVACEModel#blocks.
  WanVACEModel.conditional_named_scope: WanVACEModel#conditional_named_scope().
  WanVACETransformerBlock.compute_kv: WanVACETransformerBlock#compute_kv().
  WanVACEModel.__init__: WanVACEModel#__init__().
  BlockSizes: BlockSizes.
  WanVACETransformerBlock.proj_in: WanVACETransformerBlock#proj_in.
  WanVACETransformerBlock.attn2: WanVACETransformerBlock#attn2.
  WanVACETransformerBlock.proj_out: WanVACETransformerBlock#proj_out.
  WanVACEModel.condition_embedder: WanVACEModel#condition_embedder.
  WanVACEModel.names_which_can_be_saved: WanVACEModel#names_which_can_be_saved.
  WanVACEModel.scan_layers: WanVACEModel#scan_layers.
  WanVACETransformerBlock.norm1: WanVACETransformerBlock#norm1.
  WanVACETransformerBlock.attn1: WanVACETransformerBlock#attn1.
  WanVACETransformerBlock.norm2: WanVACETransformerBlock#norm2.
  WanVACETransformerBlock.ffn: WanVACETransformerBlock#ffn.
  WanVACETransformerBlock.norm3: WanVACETransformerBlock#norm3.
  WanVACEModel.rope: WanVACEModel#rope.
  WanVACEModel.norm_out: WanVACEModel#norm_out.
  WanVACETransformerBlock.__init__: WanVACETransformerBlock#__init__().
  WanVACEModel: WanVACEModel#
  WanVACEModel.names_which_can_be_offloaded: WanVACEModel#names_which_can_be_offloaded.
  WanVACEModel.vace_blocks: WanVACEModel#vace_blocks.
  WanVACETransformerBlock: WanVACETransformerBlock#
  WanVACETransformerBlock.enable_jax_named_scopes: WanVACETransformerBlock#enable_jax_named_scopes.
  WanVACETransformerBlock.apply_input_projection: WanVACETransformerBlock#apply_input_projection.
  WanVACETransformerBlock.apply_output_projection: WanVACETransformerBlock#apply_output_projection.
  WanVACETransformerBlock.adaln_scale_shift_table: WanVACETransformerBlock#adaln_scale_shift_table.
  WanVACEModel.enable_jax_named_scopes: WanVACEModel#enable_jax_named_scopes.
  WanVACEModel.patch_embedding: WanVACEModel#patch_embedding.
  WanVACEModel.vace_patch_embedding: WanVACEModel#vace_patch_embedding.
  WanVACEModel.proj_out: WanVACEModel#proj_out.
  WanVACEModel.scale_shift_table: WanVACEModel#scale_shift_table.
  WanVACEModel.layer_forward: WanVACEModel#layer_forward().
  WanVACEModel.layer_forward_vace: WanVACEModel#layer_forward_vace().
  WanVACEModel.num_layers: WanVACEModel#num_layers.
---
# Module: [`src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py)

## Classes
### `WanVACEModel`  ·  implements/extends WanModel
- def: [`src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py:293`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L293)
- doc: Extension of Wan to include VACE conditioning.
- signature: `class WanVACEModel(WanModel):`
- members:
  - `__init__(self, rngs: nnx.Rngs, vace_layers: list[int], vace_in_channels: int, model_type="t2v", patch_size: Tuple[int, ...] = (1, 2, 2), num_attention_heads: int = 40, attention_head_dim: int = 128, in_channels: int = 16, out_channels: int = 16, text_dim: int = 4096, freq_dim: int = 256, ffn_dim: int = 13824, num_layers: int = 40, dropout: float = 0, cross_attn_norm: bool = True, qk_norm: Optional[str] = "rms_norm_across_heads", eps: float = 0.000001, image_dim: Optional[int] = None, added_kv_proj_dim: Optional[int] = None, rope_max_seq_len: int = 1024, pos_embed_seq_len: Optional[int] = None, flash_min_seq_length: int = 4096, flash_block_sizes: BlockSizes = None, mesh: jax.sharding.Mesh = None, dtype: jnp.dtype = jnp.float32, weights_dtype: jnp.dtype = jnp.float32, precision: jax.lax.Precision = None, attention: str = "dot_product", remat_policy: str = "None", names_which_can_be_saved: list[str] = [], names_which_can_be_offloaded: list[str] = [], mask_padding_tokens: bool = True, scan_layers: bool = True, enable_jax_named_scopes: bool = False, attention_config: Optional[dict] = None)` — [`L297`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L297) — Initializes the VACE model.
  - `compute_kv_cache(self, encoder_hidden_states: jax.Array, encoder_hidden_states_image: Optional[jax.Array] = None, timestep: Optional[jax.Array] = None)` — [`L483`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L483)
  - `conditional_named_scope(self, name: str)` — [`L479`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L479) — Return a JAX named scope if enabled, otherwise a null context.
  - `layer_forward(hidden_states, control_hidden_states, rngs)` — [`L621`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L621)
  - `layer_forward_vace(hidden_states, rngs)` — [`L650`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L650)
  - `blocks` — [`L415`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L415)
  - `condition_embedder` — [`L372`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L372)
  - `enable_jax_named_scopes` — [`L346`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L346)
  - `gradient_checkpoint` — [`L383`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L383)
  - `names_which_can_be_offloaded` — [`L384`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L384)
  - `names_which_can_be_saved` — [`L385`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L385)
  - `norm_out` — [`L463`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L463)
  - `num_layers` — [`L344`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L344)
  - `patch_embedding` — [`L356`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L356)
  - `proj_out` — [`L464`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L464)
  - `rope` — [`L355`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L355)
  - `scale_shift_table` — [`L474`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L474)
  - `scan_layers` — [`L345`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L345)
  - `vace_blocks` — [`L446`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L446)
  - `vace_patch_embedding` — [`L448`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L448)
- protocol/private: `__call__`[`L544`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L544)
- uses (calls/refs, reference-scoped): [`config`](../../../configuration_utils.md#ConfigMixin.config), [`register_to_config`](../../../configuration_utils.md#register_to_config), [`FP32LayerNorm`](../../normalization_flax.md#FP32LayerNorm), [`GradientCheckpointType`](../../gradient_checkpoint.md#GradientCheckpointType), [`apply`](../../gradient_checkpoint.md#GradientCheckpointType.apply), [`WanModel`](transformer_wan.md#WanModel), [`WanTransformerBlock`](transformer_wan.md#WanTransformerBlock), [`from_str`](../../gradient_checkpoint.md#GradientCheckpointType.from_str), [`WanRotaryPosEmbed`](transformer_wan.md#WanRotaryPosEmbed), [`WanTimeTextImageEmbedding`](transformer_wan.md#WanTimeTextImageEmbedding), [`BlockSizes`](transformer_wan_vace.md#BlockSizes), [`WanVACETransformerBlock`](transformer_wan_vace.md#WanVACETransformerBlock)
- used by: [`compute_kv_cache`](transformer_wan.md#WanModel.compute_kv_cache), [`WanModel`](transformer_wan.md#WanModel), [`conditional_named_scope`](transformer_wan.md#WanModel.conditional_named_scope)

### `WanVACETransformerBlock`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py:37`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L37)
- doc: Attention block for VACE.
- signature: `class WanVACETransformerBlock(nnx.Module):`
- members:
  - `__init__(self, rngs: nnx.Rngs, *, dim: int, ffn_dim: int, num_heads: int, qk_norm: str = "rms_norm_across_heads", cross_attn_norm: bool = False, eps: float = 0.000001, flash_min_seq_length: int = 4096, flash_block_sizes: BlockSizes | None = None, mesh: jax.sharding.Mesh | None = None, dtype: jnp.dtype = jnp.float32, weights_dtype: jnp.dtype = jnp.float32, precision: jax.lax.Precision | None = None, attention: str = "dot_product", dropout: float = 0, mask_padding_tokens: bool = True, enable_jax_named_scopes: bool = False, apply_input_projection: bool = False, apply_output_projection: bool = False, attention_config: Optional[dict] = None)` — [`L47`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L47) — Sets up the model.
  - `compute_kv(self, encoder_hidden_states: jax.Array, encoder_attention_mask: Optional[jax.Array] = None)` — [`L205`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L205)
  - `conditional_named_scope(self, name: str)` — [`L201`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L201) — Return a JAX named scope if enabled, otherwise a null context.
  - `adaln_scale_shift_table` — [`L197`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L197)
  - `apply_input_projection` — [`L97`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L97)
  - `apply_output_projection` — [`L98`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L98)
  - `attn1` — [`L121`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L121)
  - `attn2` — [`L144`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L144)
  - `enable_jax_named_scopes` — [`L96`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L96)
  - `ffn` — [`L169`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L169)
  - `norm1` — [`L120`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L120)
  - `norm2` — [`L166`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L166)
  - `norm3` — [`L181`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L181)
  - `proj_in` — [`L107`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L107)
  - `proj_out` — [`L184`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L184)
- protocol/private: `__call__`[`L208`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L208)
- uses (calls/refs, reference-scoped): [`compute_kv`](../../attention_flax.md#FlaxWanAttention.compute_kv), [`FP32LayerNorm`](../../normalization_flax.md#FP32LayerNorm), [`FlaxWanAttention`](../../attention_flax.md#FlaxWanAttention), [`BlockSizes`](transformer_wan_vace.md#BlockSizes), [`WanFeedForward`](transformer_wan.md#WanFeedForward)
- used by: [`blocks`](transformer_wan_vace.md#WanVACEModel.blocks)

## Module values
- `BlockSizes` — [`L34`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan_vace.py#L34)

