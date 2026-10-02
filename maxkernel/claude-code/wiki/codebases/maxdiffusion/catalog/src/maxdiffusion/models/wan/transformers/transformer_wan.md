---
title: 'Module: src/maxdiffusion/models/wan/transformers/transformer_wan.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/wan/transformers/transformer_wan.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.wan.transformers.transformer_wan`/
symbols:
  WanModel.__call__: WanModel#__call__().
  WanModel._run_all_blocks: WanModel#_run_all_blocks().
  WanTransformerBlock.__call__: WanTransformerBlock#__call__().
  WanTransformerBlock.conditional_named_scope: WanTransformerBlock#conditional_named_scope().
  WanTimeTextImageEmbedding.__call__: WanTimeTextImageEmbedding#__call__().
  WanModel.compute_kv_cache: WanModel#compute_kv_cache().
  WanModel.blocks: WanModel#blocks.
  WanRotaryPosEmbed.__call__: WanRotaryPosEmbed#__call__().
  WanFeedForward.__call__: WanFeedForward#__call__().
  WanModel: WanModel#
  WanModel.condition_embedder: WanModel#condition_embedder.
  WanModel.conditional_named_scope: WanModel#conditional_named_scope().
  WanTimeTextImageEmbedding.image_embedder: WanTimeTextImageEmbedding#image_embedder.
  WanFeedForward.act_fn: WanFeedForward#act_fn.
  WanTransformerBlock: WanTransformerBlock#
  WanModel.gradient_checkpoint: WanModel#gradient_checkpoint.
  WanRotaryPosEmbed: WanRotaryPosEmbed#
  WanTimeTextImageEmbedding: WanTimeTextImageEmbedding#
  WanTransformerBlock.compute_kv: WanTransformerBlock#compute_kv().
  WanModel.__init__: WanModel#__init__().
  BlockSizes: BlockSizes.
  WanTimeTextImageEmbedding.timesteps_proj: WanTimeTextImageEmbedding#timesteps_proj.
  WanTimeTextImageEmbedding.time_embedder: WanTimeTextImageEmbedding#time_embedder.
  WanTimeTextImageEmbedding.act_fn: WanTimeTextImageEmbedding#act_fn.
  WanTimeTextImageEmbedding.text_embedder: WanTimeTextImageEmbedding#text_embedder.
  WanTransformerBlock.attn2: WanTransformerBlock#attn2.
  WanModel.scan_layers: WanModel#scan_layers.
  get_frequencies: get_frequencies().
  WanTransformerBlock.norm1: WanTransformerBlock#norm1.
  WanTransformerBlock.attn1: WanTransformerBlock#attn1.
  WanTransformerBlock.norm2: WanTransformerBlock#norm2.
  WanTransformerBlock.ffn: WanTransformerBlock#ffn.
  WanTransformerBlock.norm3: WanTransformerBlock#norm3.
  WanModel.rope: WanModel#rope.
  WanModel.init_block: WanModel#init_block().
  WanModel.norm_out: WanModel#norm_out.
  WanFeedForward: WanFeedForward#
  ApproximateGELU.__call__: ApproximateGELU#__call__().
  WanFeedForward.conditional_named_scope: WanFeedForward#conditional_named_scope().
  WanTransformerBlock.__init__: WanTransformerBlock#__init__().
  WanTimeTextImageEmbedding.time_proj: WanTimeTextImageEmbedding#time_proj.
  WanFeedForward.drop_out: WanFeedForward#drop_out.
  WanTransformerBlock.adaln_scale_shift_table: WanTransformerBlock#adaln_scale_shift_table.
  WanModel.names_which_can_be_offloaded: WanModel#names_which_can_be_offloaded.
  WanModel.names_which_can_be_saved: WanModel#names_which_can_be_saved.
  WanModel.scale_shift_table: WanModel#scale_shift_table.
  WanRotaryPosEmbed.attention_head_dim: WanRotaryPosEmbed#attention_head_dim.
  WanRotaryPosEmbed.patch_size: WanRotaryPosEmbed#patch_size.
  WanRotaryPosEmbed.max_seq_len: WanRotaryPosEmbed#max_seq_len.
  WanRotaryPosEmbed.theta: WanRotaryPosEmbed#theta.
  ApproximateGELU: ApproximateGELU#
  ApproximateGELU.proj: ApproximateGELU#proj.
  WanFeedForward.enable_jax_named_scopes: WanFeedForward#enable_jax_named_scopes.
  WanFeedForward.proj_out: WanFeedForward#proj_out.
  WanTransformerBlock.enable_jax_named_scopes: WanTransformerBlock#enable_jax_named_scopes.
  WanModel.num_layers: WanModel#num_layers.
  WanModel.enable_jax_named_scopes: WanModel#enable_jax_named_scopes.
  WanModel.patch_embedding: WanModel#patch_embedding.
  WanModel.proj_out: WanModel#proj_out.
  WanModel._compute_kv: WanModel#_compute_kv().
  WanModel.scan_fn: WanModel#scan_fn().
  WanModel.layer_forward: WanModel#layer_forward().
  WanRotaryPosEmbed.__init__: WanRotaryPosEmbed#__init__().
  WanTimeTextImageEmbedding.__init__: WanTimeTextImageEmbedding#__init__().
  ApproximateGELU.__init__: ApproximateGELU#__init__().
  WanFeedForward.__init__: WanFeedForward#__init__().
---
# Module: [`src/maxdiffusion/models/wan/transformers/transformer_wan.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py)

## Classes
### `ApproximateGELU`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/wan/transformers/transformer_wan.py:220`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L220)
- doc: The approximate form of the Gaussian Error Linear Unit (GELU). For more details, see section 2 of this
- signature: `class ApproximateGELU(nnx.Module):`
- members:
  - `proj` — [`L236`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L236)
- protocol/private: `__call__`[`L254`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L254), `__init__`[`L226`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L226)
- used by: [`act_fn`](transformer_wan.md#WanFeedForward.act_fn)

### `WanFeedForward`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/wan/transformers/transformer_wan.py:260`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L260)
- signature: `class WanFeedForward(nnx.Module):`
- members:
  - `conditional_named_scope(self, name: str)` — [`L315`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L315) — Return a JAX named scope if enabled, otherwise a null context.
  - `act_fn` — [`L283`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L283)
  - `drop_out` — [`L297`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L297)
  - `enable_jax_named_scopes` — [`L282`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L282)
  - `proj_out` — [`L298`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L298)
- protocol/private: `__call__`[`L319`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L319), `__init__`[`L262`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L262)
- uses (calls/refs, reference-scoped): `rate`, [`ApproximateGELU`](transformer_wan.md#ApproximateGELU)
- used by: [`ffn`](transformer_wan_vace.md#WanVACETransformerBlock.ffn), [`ffn`](transformer_wan.md#WanTransformerBlock.ffn)

### `WanModel`  ·  implements/extends ConfigMixin, FlaxModelMixin, Module
- def: [`src/maxdiffusion/models/wan/transformers/transformer_wan.py:538`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L538)
- signature: `class WanModel(nnx.Module, FlaxModelMixin, ConfigMixin):`
- members:
  - `compute_kv_cache(self, encoder_hidden_states: jax.Array, encoder_hidden_states_image: Optional[jax.Array] = None, timestep: Optional[jax.Array] = None)` — [`L702`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L702)
  - `conditional_named_scope(self, name: str)` — [`L698`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L698) — Return a JAX named scope if enabled, otherwise a null context.
  - `init_block(rngs)` — [`L627`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L627)
  - `layer_forward(hidden_states, l_kv)` — [`L895`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L895)
  - `scan_fn(carry, block_input)` — [`L846`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L846)
  - `blocks` — [`L655`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L655)
  - `condition_embedder` — [`L609`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L609)
  - `enable_jax_named_scopes` — [`L582`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L582)
  - `gradient_checkpoint` — [`L651`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L651)
  - `names_which_can_be_offloaded` — [`L652`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L652)
  - `names_which_can_be_saved` — [`L653`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L653)
  - `norm_out` — [`L682`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L682)
  - `num_layers` — [`L580`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L580)
  - `patch_embedding` — [`L592`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L592)
  - `proj_out` — [`L683`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L683)
  - `rope` — [`L591`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L591)
  - `scale_shift_table` — [`L693`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L693)
  - `scan_layers` — [`L581`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L581)
- protocol/private: `__call__`[`L758`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L758), `__init__`[`L541`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L541), `_compute_kv`[`L740`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L740), `_run_all_blocks`[`L843`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L843)
- uses (calls/refs, reference-scoped): [`config`](../../../configuration_utils.md#ConfigMixin.config), [`ConfigMixin`](../../../configuration_utils.md#ConfigMixin), [`FlaxModelMixin`](../../modeling_flax_utils.md#FlaxModelMixin), [`register_to_config`](../../../configuration_utils.md#register_to_config), [`FP32LayerNorm`](../../normalization_flax.md#FP32LayerNorm), [`GradientCheckpointType`](../../gradient_checkpoint.md#GradientCheckpointType), [`compute_kv_cache`](transformer_wan_vace.md#WanVACEModel.compute_kv_cache), [`apply`](../../gradient_checkpoint.md#GradientCheckpointType.apply), [`WanTransformerBlock`](transformer_wan.md#WanTransformerBlock), [`conditional_named_scope`](transformer_wan_vace.md#WanVACEModel.conditional_named_scope), [`from_str`](../../gradient_checkpoint.md#GradientCheckpointType.from_str), [`WanRotaryPosEmbed`](transformer_wan.md#WanRotaryPosEmbed), [`WanTimeTextImageEmbedding`](transformer_wan.md#WanTimeTextImageEmbedding), [`BlockSizes`](transformer_wan.md#BlockSizes), [`act_fn`](transformer_wan.md#WanTimeTextImageEmbedding.act_fn), [`text_embedder`](transformer_wan.md#WanTimeTextImageEmbedding.text_embedder), [`time_embedder`](transformer_wan.md#WanTimeTextImageEmbedding.time_embedder), [`timesteps_proj`](transformer_wan.md#WanTimeTextImageEmbedding.timesteps_proj), [`WanVACEModel`](transformer_wan_vace.md#WanVACEModel), [`time_proj`](transformer_wan.md#WanTimeTextImageEmbedding.time_proj)
- used by: [`ConfigMixin`](../../../configuration_utils.md#ConfigMixin), [`FlaxModelMixin`](../../modeling_flax_utils.md#FlaxModelMixin), [`WanVACEModel`](transformer_wan_vace.md#WanVACEModel)

### `WanRotaryPosEmbed`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/wan/transformers/transformer_wan.py:63`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L63)
- signature: `class WanRotaryPosEmbed(nnx.Module):`
- members:
  - `attention_head_dim` — [`L72`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L72)
  - `max_seq_len` — [`L74`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L74)
  - `patch_size` — [`L73`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L73)
  - `theta` — [`L75`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L75)
- protocol/private: `__call__`[`L77`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L77), `__init__`[`L65`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L65)
- uses (calls/refs, reference-scoped): [`get_frequencies`](transformer_wan.md#get_frequencies)
- used by: [`rope`](transformer_wan_animate.md#WanAnimateTransformer3DModel.rope), [`rope`](transformer_wan_vace.md#WanVACEModel.rope), [`rope`](transformer_wan.md#WanModel.rope)

### `WanTimeTextImageEmbedding`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/wan/transformers/transformer_wan.py:98`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L98)
- signature: `class WanTimeTextImageEmbedding(nnx.Module):`
- members:
  - `act_fn` — [`L123`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L123)
  - `image_embedder` — [`L147`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L147)
  - `text_embedder` — [`L140`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L140)
  - `time_embedder` — [`L115`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L115)
  - `time_proj` — [`L124`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L124)
  - `timesteps_proj` — [`L114`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L114)
- protocol/private: `__call__`[`L160`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L160), `__init__`[`L100`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L100)
- uses (calls/refs, reference-scoped): [`get_activation`](../../modeling_flax_utils.md#get_activation), [`NNXTimestepEmbedding`](../../embeddings_flax.md#NNXTimestepEmbedding), [`NNXFlaxTimesteps`](../../embeddings_flax.md#NNXFlaxTimesteps), [`NNXPixArtAlphaTextProjection`](../../embeddings_flax.md#NNXPixArtAlphaTextProjection), [`NNXWanImageEmbedding`](../../embeddings_flax.md#NNXWanImageEmbedding)
- used by: [`__call__`](transformer_wan.md#WanModel.__call__), [`condition_embedder`](transformer_wan.md#WanModel.condition_embedder), [`condition_embedder`](transformer_wan_vace.md#WanVACEModel.condition_embedder), [`condition_embedder`](transformer_wan_animate.md#WanAnimateTransformer3DModel.condition_embedder)

### `WanTransformerBlock`  ·  implements/extends Module
- def: [`src/maxdiffusion/models/wan/transformers/transformer_wan.py:333`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L333)
- signature: `class WanTransformerBlock(nnx.Module):`
- members:
  - `compute_kv(self, encoder_hidden_states: jax.Array, encoder_attention_mask: Optional[jax.Array] = None)` — [`L530`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L530)
  - `conditional_named_scope(self, name: str)` — [`L436`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L436) — Return a JAX named scope if enabled, otherwise a null context.
  - `adaln_scale_shift_table` — [`L432`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L432)
  - `attn1` — [`L368`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L368)
  - `attn2` — [`L391`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L391)
  - `enable_jax_named_scopes` — [`L358`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L358)
  - `ffn` — [`L418`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L418)
  - `norm1` — [`L367`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L367)
  - `norm2` — [`L415`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L415)
  - `norm3` — [`L429`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L429)
- protocol/private: `__call__`[`L440`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L440), `__init__`[`L335`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L335)
- uses (calls/refs, reference-scoped): [`compute_kv`](../../attention_flax.md#FlaxWanAttention.compute_kv), [`FP32LayerNorm`](../../normalization_flax.md#FP32LayerNorm), [`FlaxWanAttention`](../../attention_flax.md#FlaxWanAttention), [`BlockSizes`](transformer_wan.md#BlockSizes), [`WanFeedForward`](transformer_wan.md#WanFeedForward)
- used by: [`blocks`](transformer_wan_animate.md#WanAnimateTransformer3DModel.blocks), [`blocks`](transformer_wan.md#WanModel.blocks), [`names_which_can_be_saved`](transformer_wan_vace.md#WanVACEModel.names_which_can_be_saved), [`init_block`](transformer_wan_animate.md#WanAnimateTransformer3DModel.init_block), [`init_block`](transformer_wan.md#WanModel.init_block)

## Functions
- `get_frequencies(max_seq_len: int, theta: int, attention_head_dim: int)` — [`L43`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L43)

## Module values
- `BlockSizes` — [`L40`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py#L40)

