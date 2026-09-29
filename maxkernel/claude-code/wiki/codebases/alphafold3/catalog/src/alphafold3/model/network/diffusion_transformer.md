---
title: 'Module: src/alphafold3/model/network/diffusion_transformer.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/network/diffusion_transformer.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.network.diffusion_transformer`/
symbols:
  self_attention: self_attention().
  cross_attention: cross_attention().
  CrossAttTransformer.__call__: CrossAttTransformer#__call__().
  CrossAttTransformer.block: CrossAttTransformer#block().
  transition_block: transition_block().
  Transformer.block: Transformer#block().
  Transformer.super_block: Transformer#super_block().
  Transformer.__call__: Transformer#__call__().
  adaptive_zero_init: adaptive_zero_init().
  adaptive_layernorm: adaptive_layernorm().
  Transformer.config: Transformer#config.
  CrossAttentionConfig.num_head: CrossAttentionConfig#num_head.
  Transformer.__init__: Transformer#__init__().
  CrossAttTransformer.__init__: CrossAttTransformer#__init__().
  CrossAttTransformer.config: CrossAttTransformer#config.
  Transformer.Config.attention: Transformer#Config#attention.
  CrossAttTransformer.Config.attention: CrossAttTransformer#Config#attention.
  Transformer.Config.super_block_size: Transformer#Config#super_block_size.
  CrossAttTransformer: CrossAttTransformer#
  SelfAttentionConfig: SelfAttentionConfig#
  SelfAttentionConfig.num_head: SelfAttentionConfig#num_head.
  CrossAttTransformer.Config: CrossAttTransformer#Config#
  SelfAttentionConfig.key_dim: SelfAttentionConfig#key_dim.
  SelfAttentionConfig.value_dim: SelfAttentionConfig#value_dim.
  Transformer: Transformer#
  Transformer.Config: Transformer#Config#
  Transformer.Config.num_blocks: Transformer#Config#num_blocks.
  Transformer.global_config: Transformer#global_config.
  CrossAttentionConfig: CrossAttentionConfig#
  CrossAttentionConfig.key_dim: CrossAttentionConfig#key_dim.
  CrossAttentionConfig.value_dim: CrossAttentionConfig#value_dim.
  CrossAttTransformer.Config.num_blocks: CrossAttTransformer#Config#num_blocks.
  CrossAttTransformer.global_config: CrossAttTransformer#global_config.
  Transformer.Config.num_intermediate_factor: Transformer#Config#num_intermediate_factor.
  CrossAttTransformer.Config.num_intermediate_factor: CrossAttTransformer#Config#num_intermediate_factor.
  Transformer.Config.block_remat: Transformer#Config#block_remat.
---
# Module: [`src/alphafold3/model/network/diffusion_transformer.py`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py)

## Classes
### `Config`  ·  implements/extends BaseConfig
- def: [`src/alphafold3/model/network/diffusion_transformer.py:343`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L343)
- signature: `class Config(base_config.BaseConfig):`
- members:
  - `attention` — [`L193`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L193)
  - `attention` — [`L346`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L346) — documented in [alphafold3-model-network-diffusion_transformer](../../../../../concepts/alphafold3-model-network-diffusion_transformer.md)
  - `block_remat` — [`L195`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L195)
  - `num_blocks` — [`L194`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L194)
  - `num_blocks` — [`L345`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L345) — documented in [alphafold3-model-network-diffusion_transformer](../../../../../concepts/alphafold3-model-network-diffusion_transformer.md)
  - `num_intermediate_factor` — [`L197`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L197)
  - `num_intermediate_factor` — [`L344`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L344) — documented in [alphafold3-model-network-diffusion_transformer](../../../../../concepts/alphafold3-model-network-diffusion_transformer.md)
  - `super_block_size` — [`L196`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L196)
- uses (calls/refs, reference-scoped): [`SelfAttentionConfig`](diffusion_transformer.md#SelfAttentionConfig), [`CrossAttentionConfig`](diffusion_transformer.md#CrossAttentionConfig)
- used by: [`__call__`](diffusion_transformer.md#CrossAttTransformer.__call__), [`block`](diffusion_transformer.md#CrossAttTransformer.block), [`block`](diffusion_transformer.md#Transformer.block), [`super_block`](diffusion_transformer.md#Transformer.super_block), [`__call__`](diffusion_transformer.md#Transformer.__call__), [`AtomCrossAttDecoderConfig`](atom_cross_attention.md#AtomCrossAttDecoderConfig), [`atom_transformer`](atom_cross_attention.md#AtomCrossAttEncoderConfig.atom_transformer), [`__init__`](diffusion_transformer.md#CrossAttTransformer.__init__)

### `CrossAttTransformer`
- def: [`src/alphafold3/model/network/diffusion_transformer.py:340`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L340) — documented in [alphafold3-model-network-atom_cross_attention](../../../../../concepts/alphafold3-model-network-atom_cross_attention.md)
- doc: Transformer that applies cross attention between two sets of subsets.
- signature: `class CrossAttTransformer(hk.Module):`
- members:
  - `block(queries_act, pair_logits)` — [`L368`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L368)
  - `config` — [`L355`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L355)
  - `global_config` — [`L356`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L356)
- protocol/private: `__call__`[`L358`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L358), `__init__`[`L348`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L348)
- uses (calls/refs, reference-scoped): [`Linear`](../components/haiku_modules.md#Linear), [`convert`](../atom_layout/atom_layout.md#convert), [`GatherInfo`](../atom_layout/atom_layout.md#GatherInfo), [`LayerNorm`](../components/haiku_modules.md#LayerNorm), [`GlobalConfig`](../model_config.md#GlobalConfig), [`cross_attention`](diffusion_transformer.md#cross_attention), [`transition_block`](diffusion_transformer.md#transition_block), [`num_head`](diffusion_transformer.md#CrossAttentionConfig.num_head), [`attention`](diffusion_transformer.md#CrossAttTransformer.Config.attention), [`Config`](diffusion_transformer.md#CrossAttTransformer.Config), [`num_blocks`](diffusion_transformer.md#CrossAttTransformer.Config.num_blocks), [`num_intermediate_factor`](diffusion_transformer.md#CrossAttTransformer.Config.num_intermediate_factor)
- used by: [`atom_cross_att_encoder`](atom_cross_attention.md#atom_cross_att_encoder), [`atom_cross_att_decoder`](atom_cross_attention.md#atom_cross_att_decoder), [`AtomCrossAttDecoderConfig`](atom_cross_attention.md#AtomCrossAttDecoderConfig), [`atom_transformer`](atom_cross_attention.md#AtomCrossAttEncoderConfig.atom_transformer)

### `CrossAttentionConfig`  ·  implements/extends BaseConfig
- def: [`src/alphafold3/model/network/diffusion_transformer.py:266`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L266) — documented in [alphafold3-model-network-diffusion_transformer](../../../../../concepts/alphafold3-model-network-diffusion_transformer.md)
- signature: `class CrossAttentionConfig(base_config.BaseConfig):`
- members:
  - `key_dim` — [`L268`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L268)
  - `num_head` — [`L267`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L267)
  - `value_dim` — [`L269`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L269)
- used by: [`cross_attention`](diffusion_transformer.md#cross_attention), [`__call__`](diffusion_transformer.md#CrossAttTransformer.__call__), [`attention`](diffusion_transformer.md#CrossAttTransformer.Config.attention)

### `SelfAttentionConfig`  ·  implements/extends BaseConfig
- def: [`src/alphafold3/model/network/diffusion_transformer.py:123`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L123) — documented in [alphafold3-model-network-diffusion_transformer](../../../../../concepts/alphafold3-model-network-diffusion_transformer.md)
- signature: `class SelfAttentionConfig(base_config.BaseConfig):`
- members:
  - `key_dim` — [`L125`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L125) — documented in [alphafold3-model-network-diffusion_transformer](../../../../../concepts/alphafold3-model-network-diffusion_transformer.md)
  - `num_head` — [`L124`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L124) — documented in [alphafold3-model-network-diffusion_transformer](../../../../../concepts/alphafold3-model-network-diffusion_transformer.md)
  - `value_dim` — [`L126`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L126) — documented in [alphafold3-model-network-diffusion_transformer](../../../../../concepts/alphafold3-model-network-diffusion_transformer.md)
- used by: [`__call__`](modules.md#PairFormerIteration.__call__), [`self_attention`](diffusion_transformer.md#self_attention), [`super_block`](diffusion_transformer.md#Transformer.super_block), [`single_attention`](modules.md#PairFormerIteration.Config.single_attention), [`attention`](diffusion_transformer.md#Transformer.Config.attention)

### `Transformer`
- def: [`src/alphafold3/model/network/diffusion_transformer.py:189`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L189)
- doc: Simple transformer stack.
- signature: `class Transformer(hk.Module):`
- members:
  - `block(act, pair_logits)` — [`L216`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L216) — documented in [alphafold3-model-network-diffusion_transformer](../../../../../concepts/alphafold3-model-network-diffusion_transformer.md)
  - `super_block(act)` — [`L248`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L248)
  - `config` — [`L206`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L206)
  - `global_config` — [`L207`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L207)
- protocol/private: `__call__`[`L209`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L209), `__init__`[`L199`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L199)
- uses (calls/refs, reference-scoped): [`Linear`](../components/haiku_modules.md#Linear), [`LayerNorm`](../components/haiku_modules.md#LayerNorm), [`GlobalConfig`](../model_config.md#GlobalConfig), [`self_attention`](diffusion_transformer.md#self_attention), [`transition_block`](diffusion_transformer.md#transition_block), [`attention`](diffusion_transformer.md#Transformer.Config.attention), [`super_block_size`](diffusion_transformer.md#Transformer.Config.super_block_size), [`num_head`](diffusion_transformer.md#SelfAttentionConfig.num_head), [`Config`](diffusion_transformer.md#Transformer.Config), [`num_blocks`](diffusion_transformer.md#Transformer.Config.num_blocks), [`num_intermediate_factor`](diffusion_transformer.md#Transformer.Config.num_intermediate_factor)
- used by: [`__call__`](diffusion_head.md#DiffusionHead.__call__), [`transformer`](diffusion_head.md#DiffusionHead.Config.transformer)

## Functions
- `adaptive_layernorm(x, single_cond, name)` — [`L32`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L32) — Adaptive LayerNorm. — documented in [alphafold3-model-network-diffusion_transformer](../../../../../concepts/alphafold3-model-network-diffusion_transformer.md)
- `adaptive_zero_init(x, num_channels, single_cond, global_config: model_config.GlobalConfig, name)` — [`L63`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L63) — Adaptive zero init, from AdaLN-zero. — documented in [alphafold3-model-network-diffusion_transformer](../../../../../concepts/alphafold3-model-network-diffusion_transformer.md)
- `cross_attention(x_q: jnp.ndarray, x_k: jnp.ndarray, mask_q: jnp.ndarray, mask_k: jnp.ndarray, config: CrossAttentionConfig, global_config: model_config.GlobalConfig, pair_logits: jnp.ndarray | None = None, single_cond_q: jnp.ndarray | None = None, single_cond_k: jnp.ndarray | None = None, name: str = '')` — [`L272`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L272) — Multihead self-attention. — documented in [alphafold3-model-network-diffusion_transformer](../../../../../concepts/alphafold3-model-network-diffusion_transformer.md)
- `self_attention(x: jnp.ndarray, mask: jnp.ndarray, pair_logits: jnp.ndarray | None, config: SelfAttentionConfig, global_config: model_config.GlobalConfig, single_cond: jnp.ndarray | None = None, name: str = '')` — [`L129`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L129) — Multihead self-attention. — documented in [alphafold3-model-network-diffusion_transformer](../../../../../concepts/alphafold3-model-network-diffusion_transformer.md)
- `transition_block(x: jnp.ndarray, num_intermediate_factor: int, global_config: model_config.GlobalConfig, single_cond: jnp.ndarray | None = None, use_glu_kernel: bool = True, name: str = '')` — [`L87`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/diffusion_transformer.py#L87) — Transition Block. — documented in [alphafold3-model-network-diffusion_transformer](../../../../../concepts/alphafold3-model-network-diffusion_transformer.md)

