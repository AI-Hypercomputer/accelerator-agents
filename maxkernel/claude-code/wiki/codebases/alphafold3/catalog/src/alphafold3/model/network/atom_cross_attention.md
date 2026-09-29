---
title: 'Module: src/alphafold3/model/network/atom_cross_attention.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/network/atom_cross_attention.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.network.atom_cross_attention`/
symbols:
  atom_cross_att_encoder: atom_cross_att_encoder().
  atom_cross_att_decoder: atom_cross_att_decoder().
  _per_atom_conditioning: _per_atom_conditioning().
  AtomCrossAttEncoderConfig.per_atom_pair_channels: AtomCrossAttEncoderConfig#per_atom_pair_channels.
  AtomCrossAttDecoderConfig: AtomCrossAttDecoderConfig#
  AtomCrossAttEncoderConfig.per_atom_channels: AtomCrossAttEncoderConfig#per_atom_channels.
  AtomCrossAttEncoderConfig: AtomCrossAttEncoderConfig#
  AtomCrossAttEncoderConfig.atom_transformer: AtomCrossAttEncoderConfig#atom_transformer.
  AtomCrossAttEncoderOutput: AtomCrossAttEncoderOutput#
  AtomCrossAttEncoderOutput.pair_cond: AtomCrossAttEncoderOutput#pair_cond.
  AtomCrossAttEncoderOutput.queries_mask: AtomCrossAttEncoderOutput#queries_mask.
  AtomCrossAttEncoderOutput.token_act: AtomCrossAttEncoderOutput#token_act.
  AtomCrossAttEncoderOutput.skip_connection: AtomCrossAttEncoderOutput#skip_connection.
  AtomCrossAttEncoderOutput.queries_single_cond: AtomCrossAttEncoderOutput#queries_single_cond.
  AtomCrossAttEncoderOutput.keys_mask: AtomCrossAttEncoderOutput#keys_mask.
  AtomCrossAttEncoderOutput.keys_single_cond: AtomCrossAttEncoderOutput#keys_single_cond.
  AtomCrossAttDecoderConfig.per_atom_channels: AtomCrossAttDecoderConfig#per_atom_channels.
  AtomCrossAttEncoderConfig.per_token_channels: AtomCrossAttEncoderConfig#per_token_channels.
  AtomCrossAttDecoderConfig.atom_transformer: AtomCrossAttDecoderConfig#atom_transformer.
---
# Module: [`src/alphafold3/model/network/atom_cross_attention.py`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/atom_cross_attention.py)

## Classes
### `AtomCrossAttDecoderConfig`  ·  implements/extends BaseConfig
- def: [`src/alphafold3/model/network/atom_cross_attention.py:363`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/atom_cross_attention.py#L363)
- signature: `class AtomCrossAttDecoderConfig(base_config.BaseConfig):`
- members:
  - `atom_transformer` — [`L365`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/atom_cross_attention.py#L365)
  - `per_atom_channels` — [`L364`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/atom_cross_attention.py#L364)
- uses (calls/refs, reference-scoped): [`Config`](diffusion_head.md#DiffusionHead.Config), [`CrossAttTransformer`](diffusion_transformer.md#CrossAttTransformer), [`Config`](diffusion_transformer.md#CrossAttTransformer.Config)
- used by: [`atom_cross_att_decoder`](atom_cross_attention.md#atom_cross_att_decoder), [`Config`](diffusion_head.md#DiffusionHead.Config)

### `AtomCrossAttEncoderConfig`  ·  implements/extends BaseConfig
- def: [`src/alphafold3/model/network/atom_cross_attention.py:34`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/atom_cross_attention.py#L34) — documented in [alphafold3-model-network-atom_cross_attention](../../../../../concepts/alphafold3-model-network-atom_cross_attention.md)
- signature: `class AtomCrossAttEncoderConfig(base_config.BaseConfig):`
- members:
  - `atom_transformer` — [`L37`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/atom_cross_attention.py#L37)
  - `per_atom_channels` — [`L36`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/atom_cross_attention.py#L36)
  - `per_atom_pair_channels` — [`L40`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/atom_cross_attention.py#L40)
  - `per_token_channels` — [`L35`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/atom_cross_attention.py#L35)
- uses (calls/refs, reference-scoped): [`Config`](diffusion_head.md#DiffusionHead.Config), [`CrossAttTransformer`](diffusion_transformer.md#CrossAttTransformer), [`Config`](diffusion_transformer.md#CrossAttTransformer.Config)
- used by: [`atom_cross_att_encoder`](atom_cross_attention.md#atom_cross_att_encoder), [`_per_atom_conditioning`](atom_cross_attention.md#_per_atom_conditioning), [`Config`](diffusion_head.md#DiffusionHead.Config), [`per_atom_conditioning`](evoformer.md#Evoformer.Config.per_atom_conditioning)

### `AtomCrossAttEncoderOutput`
- def: [`src/alphafold3/model/network/atom_cross_attention.py:109`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/atom_cross_attention.py#L109) — documented in [alphafold3-model-network-atom_cross_attention](../../../../../concepts/alphafold3-model-network-atom_cross_attention.md)
- signature: `class AtomCrossAttEncoderOutput:`
- members:
  - `keys_mask` — [`L114`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/atom_cross_attention.py#L114) — documented in [alphafold3-model-network-atom_cross_attention](../../../../../concepts/alphafold3-model-network-atom_cross_attention.md)
  - `keys_single_cond` — [`L115`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/atom_cross_attention.py#L115) — documented in [alphafold3-model-network-atom_cross_attention](../../../../../concepts/alphafold3-model-network-atom_cross_attention.md)
  - `pair_cond` — [`L116`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/atom_cross_attention.py#L116) — documented in [alphafold3-model-network-atom_cross_attention](../../../../../concepts/alphafold3-model-network-atom_cross_attention.md)
  - `queries_mask` — [`L112`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/atom_cross_attention.py#L112) — documented in [alphafold3-model-network-atom_cross_attention](../../../../../concepts/alphafold3-model-network-atom_cross_attention.md)
  - `queries_single_cond` — [`L113`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/atom_cross_attention.py#L113) — documented in [alphafold3-model-network-atom_cross_attention](../../../../../concepts/alphafold3-model-network-atom_cross_attention.md)
  - `skip_connection` — [`L111`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/atom_cross_attention.py#L111) — documented in [alphafold3-model-network-atom_cross_attention](../../../../../concepts/alphafold3-model-network-atom_cross_attention.md)
  - `token_act` — [`L110`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/atom_cross_attention.py#L110) — documented in [alphafold3-model-network-atom_cross_attention](../../../../../concepts/alphafold3-model-network-atom_cross_attention.md)
- used by: [`atom_cross_att_encoder`](atom_cross_attention.md#atom_cross_att_encoder), [`atom_cross_att_decoder`](atom_cross_attention.md#atom_cross_att_decoder), [`__call__`](diffusion_head.md#DiffusionHead.__call__), [`create_target_feat_embedding`](../model.md#create_target_feat_embedding)

## Functions
- `_per_atom_conditioning(config: AtomCrossAttEncoderConfig, batch: feat_batch.Batch, name: str)` — [`L43`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/atom_cross_attention.py#L43) — computes single and pair conditioning for all atoms in each token. — documented in [alphafold3-model-network-atom_cross_attention](../../../../../concepts/alphafold3-model-network-atom_cross_attention.md)
- `atom_cross_att_decoder(token_act: jnp.ndarray, enc: AtomCrossAttEncoderOutput, config: AtomCrossAttDecoderConfig, global_config: model_config.GlobalConfig, batch: feat_batch.Batch, name: str)` — [`L370`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/atom_cross_attention.py#L370) — Mapping to per-atom features and self-attention on subsets. — documented in [alphafold3-model-feat_batch](../../../../../concepts/alphafold3-model-feat_batch.md)
- `atom_cross_att_encoder(token_atoms_act: jnp.ndarray | None, trunk_single_cond: jnp.ndarray | None, trunk_pair_cond: jnp.ndarray | None, config: AtomCrossAttEncoderConfig, global_config: model_config.GlobalConfig, batch: feat_batch.Batch, name: str)` — [`L126`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/atom_cross_attention.py#L126) — Cross-attention on flat atom subsets and mapping to per-token features. — documented in [alphafold3-model-feat_batch](../../../../../concepts/alphafold3-model-feat_batch.md)

