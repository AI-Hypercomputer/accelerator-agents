---
title: 'Module: src/alphafold3/model/network/evoformer.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/network/evoformer.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.network.evoformer`/Evoformer#
symbols:
  Evoformer.__call__: __call__().
  Evoformer._embed_bonds: _embed_bonds().
  Evoformer._embed_process_msa: _embed_process_msa().
  Evoformer._relative_encoding: _relative_encoding().
  Evoformer._embed_template_pair: _embed_template_pair().
  Evoformer.config: config.
  Evoformer.pairformer_fn: pairformer_fn().
  Evoformer._seq_pair_embedding: _seq_pair_embedding().
  Evoformer.evoformer_fn: evoformer_fn().
  Evoformer.Config.pair_channel: Config#pair_channel.
  Evoformer.Config.pairformer: Config#pairformer.
  Evoformer.Config.msa_stack: Config#msa_stack.
  Evoformer.global_config: global_config.
  Evoformer.PairformerConfig: PairformerConfig#
  Evoformer.Config.template: Config#template.
  Evoformer.__init__: __init__().
  Evoformer: ''
  Evoformer.Config.per_atom_conditioning: Config#per_atom_conditioning.
  Evoformer.Config: Config#
  Evoformer.Config.seq_channel: Config#seq_channel.
  Evoformer.Config.msa_channel: Config#msa_channel.
  Evoformer.Config.max_relative_chain: Config#max_relative_chain.
  Evoformer.Config.max_relative_idx: Config#max_relative_idx.
  Evoformer.Config.num_msa: Config#num_msa.
  Evoformer.PairformerConfig.block_remat: PairformerConfig#block_remat.
  Evoformer.PairformerConfig.remat_block_size: PairformerConfig#remat_block_size.
---
# Module: [`src/alphafold3/model/network/evoformer.py`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py)

## Classes
### `Config`  ·  implements/extends BaseConfig
- def: [`src/alphafold3/model/network/evoformer.py:46`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py#L46)
- doc: Configuration for Evoformer.
- signature: `class Config(base_config.BaseConfig):`
- members:
  - `max_relative_chain` — [`L49`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py#L49)
  - `max_relative_idx` — [`L52`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py#L52)
  - `msa_channel` — [`L50`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py#L50)
  - `msa_stack` — [`L74`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py#L74) — documented in [alphafold3-model-network-evoformer](../../../../../concepts/alphafold3-model-network-evoformer.md)
  - `num_msa` — [`L53`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py#L53)
  - `pair_channel` — [`L54`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py#L54) — documented in [alphafold3-model-network-evoformer](../../../../../concepts/alphafold3-model-network-evoformer.md)
  - `pairformer` — [`L55`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py#L55) — documented in [alphafold3-model-network-evoformer](../../../../../concepts/alphafold3-model-network-evoformer.md)
  - `per_atom_conditioning` — [`L60`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py#L60)
  - `seq_channel` — [`L51`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py#L51) — documented in [alphafold3-model-network-evoformer](../../../../../concepts/alphafold3-model-network-evoformer.md)
  - `template` — [`L71`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py#L71) — documented in [alphafold3-model-network-evoformer](../../../../../concepts/alphafold3-model-network-evoformer.md)
- uses (calls/refs, reference-scoped): [`AtomCrossAttEncoderConfig`](atom_cross_attention.md#AtomCrossAttEncoderConfig), [`PairformerConfig`](evoformer.md#Evoformer.PairformerConfig), [`Evoformer`](evoformer.md#Evoformer), [`Config`](template_modules.md#TemplateEmbedding.Config), [`TemplateEmbedding`](template_modules.md#TemplateEmbedding), [`Config`](modules.md#EvoformerIteration.Config), [`EvoformerIteration`](modules.md#EvoformerIteration)
- used by: [`__call__`](../model.md#Model.__call__), [`__call__`](evoformer.md#Evoformer.__call__), [`_embed_bonds`](evoformer.md#Evoformer._embed_bonds), [`_embed_process_msa`](evoformer.md#Evoformer._embed_process_msa), [`create_target_feat_embedding`](../model.md#create_target_feat_embedding), [`_embed_template_pair`](evoformer.md#Evoformer._embed_template_pair), [`_relative_encoding`](evoformer.md#Evoformer._relative_encoding), [`pairformer_fn`](evoformer.md#Evoformer.pairformer_fn), [`_seq_pair_embedding`](evoformer.md#Evoformer._seq_pair_embedding), [`evoformer_fn`](evoformer.md#Evoformer.evoformer_fn), [`evoformer`](../model.md#Model.Config.evoformer), [`__init__`](evoformer.md#Evoformer.__init__)

### `Evoformer`
- def: [`src/alphafold3/model/network/evoformer.py:39`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py#L39)
- doc: Creates 'single' and 'pair' embeddings.
- signature: `class Evoformer(hk.Module):`
- members:
  - `_embed_bonds(self, batch: feat_batch.Batch, pair_activations: jnp.ndarray)` — [`L129`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py#L129) — Embeds bond features and merges into pair activations. — documented in [alphafold3-model-feat_batch](../../../../../concepts/alphafold3-model-feat_batch.md)
  - `_embed_process_msa(self, msa_batch: features.MSA, pair_activations: jnp.ndarray, pair_mask: jnp.ndarray, key: jnp.ndarray, target_feat: jnp.ndarray)` — [`L209`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py#L209) — Processes MSA and returns updated pair activations. — documented in [alphafold3-model-network-evoformer](../../../../../concepts/alphafold3-model-network-evoformer.md)
  - `_embed_template_pair(self, batch: feat_batch.Batch, pair_activations: jnp.ndarray, pair_mask: jnp.ndarray, key: jnp.ndarray)` — [`L180`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py#L180) — Embeds Templates and merges into pair activations. — documented in [alphafold3-model-feat_batch](../../../../../concepts/alphafold3-model-feat_batch.md)
  - `_relative_encoding(self, batch: feat_batch.Batch, pair_activations: jnp.ndarray)` — [`L86`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py#L86) — Add relative position encodings. — documented in [alphafold3-model-feat_batch](../../../../../concepts/alphafold3-model-feat_batch.md)
  - `_seq_pair_embedding(self, token_features: features.TokenFeatures, target_feat: jnp.ndarray)` — [`L103`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py#L103) — Generated Pair embedding from sequence. — documented in [alphafold3-model-network-evoformer](../../../../../concepts/alphafold3-model-network-evoformer.md)
  - `evoformer_fn(x)` — [`L236`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py#L236) — documented in [alphafold3-model-network-evoformer](../../../../../concepts/alphafold3-model-network-evoformer.md)
  - `pairformer_fn(x)` — [`L319`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py#L319)
  - `config` — [`L83`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py#L83)
  - `global_config` — [`L84`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py#L84)
- protocol/private: `__call__`[`L252`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py#L252), `__init__`[`L76`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py#L76)
- uses (calls/refs, reference-scoped): [`Linear`](../components/haiku_modules.md#Linear), [`LayerNorm`](../components/haiku_modules.md#LayerNorm), [`gather_idxs`](../atom_layout/atom_layout.md#GatherInfo.gather_idxs), [`gather_mask`](../atom_layout/atom_layout.md#GatherInfo.gather_mask), [`GlobalConfig`](../model_config.md#GlobalConfig), [`Batch`](../feat_batch.md#Batch), [`final_init`](../model_config.md#GlobalConfig.final_init), [`create_relative_encoding`](featurization.md#create_relative_encoding), [`token_features`](../feat_batch.md#Batch.token_features), [`MSA`](../features.md#MSA), [`shuffle_msa`](featurization.md#shuffle_msa), [`create_msa_feat`](featurization.md#create_msa_feat), [`mask`](../features.md#MSA.mask), [`mask`](../features.md#TokenFeatures.mask), [`TokenFeatures`](../features.md#TokenFeatures), [`aatype`](../features.md#TokenFeatures.aatype), [`msa`](../feat_batch.md#Batch.msa), [`msa_stack`](evoformer.md#Evoformer.Config.msa_stack), [`pair_channel`](evoformer.md#Evoformer.Config.pair_channel), [`pairformer`](evoformer.md#Evoformer.Config.pairformer), [`asym_id`](../features.md#TokenFeatures.asym_id), [`bfloat16_context`](../components/utils.md#bfloat16_context), [`token_index`](../features.md#TokenFeatures.token_index), [`PairFormerIteration`](modules.md#PairFormerIteration), [`template`](evoformer.md#Evoformer.Config.template), [`truncate_msa_batch`](featurization.md#truncate_msa_batch), [`ligand_ligand_bond_info`](../feat_batch.md#Batch.ligand_ligand_bond_info), [`polymer_ligand_bond_info`](../feat_batch.md#Batch.polymer_ligand_bond_info), [`templates`](../feat_batch.md#Batch.templates), [`tokens_to_ligand_ligand_bonds`](../features.md#LigandLigandBondInfo.tokens_to_ligand_ligand_bonds), [`tokens_to_polymer_ligand_bonds`](../features.md#PolymerLigandBondInfo.tokens_to_polymer_ligand_bonds), [`bfloat16`](../model_config.md#GlobalConfig.bfloat16), [`num_layer`](modules.md#PairFormerIteration.Config.num_layer), [`Config`](evoformer.md#Evoformer.Config), [`TemplateEmbedding`](template_modules.md#TemplateEmbedding), [`seq_channel`](evoformer.md#Evoformer.Config.seq_channel), [`EvoformerIteration`](modules.md#EvoformerIteration), [`msa_channel`](evoformer.md#Evoformer.Config.msa_channel), [`max_relative_chain`](evoformer.md#Evoformer.Config.max_relative_chain), [`max_relative_idx`](evoformer.md#Evoformer.Config.max_relative_idx)  (+2 more)
- used by: [`__call__`](../model.md#Model.__call__), [`create_target_feat_embedding`](../model.md#create_target_feat_embedding), [`evoformer`](../model.md#Model.Config.evoformer), [`pairformer`](evoformer.md#Evoformer.Config.pairformer)

### `PairformerConfig`  ·  implements/extends Config
- def: [`src/alphafold3/model/network/evoformer.py:42`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py#L42)
- signature: `class PairformerConfig(modules.PairFormerIteration.Config):`
- members:
  - `block_remat` — [`L43`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py#L43)
  - `remat_block_size` — [`L44`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/evoformer.py#L44)
- uses (calls/refs, reference-scoped): [`PairFormerIteration`](modules.md#PairFormerIteration), [`Config`](modules.md#PairFormerIteration.Config)
- used by: [`pairformer`](evoformer.md#Evoformer.Config.pairformer)

