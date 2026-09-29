---
title: 'Module: src/alphafold3/model/network/template_modules.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/network/template_modules.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.network.template_modules`/
symbols:
  SingleTemplateEmbedding.construct_input: SingleTemplateEmbedding#construct_input().
  TemplateEmbedding.__call__: TemplateEmbedding#__call__().
  SingleTemplateEmbedding.__call__: SingleTemplateEmbedding#__call__().
  dgram_from_positions: dgram_from_positions().
  make_backbone_rigid: make_backbone_rigid().
  TemplateEmbedding.Config.template_stack: TemplateEmbedding#Config#template_stack.
  SingleTemplateEmbedding.__init__: SingleTemplateEmbedding#__init__().
  SingleTemplateEmbedding.template_iteration_fn: SingleTemplateEmbedding#template_iteration_fn().
  TemplateEmbedding.__init__: TemplateEmbedding#__init__().
  TemplateEmbedding.Config.dgram_features: TemplateEmbedding#Config#dgram_features.
  DistogramFeaturesConfig: DistogramFeaturesConfig#
  TemplateEmbedding: TemplateEmbedding#
  TemplateEmbedding.Config: TemplateEmbedding#Config#
  SingleTemplateEmbedding.config: SingleTemplateEmbedding#config.
  TemplateEmbedding.Config.num_channels: TemplateEmbedding#Config#num_channels.
  TemplateEmbedding.config: TemplateEmbedding#config.
  DistogramFeaturesConfig.min_bin: DistogramFeaturesConfig#min_bin.
  DistogramFeaturesConfig.max_bin: DistogramFeaturesConfig#max_bin.
  DistogramFeaturesConfig.num_bins: DistogramFeaturesConfig#num_bins.
  TemplateEmbedding.global_config: TemplateEmbedding#global_config.
  TemplateEmbedding.scan_fn: TemplateEmbedding#scan_fn().
  SingleTemplateEmbedding.global_config: SingleTemplateEmbedding#global_config.
  SingleTemplateEmbedding: SingleTemplateEmbedding#
---
# Module: [`src/alphafold3/model/network/template_modules.py`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/template_modules.py)

## Classes
### `Config`  ·  implements/extends BaseConfig
- def: [`src/alphafold3/model/network/template_modules.py:121`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/template_modules.py#L121)
- signature: `class Config(base_config.BaseConfig):`
- members:
  - `dgram_features` — [`L127`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/template_modules.py#L127) — documented in [alphafold3-model-network-template_modules](../../../../../concepts/alphafold3-model-network-template_modules.md)
  - `num_channels` — [`L122`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/template_modules.py#L122) — documented in [alphafold3-model-network-template_modules](../../../../../concepts/alphafold3-model-network-template_modules.md)
  - `template_stack` — [`L123`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/template_modules.py#L123) — documented in [alphafold3-model-network-template_modules](../../../../../concepts/alphafold3-model-network-template_modules.md)
- uses (calls/refs, reference-scoped): [`PairFormerIteration`](modules.md#PairFormerIteration), [`Config`](modules.md#PairFormerIteration.Config), [`DistogramFeaturesConfig`](template_modules.md#DistogramFeaturesConfig)
- used by: [`construct_input`](template_modules.md#SingleTemplateEmbedding.construct_input), [`__call__`](template_modules.md#TemplateEmbedding.__call__), [`__call__`](template_modules.md#SingleTemplateEmbedding.__call__), [`__init__`](template_modules.md#SingleTemplateEmbedding.__init__), [`template`](evoformer.md#Evoformer.Config.template), [`template_iteration_fn`](template_modules.md#SingleTemplateEmbedding.template_iteration_fn), [`__init__`](template_modules.md#TemplateEmbedding.__init__)

### `DistogramFeaturesConfig`  ·  implements/extends BaseConfig
- def: [`src/alphafold3/model/network/template_modules.py:36`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/template_modules.py#L36) — documented in [alphafold3-model-network-template_modules](../../../../../concepts/alphafold3-model-network-template_modules.md)
- signature: `class DistogramFeaturesConfig(base_config.BaseConfig):`
- members:
  - `max_bin` — [`L41`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/template_modules.py#L41) — documented in [alphafold3-model-network-template_modules](../../../../../concepts/alphafold3-model-network-template_modules.md)
  - `min_bin` — [`L38`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/template_modules.py#L38) — documented in [alphafold3-model-network-template_modules](../../../../../concepts/alphafold3-model-network-template_modules.md)
  - `num_bins` — [`L43`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/template_modules.py#L43) — documented in [alphafold3-model-network-template_modules](../../../../../concepts/alphafold3-model-network-template_modules.md)
- used by: [`dgram_from_positions`](template_modules.md#dgram_from_positions), [`dgram_features`](confidence_head.md#ConfidenceHead.Config.dgram_features), [`dgram_features`](template_modules.md#TemplateEmbedding.Config.dgram_features)

### `SingleTemplateEmbedding`
- def: [`src/alphafold3/model/network/template_modules.py:217`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/template_modules.py#L217) — documented in [alphafold3-model-network-template_modules](../../../../../concepts/alphafold3-model-network-template_modules.md)
- members:
  - `__call__(self, query_embedding: jnp.ndarray, templates: features.Templates, padding_mask_2d: jnp.ndarray, multichain_mask_2d: jnp.ndarray, key: jnp.ndarray)` — [`L230`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/template_modules.py#L230) — Build the single template embedding graph. — documented in [alphafold3-model-network-template_modules](../../../../../concepts/alphafold3-model-network-template_modules.md)
  - `construct_input(query_embedding, templates: features.Templates, multichain_mask_2d)` — [`L260`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/template_modules.py#L260) — documented in [alphafold3-jax-geometry-rigid_matrix_vector](../../../../../concepts/alphafold3-jax-geometry-rigid_matrix_vector.md)
  - `template_iteration_fn(x)` — [`L349`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/template_modules.py#L349) — documented in [alphafold3-model-network-template_modules](../../../../../concepts/alphafold3-model-network-template_modules.md)
  - `config` — [`L227`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/template_modules.py#L227)
  - `global_config` — [`L228`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/template_modules.py#L228)
- protocol/private: `__init__`[`L220`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/template_modules.py#L220)
- uses (calls/refs, reference-scoped): [`Linear`](../components/haiku_modules.md#Linear), [`LayerNorm`](../components/haiku_modules.md#LayerNorm), [`GlobalConfig`](../model_config.md#GlobalConfig), [`dgram_from_positions`](template_modules.md#dgram_from_positions), [`translation`](../../jax/geometry/rigid_matrix_vector.md#Rigid3Array.translation), [`make_backbone_rigid`](template_modules.md#make_backbone_rigid), [`aatype`](../features.md#Templates.aatype), [`Templates`](../features.md#Templates), [`template_stack`](template_modules.md#TemplateEmbedding.Config.template_stack), [`atom_mask`](../features.md#Templates.atom_mask), [`atom_positions`](../features.md#Templates.atom_positions), [`PairFormerIteration`](modules.md#PairFormerIteration), [`pseudo_beta_fn`](../scoring/scoring.md#pseudo_beta_fn), [`from_array`](../../jax/geometry/vector.md#Vec3Array.from_array), [`Vec3Array`](../../jax/geometry/__init__.md#Vec3Array), [`num_layer`](modules.md#PairFormerIteration.Config.num_layer), [`RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX`](../protein_data_processing.md#RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX), [`dgram_features`](template_modules.md#TemplateEmbedding.Config.dgram_features), [`Config`](template_modules.md#TemplateEmbedding.Config), [`TemplateEmbedding`](template_modules.md#TemplateEmbedding), [`num_channels`](template_modules.md#TemplateEmbedding.Config.num_channels)
- used by: [`__call__`](template_modules.md#TemplateEmbedding.__call__)

### `TemplateEmbedding`
- def: [`src/alphafold3/model/network/template_modules.py:118`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/template_modules.py#L118)
- doc: Embed a set of templates.
- signature: `class TemplateEmbedding(hk.Module):`
- members:
  - `__call__(self, query_embedding: jnp.ndarray, templates: features.Templates, padding_mask_2d: jnp.ndarray, multichain_mask_2d: jnp.ndarray, key: jnp.ndarray)` — [`L139`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/template_modules.py#L139) — Generate an embedding for a set of templates. — documented in [alphafold3-model-network-template_modules](../../../../../concepts/alphafold3-model-network-template_modules.md)
  - `scan_fn(carry, x)` — [`L189`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/template_modules.py#L189)
  - `config` — [`L136`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/template_modules.py#L136)
  - `global_config` — [`L137`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/template_modules.py#L137)
- protocol/private: `__init__`[`L129`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/template_modules.py#L129)
- uses (calls/refs, reference-scoped): [`Linear`](../components/haiku_modules.md#Linear), [`GlobalConfig`](../model_config.md#GlobalConfig), [`aatype`](../features.md#Templates.aatype), [`Templates`](../features.md#Templates), [`atom_mask`](../features.md#Templates.atom_mask), [`atom_positions`](../features.md#Templates.atom_positions), [`Config`](template_modules.md#TemplateEmbedding.Config), [`num_channels`](template_modules.md#TemplateEmbedding.Config.num_channels), [`SingleTemplateEmbedding`](template_modules.md#SingleTemplateEmbedding)
- used by: [`_embed_template_pair`](evoformer.md#Evoformer._embed_template_pair), [`__init__`](template_modules.md#SingleTemplateEmbedding.__init__), [`template`](evoformer.md#Evoformer.Config.template)

## Functions
- `dgram_from_positions(positions, config: DistogramFeaturesConfig)` — [`L46`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/template_modules.py#L46) — Compute distogram from amino acid positions. — documented in [alphafold3-model-network-template_modules](../../../../../concepts/alphafold3-model-network-template_modules.md)
- `make_backbone_rigid(positions: geometry.Vec3Array, mask: jnp.ndarray, group_indices: jnp.ndarray)` — [`L76`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/template_modules.py#L76) — Make backbone Rigid3Array and mask. — documented in [alphafold3-model-network-template_modules](../../../../../concepts/alphafold3-model-network-template_modules.md)

