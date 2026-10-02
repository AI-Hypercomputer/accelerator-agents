---
title: 'Module: src/alphafold3/model/network/confidence_head.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/network/confidence_head.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.network.confidence_head`/
symbols:
  ConfidenceHead.__call__: ConfidenceHead#__call__().
  ConfidenceHead._embed_features: ConfidenceHead#_embed_features().
  ConfidenceHead.config: ConfidenceHead#config.
  ConfidenceHead.pairformer_fn: ConfidenceHead#pairformer_fn().
  ConfidenceHead.Config.pae: ConfidenceHead#Config#pae.
  ConfidenceHead.Config.pairformer: ConfidenceHead#Config#pairformer.
  ConfidenceHead.global_config: ConfidenceHead#global_config.
  ConfidenceHead.__init__: ConfidenceHead#__init__().
  ConfidenceHead.Config.dgram_features: ConfidenceHead#Config#dgram_features.
  ConfidenceHead._get_tmscore_adjusted_pae: ConfidenceHead#_get_tmscore_adjusted_pae().
  ConfidenceHead: ConfidenceHead#
  ConfidenceHead.PAEConfig.num_bins: ConfidenceHead#PAEConfig#num_bins.
  ConfidenceHead.Config: ConfidenceHead#Config#
  ConfidenceHead.Config.num_plddt_bins: ConfidenceHead#Config#num_plddt_bins.
  ConfidenceHead.Config.num_bins: ConfidenceHead#Config#num_bins.
  ConfidenceHead.get_tmscore_adjusted_pae: ConfidenceHead#get_tmscore_adjusted_pae().
  ConfidenceHead.PAEConfig: ConfidenceHead#PAEConfig#
  ConfidenceHead.PAEConfig.max_error_bin: ConfidenceHead#PAEConfig#max_error_bin.
  ConfidenceHead.Config.max_error_bin: ConfidenceHead#Config#max_error_bin.
  _safe_norm: _safe_norm().
  ConfidenceHead.Config.no_embedding_prob: ConfidenceHead#Config#no_embedding_prob.
---
# Module: [`src/alphafold3/model/network/confidence_head.py`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/confidence_head.py)

## Classes
### `ConfidenceHead`
- def: [`src/alphafold3/model/network/confidence_head.py:38`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/confidence_head.py#L38)
- doc: Head to predict the distance errors in a prediction.
- signature: `class ConfidenceHead(hk.Module):`
- members:
  - `__call__(self, dense_atom_positions: jnp.ndarray, embeddings: dict[str, jnp.ndarray], seq_mask: jnp.ndarray, token_atoms_to_pseudo_beta: atom_layout.GatherInfo, asym_id: jnp.ndarray)` — [`L101`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/confidence_head.py#L101) — Builds ConfidenceHead module.
  - `get_tmscore_adjusted_pae(num_interface_tokens, bin_centers, pae_probs)` — [`L289`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/confidence_head.py#L289)
  - `pairformer_fn(act)` — [`L144`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/confidence_head.py#L144)
  - `config` — [`L69`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/confidence_head.py#L69)
  - `global_config` — [`L70`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/confidence_head.py#L70)
- protocol/private: `__init__`[`L62`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/confidence_head.py#L62), `_embed_features`[`L72`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/confidence_head.py#L72), `_get_tmscore_adjusted_pae`[`L281`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/confidence_head.py#L281)
- uses (calls/refs, reference-scoped): [`Linear`](../components/haiku_modules.md#Linear), [`convert`](../atom_layout/atom_layout.md#convert), [`GatherInfo`](../atom_layout/atom_layout.md#GatherInfo), [`LayerNorm`](../components/haiku_modules.md#LayerNorm), [`GlobalConfig`](../model_config.md#GlobalConfig), [`final_init`](../model_config.md#GlobalConfig.final_init), [`dgram_from_positions`](template_modules.md#dgram_from_positions), [`pae`](confidence_head.md#ConfidenceHead.Config.pae), [`pairformer`](confidence_head.md#ConfidenceHead.Config.pairformer), [`bfloat16_context`](../components/utils.md#bfloat16_context), [`PairFormerIteration`](modules.md#PairFormerIteration), [`bfloat16`](../model_config.md#GlobalConfig.bfloat16), [`num_layer`](modules.md#PairFormerIteration.Config.num_layer), [`dgram_features`](confidence_head.md#ConfidenceHead.Config.dgram_features), [`Config`](confidence_head.md#ConfidenceHead.Config), [`num_bins`](confidence_head.md#ConfidenceHead.Config.num_bins), [`num_bins`](confidence_head.md#ConfidenceHead.PAEConfig.num_bins), [`num_plddt_bins`](confidence_head.md#ConfidenceHead.Config.num_plddt_bins), [`max_error_bin`](confidence_head.md#ConfidenceHead.Config.max_error_bin), [`max_error_bin`](confidence_head.md#ConfidenceHead.PAEConfig.max_error_bin)
- used by: [`__call__`](../model.md#Model.__call__), [`pae`](confidence_head.md#ConfidenceHead.Config.pae), [`confidence`](../model.md#Model.HeadsConfig.confidence)

### `Config`  ·  implements/extends BaseConfig
- def: [`src/alphafold3/model/network/confidence_head.py:45`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/confidence_head.py#L45)
- doc: Configuration for ConfidenceHead.
- signature: `class Config(base_config.BaseConfig):`
- members:
  - `dgram_features` — [`L58`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/confidence_head.py#L58)
  - `max_error_bin` — [`L53`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/confidence_head.py#L53)
  - `no_embedding_prob` — [`L56`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/confidence_head.py#L56)
  - `num_bins` — [`L55`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/confidence_head.py#L55)
  - `num_plddt_bins` — [`L54`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/confidence_head.py#L54)
  - `pae` — [`L57`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/confidence_head.py#L57)
  - `pairformer` — [`L48`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/confidence_head.py#L48)
- uses (calls/refs, reference-scoped): [`PairFormerIteration`](modules.md#PairFormerIteration), [`Config`](modules.md#PairFormerIteration.Config), [`ConfidenceHead`](confidence_head.md#ConfidenceHead), [`DistogramFeaturesConfig`](template_modules.md#DistogramFeaturesConfig), [`PAEConfig`](confidence_head.md#ConfidenceHead.PAEConfig)
- used by: [`__call__`](confidence_head.md#ConfidenceHead.__call__), [`_embed_features`](confidence_head.md#ConfidenceHead._embed_features), [`pairformer_fn`](confidence_head.md#ConfidenceHead.pairformer_fn), [`confidence`](../model.md#Model.HeadsConfig.confidence), [`__init__`](confidence_head.md#ConfidenceHead.__init__)

### `PAEConfig`  ·  implements/extends BaseConfig
- def: [`src/alphafold3/model/network/confidence_head.py:41`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/confidence_head.py#L41)
- signature: `class PAEConfig(base_config.BaseConfig):`
- members:
  - `max_error_bin` — [`L42`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/confidence_head.py#L42)
  - `num_bins` — [`L43`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/confidence_head.py#L43)
- used by: [`__call__`](confidence_head.md#ConfidenceHead.__call__), [`pae`](confidence_head.md#ConfidenceHead.Config.pae)

## Functions
- `_safe_norm(x, keepdims, axis, eps=1e-8)` — [`L34`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/confidence_head.py#L34)

