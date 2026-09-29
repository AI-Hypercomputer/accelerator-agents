---
title: 'Module: src/alphafold3/model/network/featurization.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/network/featurization.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.network.featurization`/
symbols:
  create_target_feat: create_target_feat().
  create_relative_encoding: create_relative_encoding().
  shuffle_msa: shuffle_msa().
  create_msa_feat: create_msa_feat().
  truncate_msa_batch: truncate_msa_batch().
  _grid_keys: _grid_keys().
  _padding_consistent_rng: _padding_consistent_rng().
  _padding_consistent_rng.inner: _padding_consistent_rng().inner().
  gumbel_argsort_sample_idx: gumbel_argsort_sample_idx().
---
# Module: [`src/alphafold3/model/network/featurization.py`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/featurization.py)

## Functions
- `_grid_keys(key, shape)` — [`L32`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/featurization.py#L32) — Generate a grid of rng keys that is consistent with different padding.
- `_padding_consistent_rng(f)` — [`L53`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/featurization.py#L53) — Modify any element-wise random function to be consistent with padding.
- `create_msa_feat(msa: features.MSA)` — [`L120`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/featurization.py#L120) — Create and concatenate MSA features. — documented in [alphafold3-model-network-evoformer](../../../../../concepts/alphafold3-model-network-evoformer.md)
- `create_relative_encoding(seq_features: features.TokenFeatures, max_relative_idx: int, max_relative_chain: int)` — [`L177`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/featurization.py#L177) — Add relative position encodings. — documented in [alphafold3-model-network-diffusion_head](../../../../../concepts/alphafold3-model-network-diffusion_head.md)
- `create_target_feat(batch: feat_batch.Batch, append_per_atom_features: bool)` — [`L145`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/featurization.py#L145) — Make target feat.
- `gumbel_argsort_sample_idx(key: jnp.ndarray, logits: jnp.ndarray)` — [`L90`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/featurization.py#L90) — Samples with replacement from a distribution given by 'logits'.
- `inner(key, shape, **kwargs)` — [`L76`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/featurization.py#L76)
- `shuffle_msa(key: jax.Array, msa: features.MSA)` — [`L260`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/featurization.py#L260) — Shuffle MSA randomly, return batch with shuffled MSA. — documented in [alphafold3-model-network-evoformer](../../../../../concepts/alphafold3-model-network-evoformer.md)
- `truncate_msa_batch(msa: features.MSA, num_msa: int)` — [`L140`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/featurization.py#L140) — documented in [alphafold3-model-network-evoformer](../../../../../concepts/alphafold3-model-network-evoformer.md)

