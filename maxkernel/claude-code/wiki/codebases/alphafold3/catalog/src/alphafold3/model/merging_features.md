---
title: 'Module: src/alphafold3/model/merging_features.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/merging_features.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.merging_features`/
symbols:
  merge_paired_and_unpaired_msa: merge_paired_and_unpaired_msa().
  BatchDict.BatchDict: BatchDict.BatchDict.
  _pad_features_to_max: _pad_features_to_max().
  merge_msa_features: merge_msa_features().
  NUM_SEQ_NUM_RES_MSA_FEATURES: NUM_SEQ_NUM_RES_MSA_FEATURES.
  NUM_SEQ_MSA_FEATURES: NUM_SEQ_MSA_FEATURES.
  MSA_PAD_VALUES: MSA_PAD_VALUES.
  xnp_ndarray.xnp_ndarray: xnp_ndarray.xnp_ndarray.
---
# Module: [`src/alphafold3/model/merging_features.py`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/merging_features.py)

## Functions
- `_pad_features_to_max(feat_name: str, chains: list[BatchDict], axis: int)` — [`L42`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/merging_features.py#L42) — Pad a set of features to the maximum size amongst all chains.
- `merge_msa_features(feat_name: str, chains: list[BatchDict])` — [`L73`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/merging_features.py#L73) — Merges MSA features with shape (NUM_SEQ, NUM_RES) across chains.
- `merge_paired_and_unpaired_msa(example: BatchDict)` — [`L87`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/merging_features.py#L87) — Concatenates the paired (all_seq) MSA features with the unpaired ones.

## Module values
- `BatchDict` — [`L39`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/merging_features.py#L39)
- `MSA_PAD_VALUES` — [`L35`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/merging_features.py#L35)
- `NUM_SEQ_MSA_FEATURES` — [`L34`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/merging_features.py#L34)
- `NUM_SEQ_NUM_RES_MSA_FEATURES` — [`L33`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/merging_features.py#L33)
- `xnp_ndarray` — [`L38`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/merging_features.py#L38)

