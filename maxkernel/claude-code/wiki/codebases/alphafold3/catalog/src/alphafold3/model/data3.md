---
title: 'Module: src/alphafold3/model/data3.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/data3.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.data3`/
symbols:
  fix_template_features: fix_template_features().
  empty_template_features: empty_template_features().
  FeatureDict.FeatureDict: FeatureDict.FeatureDict.
  get_profile_features: get_profile_features().
  _get_timestamp: _get_timestamp().
---
# Module: [`src/alphafold3/model/data3.py`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/data3.py)

## Functions
- `_get_timestamp(date_str: str)` — [`L124`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/data3.py#L124)
- `empty_template_features(num_res: int)` — [`L101`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/data3.py#L101) — Creates a fully masked out template features to allow padding to work.
- `fix_template_features(template_features: FeatureDict, num_res: int)` — [`L50`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/data3.py#L50) — Convert template features to AlphaFold 3 format.
- `get_profile_features(msa: np.ndarray, deletion_matrix: np.ndarray)` — [`L35`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/data3.py#L35) — Returns the MSA profile and deletion_mean features.

## Module values
- `FeatureDict` — [`L32`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/data3.py#L32)

