---
title: 'Module: src/alphafold3/model/network/distogram_head.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/network/distogram_head.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.network.distogram_head`/
symbols:
  DistogramHead.__call__: DistogramHead#__call__().
  DistogramHead.__init__: DistogramHead#__init__().
  DistogramHead.config: DistogramHead#config.
  DistogramHead: DistogramHead#
  DistogramHead.Config: DistogramHead#Config#
  DistogramHead.Config.num_bins: DistogramHead#Config#num_bins.
  DistogramHead.Config.first_break: DistogramHead#Config#first_break.
  DistogramHead.Config.last_break: DistogramHead#Config#last_break.
  DistogramHead.global_config: DistogramHead#global_config.
  _CONTACT_THRESHOLD._CONTACT_THRESHOLD: _CONTACT_THRESHOLD._CONTACT_THRESHOLD.
  _CONTACT_EPSILON._CONTACT_EPSILON: _CONTACT_EPSILON._CONTACT_EPSILON.
---
# Module: [`src/alphafold3/model/network/distogram_head.py`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/distogram_head.py)

## Classes
### `Config`  ·  implements/extends BaseConfig
- def: [`src/alphafold3/model/network/distogram_head.py:40`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/distogram_head.py#L40)
- signature: `class Config(base_config.BaseConfig):`
- members:
  - `first_break` — [`L41`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/distogram_head.py#L41)
  - `last_break` — [`L42`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/distogram_head.py#L42)
  - `num_bins` — [`L43`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/distogram_head.py#L43)
- used by: [`__call__`](distogram_head.md#DistogramHead.__call__), [`distogram`](../model.md#Model.HeadsConfig.distogram), [`__init__`](distogram_head.md#DistogramHead.__init__)

### `DistogramHead`
- def: [`src/alphafold3/model/network/distogram_head.py:37`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/distogram_head.py#L37)
- doc: Distogram head.
- signature: `class DistogramHead(hk.Module):`
- members:
  - `config` — [`L52`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/distogram_head.py#L52)
  - `global_config` — [`L53`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/distogram_head.py#L53)
- protocol/private: `__call__`[`L55`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/distogram_head.py#L55), `__init__`[`L45`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/distogram_head.py#L45)
- uses (calls/refs, reference-scoped): [`Linear`](../components/haiku_modules.md#Linear), [`GlobalConfig`](../model_config.md#GlobalConfig), [`Batch`](../feat_batch.md#Batch), [`final_init`](../model_config.md#GlobalConfig.final_init), [`token_features`](../feat_batch.md#Batch.token_features), [`mask`](../features.md#TokenFeatures.mask), [`Config`](distogram_head.md#DistogramHead.Config), [`num_bins`](distogram_head.md#DistogramHead.Config.num_bins), [`_CONTACT_EPSILON`](distogram_head.md#_CONTACT_EPSILON._CONTACT_EPSILON), [`_CONTACT_THRESHOLD`](distogram_head.md#_CONTACT_THRESHOLD._CONTACT_THRESHOLD), [`first_break`](distogram_head.md#DistogramHead.Config.first_break), [`last_break`](distogram_head.md#DistogramHead.Config.last_break)
- used by: [`__call__`](../model.md#Model.__call__), [`distogram`](../model.md#Model.HeadsConfig.distogram)

## Module values
- `_CONTACT_EPSILON` — [`L34`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/distogram_head.py#L34)
- `_CONTACT_THRESHOLD` — [`L33`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/network/distogram_head.py#L33)

