---
title: 'Module: src/alphafold3/model/feat_batch.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/feat_batch.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.feat_batch`/Batch#
symbols:
  Batch.from_data_dict: from_data_dict().
  Batch.as_data_dict: as_data_dict().
  Batch.atom_cross_att: atom_cross_att.
  Batch: ''
  Batch.ref_structure: ref_structure.
  Batch.token_features: token_features.
  Batch.convert_model_output: convert_model_output.
  Batch.msa: msa.
  Batch.predicted_structure_info: predicted_structure_info.
  Batch.templates: templates.
  Batch.polymer_ligand_bond_info: polymer_ligand_bond_info.
  Batch.ligand_ligand_bond_info: ligand_ligand_bond_info.
  Batch.num_res: num_res().
  Batch.pseudo_beta_info: pseudo_beta_info.
  Batch.frames: frames.
---
# Module: [`src/alphafold3/model/feat_batch.py`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/feat_batch.py)

## Classes
### `Batch`
- def: [`src/alphafold3/model/feat_batch.py:29`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/feat_batch.py#L29) — documented in [alphafold3-model-feat_batch](../../../../concepts/alphafold3-model-feat_batch.md)
- doc: Dataclass containing batch.
- signature: `class Batch:`
- members:
  - `as_data_dict(self)` — [`L71`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/feat_batch.py#L71) — Converts batch object to dictionary. — documented in [alphafold3-model-feat_batch](../../../../concepts/alphafold3-model-feat_batch.md)
  - `from_data_dict(cls, batch: features.BatchDict)` — [`L49`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/feat_batch.py#L49) — Construct batch object from dictionary. — documented in [alphafold3-model-feat_batch](../../../../concepts/alphafold3-model-feat_batch.md)
  - `num_res(self)` — [`L45`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/feat_batch.py#L45)
  - `atom_cross_att` — [`L40`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/feat_batch.py#L40) — documented in [alphafold3-model-feat_batch](../../../../concepts/alphafold3-model-feat_batch.md)
  - `convert_model_output` — [`L41`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/feat_batch.py#L41) — documented in [alphafold3-model-feat_batch](../../../../concepts/alphafold3-model-feat_batch.md)
  - `frames` — [`L42`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/feat_batch.py#L42)
  - `ligand_ligand_bond_info` — [`L38`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/feat_batch.py#L38) — documented in [alphafold3-model-network-evoformer](../../../../concepts/alphafold3-model-network-evoformer.md)
  - `msa` — [`L32`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/feat_batch.py#L32) — documented in [alphafold3-model-feat_batch](../../../../concepts/alphafold3-model-feat_batch.md)
  - `polymer_ligand_bond_info` — [`L37`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/feat_batch.py#L37) — documented in [alphafold3-model-network-evoformer](../../../../concepts/alphafold3-model-network-evoformer.md)
  - `predicted_structure_info` — [`L36`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/feat_batch.py#L36) — documented in [alphafold3-model-network-diffusion_head](../../../../concepts/alphafold3-model-network-diffusion_head.md)
  - `pseudo_beta_info` — [`L39`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/feat_batch.py#L39)
  - `ref_structure` — [`L35`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/feat_batch.py#L35) — documented in [alphafold3-model-feat_batch](../../../../concepts/alphafold3-model-feat_batch.md)
  - `templates` — [`L33`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/feat_batch.py#L33) — documented in [alphafold3-model-network-evoformer](../../../../concepts/alphafold3-model-network-evoformer.md)
  - `token_features` — [`L34`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/feat_batch.py#L34) — documented in [alphafold3-model-feat_batch](../../../../concepts/alphafold3-model-feat_batch.md)
- uses (calls/refs, reference-scoped): [`as_data_dict`](features.md#TokenFeatures.as_data_dict), [`BatchDict`](features.md#BatchDict.BatchDict), [`as_data_dict`](features.md#AtomCrossAtt.as_data_dict), [`as_data_dict`](features.md#MSA.as_data_dict), [`as_data_dict`](features.md#RefStructure.as_data_dict), [`as_data_dict`](features.md#ConvertModelOutput.as_data_dict), [`MSA`](features.md#MSA), [`as_data_dict`](features.md#PolymerLigandBondInfo.as_data_dict), [`as_data_dict`](features.md#Templates.as_data_dict), [`Templates`](features.md#Templates), [`as_data_dict`](features.md#LigandLigandBondInfo.as_data_dict), [`as_data_dict`](features.md#PredictedStructureInfo.as_data_dict), [`as_data_dict`](features.md#PseudoBetaInfo.as_data_dict), [`from_data_dict`](features.md#AtomCrossAtt.from_data_dict), [`from_data_dict`](features.md#LigandLigandBondInfo.from_data_dict), [`from_data_dict`](features.md#PolymerLigandBondInfo.from_data_dict), [`from_data_dict`](features.md#PseudoBetaInfo.from_data_dict), [`TokenFeatures`](features.md#TokenFeatures), [`aatype`](features.md#TokenFeatures.aatype), [`RefStructure`](features.md#RefStructure), [`as_data_dict`](features.md#Frames.as_data_dict), [`from_data_dict`](features.md#ConvertModelOutput.from_data_dict), [`AtomCrossAtt`](features.md#AtomCrossAtt), [`ConvertModelOutput`](features.md#ConvertModelOutput), [`Frames`](features.md#Frames), [`LigandLigandBondInfo`](features.md#LigandLigandBondInfo), [`PolymerLigandBondInfo`](features.md#PolymerLigandBondInfo), [`PredictedStructureInfo`](features.md#PredictedStructureInfo), [`PseudoBetaInfo`](features.md#PseudoBetaInfo), [`from_data_dict`](features.md#Frames.from_data_dict), [`from_data_dict`](features.md#MSA.from_data_dict), [`from_data_dict`](features.md#PredictedStructureInfo.from_data_dict), [`from_data_dict`](features.md#RefStructure.from_data_dict), [`from_data_dict`](features.md#Templates.from_data_dict), [`from_data_dict`](features.md#TokenFeatures.from_data_dict)
- used by: [`process_structure`](pipeline/pipeline.md#WholePdbPipeline.process_structure), [`atom_cross_att_encoder`](network/atom_cross_attention.md#atom_cross_att_encoder), [`__call__`](model.md#Model.__call__), [`get_inference_result`](model.md#Model.get_inference_result), [`atom_cross_att_decoder`](network/atom_cross_attention.md#atom_cross_att_decoder), [`__call__`](network/evoformer.md#Evoformer.__call__), [`__call__`](network/diffusion_head.md#DiffusionHead.__call__), [`get_predicted_structure`](model.md#get_predicted_structure), [`_conditioning`](network/diffusion_head.md#DiffusionHead._conditioning), [`_embed_bonds`](network/evoformer.md#Evoformer._embed_bonds), [`__call__`](network/distogram_head.md#DistogramHead.__call__), [`_per_atom_conditioning`](network/atom_cross_attention.md#_per_atom_conditioning), [`create_target_feat`](network/featurization.md#create_target_feat), [`create_target_feat_embedding`](model.md#create_target_feat_embedding), [`_embed_template_pair`](network/evoformer.md#Evoformer._embed_template_pair), [`_relative_encoding`](network/evoformer.md#Evoformer._relative_encoding), [`sample`](network/diffusion_head.md#sample), [`pairformer_fn`](network/evoformer.md#Evoformer.pairformer_fn), [`process_item`](pipeline/pipeline.md#WholePdbPipeline.process_item), [`_sample_diffusion`](model.md#Model._sample_diffusion)

