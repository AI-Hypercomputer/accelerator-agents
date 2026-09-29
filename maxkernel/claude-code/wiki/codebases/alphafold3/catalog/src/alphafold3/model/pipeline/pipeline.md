---
title: 'Module: src/alphafold3/model/pipeline/pipeline.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/pipeline/pipeline.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.pipeline.pipeline`/
symbols:
  WholePdbPipeline.process_structure: WholePdbPipeline#process_structure().
  WholePdbPipeline._config: WholePdbPipeline#_config.
  WholePdbPipeline.process_item: WholePdbPipeline#process_item().
  WholePdbPipeline.Config.max_total_residues: WholePdbPipeline#Config#max_total_residues.
  WholePdbPipeline.Config.min_total_residues: WholePdbPipeline#Config#min_total_residues.
  WholePdbPipeline.Config.atom_cross_att_queries_subset_size: WholePdbPipeline#Config#atom_cross_att_queries_subset_size.
  WholePdbPipeline.__init__: WholePdbPipeline#__init__().
  TotalNumResOutOfRangeError: TotalNumResOutOfRangeError#
  WholePdbPipeline.Config.max_templates: WholePdbPipeline#Config#max_templates.
  WholePdbPipeline.Config.drop_ligand_leaving_atoms: WholePdbPipeline#Config#drop_ligand_leaving_atoms.
  WholePdbPipeline.Config.deterministic_frames: WholePdbPipeline#Config#deterministic_frames.
  WholePdbPipeline.Config.fix_standalone_glycans: WholePdbPipeline#Config#fix_standalone_glycans.
  _DETERMINISTIC_FRAMES_RANDOM_SEED: _DETERMINISTIC_FRAMES_RANDOM_SEED.
  calculate_bucket_size: calculate_bucket_size().
  NanDataError: NanDataError#
  MmcifNumChainsError: MmcifNumChainsError#
  WholePdbPipeline.Config: WholePdbPipeline#Config#
  WholePdbPipeline.Config.max_atoms_per_token: WholePdbPipeline#Config#max_atoms_per_token.
  WholePdbPipeline.Config.pad_num_chains: WholePdbPipeline#Config#pad_num_chains.
  WholePdbPipeline.Config.buckets: WholePdbPipeline#Config#buckets.
  WholePdbPipeline.Config.msa_crop_size: WholePdbPipeline#Config#msa_crop_size.
  WholePdbPipeline.Config.ref_max_modified_date: WholePdbPipeline#Config#ref_max_modified_date.
  WholePdbPipeline.Config.max_paired_sequence_per_species: WholePdbPipeline#Config#max_paired_sequence_per_species.
  WholePdbPipeline.Config.average_num_atoms_per_token: WholePdbPipeline#Config#average_num_atoms_per_token.
  WholePdbPipeline.Config.atom_cross_att_keys_subset_size: WholePdbPipeline#Config#atom_cross_att_keys_subset_size.
  WholePdbPipeline.Config.flatten_non_standard_residues: WholePdbPipeline#Config#flatten_non_standard_residues.
  WholePdbPipeline.Config.conformer_max_iterations: WholePdbPipeline#Config#conformer_max_iterations.
  WholePdbPipeline.Config.resolve_msa_overlaps: WholePdbPipeline#Config#resolve_msa_overlaps.
  WholePdbPipeline: WholePdbPipeline#
  WholePdbPipeline.Config.max_template_date: WholePdbPipeline#Config#max_template_date.
---
# Module: [`src/alphafold3/model/pipeline/pipeline.py`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py)

## Classes
### `Config`  ·  implements/extends BaseConfig
- def: [`src/alphafold3/model/pipeline/pipeline.py:89`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L89)
- doc: Configuration object for `WholePdbPipeline`.
- signature: `class Config(base_config.BaseConfig):`
- members:
  - `atom_cross_att_keys_subset_size` — [`L154`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L154)
  - `atom_cross_att_queries_subset_size` — [`L153`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L153)
  - `average_num_atoms_per_token` — [`L152`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L152)
  - `buckets` — [`L143`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L143)
  - `conformer_max_iterations` — [`L157`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L157)
  - `deterministic_frames` — [`L156`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L156)
  - `drop_ligand_leaving_atoms` — [`L151`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L151)
  - `fix_standalone_glycans` — [`L159`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L159)
  - `flatten_non_standard_residues` — [`L155`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L155)
  - `max_atoms_per_token` — [`L141`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L141)
  - `max_paired_sequence_per_species` — [`L150`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L150)
  - `max_template_date` — [`L147`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L147)
  - `max_templates` — [`L149`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L149)
  - `max_total_residues` — [`L144`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L144)
  - `min_total_residues` — [`L145`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L145)
  - `msa_crop_size` — [`L146`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L146)
  - `pad_num_chains` — [`L142`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L142)
  - `ref_max_modified_date` — [`L148`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L148)
  - `resolve_msa_overlaps` — [`L158`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L158)
- used by: [`process_structure`](pipeline.md#WholePdbPipeline.process_structure), [`__init__`](pipeline.md#WholePdbPipeline.__init__)

### `MmcifNumChainsError`  ·  implements/extends Exception
- def: [`src/alphafold3/model/pipeline/pipeline.py:82`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L82)
- doc: Raised if the mmcif file contains too many / too few chains.
- signature: `class MmcifNumChainsError(Exception):`
- used by: [`process_structure`](pipeline.md#WholePdbPipeline.process_structure)

### `NanDataError`  ·  implements/extends Exception
- def: [`src/alphafold3/model/pipeline/pipeline.py:74`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L74)
- doc: Raised if the data pipeline produces data containing nans.
- signature: `class NanDataError(Exception):`
- used by: [`process_item`](pipeline.md#WholePdbPipeline.process_item)

### `TotalNumResOutOfRangeError`  ·  implements/extends Exception
- def: [`src/alphafold3/model/pipeline/pipeline.py:78`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L78)
- doc: Raised if total number of residues for all chains outside allowed range.
- signature: `class TotalNumResOutOfRangeError(Exception):`
- used by: [`process_structure`](pipeline.md#WholePdbPipeline.process_structure)

### `WholePdbPipeline`
- def: [`src/alphafold3/model/pipeline/pipeline.py:86`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L86)
- doc: Processes an entire mmcif entity and merges the content.
- signature: `class WholePdbPipeline:`
- members:
  - `__init__(self, *, config: Config)` — [`L161`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L161) — Initializes WholePdb data pipeline.
  - `process_item(self, fold_input: folding_input.Input, random_state: np.random.RandomState, ccd: chemical_components.Ccd, random_seed: int | None = None)` — [`L440`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L440) — Takes requests from in_queue, adds (key, serialized ex) to out_queue.
  - `process_structure(self, struct: structure.Structure, random_state: np.random.RandomState, ccd: chemical_components.Ccd, unpaired_msa_by_chain_id: Mapping[str, str], paired_msa_by_chain_id: Mapping[str, str], templates_by_chain_id: Mapping[str, Sequence[folding_input.Template]], random_seed: int | None = None)` — [`L169`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L169) — Computes features for a structure and associated MSAs/templates.
- protocol/private: `_config`[`L167`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L167)
- uses (calls/refs, reference-scoped): [`compute_features`](../features.md#TokenFeatures.compute_features), [`compute_features`](../features.md#MSA.compute_features), [`atom_name`](../atom_layout/atom_layout.md#AtomLayout.atom_name), [`clean_structure`](structure_cleaning.md#clean_structure), [`as_data_dict`](../feat_batch.md#Batch.as_data_dict), [`compute_features`](../features.md#Templates.compute_features), [`Structure`](../../structure/structure.md#Structure), [`copy_and_update_globals`](../../structure/structure.md#Structure.copy_and_update_globals), [`compute_features`](../features.md#AtomCrossAtt.compute_features), [`compute_features`](../features.md#Frames.compute_features), [`compute_features`](../features.md#PolymerLigandBondInfo.compute_features), [`num_tokens`](../features.md#PaddingShapes.num_tokens), [`compute_features`](../features.md#RefStructure.compute_features), [`BatchDict`](../features.md#BatchDict.BatchDict), [`create_empty_output_struc_and_layout`](structure_cleaning.md#create_empty_output_struc_and_layout), [`atom_cross_att`](../feat_batch.md#Batch.atom_cross_att), [`compute_features`](../features.md#LigandLigandBondInfo.compute_features), [`compute_features`](../features.md#PseudoBetaInfo.compute_features), [`tokenizer`](../features.md#tokenizer), [`Batch`](../feat_batch.md#Batch), [`ref_structure`](../feat_batch.md#Batch.ref_structure), [`compute_features`](../features.md#ConvertModelOutput.compute_features), [`compute_features`](../features.md#PredictedStructureInfo.compute_features), [`get_polymer_ligand_and_ligand_ligand_bonds`](inter_chain_bonds.md#get_polymer_ligand_and_ligand_ligand_bonds), [`populate_missing_ccd_data`](../../structure/chemical_components.md#populate_missing_ccd_data), [`token_features`](../feat_batch.md#Batch.token_features), [`chemical_components_data`](../../structure/structure.md#Structure.chemical_components_data), [`name`](../../structure/structure.md#Structure.name), [`MSA`](../features.md#MSA), [`PaddingShapes`](../features.md#PaddingShapes), [`convert_model_output`](../feat_batch.md#Batch.convert_model_output), [`num_chains`](../../structure/structure.md#Structure.num_chains), [`Templates`](../features.md#Templates), [`TokenFeatures`](../features.md#TokenFeatures), [`msa`](../feat_batch.md#Batch.msa), [`predicted_structure_info`](../feat_batch.md#Batch.predicted_structure_info), [`RefStructure`](../features.md#RefStructure), [`ligand_ligand_bond_info`](../feat_batch.md#Batch.ligand_ligand_bond_info), [`polymer_ligand_bond_info`](../feat_batch.md#Batch.polymer_ligand_bond_info), [`templates`](../feat_batch.md#Batch.templates)  (+37 more)

## Functions
- `calculate_bucket_size(num_tokens: int, buckets: Sequence[int] | None)` — [`L43`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L43) — Calculates the bucket size to pad the data to.

## Module values
- `_DETERMINISTIC_FRAMES_RANDOM_SEED` — [`L40`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/pipeline.py#L40)

