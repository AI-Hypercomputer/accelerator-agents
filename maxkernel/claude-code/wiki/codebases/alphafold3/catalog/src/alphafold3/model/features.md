---
title: 'Module: src/alphafold3/model/features.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/features.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.features`/
symbols:
  TokenFeatures.compute_features: TokenFeatures#compute_features().
  MSA.compute_features: MSA#compute_features().
  Templates.compute_features: Templates#compute_features().
  TokenFeatures.as_data_dict: TokenFeatures#as_data_dict().
  AtomCrossAtt.compute_features: AtomCrossAtt#compute_features().
  Frames.compute_features: Frames#compute_features().
  PolymerLigandBondInfo.compute_features: PolymerLigandBondInfo#compute_features().
  PaddingShapes.num_tokens: PaddingShapes#num_tokens.
  xnp_ndarray.xnp_ndarray: xnp_ndarray.xnp_ndarray.
  RefStructure.compute_features: RefStructure#compute_features().
  BatchDict.BatchDict: BatchDict.BatchDict.
  _pad_to: _pad_to().
  tokenizer: tokenizer().
  LigandLigandBondInfo.compute_features: LigandLigandBondInfo#compute_features().
  PseudoBetaInfo.compute_features: PseudoBetaInfo#compute_features().
  MSA.index_msa_rows: MSA#index_msa_rows().
  _compute_asym_entity_and_sym_id: _compute_asym_entity_and_sym_id().
  MSA.as_data_dict: MSA#as_data_dict().
  RefStructure.as_data_dict: RefStructure#as_data_dict().
  AtomCrossAtt.as_data_dict: AtomCrossAtt#as_data_dict().
  PredictedStructureInfo.compute_features: PredictedStructureInfo#compute_features().
  get_reference: get_reference().
  ConvertModelOutput.compute_features: ConvertModelOutput#compute_features().
  ConvertModelOutput.as_data_dict: ConvertModelOutput#as_data_dict().
  MSA: MSA#
  PaddingShapes: PaddingShapes#
  RefStructure.positions: RefStructure#positions.
  Templates.as_data_dict: Templates#as_data_dict().
  PolymerLigandBondInfo.as_data_dict: PolymerLigandBondInfo#as_data_dict().
  Templates.aatype: Templates#aatype.
  AtomCrossAtt.queries_to_keys: AtomCrossAtt#queries_to_keys.
  AtomCrossAtt.token_atoms_to_queries: AtomCrossAtt#token_atoms_to_queries.
  Templates: Templates#
  MSA.mask: MSA#mask.
  TokenFeatures.mask: TokenFeatures#mask.
  PredictedStructureInfo.as_data_dict: PredictedStructureInfo#as_data_dict().
  PolymerLigandBondInfo.from_data_dict: PolymerLigandBondInfo#from_data_dict().
  LigandLigandBondInfo.from_data_dict: LigandLigandBondInfo#from_data_dict().
  LigandLigandBondInfo.as_data_dict: LigandLigandBondInfo#as_data_dict().
  PseudoBetaInfo.from_data_dict: PseudoBetaInfo#from_data_dict().
  PseudoBetaInfo.as_data_dict: PseudoBetaInfo#as_data_dict().
  AtomCrossAtt.from_data_dict: AtomCrossAtt#from_data_dict().
  TokenFeatures: TokenFeatures#
  MSA.rows: MSA#rows.
  MSA.deletion_matrix: MSA#deletion_matrix.
  MSA.profile: MSA#profile.
  MSA.deletion_mean: MSA#deletion_mean.
  TokenFeatures.aatype: TokenFeatures#aatype.
  RefStructure.mask: RefStructure#mask.
  RefStructure: RefStructure#
  MSA.num_alignments: MSA#num_alignments.
  Templates.atom_positions: Templates#atom_positions.
  Templates.atom_mask: Templates#atom_mask.
  TokenFeatures.token_index: TokenFeatures#token_index.
  TokenFeatures.asym_id: TokenFeatures#asym_id.
  PredictedStructureInfo.atom_mask: PredictedStructureInfo#atom_mask.
  AtomCrossAtt.queries_to_token_atoms: AtomCrossAtt#queries_to_token_atoms.
  _reduce_template_features: _reduce_template_features().
  ConvertModelOutput.from_data_dict: ConvertModelOutput#from_data_dict().
  Frames.as_data_dict: Frames#as_data_dict().
  TokenFeatures.residue_index: TokenFeatures#residue_index.
  TokenFeatures.entity_id: TokenFeatures#entity_id.
  TokenFeatures.sym_id: TokenFeatures#sym_id.
  RefStructure.element: RefStructure#element.
  RefStructure.ref_space_uid: RefStructure#ref_space_uid.
  ConvertModelOutput.flat_output_layout: ConvertModelOutput#flat_output_layout.
  ConvertModelOutput.empty_output_struc: ConvertModelOutput#empty_output_struc.
  AtomCrossAtt.tokens_to_queries: AtomCrossAtt#tokens_to_queries.
  _unwrap: _unwrap().
  PredictedStructureInfo: PredictedStructureInfo#
  PolymerLigandBondInfo: PolymerLigandBondInfo#
  LigandLigandBondInfo: LigandLigandBondInfo#
  PseudoBetaInfo: PseudoBetaInfo#
  ConvertModelOutput: ConvertModelOutput#
  AtomCrossAtt: AtomCrossAtt#
  Frames: Frames#
  Chains.sym_id: Chains#sym_id.
  TokenFeatures.seq_length: TokenFeatures#seq_length.
  TokenFeatures.is_protein: TokenFeatures#is_protein.
  TokenFeatures.is_rna: TokenFeatures#is_rna.
  TokenFeatures.is_dna: TokenFeatures#is_dna.
  TokenFeatures.is_ligand: TokenFeatures#is_ligand.
  TokenFeatures.is_nonstandard_polymer_chain: TokenFeatures#is_nonstandard_polymer_chain.
  TokenFeatures.is_water: TokenFeatures#is_water.
  PolymerLigandBondInfo.tokens_to_polymer_ligand_bonds: PolymerLigandBondInfo#tokens_to_polymer_ligand_bonds.
  LigandLigandBondInfo.tokens_to_ligand_ligand_bonds: LigandLigandBondInfo#tokens_to_ligand_ligand_bonds.
  RefStructure.charge: RefStructure#charge.
  RefStructure.atom_name_chars: RefStructure#atom_name_chars.
  ConvertModelOutput.token_atoms_layout: ConvertModelOutput#token_atoms_layout.
  AtomCrossAtt.tokens_to_keys: AtomCrossAtt#tokens_to_keys.
  PaddingShapes.msa_size: PaddingShapes#msa_size.
  PaddingShapes.num_atoms: PaddingShapes#num_atoms.
  Chains: Chains#
  Chains.chain_id: Chains#chain_id.
  MSA.from_data_dict: MSA#from_data_dict().
  Templates.from_data_dict: Templates#from_data_dict().
  TokenFeatures.from_data_dict: TokenFeatures#from_data_dict().
  PredictedStructureInfo.residue_center_index: PredictedStructureInfo#residue_center_index.
  PredictedStructureInfo.from_data_dict: PredictedStructureInfo#from_data_dict().
  PolymerLigandBondInfo.token_atoms_to_bonds: PolymerLigandBondInfo#token_atoms_to_bonds.
  PseudoBetaInfo.token_atoms_to_pseudo_beta: PseudoBetaInfo#token_atoms_to_pseudo_beta.
  random_augmentation: random_augmentation().
  RefStructure.from_data_dict: RefStructure#from_data_dict().
  ConvertModelOutput.polymer_ligand_bonds: ConvertModelOutput#polymer_ligand_bonds.
  ConvertModelOutput.ligand_ligand_bonds: ConvertModelOutput#ligand_ligand_bonds.
  Frames.mask: Frames#mask.
  Frames.from_data_dict: Frames#from_data_dict().
  _DEFAULT_BLANK_REF: _DEFAULT_BLANK_REF.
  Chains.asym_id: Chains#asym_id.
  Chains.entity_id: Chains#entity_id.
  MSA.safe_cast_int8: MSA#safe_cast_int8().
  _STANDARD_RESIDUES: _STANDARD_RESIDUES.
  PaddingShapes.num_chains: PaddingShapes#num_chains.
  PaddingShapes.num_templates: PaddingShapes#num_templates.
  random_rotation: random_rotation().
  _get_reference_positions_from_ccd_cif: _get_reference_positions_from_ccd_cif().
---
# Module: [`src/alphafold3/model/features.py`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py)

## Classes
### `AtomCrossAtt`
- def: [`src/alphafold3/model/features.py:1910`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1910)
- doc: Operate on flat atoms.
- signature: `class AtomCrossAtt:`
- members:
  - `as_data_dict(self)` — [`L2039`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L2039)
  - `compute_features(cls, all_token_atoms_layout: atom_layout.AtomLayout, queries_subset_size: int, keys_subset_size: int, padding_shapes: PaddingShapes)` — [`L1920`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1920) — Computes gather indices and meta data to work with a flat atom list.
  - `from_data_dict(cls, batch: BatchDict)` — [`L2020`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L2020) — documented in [alphafold3-model-feat_batch](../../../../concepts/alphafold3-model-feat_batch.md)
  - `queries_to_keys` — [`L1916`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1916) — documented in [alphafold3-model-network-atom_cross_attention](../../../../concepts/alphafold3-model-network-atom_cross_attention.md)
  - `queries_to_token_atoms` — [`L1917`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1917) — documented in [alphafold3-model-network-atom_cross_attention](../../../../concepts/alphafold3-model-network-atom_cross_attention.md)
  - `token_atoms_to_queries` — [`L1913`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1913) — documented in [alphafold3-model-network-atom_cross_attention](../../../../concepts/alphafold3-model-network-atom_cross_attention.md)
  - `tokens_to_keys` — [`L1915`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1915)
  - `tokens_to_queries` — [`L1914`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1914)
- uses (calls/refs, reference-scoped): [`AtomLayout`](atom_layout/atom_layout.md#AtomLayout), [`atom_name`](atom_layout/atom_layout.md#AtomLayout.atom_name), [`convert`](atom_layout/atom_layout.md#convert), [`num_tokens`](features.md#PaddingShapes.num_tokens), [`GatherInfo`](atom_layout/atom_layout.md#GatherInfo), [`compute_gather_idxs`](atom_layout/atom_layout.md#compute_gather_idxs), [`BatchDict`](features.md#BatchDict.BatchDict), [`copy_and_pad_to`](atom_layout/atom_layout.md#AtomLayout.copy_and_pad_to), [`gather_idxs`](atom_layout/atom_layout.md#GatherInfo.gather_idxs), [`gather_mask`](atom_layout/atom_layout.md#GatherInfo.gather_mask), [`shape`](atom_layout/atom_layout.md#AtomLayout.shape), [`as_dict`](atom_layout/atom_layout.md#GatherInfo.as_dict), [`input_shape`](atom_layout/atom_layout.md#GatherInfo.input_shape), [`from_dict`](atom_layout/atom_layout.md#GatherInfo.from_dict), [`PaddingShapes`](features.md#PaddingShapes), [`to_array`](atom_layout/atom_layout.md#AtomLayout.to_array), [`num_atoms`](features.md#PaddingShapes.num_atoms), [`from_array`](atom_layout/atom_layout.md#AtomLayout.from_array)
- used by: [`process_structure`](pipeline/pipeline.md#WholePdbPipeline.process_structure), [`atom_cross_att_encoder`](network/atom_cross_attention.md#atom_cross_att_encoder), [`from_data_dict`](feat_batch.md#Batch.from_data_dict), [`as_data_dict`](feat_batch.md#Batch.as_data_dict), [`atom_cross_att_decoder`](network/atom_cross_attention.md#atom_cross_att_decoder), [`atom_cross_att`](feat_batch.md#Batch.atom_cross_att)

### `Chains`
- def: [`src/alphafold3/model/features.py:114`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L114)
- signature: `class Chains:`
- members:
  - `asym_id` — [`L116`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L116)
  - `chain_id` — [`L115`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L115)
  - `entity_id` — [`L117`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L117)
  - `sym_id` — [`L118`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L118)
- used by: [`compute_features`](features.md#TokenFeatures.compute_features), [`_compute_asym_entity_and_sym_id`](features.md#_compute_asym_entity_and_sym_id)

### `ConvertModelOutput`
- def: [`src/alphafold3/model/features.py:1836`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1836)
- doc: Contains information needed to convert model flat output back to structure.
- signature: `class ConvertModelOutput:`
- members:
  - `as_data_dict(self)` — [`L1892`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1892) — documented in [alphafold3-model-feat_batch](../../../../concepts/alphafold3-model-feat_batch.md)
  - `compute_features(cls, all_token_atoms_layout: atom_layout.AtomLayout, padding_shapes: PaddingShapes, flat_output_layout: atom_layout.AtomLayout, empty_output_struc: structure.Structure, polymer_ligand_bonds: atom_layout.AtomLayout, ligand_ligand_bonds: atom_layout.AtomLayout)` — [`L1857`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1857) — Pads the all_token_atoms_layout and stores other data.
  - `from_data_dict(cls, batch: BatchDict)` — [`L1881`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1881) — Construct atom layout object from dictionary.
  - `empty_output_struc` — [`L1852`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1852)
  - `flat_output_layout` — [`L1851`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1851)
  - `ligand_ligand_bonds` — [`L1854`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1854)
  - `polymer_ligand_bonds` — [`L1853`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1853)
  - `token_atoms_layout` — [`L1850`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1850)
- uses (calls/refs, reference-scoped): [`AtomLayout`](atom_layout/atom_layout.md#AtomLayout), [`Structure`](../structure/structure.md#Structure), [`num_tokens`](features.md#PaddingShapes.num_tokens), [`BatchDict`](features.md#BatchDict.BatchDict), [`copy_and_pad_to`](atom_layout/atom_layout.md#AtomLayout.copy_and_pad_to), [`shape`](atom_layout/atom_layout.md#AtomLayout.shape), [`PaddingShapes`](features.md#PaddingShapes), [`_unwrap`](features.md#_unwrap)
- used by: [`process_structure`](pipeline/pipeline.md#WholePdbPipeline.process_structure), [`from_data_dict`](feat_batch.md#Batch.from_data_dict), [`as_data_dict`](feat_batch.md#Batch.as_data_dict), [`get_predicted_structure`](model.md#get_predicted_structure), [`convert_model_output`](feat_batch.md#Batch.convert_model_output)

### `Frames`
- def: [`src/alphafold3/model/features.py:2061`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L2061)
- doc: Features for backbone frames.
- signature: `class Frames:`
- members:
  - `as_data_dict(self)` — [`L2175`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L2175)
  - `compute_features(cls, all_tokens: atom_layout.AtomLayout, all_token_atoms_layout: atom_layout.AtomLayout, ref_structure: RefStructure, padding_shapes: PaddingShapes)` — [`L2067`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L2067) — Computes features for backbone frames.
  - `from_data_dict(cls, batch: BatchDict)` — [`L2172`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L2172)
  - `mask` — [`L2064`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L2064)
- uses (calls/refs, reference-scoped): [`AtomLayout`](atom_layout/atom_layout.md#AtomLayout), [`chain_id`](atom_layout/atom_layout.md#AtomLayout.chain_id), [`res_id`](atom_layout/atom_layout.md#AtomLayout.res_id), [`convert`](atom_layout/atom_layout.md#convert), [`chain_type`](atom_layout/atom_layout.md#AtomLayout.chain_type), [`num_tokens`](features.md#PaddingShapes.num_tokens), [`xnp_ndarray`](features.md#xnp_ndarray.xnp_ndarray), [`compute_gather_idxs`](atom_layout/atom_layout.md#compute_gather_idxs), [`BatchDict`](features.md#BatchDict.BatchDict), [`copy_and_pad_to`](atom_layout/atom_layout.md#AtomLayout.copy_and_pad_to), [`_pad_to`](features.md#_pad_to), [`gather_mask`](atom_layout/atom_layout.md#GatherInfo.gather_mask), [`shape`](atom_layout/atom_layout.md#AtomLayout.shape), [`PaddingShapes`](features.md#PaddingShapes), [`positions`](features.md#RefStructure.positions), [`mask`](features.md#RefStructure.mask), [`RefStructure`](features.md#RefStructure)
- used by: [`process_structure`](pipeline/pipeline.md#WholePdbPipeline.process_structure), [`from_data_dict`](feat_batch.md#Batch.from_data_dict), [`as_data_dict`](feat_batch.md#Batch.as_data_dict), [`frames`](feat_batch.md#Batch.frames)

### `LigandLigandBondInfo`
- def: [`src/alphafold3/model/features.py:1246`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1246)
- doc: Contains information about the location of ligand-ligand bonds.
- signature: `class LigandLigandBondInfo:`
- members:
  - `as_data_dict(self)` — [`L1333`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1333)
  - `compute_features(cls, all_tokens: atom_layout.AtomLayout, bond_layout: atom_layout.AtomLayout | None, padding_shapes: PaddingShapes)` — [`L1252`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1252) — Computes the InterChainBondInfo features.
  - `from_data_dict(cls, batch: BatchDict)` — [`L1326`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1326) — documented in [alphafold3-model-feat_batch](../../../../concepts/alphafold3-model-feat_batch.md)
  - `tokens_to_ligand_ligand_bonds` — [`L1249`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1249)
- uses (calls/refs, reference-scoped): [`AtomLayout`](atom_layout/atom_layout.md#AtomLayout), [`atom_name`](atom_layout/atom_layout.md#AtomLayout.atom_name), [`chain_id`](atom_layout/atom_layout.md#AtomLayout.chain_id), [`res_id`](atom_layout/atom_layout.md#AtomLayout.res_id), [`chain_type`](atom_layout/atom_layout.md#AtomLayout.chain_type), [`num_tokens`](features.md#PaddingShapes.num_tokens), [`GatherInfo`](atom_layout/atom_layout.md#GatherInfo), [`compute_gather_idxs`](atom_layout/atom_layout.md#compute_gather_idxs), [`BatchDict`](features.md#BatchDict.BatchDict), [`copy_and_pad_to`](atom_layout/atom_layout.md#AtomLayout.copy_and_pad_to), [`as_dict`](atom_layout/atom_layout.md#GatherInfo.as_dict), [`from_dict`](atom_layout/atom_layout.md#GatherInfo.from_dict), [`PaddingShapes`](features.md#PaddingShapes)
- used by: [`process_structure`](pipeline/pipeline.md#WholePdbPipeline.process_structure), [`from_data_dict`](feat_batch.md#Batch.from_data_dict), [`as_data_dict`](feat_batch.md#Batch.as_data_dict), [`_embed_bonds`](network/evoformer.md#Evoformer._embed_bonds), [`ligand_ligand_bond_info`](feat_batch.md#Batch.ligand_ligand_bond_info)

### `MSA`
- def: [`src/alphafold3/model/features.py:412`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L412)
- doc: Dataclass containing MSA.
- signature: `class MSA:`
- members:
  - `as_data_dict(self)` — [`L700`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L700)
  - `compute_features(cls, *, all_tokens: atom_layout.AtomLayout, standard_token_idxs: np.ndarray, padding_shapes: PaddingShapes, unpaired_msa_by_chain_id: Mapping[str, str], paired_msa_by_chain_id: Mapping[str, str], logging_name: str, max_paired_sequence_per_species: int, resolve_msa_overlaps: bool = True)` — [`L426`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L426) — Compute the msa features. — documented in [alphafold3-model-features](../../../../concepts/alphafold3-model-features.md)
  - `from_data_dict(cls, batch: BatchDict)` — [`L689`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L689)
  - `index_msa_rows(self, indices: xnp_ndarray)` — [`L676`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L676)
  - `safe_cast_int8(x)` — [`L655`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L655)
  - `deletion_matrix` — [`L417`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L417)
  - `deletion_mean` — [`L421`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L421)
  - `mask` — [`L416`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L416)
  - `num_alignments` — [`L423`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L423)
  - `profile` — [`L419`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L419)
  - `rows` — [`L415`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L415)
- uses (calls/refs, reference-scoped): [`AtomLayout`](atom_layout/atom_layout.md#AtomLayout), [`atom_name`](atom_layout/atom_layout.md#AtomLayout.atom_name), [`chain_id`](atom_layout/atom_layout.md#AtomLayout.chain_id), [`make_structure`](atom_layout/atom_layout.md#make_structure), [`num_tokens`](features.md#PaddingShapes.num_tokens), [`xnp_ndarray`](features.md#xnp_ndarray.xnp_ndarray), [`res_name`](atom_layout/atom_layout.md#AtomLayout.res_name), [`BatchDict`](features.md#BatchDict.BatchDict), [`_pad_to`](features.md#_pad_to), [`shape`](atom_layout/atom_layout.md#AtomLayout.shape), [`chain_single_letter_sequence`](../structure/structure.md#Structure.chain_single_letter_sequence), [`iter_chains`](../structure/structure.md#Structure.iter_chains), [`PaddingShapes`](features.md#PaddingShapes), [`merge_paired_and_unpaired_msa`](merging_features.md#merge_paired_and_unpaired_msa), [`remove_all_gapped_rows_from_all_seqs`](msa_pairing.md#remove_all_gapped_rows_from_all_seqs), [`filter_to_entity_type`](../structure/structure.md#Structure.filter_to_entity_type), [`NUM_SEQ_NUM_RES_MSA_FEATURES`](data_constants.md#NUM_SEQ_NUM_RES_MSA_FEATURES), [`create_paired_features`](msa_pairing.md#create_paired_features), [`deduplicate_unpaired_sequences`](msa_pairing.md#deduplicate_unpaired_sequences), [`merge_msa_features`](merging_features.md#merge_msa_features), [`msa_size`](features.md#PaddingShapes.msa_size), [`get_profile_features`](data3.md#get_profile_features), [`choose_paired_unpaired_msa_crop_sizes`](msa_pairing.md#choose_paired_unpaired_msa_crop_sizes)
- used by: [`process_structure`](pipeline/pipeline.md#WholePdbPipeline.process_structure), [`from_data_dict`](feat_batch.md#Batch.from_data_dict), [`as_data_dict`](feat_batch.md#Batch.as_data_dict), [`_embed_process_msa`](network/evoformer.md#Evoformer._embed_process_msa), [`create_target_feat`](network/featurization.md#create_target_feat), [`shuffle_msa`](network/featurization.md#shuffle_msa), [`create_msa_feat`](network/featurization.md#create_msa_feat), [`msa`](feat_batch.md#Batch.msa), [`truncate_msa_batch`](network/featurization.md#truncate_msa_batch)

### `PaddingShapes`
- def: [`src/alphafold3/model/features.py:61`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L61) — documented in [alphafold3-model-features](../../../../concepts/alphafold3-model-features.md)
- signature: `class PaddingShapes:`
- members:
  - `msa_size` — [`L63`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L63)
  - `num_atoms` — [`L66`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L66)
  - `num_chains` — [`L64`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L64)
  - `num_templates` — [`L65`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L65)
  - `num_tokens` — [`L62`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L62) — documented in [alphafold3-model-features](../../../../concepts/alphafold3-model-features.md)
- used by: [`process_structure`](pipeline/pipeline.md#WholePdbPipeline.process_structure), [`compute_features`](features.md#TokenFeatures.compute_features), [`compute_features`](features.md#MSA.compute_features), [`compute_features`](features.md#Templates.compute_features), [`compute_features`](features.md#AtomCrossAtt.compute_features), [`compute_features`](features.md#Frames.compute_features), [`compute_features`](features.md#PolymerLigandBondInfo.compute_features), [`compute_features`](features.md#RefStructure.compute_features), [`compute_features`](features.md#LigandLigandBondInfo.compute_features), [`compute_features`](features.md#PseudoBetaInfo.compute_features), [`compute_features`](features.md#ConvertModelOutput.compute_features), [`compute_features`](features.md#PredictedStructureInfo.compute_features)

### `PolymerLigandBondInfo`
- def: [`src/alphafold3/model/features.py:1123`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1123)
- doc: Contains information about polymer-ligand bonds.
- signature: `class PolymerLigandBondInfo:`
- members:
  - `as_data_dict(self)` — [`L1227`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1227) — documented in [alphafold3-model-feat_batch](../../../../concepts/alphafold3-model-feat_batch.md)
  - `compute_features(cls, all_tokens: atom_layout.AtomLayout, all_token_atoms_layout: atom_layout.AtomLayout, bond_layout: atom_layout.AtomLayout | None, padding_shapes: PaddingShapes)` — [`L1132`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1132) — Computes the InterChainBondInfo features.
  - `from_data_dict(cls, batch: BatchDict)` — [`L1217`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1217) — documented in [alphafold3-model-feat_batch](../../../../concepts/alphafold3-model-feat_batch.md)
  - `token_atoms_to_bonds` — [`L1129`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1129)
  - `tokens_to_polymer_ligand_bonds` — [`L1126`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1126)
- uses (calls/refs, reference-scoped): [`AtomLayout`](atom_layout/atom_layout.md#AtomLayout), [`atom_name`](atom_layout/atom_layout.md#AtomLayout.atom_name), [`chain_id`](atom_layout/atom_layout.md#AtomLayout.chain_id), [`res_id`](atom_layout/atom_layout.md#AtomLayout.res_id), [`chain_type`](atom_layout/atom_layout.md#AtomLayout.chain_type), [`num_tokens`](features.md#PaddingShapes.num_tokens), [`GatherInfo`](atom_layout/atom_layout.md#GatherInfo), [`compute_gather_idxs`](atom_layout/atom_layout.md#compute_gather_idxs), [`BatchDict`](features.md#BatchDict.BatchDict), [`copy_and_pad_to`](atom_layout/atom_layout.md#AtomLayout.copy_and_pad_to), [`gather_idxs`](atom_layout/atom_layout.md#GatherInfo.gather_idxs), [`gather_mask`](atom_layout/atom_layout.md#GatherInfo.gather_mask), [`shape`](atom_layout/atom_layout.md#AtomLayout.shape), [`as_dict`](atom_layout/atom_layout.md#GatherInfo.as_dict), [`input_shape`](atom_layout/atom_layout.md#GatherInfo.input_shape), [`from_dict`](atom_layout/atom_layout.md#GatherInfo.from_dict), [`PaddingShapes`](features.md#PaddingShapes)
- used by: [`process_structure`](pipeline/pipeline.md#WholePdbPipeline.process_structure), [`from_data_dict`](feat_batch.md#Batch.from_data_dict), [`as_data_dict`](feat_batch.md#Batch.as_data_dict), [`_embed_bonds`](network/evoformer.md#Evoformer._embed_bonds), [`polymer_ligand_bond_info`](feat_batch.md#Batch.polymer_ligand_bond_info)

### `PredictedStructureInfo`
- def: [`src/alphafold3/model/features.py:1054`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1054)
- doc: Contains information necessary to work with predicted structure.
- signature: `class PredictedStructureInfo:`
- members:
  - `as_data_dict(self)` — [`L1108`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1108)
  - `compute_features(cls, all_tokens: atom_layout.AtomLayout, all_token_atoms_layout: atom_layout.AtomLayout, padding_shapes: PaddingShapes)` — [`L1061`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1061) — Compute the PredictedStructureInfo features.
  - `from_data_dict(cls, batch: BatchDict)` — [`L1102`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1102)
  - `atom_mask` — [`L1057`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1057)
  - `residue_center_index` — [`L1058`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1058)
- uses (calls/refs, reference-scoped): [`AtomLayout`](atom_layout/atom_layout.md#AtomLayout), [`atom_name`](atom_layout/atom_layout.md#AtomLayout.atom_name), [`num_tokens`](features.md#PaddingShapes.num_tokens), [`xnp_ndarray`](features.md#xnp_ndarray.xnp_ndarray), [`BatchDict`](features.md#BatchDict.BatchDict), [`_pad_to`](features.md#_pad_to), [`shape`](atom_layout/atom_layout.md#AtomLayout.shape), [`PaddingShapes`](features.md#PaddingShapes)
- used by: [`process_structure`](pipeline/pipeline.md#WholePdbPipeline.process_structure), [`atom_cross_att_encoder`](network/atom_cross_attention.md#atom_cross_att_encoder), [`from_data_dict`](feat_batch.md#Batch.from_data_dict), [`as_data_dict`](feat_batch.md#Batch.as_data_dict), [`__call__`](network/diffusion_head.md#DiffusionHead.__call__), [`sample`](network/diffusion_head.md#sample), [`predicted_structure_info`](feat_batch.md#Batch.predicted_structure_info)

### `PseudoBetaInfo`
- def: [`src/alphafold3/model/features.py:1349`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1349)
- doc: Contains information for extracting pseudo-beta and equivalent atoms.
- signature: `class PseudoBetaInfo:`
- members:
  - `as_data_dict(self)` — [`L1455`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1455)
  - `compute_features(cls, all_token_atoms_layout: atom_layout.AtomLayout, ccd: chemical_components.Ccd, padding_shapes: PaddingShapes, logging_name: str)` — [`L1355`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1355) — Compute the PseudoBetaInfo features.
  - `from_data_dict(cls, batch: BatchDict)` — [`L1448`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1448) — documented in [alphafold3-model-feat_batch](../../../../concepts/alphafold3-model-feat_batch.md)
  - `token_atoms_to_pseudo_beta` — [`L1352`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1352)
- uses (calls/refs, reference-scoped): [`AtomLayout`](atom_layout/atom_layout.md#AtomLayout), [`atom_name`](atom_layout/atom_layout.md#AtomLayout.atom_name), [`chain_type`](atom_layout/atom_layout.md#AtomLayout.chain_type), [`num_tokens`](features.md#PaddingShapes.num_tokens), [`GatherInfo`](atom_layout/atom_layout.md#GatherInfo), [`compute_gather_idxs`](atom_layout/atom_layout.md#compute_gather_idxs), [`res_name`](atom_layout/atom_layout.md#AtomLayout.res_name), [`BatchDict`](features.md#BatchDict.BatchDict), [`copy_and_pad_to`](atom_layout/atom_layout.md#AtomLayout.copy_and_pad_to), [`shape`](atom_layout/atom_layout.md#AtomLayout.shape), [`as_dict`](atom_layout/atom_layout.md#GatherInfo.as_dict), [`from_dict`](atom_layout/atom_layout.md#GatherInfo.from_dict), [`PaddingShapes`](features.md#PaddingShapes)
- used by: [`process_structure`](pipeline/pipeline.md#WholePdbPipeline.process_structure), [`from_data_dict`](feat_batch.md#Batch.from_data_dict), [`as_data_dict`](feat_batch.md#Batch.as_data_dict), [`pseudo_beta_info`](feat_batch.md#Batch.pseudo_beta_info)

### `RefStructure`
- def: [`src/alphafold3/model/features.py:1671`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1671)
- doc: Contains ref structure information.
- signature: `class RefStructure:`
- members:
  - `as_data_dict(self)` — [`L1817`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1817) — documented in [alphafold3-model-feat_batch](../../../../concepts/alphafold3-model-feat_batch.md)
  - `compute_features(cls, all_token_atoms_layout: atom_layout.AtomLayout, ccd: chemical_components.Ccd, padding_shapes: PaddingShapes, chemical_components_data: struc_chem_comps.ChemicalComponentsData, random_state: np.random.RandomState, ref_max_modified_date: datetime.date, conformer_max_iterations: int | None, ligand_ligand_bonds: atom_layout.AtomLayout | None = None)` — [`L1688`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1688) — Reference structure information for each residue.
  - `from_data_dict(cls, batch: BatchDict)` — [`L1807`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1807)
  - `atom_name_chars` — [`L1683`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1683)
  - `charge` — [`L1681`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1681) — documented in [alphafold3-model-network-atom_cross_attention](../../../../concepts/alphafold3-model-network-atom_cross_attention.md)
  - `element` — [`L1679`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1679) — documented in [alphafold3-model-network-atom_cross_attention](../../../../concepts/alphafold3-model-network-atom_cross_attention.md)
  - `mask` — [`L1677`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1677)
  - `positions` — [`L1675`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1675)
  - `ref_space_uid` — [`L1685`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1685)
- uses (calls/refs, reference-scoped): [`AtomLayout`](atom_layout/atom_layout.md#AtomLayout), [`atom_name`](atom_layout/atom_layout.md#AtomLayout.atom_name), [`chain_id`](atom_layout/atom_layout.md#AtomLayout.chain_id), [`res_id`](atom_layout/atom_layout.md#AtomLayout.res_id), [`num_tokens`](features.md#PaddingShapes.num_tokens), [`xnp_ndarray`](features.md#xnp_ndarray.xnp_ndarray), [`res_name`](atom_layout/atom_layout.md#AtomLayout.res_name), [`BatchDict`](features.md#BatchDict.BatchDict), [`shape`](atom_layout/atom_layout.md#AtomLayout.shape), [`ChemicalComponentsData`](../structure/chemical_components.md#ChemicalComponentsData), [`get_reference`](features.md#get_reference), [`PaddingShapes`](features.md#PaddingShapes), [`_DEFAULT_BLANK_REF`](features.md#_DEFAULT_BLANK_REF), [`_STANDARD_RESIDUES`](features.md#_STANDARD_RESIDUES)
- used by: [`process_structure`](pipeline/pipeline.md#WholePdbPipeline.process_structure), [`atom_cross_att_encoder`](network/atom_cross_attention.md#atom_cross_att_encoder), [`from_data_dict`](feat_batch.md#Batch.from_data_dict), [`as_data_dict`](feat_batch.md#Batch.as_data_dict), [`compute_features`](features.md#Frames.compute_features), [`_per_atom_conditioning`](network/atom_cross_attention.md#_per_atom_conditioning), [`create_target_feat`](network/featurization.md#create_target_feat), [`ref_structure`](feat_batch.md#Batch.ref_structure)

### `Templates`
- def: [`src/alphafold3/model/features.py:719`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L719) — documented in [alphafold3-model-network-template_modules](../../../../concepts/alphafold3-model-network-template_modules.md)
- doc: Dataclass containing templates.
- signature: `class Templates:`
- members:
  - `as_data_dict(self)` — [`L871`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L871)
  - `compute_features(cls, all_tokens: atom_layout.AtomLayout, standard_token_idxs: np.ndarray, padding_shapes: PaddingShapes, templates_by_chain_id: Mapping[str, Sequence[folding_input.Template]], max_templates: int, logging_name: str)` — [`L730`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L730) — Compute the template features. — documented in [alphafold3-model-features](../../../../concepts/alphafold3-model-features.md)
  - `from_data_dict(cls, batch: BatchDict)` — [`L863`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L863) — Make Template from batch dictionary.
  - `aatype` — [`L723`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L723) — documented in [alphafold3-model-network-template_modules](../../../../concepts/alphafold3-model-network-template_modules.md)
  - `atom_mask` — [`L727`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L727) — documented in [alphafold3-model-network-template_modules](../../../../concepts/alphafold3-model-network-template_modules.md)
  - `atom_positions` — [`L725`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L725) — documented in [alphafold3-model-network-template_modules](../../../../concepts/alphafold3-model-network-template_modules.md)
- uses (calls/refs, reference-scoped): [`AtomLayout`](atom_layout/atom_layout.md#AtomLayout), [`atom_name`](atom_layout/atom_layout.md#AtomLayout.atom_name), [`chain_id`](atom_layout/atom_layout.md#AtomLayout.chain_id), [`make_structure`](atom_layout/atom_layout.md#make_structure), [`num_tokens`](features.md#PaddingShapes.num_tokens), [`xnp_ndarray`](features.md#xnp_ndarray.xnp_ndarray), [`res_name`](atom_layout/atom_layout.md#AtomLayout.res_name), [`BatchDict`](features.md#BatchDict.BatchDict), [`_pad_to`](features.md#_pad_to), [`shape`](atom_layout/atom_layout.md#AtomLayout.shape), [`chain_single_letter_sequence`](../structure/structure.md#Structure.chain_single_letter_sequence), [`iter_chains`](../structure/structure.md#Structure.iter_chains), [`from_mmcif`](../structure/parsing.md#from_mmcif), [`PaddingShapes`](features.md#PaddingShapes), [`fix_template_features`](data3.md#fix_template_features), [`empty_template_features`](data3.md#empty_template_features), [`_reduce_template_features`](features.md#_reduce_template_features), [`TEMPLATE_FEATURES`](data_constants.md#TEMPLATE_FEATURES)
- used by: [`process_structure`](pipeline/pipeline.md#WholePdbPipeline.process_structure), [`from_data_dict`](feat_batch.md#Batch.from_data_dict), [`as_data_dict`](feat_batch.md#Batch.as_data_dict), [`construct_input`](network/template_modules.md#SingleTemplateEmbedding.construct_input), [`__call__`](network/template_modules.md#TemplateEmbedding.__call__), [`__call__`](network/template_modules.md#SingleTemplateEmbedding.__call__), [`templates`](feat_batch.md#Batch.templates)

### `TokenFeatures`
- def: [`src/alphafold3/model/features.py:905`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L905) — documented in [alphafold3-model-network-evoformer](../../../../concepts/alphafold3-model-network-evoformer.md)
- doc: Dataclass containing features for tokens.
- signature: `class TokenFeatures:`
- members:
  - `as_data_dict(self)` — [`L1027`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1027) — documented in [alphafold3-model-feat_batch](../../../../concepts/alphafold3-model-feat_batch.md)
  - `compute_features(cls, all_tokens: atom_layout.AtomLayout, padding_shapes: PaddingShapes)` — [`L932`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L932) — Compute the per-token features. — documented in [alphafold3-model-features](../../../../concepts/alphafold3-model-features.md)
  - `from_data_dict(cls, batch: BatchDict)` — [`L1009`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1009)
  - `aatype` — [`L910`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L910) — documented in [alphafold3-model-network-evoformer](../../../../concepts/alphafold3-model-network-evoformer.md)
  - `asym_id` — [`L919`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L919)
  - `entity_id` — [`L920`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L920)
  - `is_dna` — [`L926`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L926)
  - `is_ligand` — [`L927`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L927)
  - `is_nonstandard_polymer_chain` — [`L928`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L928)
  - `is_protein` — [`L924`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L924)
  - `is_rna` — [`L925`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L925)
  - `is_water` — [`L929`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L929)
  - `mask` — [`L911`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L911)
  - `residue_index` — [`L908`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L908) — documented in [alphafold3-model-network-evoformer](../../../../concepts/alphafold3-model-network-evoformer.md)
  - `seq_length` — [`L912`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L912)
  - `sym_id` — [`L921`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L921)
  - `token_index` — [`L909`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L909)
- uses (calls/refs, reference-scoped): [`AtomLayout`](atom_layout/atom_layout.md#AtomLayout), [`atom_name`](atom_layout/atom_layout.md#AtomLayout.atom_name), [`chain_id`](atom_layout/atom_layout.md#AtomLayout.chain_id), [`res_id`](atom_layout/atom_layout.md#AtomLayout.res_id), [`chain_type`](atom_layout/atom_layout.md#AtomLayout.chain_type), [`num_tokens`](features.md#PaddingShapes.num_tokens), [`xnp_ndarray`](features.md#xnp_ndarray.xnp_ndarray), [`res_name`](atom_layout/atom_layout.md#AtomLayout.res_name), [`BatchDict`](features.md#BatchDict.BatchDict), [`_pad_to`](features.md#_pad_to), [`shape`](atom_layout/atom_layout.md#AtomLayout.shape), [`_compute_asym_entity_and_sym_id`](features.md#_compute_asym_entity_and_sym_id), [`PaddingShapes`](features.md#PaddingShapes), [`sym_id`](features.md#Chains.sym_id), [`chain_id`](features.md#Chains.chain_id), [`asym_id`](features.md#Chains.asym_id), [`entity_id`](features.md#Chains.entity_id)
- used by: [`process_structure`](pipeline/pipeline.md#WholePdbPipeline.process_structure), [`from_data_dict`](feat_batch.md#Batch.from_data_dict), [`as_data_dict`](feat_batch.md#Batch.as_data_dict), [`__call__`](network/evoformer.md#Evoformer.__call__), [`__call__`](network/diffusion_head.md#DiffusionHead.__call__), [`_embed_bonds`](network/evoformer.md#Evoformer._embed_bonds), [`__call__`](network/distogram_head.md#DistogramHead.__call__), [`create_target_feat`](network/featurization.md#create_target_feat), [`_embed_template_pair`](network/evoformer.md#Evoformer._embed_template_pair), [`create_relative_encoding`](network/featurization.md#create_relative_encoding), [`pairformer_fn`](network/evoformer.md#Evoformer.pairformer_fn), [`token_features`](feat_batch.md#Batch.token_features), [`_seq_pair_embedding`](network/evoformer.md#Evoformer._seq_pair_embedding), [`num_res`](feat_batch.md#Batch.num_res)

## Functions
- `_compute_asym_entity_and_sym_id(all_tokens: atom_layout.AtomLayout)` — [`L128`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L128) — Compute asym_id, entity_id and sym_id. — documented in [alphafold3-model-features](../../../../concepts/alphafold3-model-features.md)
- `_get_reference_positions_from_ccd_cif(ccd_cif: cif_dict.CifDict, ref_max_modified_date: datetime.date, logging_name: str)` — [`L1505`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1505) — Creates reference positions from a CCD mmcif data block.
- `_pad_to(arr: np.ndarray, shape: tuple[int | None, ...], **kwargs)` — [`L69`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L69) — Pads an array to a given shape. Wrapper around np.pad(). — documented in [alphafold3-model-features](../../../../concepts/alphafold3-model-features.md)
- `_reduce_template_features(template_features: data3.FeatureDict, max_templates: int)` — [`L886`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L886) — Reduces template features to max num templates and defined feature set.
- `_unwrap(obj)` — [`L105`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L105) — Unwrap an object from a zero-dim np.ndarray.
- `get_reference(res_name: str, chemical_components_data: struc_chem_comps.ChemicalComponentsData, ccd: chemical_components.Ccd, random_state: np.random.RandomState, ref_max_modified_date: datetime.date, conformer_max_iterations: int | None)` — [`L1544`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1544) — Reference structure for residue from CCD or SMILES.
- `random_augmentation(positions: np.ndarray, random_state: np.random.RandomState)` — [`L1490`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1490) — Center then apply random translation and rotation.
- `random_rotation(random_state: np.random.RandomState)` — [`L1479`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1479)
- `tokenizer(flat_output_layout: atom_layout.AtomLayout, ccd: chemical_components.Ccd, max_atoms_per_token: int, flatten_non_standard_residues: bool, logging_name: str)` — [`L173`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L173) — Maps a flat atom layout to tokens for evoformer.

## Module values
- `BatchDict` — [`L52`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L52)
- `_DEFAULT_BLANK_REF` — [`L1470`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L1470)
- `_STANDARD_RESIDUES` — [`L54`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L54)
- `xnp_ndarray` — [`L51`](../../../../../../../raw/code/alphafold3/src/alphafold3/model/features.py#L51)

