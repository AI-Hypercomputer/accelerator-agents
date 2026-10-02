---
title: 'Module: src/alphafold3/model/atom_layout/atom_layout.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/atom_layout/atom_layout.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.atom_layout.atom_layout`/
symbols:
  AtomLayout: AtomLayout#
  AtomLayout.atom_name: AtomLayout#atom_name.
  AtomLayout.chain_id: AtomLayout#chain_id.
  residues_from_structure: residues_from_structure().
  make_flat_atom_layout: make_flat_atom_layout().
  AtomLayout.res_id: AtomLayout#res_id.
  atom_layout_from_structure: atom_layout_from_structure().
  make_structure: make_structure().
  convert: convert().
  AtomLayout.chain_type: AtomLayout#chain_type.
  GatherInfo: GatherInfo#
  AtomLayout.res_name: AtomLayout#res_name.
  compute_gather_idxs: compute_gather_idxs().
  AtomLayout.copy_and_pad_to: AtomLayout#copy_and_pad_to().
  GatherInfo.gather_idxs: GatherInfo#gather_idxs.
  GatherInfo.gather_mask: GatherInfo#gather_mask.
  Residues.__getitem__: Residues#__getitem__().
  AtomLayout.shape: AtomLayout#shape().
  fill_in_optional_fields: fill_in_optional_fields().
  AtomLayout.__getitem__: AtomLayout#__getitem__().
  GatherInfo.as_dict: GatherInfo#as_dict().
  GatherInfo.input_shape: GatherInfo#input_shape.
  AtomLayout.atom_element: AtomLayout#atom_element.
  Residues.res_name: Residues#res_name.
  GatherInfo.from_dict: GatherInfo#from_dict().
  GatherInfo.__getitem__: GatherInfo#__getitem__().
  get_bonded_atoms: get_bonded_atoms().
  fill_in_optional_fields._convert_str_array: fill_in_optional_fields()._convert_str_array().
  Residues: Residues#
  GatherInfo.as_np_or_jnp: GatherInfo#as_np_or_jnp().
  _assert_all_arrays_have_same_shape: _assert_all_arrays_have_same_shape().
  AtomLayout.to_array: AtomLayout#to_array().
  residues_from_structure._get_smiles: residues_from_structure()._get_smiles().
  guess_deprotonation: guess_deprotonation().
  Residues.smiles_string: Residues#smiles_string.
  xnp_ndarray.xnp_ndarray: xnp_ndarray.xnp_ndarray.
  Residues.res_id: Residues#res_id.
  Residues.chain_id: Residues#chain_id.
  Residues.deprotonation: Residues#deprotonation.
  AtomLayout.__post_init__: AtomLayout#__post_init__().
  AtomLayout.__eq__: AtomLayout#__eq__().
  Residues.chain_type: Residues#chain_type.
  Residues.is_end_terminus: Residues#is_end_terminus.
  Residues.__post_init__: Residues#__post_init__().
  GatherInfo.__post_init__: GatherInfo#__post_init__().
  Residues.is_start_terminus: Residues#is_start_terminus.
  AtomLayout.from_array: AtomLayout#from_array().
  GatherInfo.shape: GatherInfo#shape().
  NumpyIndex.NumpyIndex: NumpyIndex.NumpyIndex.
  Residues.__eq__: Residues#__eq__().
  Residues.shape: Residues#shape().
  get_link_drop_atoms: get_link_drop_atoms().
---
# Module: [`src/alphafold3/model/atom_layout/atom_layout.py`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py)

## Classes
### `AtomLayout`
- def: [`src/alphafold3/model/atom_layout/atom_layout.py:45`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L45) — documented in [alphafold3-model-atom_layout](../../../../../concepts/alphafold3-model-atom_layout.md)
- doc: Atom layout in a fixed shape (usually 1-dim or 2-dim).
- signature: `class AtomLayout:`
- members:
  - `__post_init__(self)` — [`L84`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L84) — Assert all arrays have the same shape.
  - `copy_and_pad_to(self, shape: tuple[int, ...])` — [`L146`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L146) — Copies and pads the layout to the requested shape. — documented in [alphafold3-model-atom_layout](../../../../../concepts/alphafold3-model-atom_layout.md)
  - `from_array(cls, arr: np.ndarray)` — [`L220`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L220) — Creates an AtomLayout object from a numpy array with shape (6, ...).
  - `shape(self)` — [`L239`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L239) — documented in [alphafold3-model-atom_layout](../../../../../concepts/alphafold3-model-atom_layout.md)
  - `to_array(self)` — [`L192`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L192) — Stacks the fields to a numpy array with shape (6, <layout_shape>).
  - `atom_element` — [`L80`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L80)
  - `atom_name` — [`L77`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L77) — documented in [alphafold3-model-atom_layout](../../../../../concepts/alphafold3-model-atom_layout.md)
  - `chain_id` — [`L79`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L79) — documented in [alphafold3-model-atom_layout](../../../../../concepts/alphafold3-model-atom_layout.md)
  - `chain_type` — [`L82`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L82)
  - `res_id` — [`L78`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L78)
  - `res_name` — [`L81`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L81)
- protocol/private: `__eq__`[`L121`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L121), `__getitem__`[`L107`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L107)
- uses (calls/refs, reference-scoped): [`_assert_all_arrays_have_same_shape`](atom_layout.md#_assert_all_arrays_have_same_shape), [`NumpyIndex`](atom_layout.md#NumpyIndex.NumpyIndex)
- used by: [`process_structure`](../pipeline/pipeline.md#WholePdbPipeline.process_structure), [`compute_features`](../features.md#TokenFeatures.compute_features), [`compute_features`](../features.md#MSA.compute_features), [`clean_structure`](../pipeline/structure_cleaning.md#clean_structure), [`get_bond_layout`](../pipeline/inter_chain_bonds.md#get_bond_layout), [`compute_features`](../features.md#Templates.compute_features), [`make_flat_atom_layout`](atom_layout.md#make_flat_atom_layout), [`get_predicted_structure`](../model.md#get_predicted_structure), [`atom_layout_from_structure`](atom_layout.md#atom_layout_from_structure), [`compute_features`](../features.md#AtomCrossAtt.compute_features), [`compute_features`](../features.md#Frames.compute_features), [`make_structure`](atom_layout.md#make_structure), [`compute_features`](../features.md#PolymerLigandBondInfo.compute_features), [`compute_features`](../features.md#RefStructure.compute_features), [`compute_gather_idxs`](atom_layout.md#compute_gather_idxs), [`create_empty_output_struc_and_layout`](../pipeline/structure_cleaning.md#create_empty_output_struc_and_layout), [`compute_features`](../features.md#LigandLigandBondInfo.compute_features), [`compute_features`](../features.md#PseudoBetaInfo.compute_features), [`fill_in_optional_fields`](atom_layout.md#fill_in_optional_fields), [`tokenizer`](../features.md#tokenizer), [`_compute_asym_entity_and_sym_id`](../features.md#_compute_asym_entity_and_sym_id), [`_get_leaving_atom_mask`](../pipeline/structure_cleaning.md#_get_leaving_atom_mask), [`compute_features`](../features.md#ConvertModelOutput.compute_features), [`compute_features`](../features.md#PredictedStructureInfo.compute_features), [`get_polymer_ligand_and_ligand_ligand_bonds`](../pipeline/inter_chain_bonds.md#get_polymer_ligand_and_ligand_ligand_bonds), [`get_ligand_ligand_bonds`](../pipeline/inter_chain_bonds.md#get_ligand_ligand_bonds), [`get_polymer_ligand_bonds`](../pipeline/inter_chain_bonds.md#get_polymer_ligand_bonds), [`get_bonded_atoms`](atom_layout.md#get_bonded_atoms), [`_remove_multi_bonds`](../pipeline/inter_chain_bonds.md#_remove_multi_bonds), [`_assert_all_arrays_have_same_shape`](atom_layout.md#_assert_all_arrays_have_same_shape), [`flat_output_layout`](../features.md#ConvertModelOutput.flat_output_layout), [`token_atoms_layout`](../features.md#ConvertModelOutput.token_atoms_layout), [`ligand_ligand_bonds`](../features.md#ConvertModelOutput.ligand_ligand_bonds), [`polymer_ligand_bonds`](../features.md#ConvertModelOutput.polymer_ligand_bonds)

### `GatherInfo`
- def: [`src/alphafold3/model/atom_layout/atom_layout.py:314`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L314) — documented in [alphafold3-model-feat_batch](../../../../../concepts/alphafold3-model-feat_batch.md)
- doc: Gather indices to translate from one atom layout to another.
- signature: `class GatherInfo:`
- members:
  - `as_dict(self, key_prefix: str | None = None)` — [`L361`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L361)
  - `as_np_or_jnp(self, xnp: types.ModuleType)` — [`L354`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L354)
  - `from_dict(cls, d: Mapping[str, xnp_ndarray], key_prefix: str | None = None)` — [`L373`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L373) — Creates GatherInfo from a given dictionary.
  - `shape(self)` — [`L351`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L351)
  - `gather_idxs` — [`L331`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L331) — documented in [alphafold3-model-feat_batch](../../../../../concepts/alphafold3-model-feat_batch.md)
  - `gather_mask` — [`L332`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L332) — documented in [alphafold3-model-feat_batch](../../../../../concepts/alphafold3-model-feat_batch.md)
  - `input_shape` — [`L333`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L333) — documented in [alphafold3-model-feat_batch](../../../../../concepts/alphafold3-model-feat_batch.md)
- protocol/private: `__getitem__`[`L343`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L343), `__post_init__`[`L335`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L335)
- uses (calls/refs, reference-scoped): [`xnp_ndarray`](atom_layout.md#xnp_ndarray.xnp_ndarray), [`NumpyIndex`](atom_layout.md#NumpyIndex.NumpyIndex)
- used by: [`atom_cross_att_encoder`](../network/atom_cross_attention.md#atom_cross_att_encoder), [`atom_cross_att_decoder`](../network/atom_cross_attention.md#atom_cross_att_decoder), [`__call__`](../network/confidence_head.md#ConfidenceHead.__call__), [`get_predicted_structure`](../model.md#get_predicted_structure), [`compute_features`](../features.md#AtomCrossAtt.compute_features), [`compute_features`](../features.md#Frames.compute_features), [`convert`](atom_layout.md#convert), [`compute_features`](../features.md#PolymerLigandBondInfo.compute_features), [`compute_gather_idxs`](atom_layout.md#compute_gather_idxs), [`_embed_bonds`](../network/evoformer.md#Evoformer._embed_bonds), [`fill_in_optional_fields`](atom_layout.md#fill_in_optional_fields), [`__call__`](../network/diffusion_transformer.md#CrossAttTransformer.__call__), [`as_data_dict`](../features.md#AtomCrossAtt.as_data_dict), [`as_data_dict`](../features.md#PolymerLigandBondInfo.as_data_dict), [`_convert_str_array`](atom_layout.md#fill_in_optional_fields._convert_str_array), [`queries_to_keys`](../features.md#AtomCrossAtt.queries_to_keys), [`_assert_all_arrays_have_same_shape`](atom_layout.md#_assert_all_arrays_have_same_shape), [`token_atoms_to_queries`](../features.md#AtomCrossAtt.token_atoms_to_queries), [`as_data_dict`](../features.md#LigandLigandBondInfo.as_data_dict), [`as_data_dict`](../features.md#PseudoBetaInfo.as_data_dict), [`from_data_dict`](../features.md#AtomCrossAtt.from_data_dict), [`from_data_dict`](../features.md#LigandLigandBondInfo.from_data_dict), [`from_data_dict`](../features.md#PolymerLigandBondInfo.from_data_dict), [`from_data_dict`](../features.md#PseudoBetaInfo.from_data_dict), [`queries_to_token_atoms`](../features.md#AtomCrossAtt.queries_to_token_atoms), [`tokens_to_queries`](../features.md#AtomCrossAtt.tokens_to_queries), [`tokens_to_keys`](../features.md#AtomCrossAtt.tokens_to_keys), [`tokens_to_ligand_ligand_bonds`](../features.md#LigandLigandBondInfo.tokens_to_ligand_ligand_bonds), [`tokens_to_polymer_ligand_bonds`](../features.md#PolymerLigandBondInfo.tokens_to_polymer_ligand_bonds), [`token_atoms_to_bonds`](../features.md#PolymerLigandBondInfo.token_atoms_to_bonds), [`token_atoms_to_pseudo_beta`](../features.md#PseudoBetaInfo.token_atoms_to_pseudo_beta)

### `Residues`
- def: [`src/alphafold3/model/atom_layout/atom_layout.py:244`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L244)
- doc: List of residues with meta data.
- signature: `class Residues:`
- members:
  - `__post_init__(self)` — [`L268`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L268) — Assert all arrays are 1D have the same shape.
  - `shape(self)` — [`L309`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L309)
  - `chain_id` — [`L261`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L261)
  - `chain_type` — [`L262`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L262)
  - `deprotonation` — [`L265`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L265)
  - `is_end_terminus` — [`L264`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L264)
  - `is_start_terminus` — [`L263`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L263)
  - `res_id` — [`L260`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L260)
  - `res_name` — [`L259`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L259) — documented in [alphafold3-model-atom_layout](../../../../../concepts/alphafold3-model-atom_layout.md)
  - `smiles_string` — [`L266`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L266)
- protocol/private: `__eq__`[`L302`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L302), `__getitem__`[`L286`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L286)
- uses (calls/refs, reference-scoped): [`_assert_all_arrays_have_same_shape`](atom_layout.md#_assert_all_arrays_have_same_shape), [`NumpyIndex`](atom_layout.md#NumpyIndex.NumpyIndex)
- used by: [`residues_from_structure`](atom_layout.md#residues_from_structure), [`make_flat_atom_layout`](atom_layout.md#make_flat_atom_layout), [`make_structure`](atom_layout.md#make_structure), [`_assert_all_arrays_have_same_shape`](atom_layout.md#_assert_all_arrays_have_same_shape), [`guess_deprotonation`](atom_layout.md#guess_deprotonation)

## Functions
- `_assert_all_arrays_have_same_shape(*, obj: AtomLayout | Residues | GatherInfo, expected_shape: tuple[int, ...], attribute_names: Sequence[str])` — [`L1092`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L1092) — Checks that given attributes of the object have the expected shape.
- `_convert_str_array(gather: GatherInfo, arr: np.ndarray)` — [`L421`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L421)
- `_get_smiles(res_name)` — [`L520`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L520) — Get SMILES string from chemical components.
- `atom_layout_from_structure(struct: structure.Structure, *, fix_non_standard_polymer_res: bool = False)` — [`L463`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L463) — Extract AtomLayout from a Structure. — documented in [alphafold3-model-atom_layout](../../../../../concepts/alphafold3-model-atom_layout.md)
- `compute_gather_idxs(*, source_layout: AtomLayout, target_layout: AtomLayout, fill_value: int = 0)` — [`L893`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L893) — Produce gather indices and mask to convert from source layout to target. — documented in [alphafold3-model-atom_layout](../../../../../concepts/alphafold3-model-atom_layout.md)
- `convert(gather_info: GatherInfo, arr: xnp_ndarray, *, layout_axes: tuple[int, ...] = (0,))` — [`L933`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L933) — Convert an array from one atom layout to another. — documented in [alphafold3-model-network-atom_cross_attention](../../../../../concepts/alphafold3-model-network-atom_cross_attention.md)
- `fill_in_optional_fields(minimal_atom_layout: AtomLayout, reference_atoms: AtomLayout)` — [`L387`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L387) — Fill in the optional fields (atom_element, res_name, chain_type).
- `get_bonded_atoms(polymer_ligand_bonds: AtomLayout, ligand_ligand_bonds: AtomLayout, res_id: int, chain_id: str)` — [`L704`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L704) — Finds the res_name on the opposite end of the bond, if a bond exists. — documented in [alphafold3-model-atom_layout](../../../../../concepts/alphafold3-model-atom_layout.md)
- `get_link_drop_atoms(res_name: str, chain_type: str, *, is_start_terminus: bool, is_end_terminus: bool, bonded_atoms: set[str], drop_ligand_leaving_atoms: bool = False, fix_standalone_glycans: bool = False)` — [`L644`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L644) — Returns set of atoms that are dropped when this res_name gets linked.
- `guess_deprotonation(residues: Residues)` — [`L436`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L436) — Convenience function to create a plausible deprotonation field.
- `make_flat_atom_layout(residues: Residues, ccd: chemical_components.Ccd, polymer_ligand_bonds: AtomLayout | None = None, ligand_ligand_bonds: AtomLayout | None = None, *, with_hydrogens: bool = False, skip_unk_residues: bool = True, drop_ligand_leaving_atoms: bool = False, fix_standalone_glycans: bool = False)` — [`L756`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L756) — Make a flat atom layout for given residues. — documented in [alphafold3-model-atom_layout](../../../../../concepts/alphafold3-model-atom_layout.md)
- `make_structure(flat_layout: AtomLayout, atom_coords: np.ndarray, name: str, *, atom_b_factors: np.ndarray | None = None, all_physical_residues: Residues | None = None)` — [`L1003`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L1003) — Returns a Structure from a flat layout and atom coordinates.
- `residues_from_structure(struct: structure.Structure, *, include_missing_residues: bool = True, fix_non_standard_polymer_res: bool = False)` — [`L512`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L512) — Create a Residues object from a Structure object. — documented in [alphafold3-model-atom_layout](../../../../../concepts/alphafold3-model-atom_layout.md)

## Module values
- `NumpyIndex` — [`L41`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L41)
- `xnp_ndarray` — [`L40`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/atom_layout/atom_layout.py#L40)

