---
title: 'Module: src/alphafold3/structure/structure.py'
type: catalog
provenance: extracted
module: src/alphafold3/structure/structure.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.structure.structure`/
symbols:
  Structure._atoms: Structure#_atoms.
  Structure._chains: Structure#_chains.
  Structure.copy_and_update: Structure#copy_and_update().
  Structure._residues: Structure#_residues.
  Structure.iter_atoms: Structure#iter_atoms().
  Structure.filter: Structure#filter().
  Structure.copy_and_update_atoms: Structure#copy_and_update_atoms().
  Structure: Structure#
  Structure.order_and_drop_atoms_to_match: Structure#order_and_drop_atoms_to_match().
  Structure._cascade_delete: Structure#_cascade_delete().
  Structure.copy_and_update_globals: Structure#copy_and_update_globals().
  Structure.author_naming_scheme: Structure#author_naming_scheme().
  Structure.reset_author_naming_scheme: Structure#reset_author_naming_scheme().
  Structure.merge_chains: Structure#merge_chains().
  Structure.iter_bonds: Structure#iter_bonds().
  Structure._apply_bioassembly_transform: Structure#_apply_bioassembly_transform().
  Structure.chain_id: Structure#chain_id().
  concat: concat().
  Structure.generate_bioassembly: Structure#generate_bioassembly().
  Structure.to_mmcif_dict: Structure#to_mmcif_dict().
  Structure.all_residues: Structure#all_residues().
  Structure.iter_residues: Structure#iter_residues().
  _UNSET: _UNSET.
  Structure.chain_res_name_sequence: Structure#chain_res_name_sequence().
  Structure.add_bonds: Structure#add_bonds().
  Structure.reorder_chains: Structure#reorder_chains().
  Structure._to_mmcif_header: Structure#_to_mmcif_header().
  stack: stack().
  make_empty_structure: make_empty_structure().
  Structure.rename_chain_ids: Structure#rename_chain_ids().
  Structure.chain_single_letter_sequence: Structure#chain_single_letter_sequence().
  Structure.to_res_arrays: Structure#to_res_arrays().
  Structure._bonds: Structure#_bonds.
  Structure.atom_ids: Structure#atom_ids().
  _UnsetType: _UnsetType.
  Structure._validate_consistent_table_ordering: Structure#_validate_consistent_table_ordering().
  Structure.chains: Structure#chains().
  Structure.copy_and_update_residues: Structure#copy_and_update_residues().
  Structure.fix_non_standard_polymer_res: Structure#fix_non_standard_polymer_res().
  Structure.res_id: Structure#res_id().
  Structure.atoms_table: Structure#atoms_table().
  Structure._atom_mask: Structure#_atom_mask().
  Structure.select: Structure#select().
  Structure.present_chains: Structure#present_chains().
  Structure.present_residues: Structure#present_residues().
  Structure._apply_atom_index_array: Structure#_apply_atom_index_array().
  Structure.iter_chains: Structure#iter_chains().
  Structure.copy_and_update_from_res_arrays: Structure#copy_and_update_from_res_arrays().
  Structure.remap_res_id: Structure#remap_res_id().
  _LeadingDimSlice.__getitem__: _LeadingDimSlice#__getitem__().
  Structure.atom_name: Structure#atom_name().
  Structure.res_name: Structure#res_name().
  Structure._validate_table_foreign_keys: Structure#_validate_table_foreign_keys().
  Structure.without_terminal_oxygens: Structure#without_terminal_oxygens().
  Structure._keep: Structure#_keep().
  Structure.__init__: Structure#__init__().
  Structure.chemical_components_data: Structure#chemical_components_data().
  Structure.bonds: Structure#bonds().
  Structure._copy_and_update: Structure#_copy_and_update().
  Structure.coords: Structure#coords().
  Structure.rename_res_name: Structure#rename_res_name().
  Structure.name: Structure#name().
  Structure.chains_table: Structure#chains_table().
  Structure.num_residues: Structure#num_residues().
  Structure.num_atoms: Structure#num_atoms().
  Structure.residues_table: Structure#residues_table().
  Structure.filter_residues: Structure#filter_residues().
  Structure.filter_polymers_to_single_atom_per_res: Structure#filter_polymers_to_single_atom_per_res().
  Structure._iter_residues: Structure#_iter_residues().
  Structure.unstack: Structure#unstack().
  Structure.chain_type: Structure#chain_type().
  Structure.get_table: Structure#get_table().
  Structure.drop_non_standard_atoms: Structure#drop_non_standard_atoms().
  _assign_unique_chain_ids: _assign_unique_chain_ids().
  multichain_residue_index: multichain_residue_index().
  Structure.chain_boundaries: Structure#chain_boundaries().
  Structure.res_boundaries: Structure#res_boundaries().
  Structure.unresolved_residues: Structure#unresolved_residues().
  Structure.__repr__: Structure#__repr__().
  Structure.copy_and_update_coords: Structure#copy_and_update_coords().
  Structure.transform_states_to_chains: Structure#transform_states_to_chains().
  Structure.to_res_atom_lists: Structure#to_res_atom_lists().
  Structure.num_chains: Structure#num_chains().
  _get_change_indices: _get_change_indices().
  _unpack_filter_predicates: _unpack_filter_predicates().
  Structure.with_sorted_chains: Structure#with_sorted_chains().
  Structure.without_hydrogen: Structure#without_hydrogen().
  Structure.find_chains_with_unknown_sequence: Structure#find_chains_with_unknown_sequence().
  Structure._bioassembly_data: Structure#_bioassembly_data.
  Structure.atom_element: Structure#atom_element().
  Structure.atom_b_factor: Structure#atom_b_factor().
  CHAIN_FIELDS.CHAIN_FIELDS: CHAIN_FIELDS.CHAIN_FIELDS.
  ATOM_FIELDS.ATOM_FIELDS: ATOM_FIELDS.ATOM_FIELDS.
  V2_FIELDS.V2_FIELDS: V2_FIELDS.V2_FIELDS.
  Structure.label_asym_id_to_entity_id: Structure#label_asym_id_to_entity_id().
  Structure.chain_entity_id: Structure#chain_entity_id().
  Structure.chain_entity_desc: Structure#chain_entity_desc().
  Structure.chain_auth_asym_id: Structure#chain_auth_asym_id().
  Structure.res_auth_seq_id: Structure#res_auth_seq_id().
  Structure.res_insertion_code: Structure#res_insertion_code().
  Structure.filter_out: Structure#filter_out().
  Structure.filter_to_entity_type: Structure#filter_to_entity_type().
  Structure.rename_auth_asym_ids: Structure#rename_auth_asym_ids().
  Structure.rename_chains_to_match: Structure#rename_chains_to_match().
  CascadeDelete: CascadeDelete#
  Structure.bioassembly_data: Structure#bioassembly_data().
  Structure.iter_chain_ranges: Structure#iter_chain_ranges().
  StructureTables.atoms: StructureTables#atoms.
  StructureTables.bonds: StructureTables#bonds.
  Structure._iter_residue_ranges: Structure#_iter_residue_ranges().
  Structure.bonds_table: Structure#bonds_table().
  Structure.atom_x: Structure#atom_x().
  Structure.atom_y: Structure#atom_y().
  Structure.atom_z: Structure#atom_z().
  Structure.atom_occupancy: Structure#atom_occupancy().
  Structure.group_by_residue: Structure#group_by_residue().
  Structure.to_mmcif: Structure#to_mmcif().
  RESIDUE_FIELDS.RESIDUE_FIELDS: RESIDUE_FIELDS.RESIDUE_FIELDS.
  Structure.release_date: Structure#release_date().
  Structure.resolution: Structure#resolution().
  Structure.structure_method: Structure#structure_method().
  Structure._iter_atom_ranges: Structure#_iter_atom_ranges().
  enumerate_residues: enumerate_residues().
  AuthorNamingScheme.AuthorNamingScheme: AuthorNamingScheme.AuthorNamingScheme.
  Structure._name: Structure#_name.
  Structure._chemical_components_data: Structure#_chemical_components_data.
  Structure.__getitem__: Structure#__getitem__().
  Structure.num_models: Structure#num_models().
  Structure.iter_residue_ranges: Structure#iter_residue_ranges().
  Structure.group_by_chain: Structure#group_by_chain().
  Structure.get_stoichiometry: Structure#get_stoichiometry().
  Structure.filter_coords: Structure#filter_coords().
  Structure.polymer_author_chain_single_letter_sequence: Structure#polymer_author_chain_single_letter_sequence().
  Structure.split_by_chain: Structure#split_by_chain().
  StructureTables.chains: StructureTables#chains.
  StructureTables.residues: StructureTables#residues.
  Structure.all_unset: Structure#all_unset().
  _LeadingDimSlice._struc: _LeadingDimSlice#_struc.
  TABLE_FIELDS.TABLE_FIELDS: TABLE_FIELDS.TABLE_FIELDS.
  Structure.atom_key: Structure#atom_key().
  Structure.polymer_auth_asym_id_to_label_asym_id: Structure#polymer_auth_asym_id_to_label_asym_id().
  Structure.slice_leading_dims: Structure#slice_leading_dims().
  Bond: Bond#
  fix_non_standard_polymer_residues: fix_non_standard_polymer_residues().
  Structure._release_date: Structure#_release_date.
  Structure._resolution: Structure#_resolution.
  Structure._structure_method: Structure#_structure_method.
  Structure.tables: Structure#tables.
  Structure.is_protein_mask: Structure#is_protein_mask().
  Structure.is_dna_mask: Structure#is_dna_mask().
  Structure.is_rna_mask: Structure#is_rna_mask().
  Structure.is_nucleic_mask: Structure#is_nucleic_mask().
  Structure.is_ligand_mask: Structure#is_ligand_mask().
  Structure.is_water_mask: Structure#is_water_mask().
  Structure.drop_non_standard_protein_atoms: Structure#drop_non_standard_protein_atoms().
  _LeadingDimSlice.__init__: _LeadingDimSlice#__init__().
  CascadeDelete.NONE: CascadeDelete#NONE.
  CascadeDelete.CHAINS: CascadeDelete#CHAINS.
  _UnsetSentinel: _UnsetSentinel#
  _UnsetSentinel.UNSET: _UnsetSentinel#UNSET.
  Bond.from_atom: Bond#from_atom.
  Bond.dest_atom: Bond#dest_atom.
  MissingAtomError: MissingAtomError#
  GLOBAL_FIELDS: GLOBAL_FIELDS.
  StructureTables: StructureTables#
  ResArrays: ResArrays#
  Structure._to_internal_res_id: Structure#_to_internal_res_id().
  _COORDS_DECIMAL_PLACES._COORDS_DECIMAL_PLACES: _COORDS_DECIMAL_PLACES._COORDS_DECIMAL_PLACES.
  AllResidues.AllResidues: AllResidues.AllResidues.
  _LeadingDimSlice: _LeadingDimSlice#
  CascadeDelete.FULL: CascadeDelete#FULL.
  Bond.bond_info: Bond#bond_info.
  MissingAuthorResidueIdError: MissingAuthorResidueIdError#
  ARRAY_FIELDS: ARRAY_FIELDS.
  ResArrays.atom_positions: ResArrays#atom_positions.
  ResArrays.atom_mask: ResArrays#atom_mask.
  ResArrays.atom_b_factor: ResArrays#atom_b_factor.
  ResArrays.atom_occupancy: ResArrays#atom_occupancy.
  Structure._VERSION: Structure#_VERSION.
  MISSING_AUTH_SEQ_ID.MISSING_AUTH_SEQ_ID: MISSING_AUTH_SEQ_ID.MISSING_AUTH_SEQ_ID.
  _UPDATEABLE_FIELDS._UPDATEABLE_FIELDS: _UPDATEABLE_FIELDS._UPDATEABLE_FIELDS.
  SCALAR_FIELDS.SCALAR_FIELDS: SCALAR_FIELDS.SCALAR_FIELDS.
  _T: _T.
  Structure.foreign_keys: Structure#foreign_keys.
  Structure.__getstate__: Structure#__getstate__().
---
# Module: [`src/alphafold3/structure/structure.py`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py)

## Classes
### `Bond`  ·  implements/extends NamedTuple
- def: [`src/alphafold3/structure/structure.py:66`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L66)
- doc: Describes a bond between two atoms.
- signature: `class Bond(NamedTuple):`
- members:
  - `bond_info` — [`L71`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L71)
  - `dest_atom` — [`L70`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L70)
  - `from_atom` — [`L69`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L69)
- used by: [`clean_structure`](../model/pipeline/structure_cleaning.md#clean_structure), [`iter_bonds`](structure.md#Structure.iter_bonds)

### `CascadeDelete`  ·  implements/extends Enum
- def: [`src/alphafold3/structure/structure.py:51`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L51)
- signature: `class CascadeDelete(enum.Enum):`
- members:
  - `CHAINS` — [`L54`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L54)
  - `FULL` — [`L53`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L53)
  - `NONE` — [`L52`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L52)
- used by: [`filter`](structure.md#Structure.filter)

### `MissingAtomError`  ·  implements/extends Exception
- def: [`src/alphafold3/structure/structure.py:74`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L74)
- doc: Error raised when an atom is missing during alignment.
- signature: `class MissingAtomError(Exception):`
- used by: [`order_and_drop_atoms_to_match`](structure.md#Structure.order_and_drop_atoms_to_match)

### `MissingAuthorResidueIdError`  ·  implements/extends Exception
- def: [`src/alphafold3/structure/structure.py:78`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L78)
- doc: Raised when author naming data is missing for a residue.
- signature: `class MissingAuthorResidueIdError(Exception):`

### `ResArrays`
- def: [`src/alphafold3/structure/structure.py:289`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L289)
- doc: Atom-level data arrays with a residue dimension.
- signature: `class ResArrays:`
- members:
  - `atom_b_factor` — [`L302`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L302)
  - `atom_mask` — [`L301`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L301)
  - `atom_occupancy` — [`L303`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L303)
  - `atom_positions` — [`L300`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L300)
- used by: [`to_res_arrays`](structure.md#Structure.to_res_arrays)

### `Structure`
- def: [`src/alphafold3/structure/structure.py:306`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L306) — documented in [alphafold3-model-atom_layout](../../../../concepts/alphafold3-model-atom_layout.md)
- members:
  - `__getitem__(self, field: str)` — [`L631`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L631) — Gets raw field data using field name as a string.
  - `__getstate__(self)` — [`L638`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L638) — Pickle calls this on dump.
  - `_apply_atom_index_array(self, index_arr: np.ndarray, chain_boundaries: np.ndarray | None = None, res_boundaries: np.ndarray | None = None, skip_validation: bool = False)` — [`L898`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L898) — Applies index_arr to the atom table using NumPy-style array indexing.
  - `_apply_bioassembly_transform(self, transform: bioassemblies.Transform)` — [`L2842`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2842) — Applies a bioassembly transform to this structure.
  - `_atom_mask(self, entities: Set[str])` — [`L685`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L685) — Boolean label indicating if each atom is from entities or not.
  - `_cascade_delete(self, *, chains: structure_tables.Chains | None = None, residues: structure_tables.Residues | None = None, atoms: structure_tables.Atoms | None = None, bonds: structure_tables.Bonds | None = None)` — [`L1426`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1426) — Performs a cascade delete operation on the structure's tables. — documented in [alphafold3-structure](../../../../concepts/alphafold3-structure.md)
  - `_copy_and_update(self, skip_validation: bool = False, **changes: Any)` — [`L1205`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1205) — Performs a shallow copy but with specified fields updated.
  - `_iter_atom_ranges(self, boundaries: Sequence[int] | np.ndarray)` — [`L805`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L805) — Iterator for (start, end) pairs from an array of start indices.
  - `_iter_residue_ranges(self, boundaries: Sequence[int] | np.ndarray, *, count_unresolved: bool)` — [`L814`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L814) — Iterator for (start, end) pairs from an array of start indices.
  - `_validate_consistent_table_ordering(self)` — [`L377`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L377) — Validates that all tables have the same ordering. — documented in [alphafold3-structure](../../../../concepts/alphafold3-structure.md)
  - `_validate_table_foreign_keys(self)` — [`L355`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L355) — Validates that all foreign keys are present in the referred tables. — documented in [alphafold3-structure-structure_tables](../../../../concepts/alphafold3-structure-structure_tables.md)
  - `add_bonds(self, bonded_atom_pairs: Sequence[tuple[tuple[str, int, str], tuple[str, int, str]],], bond_type: str | None = None)` — [`L1908`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1908) — Returns a structure with new bonds added. — documented in [alphafold3-structure-bonds](../../../../concepts/alphafold3-structure-bonds.md)
  - `all_residues(self)` — [`L495`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L495)
  - `all_unset(fields)` — [`L1162`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1162)
  - `atom_b_factor(self)` — [`L590`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L590)
  - `atom_element(self)` — [`L574`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L574)
  - `atom_ids(self)` — [`L991`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L991) — Gets a list of atom ID tuples from Structure class arrays.
  - `atom_key(self)` — [`L566`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L566)
  - `atom_name(self)` — [`L570`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L570)
  - `atom_occupancy(self)` — [`L594`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L594)
  - `atom_x(self)` — [`L578`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L578)
  - `atom_y(self)` — [`L582`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L582)
  - `atom_z(self)` — [`L586`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L586)
  - `atoms_table(self)` — [`L417`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L417) — Atoms table. — documented in [alphafold3-structure](../../../../concepts/alphafold3-structure.md)
  - `author_naming_scheme(self)` — [`L457`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L457)
  - `bioassembly_data(self)` — [`L443`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L443) — documented in [alphafold3-structure-bioassemblies](../../../../concepts/alphafold3-structure-bioassemblies.md)
  - `bonds(self)` — [`L453`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L453)
  - `bonds_table(self)` — [`L422`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L422) — Bonds table.
  - `chain_auth_asym_id(self)` — [`L527`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L527) — Returns the chain auth asym ID for each atom in the structure.
  - `chain_boundaries(self)` — [`L598`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L598) — The indices in the atom fields where each present chain begins.
  - `chain_entity_desc(self)` — [`L520`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L520) — Returns the entity description for each atom in the structure.
  - `chain_entity_id(self)` — [`L513`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L513) — Returns the entity ID for each atom in the structure.
  - `chain_id(self)` — [`L534`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L534)
  - `chain_res_name_sequence(self, *, include_missing_residues: bool = True, fix_non_standard_polymer_res: bool = False)` — [`L2133`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2133) — A mapping from internal chain ID to a sequence of residue names.
  - `chain_single_letter_sequence(self, include_missing_residues: bool = True)` — [`L1977`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1977) — Returns a mapping from chain ID to a single letter residue sequence.
  - `chain_type(self)` — [`L539`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L539)
  - `chains(self)` — [`L2679`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2679) — Ordered internal chain IDs (label_asym_id) present in the Structure.
  - `chains_table(self)` — [`L407`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L407) — Chains table.
  - `chemical_components_data(self)` — [`L447`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L447) — documented in [alphafold3-structure](../../../../concepts/alphafold3-structure.md)
  - `coords(self)` — [`L1973`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1973) — A [..., num_atom, 3] shaped array of atom coordinates.
  - `copy_and_update(self, *, name: str | _UnsetType = _UNSET, release_date: datetime.date | None | _UnsetType = _UNSET, resolution: float | None | _UnsetType = _UNSET, structure_method: str | None | _UnsetType = _UNSET, bioassembly_data: (bioassemblies.BioassemblyData | None | _UnsetType) = _UNSET, chemical_components_data: (struc_chem_comps.ChemicalComponentsData | None | _UnsetType) = _UNSET, chains: structure_tables.Chains | None | _UnsetType = _UNSET, residues: structure_tables.Residues | None | _UnsetType = _UNSET, atoms: structure_tables.Atoms | None | _UnsetType = _UNSET, bonds: structure_tables.Bonds | None | _UnsetType = _UNSET, skip_validation: bool = False)` — [`L1141`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1141) — Performs a shallow copy but with specified fields updated. — documented in [alphafold3-structure](../../../../concepts/alphafold3-structure.md)
  - `copy_and_update_atoms(self, *, atom_name: np.ndarray | None = None, atom_element: np.ndarray | None = None, atom_x: np.ndarray | None = None, atom_y: np.ndarray | None = None, atom_z: np.ndarray | None = None, atom_b_factor: np.ndarray | None = None, atom_occupancy: np.ndarray | None = None)` — [`L1371`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1371) — Returns a shallow copy with the atoms table updated. — documented in [alphafold3-model](../../../../concepts/alphafold3-model.md)
  - `copy_and_update_coords(self, coords: np.ndarray)` — [`L1257`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1257) — Performs a shallow copy but with coordinates updated.
  - `copy_and_update_from_res_arrays(self, *, include_unresolved: bool = False, **changes: np.ndarray)` — [`L1266`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1266) — Like copy_and_update but changes are arrays of length num_residues.
  - `copy_and_update_globals(self, *, name: str | _UnsetType = _UNSET, release_date: datetime.date | _UnsetType | None = _UNSET, resolution: float | _UnsetType | None = _UNSET, structure_method: str | _UnsetType | None = _UNSET, bioassembly_data: (bioassemblies.BioassemblyData | _UnsetType | None) = _UNSET, chemical_components_data: (struc_chem_comps.ChemicalComponentsData | _UnsetType | None) = _UNSET)` — [`L1332`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1332) — Returns a shallow copy with the global columns updated. — documented in [alphafold3-structure](../../../../concepts/alphafold3-structure.md)
  - `copy_and_update_residues(self, *, res_id: np.ndarray | None = None, res_name: np.ndarray | None = None, res_auth_seq_id: np.ndarray | None = None, res_insertion_code: np.ndarray | None = None)` — [`L1403`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1403) — Returns a shallow copy with the residues table updated. — documented in [alphafold3-structure](../../../../concepts/alphafold3-structure.md)
  - `drop_non_standard_atoms(self, *, ccd: chemical_components.Ccd, drop_unk: bool, drop_non_ccd: bool, drop_terminal_oxygens: bool = False)` — [`L1860`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1860) — Drops atoms that are not in the CCD for the given residue type.
  - `drop_non_standard_protein_atoms(self, *, drop_oxt: bool = True)` — [`L1839`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1839) — Drops non-standard atom names from protein chains.
  - `filter(self, mask: np.ndarray | None = None, *, apply_per_element: bool = False, invert: bool = False, cascade_delete: CascadeDelete = CascadeDelete.CHAINS, **predicate_by_field_name: table.FilterPredicate)` — [`L1481`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1481) — Filters the structure by field values and returns a new structure. — documented in [alphafold3-structure](../../../../concepts/alphafold3-structure.md)
  - `filter_coords(self, coord_predicate: Callable[[np.ndarray], bool])` — [`L1776`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1776) — Filter a structure's atoms by a function of their coordinates.
  - `filter_out(self, *args, **kwargs)` — [`L1637`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1637) — Returns a new structure with the specified elements removed.
  - `filter_polymers_to_single_atom_per_res(self, representative_atom_by_chain_type: Mapping[str, str] = mmcif_names.RESIDUE_REPRESENTATIVE_ATOMS)` — [`L1801`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1801) — Filter to one representative atom per polymer residue, ligands unchanged.
  - `filter_residues(self, res_mask: np.ndarray)` — [`L1762`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1762) — Filter resolved residues using a boolean mask.
  - `filter_to_entity_type(self, *, protein: bool = False, rna: bool = False, dna: bool = False, dna_rna_hybrid: bool = False, ligand: bool = False, water: bool = False)` — [`L1641`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1641) — Filters the structure to only include the selected entity types.
  - `find_chains_with_unknown_sequence(self)` — [`L1893`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1893) — Returns a sequence of chain IDs that contain only unknown residues.
  - `fix_non_standard_polymer_res(self, res_mapper: Callable[[np.ndarray, str], np.ndarray] = fix_non_standard_polymer_residues)` — [`L2183`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2183) — Replaces non-standard polymer residues with standard alternatives or UNK.
  - `generate_bioassembly(self, assembly_id: str | None = None)` — [`L2863`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2863) — Generates a biological assembly as a new `Structure`. — documented in [alphafold3-structure-bioassemblies](../../../../concepts/alphafold3-structure-bioassemblies.md)
  - `get_stoichiometry(self, *, fix_non_standard_polymer_res: bool = False)` — [`L1682`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1682) — Returns the structure's stoichiometry using chain_res_name_sequence.
  - `get_table(self, table_name: str)` — [`L393`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L393)
  - `group_by_chain(self)` — [`L958`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L958) — Returns a Structure where all fields are per-chain.
  - `group_by_residue(self)` — [`L942`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L942) — Returns a Structure with one atom per residue.
  - `is_dna_mask(self)` — [`L701`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L701) — Boolean label indicating if each atom is from DNA or not.
  - `is_ligand_mask(self)` — [`L716`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L716) — Boolean label indicating if each atom is a ligand or not.
  - `is_nucleic_mask(self)` — [`L711`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L711) — Boolean label indicating if each atom is a nucleic acid or not.
  - `is_protein_mask(self)` — [`L696`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L696) — Boolean label indicating if each atom is from protein or not.
  - `is_rna_mask(self)` — [`L706`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L706) — Boolean label indicating if each atom is from RNA or not.
  - `is_water_mask(self)` — [`L721`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L721) — Boolean label indicating if each atom is from water or not.
  - `iter_atoms(self)` — [`L725`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L725) — Iterates over the atoms in the structure. — documented in [alphafold3-structure](../../../../concepts/alphafold3-structure.md)
  - `iter_bonds(self)` — [`L857`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L857) — Iterates over the atoms and bond information. — documented in [alphafold3-structure-bonds](../../../../concepts/alphafold3-structure-bonds.md)
  - `iter_chain_ranges(self)` — [`L826`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L826) — Iterates pairs of (chain_start, chain_end) indices.
  - `iter_chains(self)` — [`L846`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L846) — Iterates over the chains in the structure.
  - `iter_residue_ranges(self)` — [`L836`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L836) — Iterates pairs of (residue_start, residue_end) indices.
  - `iter_residues(self, include_unresolved: bool = False)` — [`L775`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L775) — Iterates over the residues in the structure. — documented in [alphafold3-structure](../../../../concepts/alphafold3-structure.md)
  - `label_asym_id_to_entity_id(self)` — [`L509`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L509)
  - `merge_chains(self, *, chain_groups: Sequence[Sequence[str]], chain_group_ids: Sequence[str] | None = None, chain_group_types: Sequence[str] | None = None, chain_group_entity_ids: Sequence[str] | None = None)` — [`L2317`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2317) — Merges chains in each group into a single chain. — documented in [alphafold3-structure](../../../../concepts/alphafold3-structure.md)
  - `name(self)` — [`L427`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L427) — documented in [alphafold3-structure](../../../../concepts/alphafold3-structure.md)
  - `num_atoms(self)` — [`L659`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L659)
  - `num_chains(self)` — [`L677`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L677)
  - `num_models(self)` — [`L681`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L681) — The number of models of this Structure.
  - `num_residues(self, *, count_unresolved: bool)` — [`L662`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L662) — Returns the number of residues in this Structure.
  - `order_and_drop_atoms_to_match(self, other: Structure, *, allow_missing_atoms: bool = False)` — [`L1009`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1009) — Returns a new structure with atoms ordered & dropped to match another's.
  - `polymer_auth_asym_id_to_label_asym_id(self, *, protein: bool = True, rna: bool = True, dna: bool = True, other: bool = True)` — [`L2019`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2019) — Mapping from author chain ID to internal chain ID, polymers only.
  - `polymer_author_chain_single_letter_sequence(self, *, include_missing_residues: bool = True, protein: bool = True, rna: bool = True, dna: bool = True, other: bool = True)` — [`L2089`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2089) — Mapping from author chain ID to single letter aa sequence, polymers only.
  - `present_chains(self)` — [`L608`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L608) — Returns table of chains which have at least 1 resolved atom.
  - `present_residues(self)` — [`L614`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L614) — Returns table of residues which have at least 1 resolved atom.
  - `release_date(self)` — [`L431`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L431)
  - `remap_res_id(self, res_id_map: Mapping[str, Mapping[int, int]])` — [`L2724`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2724) — Returns a copy of this structure with residue IDs remapped.
  - `rename_auth_asym_ids(self, new_id_by_old_id: Mapping[str, str])` — [`L2601`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2601) — Returns a new structure with renamed author chain IDs (auth_asym_ids).
  - `rename_chain_ids(self, new_id_by_old_id: Mapping[str, str])` — [`L2624`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2624) — Returns a new structure with renamed chain IDs (label_asym_ids). — documented in [alphafold3-structure-bioassemblies](../../../../concepts/alphafold3-structure-bioassemblies.md)
  - `rename_chains_to_match(self, other: Structure, *, fuzzy_match_non_standard_res: bool = True)` — [`L2779`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2779) — Returns a new structure with renamed chains to match another's.
  - `rename_res_name(self, res_name_map: Mapping[str, str], fail_if_not_found: bool = True)` — [`L2683`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2683) — Returns a copy of this structure with residues renamed.
  - `reorder_chains(self, new_order: Sequence[str])` — [`L2546`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2546) — Reorders tables so that the label_asym_ids are in the given order.
  - `res_auth_seq_id(self)` — [`L552`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L552) — Returns the residue auth seq ID for each atom in the structure.
  - `res_boundaries(self)` — [`L603`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L603) — The indices in the atom fields where each present residue begins.
  - `res_id(self)` — [`L544`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L544)
  - `res_insertion_code(self)` — [`L559`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L559) — Returns the residue insertion code for each atom in the structure.
  - `res_name(self)` — [`L548`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L548)
  - `reset_author_naming_scheme(self)` — [`L1738`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1738) — Remove author chain/residue ids, entity info and use internal ids.
  - `residues_table(self)` — [`L412`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L412) — Residues table.
  - `resolution(self)` — [`L435`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L435)
  - `select(field, default)` — [`L1186`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1186) — documented in [alphafold3-structure](../../../../concepts/alphafold3-structure.md)
  - `slice_leading_dims(self)` — [`L2221`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2221) — Used to create a new Structure by slicing into the leading dimensions.
  - `split_by_chain(self)` — [`L2281`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2281) — Splits a Structure into single-chain Structures, one for each chain.
  - `structure_method(self)` — [`L439`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L439)
  - `to_mmcif(self, *, coords_decimal_places: int = _COORDS_DECIMAL_PLACES)` — [`L2971`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2971) — Returns an mmCIF string representing the structure.
  - `to_mmcif_dict(self, *, coords_decimal_places: int = _COORDS_DECIMAL_PLACES)` — [`L2952`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2952) — Returns an Mmcif representing the structure.
  - `to_res_arrays(self, *, include_missing_residues: bool, atom_order: Mapping[str, int] = atom_types.ATOM37_ORDER)` — [`L2464`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2464) — Returns atom-level information in arrays containing a num_res dimension.
  - `to_res_atom_lists(self, *, include_missing_residues: bool)` — [`L2513`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2513) — Returns list of atom dictionaries grouped by residue.
  - `transform_states_to_chains(self)` — [`L2293`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2293) — Transforms states to chains.
  - `unresolved_residues(self)` — [`L622`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L622) — Returns table of residues which have at least 1 resolved atom.
  - `unstack(self, axis: int = 0)` — [`L2240`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2240) — Unstacks a multi-model structure into a list of Structures.
  - `with_sorted_chains(self)` — [`L974`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L974) — Returns a new structure with the chains are in reverse spreadsheet style.
  - `without_hydrogen(self)` — [`L1718`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1718) — Returns the structure without hydrogen atoms.
  - `without_terminal_oxygens(self)` — [`L1724`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1724) — Returns the structure without terminal oxygen atoms.
  - `foreign_keys` — [`L311`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L311)
  - `tables` — [`L309`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L309)
- protocol/private: `_VERSION`[`L338`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L338), `__init__`[`L317`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L317), `__repr__`[`L651`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L651), `_atoms`[`L348`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L348), `_bioassembly_data`[`L343`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L343), `_bonds`[`L349`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L349), `_chains`[`L346`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L346), `_chemical_components_data`[`L344`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L344), `_iter_residues`[`L1083`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1083), `_keep`[`L1871`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1871), `_name`[`L339`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L339), `_release_date`[`L340`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L340), `_residues`[`L347`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L347), `_resolution`[`L341`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L341), `_structure_method`[`L342`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L342), `_to_internal_res_id`[`L1938`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L1938), `_to_mmcif_header`[`L2919`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2919)
- uses (calls/refs, reference-scoped): [`to_mmcif_dict`](chemical_components.md#ChemicalComponentsData.to_mmcif_dict), [`to_mmcif_atom_site_and_bonds_table`](structure_tables.md#to_mmcif_atom_site_and_bonds_table), [`chain_key`](structure_tables.md#Residues.chain_key), [`chain_key`](structure_tables.md#Atoms.chain_key), [`res_key`](structure_tables.md#Atoms.res_key), [`concat`](structure.md#concat), [`Mmcif`](mmcif.md#Mmcif), [`id`](structure_tables.md#Chains.id), [`get_transforms`](bioassemblies.md#BioassemblyData.get_transforms), [`Residues`](structure_tables.md#Residues), [`_UNSET`](structure.md#_UNSET), [`Chains`](structure_tables.md#Chains), [`Atoms`](structure_tables.md#Atoms), [`name`](structure_tables.md#Residues.name), [`_UnsetType`](structure.md#_UnsetType), [`id`](structure_tables.md#Residues.id), [`ChemicalComponentsData`](chemical_components.md#ChemicalComponentsData), [`chem_comp`](chemical_components.md#ChemicalComponentsData.chem_comp), [`to_mmcif_sequence_and_entity_tables`](structure_tables.md#to_mmcif_sequence_and_entity_tables), [`rename_label_asym_ids`](bioassemblies.md#BioassemblyData.rename_label_asym_ids), [`Bonds`](structure_tables.md#Bonds), [`name`](structure_tables.md#Atoms.name), [`to_mmcif_dict`](bioassemblies.md#BioassemblyData.to_mmcif_dict), [`auth_seq_id`](structure_tables.md#Residues.auth_seq_id), [`insertion_code`](structure_tables.md#Residues.insertion_code), [`auth_asym_id`](structure_tables.md#Chains.auth_asym_id), [`b_factor`](structure_tables.md#Atoms.b_factor), [`entity_id`](structure_tables.md#Chains.entity_id), [`occupancy`](structure_tables.md#Atoms.occupancy), [`x`](structure_tables.md#Atoms.x), [`dest_atom_key`](bonds.md#Bonds.dest_atom_key), [`element`](structure_tables.md#Atoms.element), [`entity_desc`](structure_tables.md#Chains.entity_desc), [`from_atom_key`](bonds.md#Bonds.from_atom_key), [`y`](structure_tables.md#Atoms.y), [`z`](structure_tables.md#Atoms.z), [`num_models`](structure_tables.md#Atoms.num_models), [`BioassemblyData`](bioassemblies.md#BioassemblyData), [`_get_change_indices`](structure.md#_get_change_indices), [`_unpack_filter_predicates`](structure.md#_unpack_filter_predicates)  (+51 more)
- used by: [`process_structure`](../model/pipeline/pipeline.md#WholePdbPipeline.process_structure), [`from_sequences_and_bonds`](parsing.md#from_sequences_and_bonds), [`compute_features`](../model/features.md#MSA.compute_features), [`from_res_arrays`](parsing.md#from_res_arrays), [`clean_structure`](../model/pipeline/structure_cleaning.md#clean_structure), [`get_inference_result`](../model/model.md#Model.get_inference_result), [`post_process_inference_result`](../model/post_processing.md#post_process_inference_result), [`get_bond_layout`](../model/pipeline/inter_chain_bonds.md#get_bond_layout), [`residues_from_structure`](../model/atom_layout/atom_layout.md#residues_from_structure), [`compute_features`](../model/features.md#Templates.compute_features), [`from_parsed_mmcif`](parsing.md#from_parsed_mmcif), [`get_predicted_structure`](../model/model.md#get_predicted_structure), [`atom_layout_from_structure`](../model/atom_layout/atom_layout.md#atom_layout_from_structure), [`make_structure`](../model/atom_layout/atom_layout.md#make_structure), [`_get_bond_atom_arrays`](../model/pipeline/inter_chain_bonds.md#_get_bond_atom_arrays), [`create_empty_output_struc_and_layout`](../model/pipeline/structure_cleaning.md#create_empty_output_struc_and_layout), [`concat`](structure.md#concat), [`make_empty_structure`](structure.md#make_empty_structure), [`stack`](structure.md#stack), [`compare_chirality`](../model/scoring/chirality.md#compare_chirality), [`_get_leaving_atom_mask`](../model/pipeline/structure_cleaning.md#_get_leaving_atom_mask), [`fraction_disordered`](../model/confidences.md#fraction_disordered), [`__getitem__`](structure.md#_LeadingDimSlice.__getitem__), [`from_atom_arrays`](parsing.md#from_atom_arrays), [`compute_features`](../model/features.md#ConvertModelOutput.compute_features), [`from_inference_result`](../model/confidence_types.md#AtomConfidence.from_inference_result), [`from_inference_result`](../model/confidence_types.md#StructureConfidenceFull.from_inference_result), [`get_polymer_ligand_and_ligand_ligand_bonds`](../model/pipeline/inter_chain_bonds.md#get_polymer_ligand_and_ligand_ligand_bonds), [`has_clash`](../model/confidences.md#has_clash), [`from_mmcif`](parsing.md#from_mmcif), [`get_ligand_ligand_bonds`](../model/pipeline/inter_chain_bonds.md#get_ligand_ligand_bonds), [`get_polymer_ligand_bonds`](../model/pipeline/inter_chain_bonds.md#get_polymer_ligand_bonds), [`process_item`](../model/pipeline/pipeline.md#WholePdbPipeline.process_item), [`_assign_unique_chain_ids`](structure.md#_assign_unique_chain_ids), [`_mol_from_ligand_struc`](../model/scoring/chirality.md#_mol_from_ligand_struc), [`multichain_residue_index`](structure.md#multichain_residue_index), [`_get_smiles`](../model/atom_layout/atom_layout.md#residues_from_structure._get_smiles), [`predicted_structure`](../model/model.md#InferenceResult.predicted_structure), [`empty_output_struc`](../model/features.md#ConvertModelOutput.empty_output_struc), [`__init__`](structure.md#_LeadingDimSlice.__init__)

### `StructureTables`
- def: [`src/alphafold3/structure/structure.py:281`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L281)
- signature: `class StructureTables:`
- members:
  - `atoms` — [`L284`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L284)
  - `bonds` — [`L285`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L285)
  - `chains` — [`L282`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L282)
  - `residues` — [`L283`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L283)
- uses (calls/refs, reference-scoped): [`Residues`](structure_tables.md#Residues), [`Chains`](structure_tables.md#Chains), [`Atoms`](structure_tables.md#Atoms), [`Bonds`](structure_tables.md#Bonds)
- used by: [`filter`](structure.md#Structure.filter), [`_cascade_delete`](structure.md#Structure._cascade_delete), [`order_and_drop_atoms_to_match`](structure.md#Structure.order_and_drop_atoms_to_match), [`_apply_atom_index_array`](structure.md#Structure._apply_atom_index_array)

### `_LeadingDimSlice`
- def: [`src/alphafold3/structure/structure.py:2985`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2985)
- protocol/private: `__getitem__`[`L2998`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2998), `__init__`[`L2995`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2995), `_struc`[`L2996`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L2996)
- uses (calls/refs, reference-scoped): [`copy_and_update`](structure.md#Structure.copy_and_update), [`Structure`](structure.md#Structure), [`Atoms`](structure_tables.md#Atoms), [`atoms_table`](structure.md#Structure.atoms_table), [`num_atoms`](structure.md#Structure.num_atoms), [`multimodel_cols`](structure_tables.md#Atoms.multimodel_cols)
- used by: [`slice_leading_dims`](structure.md#Structure.slice_leading_dims)

### `_UnsetSentinel`  ·  implements/extends Enum
- def: [`src/alphafold3/structure/structure.py:58`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L58)
- signature: `class _UnsetSentinel(enum.Enum):`
- members:
  - `UNSET` — [`L59`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L59)
- used by: [`_UNSET`](structure.md#_UNSET), [`_UnsetType`](structure.md#_UnsetType)

## Functions
- `_assign_unique_chain_ids(strucs: Iterable[Structure])` — [`L3072`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L3072) — Creates a sequence of `Structure` objects with unique chain IDs.
- `_get_change_indices(arr: np.ndarray)` — [`L228`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L228)
- `_unpack_filter_predicates(predicate_by_field_name: Mapping[str, table.FilterPredicate])` — [`L236`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L236) — Unpacks filter kwargs into predicates for each table.
- `concat(strucs: Sequence[Structure], *, name: str | None = None, assign_unique_chain_ids: bool = True, assign_unique_entity_ids: bool = True)` — [`L3106`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L3106) — Concatenates structures along the atom dimension.
- `enumerate_residues(atom_iter: Iterable[Mapping[str, Any]], all_residues: AllResidues | None = None)` — [`L3267`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L3267) — Provides a zero-indexed enumeration of residues in an atom iterable.
- `fix_non_standard_polymer_residues(res_names: np.ndarray, chain_type: str)` — [`L176`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L176) — Remaps residue names to the closest standard protein/RNA/DNA residue.
- `make_empty_structure()` — [`L3257`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L3257) — Returns a new structure consisting of empty array fields.
- `multichain_residue_index(struc: Structure, chain_offset: int = 9000, between_chain_buffer: int = 1000)` — [`L3220`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L3220) — Compute a residue index array that is monotonic across all chains.
- `stack(strucs: Sequence[Structure], axis: int = 0)` — [`L3015`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L3015) — Stacks multiple structures into a single multi-model Structure.

## Module values
- `ARRAY_FIELDS` — [`L127`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L127)
- `ATOM_FIELDS` — [`L115`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L115)
- `AllResidues` — [`L89`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L89)
- `AuthorNamingScheme` — [`L90`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L90)
- `CHAIN_FIELDS` — [`L99`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L99)
- `GLOBAL_FIELDS` — [`L142`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L142)
- `MISSING_AUTH_SEQ_ID` — [`L95`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L95)
- `RESIDUE_FIELDS` — [`L108`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L108)
- `SCALAR_FIELDS` — [`L262`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L262)
- `TABLE_FIELDS` — [`L272`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L272)
- `V2_FIELDS` — [`L277`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L277)
- `_COORDS_DECIMAL_PLACES` — [`L47`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L47)
- `_T` — [`L259`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L259)
- `_UNSET` — [`L63`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L63)
- `_UPDATEABLE_FIELDS` — [`L152`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L152)
- `_UnsetType` — [`L62`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure.py#L62)

