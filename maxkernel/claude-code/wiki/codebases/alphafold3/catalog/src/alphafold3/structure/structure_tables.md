---
title: 'Module: src/alphafold3/structure/structure_tables.py'
type: catalog
provenance: extracted
module: src/alphafold3/structure/structure_tables.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.structure.structure_tables`/
symbols:
  tables_from_atom_arrays: tables_from_atom_arrays().
  to_mmcif_atom_site_and_bonds_table: to_mmcif_atom_site_and_bonds_table().
  Residues.chain_key: Residues#chain_key.
  Atoms.chain_key: Atoms#chain_key.
  Atoms.res_key: Atoms#res_key.
  Atoms.from_defaults: Atoms#from_defaults().
  Chains.id: Chains#id.
  Residues: Residues#
  _default: _default().
  Chains: Chains#
  Atoms: Atoms#
  Residues.name: Residues#name.
  Residues.id: Residues#id.
  Residues.from_defaults: Residues#from_defaults().
  to_mmcif_sequence_and_entity_tables: to_mmcif_sequence_and_entity_tables().
  Chains.from_defaults: Chains#from_defaults().
  Bonds: Bonds.
  Atoms.name: Atoms#name.
  Residues.auth_seq_id: Residues#auth_seq_id.
  Residues.insertion_code: Residues#insertion_code.
  to_mmcif_atom_site_and_bonds_table.tile_for_models: to_mmcif_atom_site_and_bonds_table().tile_for_models().
  Atoms.x: Atoms#x.
  Atoms.b_factor: Atoms#b_factor.
  Atoms.occupancy: Atoms#occupancy.
  Chains.auth_asym_id: Chains#auth_asym_id.
  Chains.entity_id: Chains#entity_id.
  Atoms.element: Atoms#element.
  Atoms.y: Atoms#y.
  Atoms.z: Atoms#z.
  Chains.entity_desc: Chains#entity_desc.
  Atoms.num_models: Atoms#num_models().
  Atoms.shape: Atoms#shape().
  Atoms.ndim: Atoms#ndim().
  Atoms.copy_and_update_coords: Atoms#copy_and_update_coords().
  AuthorNamingScheme.auth_seq_id: AuthorNamingScheme#auth_seq_id.
  AuthorNamingScheme.insertion_code: AuthorNamingScheme#insertion_code.
  Atoms.make_empty: Atoms#make_empty().
  Residues.make_empty: Residues#make_empty().
  Chains.make_empty: Chains#make_empty().
  Atoms.__post_init__: Atoms#__post_init__().
  Atoms.get_value_by_index: Atoms#get_value_by_index().
  AuthorNamingScheme: AuthorNamingScheme#
  AuthorNamingScheme.auth_asym_id: AuthorNamingScheme#auth_asym_id.
  AuthorNamingScheme.entity_id: AuthorNamingScheme#entity_id.
  AuthorNamingScheme.entity_desc: AuthorNamingScheme#entity_desc.
  Atoms.multimodel_cols: Atoms#multimodel_cols.
  _flatten_author_naming_scheme_table: _flatten_author_naming_scheme_table().
  _residue_name_to_record_name: _residue_name_to_record_name().
---
# Module: [`src/alphafold3/structure/structure_tables.py`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py)

## Classes
### `Atoms`  ·  implements/extends Table
- def: [`src/alphafold3/structure/structure_tables.py:89`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L89) — documented in [alphafold3-structure-parsing](../../../../concepts/alphafold3-structure-parsing.md)
- doc: Table of atoms in a Structure.
- signature: `class Atoms(table.Table):`
- members:
  - `copy_and_update_coords(self, coords: np.ndarray)` — [`L176`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L176) — Returns a copy with the x, y and z columns updated.
  - `from_defaults(cls, *, chain_key: np.ndarray, res_key: np.ndarray, key: np.ndarray | None = None, name: np.ndarray | None = None, element: np.ndarray | None = None, x: np.ndarray | None = None, y: np.ndarray | None = None, z: np.ndarray | None = None, b_factor: np.ndarray | None = None, occupancy: np.ndarray | None = None)` — [`L137`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L137) — Create an Atoms table with minimal user inputs. — documented in [alphafold3-structure-structure_tables](../../../../concepts/alphafold3-structure-structure_tables.md)
  - `get_value_by_index(self, column_name: str, index: int)` — [`L168`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L168)
  - `make_empty(cls)` — [`L122`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L122)
  - `ndim(self)` — [`L194`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L194)
  - `num_models(self)` — [`L198`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L198) — The number of models of this Structure.
  - `shape(self)` — [`L190`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L190)
  - `b_factor` — [`L99`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L99) — documented in [alphafold3-structure-structure_tables](../../../../concepts/alphafold3-structure-structure_tables.md)
  - `chain_key` — [`L92`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L92) — documented in [alphafold3-structure-structure_tables](../../../../concepts/alphafold3-structure-structure_tables.md)
  - `element` — [`L95`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L95) — documented in [alphafold3-structure-structure_tables](../../../../concepts/alphafold3-structure-structure_tables.md)
  - `multimodel_cols` — [`L101`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L101)
  - `name` — [`L94`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L94) — documented in [alphafold3-structure-structure_tables](../../../../concepts/alphafold3-structure-structure_tables.md)
  - `occupancy` — [`L100`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L100) — documented in [alphafold3-structure-structure_tables](../../../../concepts/alphafold3-structure-structure_tables.md)
  - `res_key` — [`L93`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L93) — documented in [alphafold3-structure-structure_tables](../../../../concepts/alphafold3-structure-structure_tables.md)
  - `x` — [`L96`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L96) — documented in [alphafold3-structure-structure_tables](../../../../concepts/alphafold3-structure-structure_tables.md)
  - `y` — [`L97`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L97)
  - `z` — [`L98`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L98)
- protocol/private: `__post_init__`[`L109`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L109)
- uses (calls/refs, reference-scoped): [`_default`](structure_tables.md#_default)
- used by: [`from_sequences_and_bonds`](parsing.md#from_sequences_and_bonds), [`get_tables`](parsing.md#get_tables), [`tables_from_atom_arrays`](structure_tables.md#tables_from_atom_arrays), [`copy_and_update`](structure.md#Structure.copy_and_update), [`from_res_arrays`](parsing.md#from_res_arrays), [`iter_atoms`](structure.md#Structure.iter_atoms), [`filter`](structure.md#Structure.filter), [`copy_and_update_atoms`](structure.md#Structure.copy_and_update_atoms), [`to_mmcif_atom_site_and_bonds_table`](structure_tables.md#to_mmcif_atom_site_and_bonds_table), [`_cascade_delete`](structure.md#Structure._cascade_delete), [`order_and_drop_atoms_to_match`](structure.md#Structure.order_and_drop_atoms_to_match), [`merge_chains`](structure.md#Structure.merge_chains), [`_apply_bioassembly_transform`](structure.md#Structure._apply_bioassembly_transform), [`chain_id`](structure.md#Structure.chain_id), [`to_mmcif_dict`](structure.md#Structure.to_mmcif_dict), [`make_empty_structure`](structure.md#make_empty_structure), [`reorder_chains`](structure.md#Structure.reorder_chains), [`stack`](structure.md#stack), [`atom_ids`](structure.md#Structure.atom_ids), [`_validate_consistent_table_ordering`](structure.md#Structure._validate_consistent_table_ordering), [`res_id`](structure.md#Structure.res_id), [`atoms_table`](structure.md#Structure.atoms_table), [`_atom_mask`](structure.md#Structure._atom_mask), [`_apply_atom_index_array`](structure.md#Structure._apply_atom_index_array), [`present_chains`](structure.md#Structure.present_chains), [`present_residues`](structure.md#Structure.present_residues), [`__getitem__`](structure.md#_LeadingDimSlice.__getitem__), [`copy_and_update_from_res_arrays`](structure.md#Structure.copy_and_update_from_res_arrays), [`atom_name`](structure.md#Structure.atom_name), [`res_name`](structure.md#Structure.res_name), [`_keep`](structure.md#Structure._keep), [`_validate_table_foreign_keys`](structure.md#Structure._validate_table_foreign_keys), [`without_terminal_oxygens`](structure.md#Structure.without_terminal_oxygens), [`__init__`](structure.md#Structure.__init__), [`coords`](structure.md#Structure.coords), [`filter_polymers_to_single_atom_per_res`](structure.md#Structure.filter_polymers_to_single_atom_per_res), [`filter_residues`](structure.md#Structure.filter_residues), [`tile_for_models`](structure_tables.md#to_mmcif_atom_site_and_bonds_table.tile_for_models), [`unstack`](structure.md#Structure.unstack), [`chain_type`](structure.md#Structure.chain_type)  (+19 more)

### `AuthorNamingScheme`
- def: [`src/alphafold3/structure/structure_tables.py:58`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L58)
- doc: A mapping from internal values to author values in a mmCIF.
- signature: `class AuthorNamingScheme:`
- members:
  - `auth_asym_id` — [`L71`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L71)
  - `auth_seq_id` — [`L72`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L72)
  - `entity_desc` — [`L75`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L75)
  - `entity_id` — [`L74`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L74)
  - `insertion_code` — [`L73`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L73)
- used by: [`tables_from_atom_arrays`](structure_tables.md#tables_from_atom_arrays), [`author_naming_scheme`](structure.md#Structure.author_naming_scheme), [`AuthorNamingScheme`](structure.md#AuthorNamingScheme.AuthorNamingScheme)

### `Chains`  ·  implements/extends Table
- def: [`src/alphafold3/structure/structure_tables.py:260`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L260) — documented in [alphafold3-structure-parsing](../../../../concepts/alphafold3-structure-parsing.md)
- doc: Table of chains in a Structure.
- signature: `class Chains(table.Table):`
- members:
  - `from_defaults(cls, *, id: np.ndarray, key: np.ndarray | None = None, type: np.ndarray | None = None, auth_asym_id: np.ndarray | None = None, entity_id: np.ndarray | None = None, entity_desc: np.ndarray | None = None)` — [`L281`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L281) — Create a Chains table with minimal user inputs.
  - `make_empty(cls)` — [`L270`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L270)
  - `auth_asym_id` — [`L265`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L265) — documented in [alphafold3-structure-structure_tables](../../../../concepts/alphafold3-structure-structure_tables.md)
  - `entity_desc` — [`L267`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L267) — documented in [alphafold3-structure-structure_tables](../../../../concepts/alphafold3-structure-structure_tables.md)
  - `entity_id` — [`L266`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L266) — documented in [alphafold3-structure-structure_tables](../../../../concepts/alphafold3-structure-structure_tables.md)
  - `id` — [`L263`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L263) — documented in [alphafold3-structure-structure_tables](../../../../concepts/alphafold3-structure-structure_tables.md)
- uses (calls/refs, reference-scoped): [`_default`](structure_tables.md#_default)
- used by: [`from_sequences_and_bonds`](parsing.md#from_sequences_and_bonds), [`get_tables`](parsing.md#get_tables), [`tables_from_atom_arrays`](structure_tables.md#tables_from_atom_arrays), [`copy_and_update`](structure.md#Structure.copy_and_update), [`from_res_arrays`](parsing.md#from_res_arrays), [`iter_atoms`](structure.md#Structure.iter_atoms), [`to_mmcif_atom_site_and_bonds_table`](structure_tables.md#to_mmcif_atom_site_and_bonds_table), [`_cascade_delete`](structure.md#Structure._cascade_delete), [`order_and_drop_atoms_to_match`](structure.md#Structure.order_and_drop_atoms_to_match), [`author_naming_scheme`](structure.md#Structure.author_naming_scheme), [`reset_author_naming_scheme`](structure.md#Structure.reset_author_naming_scheme), [`merge_chains`](structure.md#Structure.merge_chains), [`_apply_bioassembly_transform`](structure.md#Structure._apply_bioassembly_transform), [`make_chains_table`](parsing.md#_ChainResBuilder.make_chains_table), [`chain_id`](structure.md#Structure.chain_id), [`all_residues`](structure.md#Structure.all_residues), [`make_empty_structure`](structure.md#make_empty_structure), [`reorder_chains`](structure.md#Structure.reorder_chains), [`rename_chain_ids`](structure.md#Structure.rename_chain_ids), [`chains`](structure.md#Structure.chains), [`to_mmcif_sequence_and_entity_tables`](structure_tables.md#to_mmcif_sequence_and_entity_tables), [`present_chains`](structure.md#Structure.present_chains), [`iter_chains`](structure.md#Structure.iter_chains), [`__init__`](structure.md#Structure.__init__), [`chains_table`](structure.md#Structure.chains_table), [`label_asym_id_to_entity_id`](structure.md#Structure.label_asym_id_to_entity_id), [`rename_auth_asym_ids`](structure.md#Structure.rename_auth_asym_ids), [`chains`](structure.md#StructureTables.chains)

### `Residues`  ·  implements/extends Table
- def: [`src/alphafold3/structure/structure_tables.py:214`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L214) — documented in [alphafold3-model-atom_layout](../../../../concepts/alphafold3-model-atom_layout.md)
- doc: Table of residues in a Structure.
- signature: `class Residues(table.Table):`
- members:
  - `from_defaults(cls, *, id: np.ndarray, chain_key: np.ndarray, key: np.ndarray | None = None, name: np.ndarray | None = None, auth_seq_id: np.ndarray | None = None, insertion_code: np.ndarray | None = None)` — [`L235`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L235) — Create a Residues table with minimal user inputs. — documented in [alphafold3-structure-structure_tables](../../../../concepts/alphafold3-structure-structure_tables.md)
  - `make_empty(cls)` — [`L224`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L224)
  - `auth_seq_id` — [`L220`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L220) — documented in [alphafold3-structure-structure_tables](../../../../concepts/alphafold3-structure-structure_tables.md)
  - `chain_key` — [`L217`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L217) — documented in [alphafold3-model-atom_layout](../../../../concepts/alphafold3-model-atom_layout.md)
  - `id` — [`L218`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L218) — documented in [alphafold3-model-atom_layout](../../../../concepts/alphafold3-model-atom_layout.md)
  - `insertion_code` — [`L221`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L221) — documented in [alphafold3-structure-structure_tables](../../../../concepts/alphafold3-structure-structure_tables.md)
  - `name` — [`L219`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L219) — documented in [alphafold3-model-atom_layout](../../../../concepts/alphafold3-model-atom_layout.md)
- uses (calls/refs, reference-scoped): [`_default`](structure_tables.md#_default)
- used by: [`from_sequences_and_bonds`](parsing.md#from_sequences_and_bonds), [`get_tables`](parsing.md#get_tables), [`tables_from_atom_arrays`](structure_tables.md#tables_from_atom_arrays), [`copy_and_update`](structure.md#Structure.copy_and_update), [`from_res_arrays`](parsing.md#from_res_arrays), [`iter_atoms`](structure.md#Structure.iter_atoms), [`to_mmcif_atom_site_and_bonds_table`](structure_tables.md#to_mmcif_atom_site_and_bonds_table), [`_cascade_delete`](structure.md#Structure._cascade_delete), [`author_naming_scheme`](structure.md#Structure.author_naming_scheme), [`make_residues_table`](parsing.md#_ChainResBuilder.make_residues_table), [`reset_author_naming_scheme`](structure.md#Structure.reset_author_naming_scheme), [`merge_chains`](structure.md#Structure.merge_chains), [`all_residues`](structure.md#Structure.all_residues), [`iter_residues`](structure.md#Structure.iter_residues), [`chain_res_name_sequence`](structure.md#Structure.chain_res_name_sequence), [`make_empty_structure`](structure.md#make_empty_structure), [`reorder_chains`](structure.md#Structure.reorder_chains), [`rename_chain_ids`](structure.md#Structure.rename_chain_ids), [`chain_single_letter_sequence`](structure.md#Structure.chain_single_letter_sequence), [`atom_ids`](structure.md#Structure.atom_ids), [`copy_and_update_residues`](structure.md#Structure.copy_and_update_residues), [`fix_non_standard_polymer_res`](structure.md#Structure.fix_non_standard_polymer_res), [`to_mmcif_sequence_and_entity_tables`](structure_tables.md#to_mmcif_sequence_and_entity_tables), [`present_residues`](structure.md#Structure.present_residues), [`remap_res_id`](structure.md#Structure.remap_res_id), [`_keep`](structure.md#Structure._keep), [`_validate_table_foreign_keys`](structure.md#Structure._validate_table_foreign_keys), [`__init__`](structure.md#Structure.__init__), [`rename_res_name`](structure.md#Structure.rename_res_name), [`residues_table`](structure.md#Structure.residues_table), [`_iter_residues`](structure.md#Structure._iter_residues), [`unresolved_residues`](structure.md#Structure.unresolved_residues), [`residues`](structure.md#StructureTables.residues)

## Functions
- `_default(candidate_value: np.ndarray | None, default_value: Sequence[Any] | np.ndarray, dtype: Any)` — [`L78`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L78)
- `_flatten_author_naming_scheme_table(res_table: Mapping[str, Mapping[int, str | None]], chain_ids: np.ndarray, res_chain_ids: np.ndarray, res_ids: np.ndarray, default_if_missing: str, table_name: str)` — [`L534`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L534) — Flattens an author naming scheme table consistently with res_ids.
- `_residue_name_to_record_name(residue_name: np.ndarray, polymer_mask: np.ndarray)` — [`L43`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L43) — Returns record names (ATOM/HETATM) given residue names and polymer mask.
- `tables_from_atom_arrays(*, res_id: np.ndarray, author_naming_scheme: AuthorNamingScheme | None = None, all_residues: Mapping[str, Sequence[tuple[str, int]]] | None = None, chain_id: np.ndarray | None = None, chain_type: np.ndarray | None = None, res_name: np.ndarray | None = None, atom_key: np.ndarray | None = None, atom_name: np.ndarray | None = None, atom_element: np.ndarray | None = None, atom_x: np.ndarray | None = None, atom_y: np.ndarray | None = None, atom_z: np.ndarray | None = None, atom_b_factor: np.ndarray | None = None, atom_occupancy: np.ndarray | None = None)` — [`L568`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L568) — Returns Structure tables constructed from atom array level data. — documented in [alphafold3-structure-structure_tables](../../../../concepts/alphafold3-structure-structure_tables.md)
- `tile_for_models(arr: np.ndarray)` — [`L496`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L496)
- `to_mmcif_atom_site_and_bonds_table(*, chains: Chains, residues: Residues, atoms: Atoms, bonds: Bonds, coords_decimal_places: int)` — [`L422`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L422) — Returns raw _atom_site and _struct_conn mmCIF tables.
- `to_mmcif_sequence_and_entity_tables(chains: Chains, residues: Residues, atom_res_key: np.ndarray)` — [`L308`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L308) — Returns raw sequence and entity mmCIF tables.

## Module values
- `Bonds` — [`L40`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/structure_tables.py#L40) — documented in [alphafold3-structure](../../../../concepts/alphafold3-structure.md)

