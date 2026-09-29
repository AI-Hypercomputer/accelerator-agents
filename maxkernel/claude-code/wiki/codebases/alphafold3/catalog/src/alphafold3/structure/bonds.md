---
title: 'Module: src/alphafold3/structure/bonds.py'
type: catalog
provenance: extracted
module: src/alphafold3/structure/bonds.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.structure.bonds`/
symbols:
  Bonds: Bonds#
  Bonds.from_atom_key: Bonds#from_atom_key.
  Bonds.dest_atom_key: Bonds#dest_atom_key.
  concat_with_atom_keys: concat_with_atom_keys().
  Bonds.restrict_to_atoms: Bonds#restrict_to_atoms().
  Bonds.make_empty: Bonds#make_empty().
  Bonds.get_atom_indices: Bonds#get_atom_indices().
  Bonds.role: Bonds#role.
  Bonds.to_mmcif_dict_from_atom_arrays: Bonds#to_mmcif_dict_from_atom_arrays().
---
# Module: [`src/alphafold3/structure/bonds.py`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bonds.py)

## Classes
### `Bonds`  ·  implements/extends Table
- def: [`src/alphafold3/structure/bonds.py:33`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bonds.py#L33) — documented in [alphafold3-structure-bonds](../../../../concepts/alphafold3-structure-bonds.md)
- doc: Table of atomic bonds.
- signature: `class Bonds(table.Table):`
- members:
  - `get_atom_indices(self, atom_key: np.ndarray)` — [`L63`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bonds.py#L63) — Returns the indices of the from/dest atoms in the atom_key array. — documented in [alphafold3-structure-bonds](../../../../concepts/alphafold3-structure-bonds.md)
  - `make_empty(cls)` — [`L54`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bonds.py#L54) — documented in [alphafold3-structure-parsing](../../../../concepts/alphafold3-structure-parsing.md)
  - `restrict_to_atoms(self, atom_key: np.ndarray)` — [`L89`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bonds.py#L89) — documented in [alphafold3-structure-bonds](../../../../concepts/alphafold3-structure-bonds.md)
  - `to_mmcif_dict_from_atom_arrays(self, atom_key: np.ndarray, chain_id: np.ndarray, res_id: np.ndarray, res_name: np.ndarray, atom_name: np.ndarray, auth_asym_id: np.ndarray, auth_seq_id: np.ndarray, insertion_code: np.ndarray)` — [`L97`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bonds.py#L97) — Returns a dict suitable for building a CifDict, representing bonds.
  - `dest_atom_key` — [`L51`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bonds.py#L51) — documented in [alphafold3-structure-bonds](../../../../concepts/alphafold3-structure-bonds.md)
  - `from_atom_key` — [`L48`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bonds.py#L48) — documented in [alphafold3-structure-bonds](../../../../concepts/alphafold3-structure-bonds.md)
  - `role` — [`L45`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bonds.py#L45) — documented in [alphafold3-structure-bonds](../../../../concepts/alphafold3-structure-bonds.md)
- used by: [`from_sequences_and_bonds`](parsing.md#from_sequences_and_bonds), [`from_res_arrays`](parsing.md#from_res_arrays), [`clean_structure`](../model/pipeline/structure_cleaning.md#clean_structure), [`get_bond_layout`](../model/pipeline/inter_chain_bonds.md#get_bond_layout), [`to_mmcif_atom_site_and_bonds_table`](structure_tables.md#to_mmcif_atom_site_and_bonds_table), [`_cascade_delete`](structure.md#Structure._cascade_delete), [`from_parsed_mmcif`](parsing.md#from_parsed_mmcif), [`iter_bonds`](structure.md#Structure.iter_bonds), [`add_bonds`](structure.md#Structure.add_bonds), [`make_empty_structure`](structure.md#make_empty_structure), [`_parse_bonds`](parsing.md#_parse_bonds), [`from_atom_arrays`](parsing.md#from_atom_arrays), [`Bonds`](structure_tables.md#Bonds), [`concat_with_atom_keys`](bonds.md#concat_with_atom_keys)

## Functions
- `concat_with_atom_keys(bonds_tables: Sequence[Bonds | None], atom_key_arrays: Sequence[np.ndarray])` — [`L175`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bonds.py#L175) — Concatenates bonds tables and atom keys simultaneously. — documented in [alphafold3-structure-bonds](../../../../concepts/alphafold3-structure-bonds.md)

