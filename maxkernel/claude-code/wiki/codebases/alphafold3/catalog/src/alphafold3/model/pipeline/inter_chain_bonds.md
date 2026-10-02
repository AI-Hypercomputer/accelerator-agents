---
title: 'Module: src/alphafold3/model/pipeline/inter_chain_bonds.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/pipeline/inter_chain_bonds.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.pipeline.inter_chain_bonds`/
symbols:
  get_bond_layout: get_bond_layout().
  _get_bond_atom_arrays: _get_bond_atom_arrays().
  get_polymer_ligand_and_ligand_ligand_bonds: get_polymer_ligand_and_ligand_ligand_bonds().
  get_ligand_ligand_bonds: get_ligand_ligand_bonds().
  get_polymer_ligand_bonds: get_polymer_ligand_bonds().
  _remove_multi_bonds: _remove_multi_bonds().
  BondAtomArrays.chain_type: BondAtomArrays#chain_type.
  BondAtomArrays.chain_id: BondAtomArrays#chain_id.
  BondAtomArrays.res_id: BondAtomArrays#res_id.
  BOND_THRESHOLD_ALL_ANGSTROM.BOND_THRESHOLD_ALL_ANGSTROM: BOND_THRESHOLD_ALL_ANGSTROM.BOND_THRESHOLD_ALL_ANGSTROM.
  BondAtomArrays.res_name: BondAtomArrays#res_name.
  BondAtomArrays.atom_name: BondAtomArrays#atom_name.
  BondAtomArrays.coords: BondAtomArrays#coords.
  BOND_THRESHOLD_GLYCANS_ANGSTROM.BOND_THRESHOLD_GLYCANS_ANGSTROM: BOND_THRESHOLD_GLYCANS_ANGSTROM.BOND_THRESHOLD_GLYCANS_ANGSTROM.
  BondAtomArrays: BondAtomArrays#
---
# Module: [`src/alphafold3/model/pipeline/inter_chain_bonds.py`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/inter_chain_bonds.py)

## Classes
### `BondAtomArrays`  ·  implements/extends NamedTuple
- def: [`src/alphafold3/model/pipeline/inter_chain_bonds.py:37`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/inter_chain_bonds.py#L37)
- signature: `class BondAtomArrays(NamedTuple):`
- members:
  - `atom_name` — [`L42`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/inter_chain_bonds.py#L42)
  - `chain_id` — [`L38`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/inter_chain_bonds.py#L38)
  - `chain_type` — [`L39`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/inter_chain_bonds.py#L39)
  - `coords` — [`L43`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/inter_chain_bonds.py#L43)
  - `res_id` — [`L40`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/inter_chain_bonds.py#L40)
  - `res_name` — [`L41`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/inter_chain_bonds.py#L41)
- used by: [`get_bond_layout`](inter_chain_bonds.md#get_bond_layout), [`_get_bond_atom_arrays`](inter_chain_bonds.md#_get_bond_atom_arrays)

## Functions
- `_get_bond_atom_arrays(struc: structure.Structure, bond_atom_indices: np.ndarray)` — [`L46`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/inter_chain_bonds.py#L46)
- `_remove_multi_bonds(bond_layout: atom_layout.AtomLayout)` — [`L112`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/inter_chain_bonds.py#L112) — Remove instances greedily.
- `get_bond_layout(bond_threshold: float = BOND_THRESHOLD_ALL_ANGSTROM, *, struct: structure.Structure, allowed_chain_types1: Collection[str], allowed_chain_types2: Collection[str], include_bond_types: Collection[str] = ('covale',), allowed_res_names: Collection[str] | None = None, allow_multiple_bonds_per_atom: bool)` — [`L211`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/inter_chain_bonds.py#L211) — Get bond_layout for all bonds between two sets of chain types.
- `get_ligand_ligand_bonds(struct: structure.Structure, only_glycan_ligands: bool, allow_multiple_bonds_per_atom: bool = False)` — [`L135`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/inter_chain_bonds.py#L135) — Return ligand-ligand inter-residue bonds.
- `get_polymer_ligand_and_ligand_ligand_bonds(struct: structure.Structure, only_glycan_ligands: bool, allow_multiple_bonds_per_atom: bool)` — [`L60`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/inter_chain_bonds.py#L60) — Return polymer-ligand & ligand-ligand inter-residue bonds.
- `get_polymer_ligand_bonds(struct: structure.Structure, only_glycan_ligands: bool, allow_multiple_bonds_per_atom: bool = False, bond_threshold: float | None = None)` — [`L171`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/inter_chain_bonds.py#L171) — Return polymer-ligand interchain bonds.

## Module values
- `BOND_THRESHOLD_ALL_ANGSTROM` — [`L34`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/inter_chain_bonds.py#L34)
- `BOND_THRESHOLD_GLYCANS_ANGSTROM` — [`L32`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/inter_chain_bonds.py#L32)

