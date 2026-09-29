---
title: 'Module: src/alphafold3/model/pipeline/structure_cleaning.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/pipeline/structure_cleaning.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.pipeline.structure_cleaning`/
symbols:
  clean_structure: clean_structure().
  create_empty_output_struc_and_layout: create_empty_output_struc_and_layout().
  _get_leaving_atom_mask: _get_leaving_atom_mask().
---
# Module: [`src/alphafold3/model/pipeline/structure_cleaning.py`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/structure_cleaning.py)

## Functions
- `_get_leaving_atom_mask(struc: structure.Structure, polymer_ligand_bonds: atom_layout.AtomLayout | None, ligand_ligand_bonds: atom_layout.AtomLayout | None, chain_id: str, chain_type: str, res_id: int, res_name: str, fix_standalone_glycans: bool)` — [`L32`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/structure_cleaning.py#L32) — Updates a drop_leaving_atoms mask with new leaving atom locations.
- `clean_structure(struc: structure.Structure, ccd: chemical_components.Ccd, *, drop_missing_sequence: bool, drop_non_standard_atoms: bool, filter_waters: bool, filter_hydrogens: bool, filter_leaving_atoms: bool, only_glycan_ligands_for_leaving_atoms: bool, covalent_bonds_only: bool, remove_polymer_polymer_bonds: bool, remove_bad_bonds: bool, fix_standalone_glycans: bool)` — [`L75`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/structure_cleaning.py#L75) — Returns a cleaned version of the input structure.
- `create_empty_output_struc_and_layout(struc: structure.Structure, ccd: chemical_components.Ccd, *, with_hydrogens: bool = False, skip_unk: bool = False, polymer_ligand_bonds: atom_layout.AtomLayout | None = None, ligand_ligand_bonds: atom_layout.AtomLayout | None = None, drop_ligand_leaving_atoms: bool = False, fix_standalone_glycans: bool = False)` — [`L254`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/pipeline/structure_cleaning.py#L254) — Make zero-coordinate structure from all physical residues.

