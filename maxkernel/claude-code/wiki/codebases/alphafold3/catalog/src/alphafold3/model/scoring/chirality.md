---
title: 'Module: src/alphafold3/model/scoring/chirality.py'
type: catalog
provenance: extracted
module: src/alphafold3/model/scoring/chirality.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.model.scoring.chirality`/
symbols:
  compare_chirality: compare_chirality().
  _mol_from_ligand_struc: _mol_from_ligand_struc().
  _find_chiral_centres: _find_chiral_centres().
  _chiral_match: _chiral_match().
  _CHIRAL_ELEMENTS: _CHIRAL_ELEMENTS.
  _maybe_mol_from_ccd: _maybe_mol_from_ccd().
---
# Module: [`src/alphafold3/model/scoring/chirality.py`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/scoring/chirality.py)

## Functions
- `_chiral_match(mol1: rd_chem.Mol, mol2: rd_chem.Mol)` — [`L58`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/scoring/chirality.py#L58) — Compares chirality of two Mols. Mol1 can match a subset of mol2.
- `_find_chiral_centres(mol: rd_chem.Mol)` — [`L33`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/scoring/chirality.py#L33) — Find chiral centres and detect their chirality.
- `_maybe_mol_from_ccd(res_name: str)` — [`L127`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/scoring/chirality.py#L127) — Creates a Mol object from CCD information if res_name is in the CCD.
- `_mol_from_ligand_struc(ligand_struc: structure.Structure, ref_mol: rd_chem.Mol)` — [`L80`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/scoring/chirality.py#L80) — Creates a Mol object from a ligand structure and reference mol.
- `compare_chirality(test_struc: structure.Structure, ref_mol_by_chain: Mapping[str, rd_chem.Mol] | None = None)` — [`L149`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/scoring/chirality.py#L149) — Compares chirality of ligands in a structure with reference molecules.

## Module values
- `_CHIRAL_ELEMENTS` — [`L30`](../../../../../../../../raw/code/alphafold3/src/alphafold3/model/scoring/chirality.py#L30)

