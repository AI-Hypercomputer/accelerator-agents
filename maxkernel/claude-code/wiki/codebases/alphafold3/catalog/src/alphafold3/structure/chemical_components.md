---
title: 'Module: src/alphafold3/structure/chemical_components.py'
type: catalog
provenance: extracted
module: src/alphafold3/structure/chemical_components.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.structure.chemical_components`/
symbols:
  ChemicalComponentsData.to_mmcif_dict: ChemicalComponentsData#to_mmcif_dict().
  ChemicalComponentsData.from_mmcif: ChemicalComponentsData#from_mmcif().
  parse_atom_data: parse_atom_data().
  get_data_for_ccd_components: get_data_for_ccd_components().
  ChemicalComponentsData: ChemicalComponentsData#
  ChemicalComponentsData.chem_comp: ChemicalComponentsData#chem_comp.
  parse_bond_data: parse_bond_data().
  populate_missing_ccd_data: populate_missing_ccd_data().
  ChemCompEntry.pdbx_smiles: ChemCompEntry#pdbx_smiles.
  ChemCompEntry: ChemCompEntry#
  _to_optional_float: _to_optional_float().
  ChemCompEntry.chem_comp_atoms: ChemCompEntry#chem_comp_atoms.
  ChemCompEntry.chem_comp_bonds: ChemCompEntry#chem_comp_bonds.
  ChemicalComponentsData.format_coords: ChemicalComponentsData#format_coords().
  _to_optional_int: _to_optional_int().
  _value_is_missing: _value_is_missing().
  ChemCompEntry.name: ChemCompEntry#name.
  ChemCompEntry.pdbx_synonyms: ChemCompEntry#pdbx_synonyms.
  ChemCompEntry.formula: ChemCompEntry#formula.
  ChemCompEntry.formula_weight: ChemCompEntry#formula_weight.
  ChemCompEntry.mon_nstd_flag: ChemCompEntry#mon_nstd_flag.
  get_all_atoms_in_entry: get_all_atoms_in_entry().
  get_res_atom_names: get_res_atom_names().
  ChemCompAtom: ChemCompAtom#
  ChemCompAtom.ordinal: ChemCompAtom#ordinal.
  ChemCompAtom.charge: ChemCompAtom#charge.
  ChemCompBond: ChemCompBond#
  ChemCompEntry.extends: ChemCompEntry#extends().
  ChemCompEntry.rdkit_mol: ChemCompEntry#rdkit_mol().
  ChemCompAtom.type_symbol: ChemCompAtom#type_symbol.
  ChemCompAtom.leaving_atom_flag: ChemCompAtom#leaving_atom_flag.
  ChemCompAtom.model_ideal_x: ChemCompAtom#model_ideal_x.
  ChemCompAtom.model_ideal_y: ChemCompAtom#model_ideal_y.
  ChemCompAtom.model_ideal_z: ChemCompAtom#model_ideal_z.
  ChemCompBond.atom_id_1: ChemCompBond#atom_id_1.
  ChemCompBond.atom_id_2: ChemCompBond#atom_id_2.
  ChemCompBond.value_order: ChemCompBond#value_order.
  ChemCompBond.aromatic_flag: ChemCompBond#aromatic_flag.
  ChemCompBond.stereo_config: ChemCompBond#stereo_config.
  MissingChemicalComponentsDataError: MissingChemicalComponentsDataError#
  _REQUIRED_MMCIF_COLUMNS: _REQUIRED_MMCIF_COLUMNS.
  ChemCompEntry.__post_init__: ChemCompEntry#__post_init__().
---
# Module: [`src/alphafold3/structure/chemical_components.py`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py)

## Classes
### `ChemCompAtom`
- def: [`src/alphafold3/structure/chemical_components.py:47`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L47) — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
- doc: Items of _chem_comp_atom category.
- signature: `class ChemCompAtom:`
- members:
  - `charge` — [`L64`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L64) — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
  - `leaving_atom_flag` — [`L65`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L65) — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
  - `model_ideal_x` — [`L66`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L66) — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
  - `model_ideal_y` — [`L67`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L67)
  - `model_ideal_z` — [`L68`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L68)
  - `ordinal` — [`L63`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L63) — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
  - `type_symbol` — [`L62`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L62) — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
- used by: [`to_mmcif_dict`](chemical_components.md#ChemicalComponentsData.to_mmcif_dict), [`parse_atom_data`](chemical_components.md#parse_atom_data), [`chem_comp_atoms`](chemical_components.md#ChemCompEntry.chem_comp_atoms)

### `ChemCompBond`
- def: [`src/alphafold3/structure/chemical_components.py:72`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L72) — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
- doc: Items of _chem_comp_bond category.
- signature: `class ChemCompBond:`
- members:
  - `aromatic_flag` — [`L88`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L88) — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
  - `atom_id_1` — [`L85`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L85) — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
  - `atom_id_2` — [`L86`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L86)
  - `stereo_config` — [`L89`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L89) — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
  - `value_order` — [`L87`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L87) — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
- used by: [`to_mmcif_dict`](chemical_components.md#ChemicalComponentsData.to_mmcif_dict), [`parse_bond_data`](chemical_components.md#parse_bond_data), [`chem_comp_bonds`](chemical_components.md#ChemCompEntry.chem_comp_bonds)

### `ChemCompEntry`
- def: [`src/alphafold3/structure/chemical_components.py:93`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L93) — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
- doc: Items of _chem_comp category.
- signature: `class ChemCompEntry:`
- members:
  - `extends(self, other: Self)` — [`L116`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L116) — Checks whether this ChemCompEntry extends another one.
  - `rdkit_mol(self)` — [`L127`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L127) — Returns an RDKit Mol, created via RDKit from entry SMILES string.
  - `chem_comp_atoms` — [`L108`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L108) — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
  - `chem_comp_bonds` — [`L109`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L109) — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
  - `formula` — [`L103`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L103) — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
  - `formula_weight` — [`L104`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L104) — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
  - `mon_nstd_flag` — [`L105`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L105) — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
  - `name` — [`L101`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L101) — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
  - `pdbx_smiles` — [`L106`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L106) — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
  - `pdbx_synonyms` — [`L102`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L102) — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
- protocol/private: `__post_init__`[`L111`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L111)
- uses (calls/refs, reference-scoped): [`_value_is_missing`](chemical_components.md#_value_is_missing), [`ChemCompAtom`](chemical_components.md#ChemCompAtom), [`ChemCompBond`](chemical_components.md#ChemCompBond)
- used by: [`to_mmcif_dict`](chemical_components.md#ChemicalComponentsData.to_mmcif_dict), [`from_mmcif`](chemical_components.md#ChemicalComponentsData.from_mmcif), [`get_data_for_ccd_components`](chemical_components.md#get_data_for_ccd_components), [`chem_comp`](chemical_components.md#ChemicalComponentsData.chem_comp), [`get_reference`](../model/features.md#get_reference), [`populate_missing_ccd_data`](chemical_components.md#populate_missing_ccd_data), [`_get_smiles`](../model/atom_layout/atom_layout.md#residues_from_structure._get_smiles), [`_add_ligand_to_chem_comp`](parsing.md#_add_ligand_to_chem_comp)

### `ChemicalComponentsData`
- def: [`src/alphafold3/structure/chemical_components.py:142`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L142) — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
- doc: Extra information for chemical components occurring in mmCIF.
- signature: `class ChemicalComponentsData:`
- members:
  - `format_coords(coords: Sequence[float | None])` — [`L281`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L281) — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
  - `from_mmcif(cls, cif: mmcif.Mmcif, fix_mse: bool, fix_unknown_dna: bool)` — [`L153`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L153) — Constructs an instance of ChemicalComponentsData from an Mmcif object. — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
  - `to_mmcif_dict(self)` — [`L213`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L213) — Returns chemical components data as a dict suitable for `mmcif.Mmcif`. — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
  - `chem_comp` — [`L150`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L150) — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
- uses (calls/refs, reference-scoped): [`parse_atom_data`](chemical_components.md#parse_atom_data), [`Mmcif`](mmcif.md#Mmcif), [`parse_bond_data`](chemical_components.md#parse_bond_data), [`pdbx_smiles`](chemical_components.md#ChemCompEntry.pdbx_smiles), [`ChemCompEntry`](chemical_components.md#ChemCompEntry), [`format_float_array`](mmcif.md#format_float_array), [`chem_comp_atoms`](chemical_components.md#ChemCompEntry.chem_comp_atoms), [`chem_comp_bonds`](chemical_components.md#ChemCompEntry.chem_comp_bonds), [`formula`](chemical_components.md#ChemCompEntry.formula), [`formula_weight`](chemical_components.md#ChemCompEntry.formula_weight), [`mon_nstd_flag`](chemical_components.md#ChemCompEntry.mon_nstd_flag), [`name`](chemical_components.md#ChemCompEntry.name), [`pdbx_synonyms`](chemical_components.md#ChemCompEntry.pdbx_synonyms), [`charge`](chemical_components.md#ChemCompAtom.charge), [`ordinal`](chemical_components.md#ChemCompAtom.ordinal), [`MissingChemicalComponentsDataError`](chemical_components.md#MissingChemicalComponentsDataError), [`aromatic_flag`](chemical_components.md#ChemCompBond.aromatic_flag), [`atom_id_1`](chemical_components.md#ChemCompBond.atom_id_1), [`atom_id_2`](chemical_components.md#ChemCompBond.atom_id_2), [`leaving_atom_flag`](chemical_components.md#ChemCompAtom.leaving_atom_flag), [`model_ideal_x`](chemical_components.md#ChemCompAtom.model_ideal_x), [`model_ideal_y`](chemical_components.md#ChemCompAtom.model_ideal_y), [`model_ideal_z`](chemical_components.md#ChemCompAtom.model_ideal_z), [`stereo_config`](chemical_components.md#ChemCompBond.stereo_config), [`type_symbol`](chemical_components.md#ChemCompAtom.type_symbol), [`value_order`](chemical_components.md#ChemCompBond.value_order), [`_REQUIRED_MMCIF_COLUMNS`](chemical_components.md#_REQUIRED_MMCIF_COLUMNS)
- used by: [`from_sequences_and_bonds`](parsing.md#from_sequences_and_bonds), [`copy_and_update`](structure.md#Structure.copy_and_update), [`copy_and_update_globals`](structure.md#Structure.copy_and_update_globals), [`_get_mmcif_header`](parsing.md#_get_mmcif_header), [`compute_features`](../model/features.md#RefStructure.compute_features), [`concat`](structure.md#concat), [`get_data_for_ccd_components`](chemical_components.md#get_data_for_ccd_components), [`_to_mmcif_header`](structure.md#Structure._to_mmcif_header), [`from_atom_arrays`](parsing.md#from_atom_arrays), [`get_reference`](../model/features.md#get_reference), [`populate_missing_ccd_data`](chemical_components.md#populate_missing_ccd_data), [`__init__`](structure.md#Structure.__init__), [`chemical_components_data`](structure.md#Structure.chemical_components_data), [`rename_res_name`](structure.md#Structure.rename_res_name), [`_get_smiles`](../model/atom_layout/atom_layout.md#residues_from_structure._get_smiles), [`chemical_components_data`](parsing.md#_MmcifHeader.chemical_components_data)

### `MissingChemicalComponentsDataError`  ·  implements/extends Exception
- def: [`src/alphafold3/structure/chemical_components.py:137`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L137) — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
- doc: Raised when chemical components data is missing from an mmCIF.
- signature: `class MissingChemicalComponentsDataError(Exception):`
- used by: [`_get_mmcif_header`](parsing.md#_get_mmcif_header), [`from_mmcif`](chemical_components.md#ChemicalComponentsData.from_mmcif)

## Functions
- `_to_optional_float(values: Sequence[str | None])` — [`L42`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L42)
- `_to_optional_int(values: Sequence[str | None])` — [`L38`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L38)
- `_value_is_missing(value: Collection[Any] | str | None)` — [`L34`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L34) — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
- `get_all_atoms_in_entry(ccd: chemical_components.Ccd, res_name: str)` — [`L501`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L501) — Get all possible atoms and bonds for this residue in a standard order.
- `get_data_for_ccd_components(ccd: chemical_components.Ccd, chemical_component_ids: Iterable[str], populate_pdbx_smiles: bool = False)` — [`L427`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L427) — Returns `ChemicalComponentsData` for chemical components known by PDB. — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
- `get_res_atom_names(ccd: chemical_components.Ccd, res_name: str)` — [`L549`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L549) — Gets the names of the atoms in a given CCD residue.
- `parse_atom_data(cif: mmcif.Mmcif)` — [`L352`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L352) — Parses _chem_comp_atom data from an Mmcif object. — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
- `parse_bond_data(cif: mmcif.Mmcif)` — [`L396`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L396) — Parses _chem_comp_bond data from an Mmcif object. — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
- `populate_missing_ccd_data(ccd: chemical_components.Ccd, chemical_components_data: ChemicalComponentsData, chemical_component_ids: Iterable[str] | None = None, populate_pdbx_smiles: bool = False)` — [`L455`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L455) — Populates missing data for the chemical components from CCD.

## Module values
- `_REQUIRED_MMCIF_COLUMNS` — [`L134`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/chemical_components.py#L134)

