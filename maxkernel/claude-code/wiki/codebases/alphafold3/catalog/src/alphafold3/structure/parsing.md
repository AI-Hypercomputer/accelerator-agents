---
title: 'Module: src/alphafold3/structure/parsing.py'
type: catalog
provenance: extracted
module: src/alphafold3/structure/parsing.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.structure.parsing`/
symbols:
  from_sequences_and_bonds: from_sequences_and_bonds().
  get_tables: get_tables().
  from_res_arrays: from_res_arrays().
  _ChainResBuilder.add_residues: _ChainResBuilder#add_residues().
  from_parsed_mmcif: from_parsed_mmcif().
  _get_mmcif_header: _get_mmcif_header().
  _ChainResBuilder.make_residues_table: _ChainResBuilder#make_residues_table().
  _ChainResBuilder.make_chains_table: _ChainResBuilder#make_chains_table().
  _create_bond_lookup: _create_bond_lookup().
  _parse_bonds: _parse_bonds().
  from_atom_arrays: from_atom_arrays().
  _get_representative_atom: _get_representative_atom().
  _get_str_model_id: _get_str_model_id().
  from_mmcif: from_mmcif().
  _generate_required_tables_if_missing: _generate_required_tables_if_missing().
  BondAtomId.BondAtomId: BondAtomId.BondAtomId.
  SequenceFormat: SequenceFormat#
  expand_sequence: expand_sequence().
  ModelID: ModelID#
  get_tables._first_model_string_array: get_tables()._first_model_string_array().
  _get_string_array_default: _get_string_array_default().
  _add_ligand_to_chem_comp: _add_ligand_to_chem_comp().
  _maybe_add_missing_scheme_tables: _maybe_add_missing_scheme_tables().
  get_tables._get_poly_seq_scheme_col: get_tables()._get_poly_seq_scheme_col().
  _ChainResBuilder.chain_id: _ChainResBuilder#chain_id.
  get_tables._requested_models_float_array: get_tables()._requested_models_float_array().
  _get_first_model_id: _get_first_model_id().
  _MmcifHeader.bioassembly_data: _MmcifHeader#bioassembly_data.
  _MmcifHeader.chemical_components_data: _MmcifHeader#chemical_components_data.
  SequenceFormat.LIGAND_SMILES: SequenceFormat#LIGAND_SMILES.
  _ChainResBuilder.chain_key: _ChainResBuilder#chain_key.
  _ChainResBuilder.chain_type: _ChainResBuilder#chain_type.
  _ChainResBuilder.res_id: _ChainResBuilder#res_id.
  get_tables._get_nonpoly_scheme_col: get_tables()._get_nonpoly_scheme_col().
  get_tables._get_branch_scheme_col: get_tables()._get_branch_scheme_col().
  _INSERTION_CODE_REMAP._INSERTION_CODE_REMAP: _INSERTION_CODE_REMAP._INSERTION_CODE_REMAP.
  _get_first_non_leaving_atom: _get_first_non_leaving_atom().
  BondIndices: BondIndices#
  BondIndices.from_indices: BondIndices#from_indices.
  BondIndices.dest_indices: BondIndices#dest_indices.
  ModelID.FIRST: ModelID#FIRST.
  _ChainResBuilder.chain_auth_asym_id: _ChainResBuilder#chain_auth_asym_id.
  _ChainResBuilder.chain_entity_id: _ChainResBuilder#chain_entity_id.
  _ChainResBuilder.chain_entity_desc: _ChainResBuilder#chain_entity_desc.
  _ChainResBuilder.res_key: _ChainResBuilder#res_key.
  _ChainResBuilder.res_auth_seq_id: _ChainResBuilder#res_auth_seq_id.
  _ChainResBuilder.res_insertion_code: _ChainResBuilder#res_insertion_code.
  _ChainResBuilder.chain_key_by_chain_id: _ChainResBuilder#chain_key_by_chain_id.
  NoAtomsError: NoAtomsError#
  SequenceFormat.FASTA: SequenceFormat#FASTA.
  SequenceFormat.CCD_CODES: SequenceFormat#CCD_CODES.
  _get_atom_element: _get_atom_element().
  _MmcifHeader: _MmcifHeader#
  _MmcifHeader.name: _MmcifHeader#name.
  _MmcifHeader.resolution: _MmcifHeader#resolution.
  _MmcifHeader.release_date: _MmcifHeader#release_date.
  _MmcifHeader.structure_method: _MmcifHeader#structure_method.
  _ChainResBuilder.res_chain_key: _ChainResBuilder#res_chain_key.
  _ChainResBuilder.res_name: _ChainResBuilder#res_name.
  _ChainResBuilder.key_for_res: _ChainResBuilder#key_for_res.
  get_tables._ligand_residue_ids: get_tables()._ligand_residue_ids().
  ChainIndex.ChainIndex: ChainIndex.ChainIndex.
  ResIndex.ResIndex: ResIndex.ResIndex.
  AtomName.AtomName: AtomName.AtomName.
  ModelID.ALL: ModelID#ALL.
  _ChainResBuilder: _ChainResBuilder#
  _ChainResBuilder.entity_id_by_chain_id: _ChainResBuilder#entity_id_by_chain_id.
  _ChainResBuilder.chain_type_by_entity_id: _ChainResBuilder#chain_type_by_entity_id.
  _ChainResBuilder.entity_desc_by_entity_id: _ChainResBuilder#entity_desc_by_entity_id.
  _ChainResBuilder._fix_mse_residues: _ChainResBuilder#_fix_mse_residues.
  _ChainResBuilder._fix_unknown_dna: _ChainResBuilder#_fix_unknown_dna.
  _get_chain_key_by_chain_id: _get_chain_key_by_chain_id().
  _guess_entity_type: _guess_entity_type().
  _ChainResBuilder.__init__: _ChainResBuilder#__init__().
---
# Module: [`src/alphafold3/structure/parsing.py`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py)

## Classes
### `BondIndices`
- def: [`src/alphafold3/structure/parsing.py:57`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L57)
- signature: `class BondIndices:`
- members:
  - `dest_indices` — [`L59`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L59)
  - `from_indices` — [`L58`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L58)
- used by: [`from_sequences_and_bonds`](parsing.md#from_sequences_and_bonds), [`_create_bond_lookup`](parsing.md#_create_bond_lookup)

### `ModelID`  ·  implements/extends Enum
- def: [`src/alphafold3/structure/parsing.py:63`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L63)
- doc: Values for specifying model IDs when parsing.
- signature: `class ModelID(enum.Enum):`
- members:
  - `ALL` — [`L67`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L67)
  - `FIRST` — [`L66`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L66)
- used by: [`from_parsed_mmcif`](parsing.md#from_parsed_mmcif), [`_get_str_model_id`](parsing.md#_get_str_model_id), [`from_mmcif`](parsing.md#from_mmcif)

### `NoAtomsError`  ·  implements/extends Exception
- def: [`src/alphafold3/structure/parsing.py:52`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L52)
- doc: Raise when the mmCIF does not have any atoms.
- signature: `class NoAtomsError(Exception):`
- used by: [`_get_str_model_id`](parsing.md#_get_str_model_id)

### `SequenceFormat`  ·  implements/extends Enum
- def: [`src/alphafold3/structure/parsing.py:71`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L71) — documented in [alphafold3-structure-parsing](../../../../concepts/alphafold3-structure-parsing.md)
- doc: The possible formats for an input sequence.
- signature: `class SequenceFormat(enum.Enum):`
- members:
  - `CCD_CODES` — [`L75`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L75)
  - `FASTA` — [`L74`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L74)
  - `LIGAND_SMILES` — [`L76`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L76)
- used by: [`from_sequences_and_bonds`](parsing.md#from_sequences_and_bonds), [`_get_representative_atom`](parsing.md#_get_representative_atom), [`expand_sequence`](parsing.md#expand_sequence)

### `_ChainResBuilder`
- def: [`src/alphafold3/structure/parsing.py:851`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L851)
- doc: Class for incrementally building chain and residue tables.
- signature: `class _ChainResBuilder:`
- members:
  - `add_residues(self, *, chain_ids: np.ndarray, chain_auth_asym_ids: np.ndarray, res_ids: np.ndarray, res_names: np.ndarray, res_auth_seq_ids: np.ndarray, res_ins_codes: np.ndarray)` — [`L889`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L889) — Adds a residue (and its chain) to the tables. — documented in [alphafold3-structure-parsing](../../../../concepts/alphafold3-structure-parsing.md)
  - `make_chains_table(self)` — [`L952`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L952) — Returns the Structure chains table. — documented in [alphafold3-structure-parsing](../../../../concepts/alphafold3-structure-parsing.md)
  - `make_residues_table(self)` — [`L975`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L975) — Returns the Structure residues table. — documented in [alphafold3-structure-parsing](../../../../concepts/alphafold3-structure-parsing.md)
  - `chain_auth_asym_id` — [`L868`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L868)
  - `chain_entity_desc` — [`L870`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L870)
  - `chain_entity_id` — [`L869`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L869)
  - `chain_id` — [`L866`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L866) — documented in [alphafold3-structure-parsing](../../../../concepts/alphafold3-structure-parsing.md)
  - `chain_key` — [`L865`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L865)
  - `chain_key_by_chain_id` — [`L880`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L880)
  - `chain_type` — [`L867`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L867)
  - `chain_type_by_entity_id` — [`L882`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L882)
  - `entity_desc_by_entity_id` — [`L883`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L883)
  - `entity_id_by_chain_id` — [`L881`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L881)
  - `key_for_res` — [`L884`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L884)
  - `res_auth_seq_id` — [`L877`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L877)
  - `res_chain_key` — [`L874`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L874)
  - `res_id` — [`L875`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L875)
  - `res_insertion_code` — [`L878`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L878)
  - `res_key` — [`L873`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L873)
  - `res_name` — [`L876`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L876)
- protocol/private: `__init__`[`L854`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L854), `_fix_mse_residues`[`L886`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L886), `_fix_unknown_dna`[`L887`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L887)
- uses (calls/refs, reference-scoped): [`chain_key`](structure_tables.md#Residues.chain_key), [`id`](structure_tables.md#Chains.id), [`Residues`](structure_tables.md#Residues), [`Chains`](structure_tables.md#Chains), [`name`](structure_tables.md#Residues.name), [`id`](structure_tables.md#Residues.id), [`auth_seq_id`](structure_tables.md#Residues.auth_seq_id), [`insertion_code`](structure_tables.md#Residues.insertion_code), [`auth_asym_id`](structure_tables.md#Chains.auth_asym_id), [`entity_id`](structure_tables.md#Chains.entity_id), [`entity_desc`](structure_tables.md#Chains.entity_desc)
- used by: [`get_tables`](parsing.md#get_tables)

### `_MmcifHeader`
- def: [`src/alphafold3/structure/parsing.py:266`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L266)
- signature: `class _MmcifHeader:`
- members:
  - `bioassembly_data` — [`L271`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L271)
  - `chemical_components_data` — [`L272`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L272)
  - `name` — [`L267`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L267)
  - `release_date` — [`L269`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L269)
  - `resolution` — [`L268`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L268)
  - `structure_method` — [`L270`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L270)
- uses (calls/refs, reference-scoped): [`ChemicalComponentsData`](chemical_components.md#ChemicalComponentsData), [`BioassemblyData`](bioassemblies.md#BioassemblyData)
- used by: [`from_parsed_mmcif`](parsing.md#from_parsed_mmcif), [`_get_mmcif_header`](parsing.md#_get_mmcif_header)

## Functions
- `_add_ligand_to_chem_comp(chem_comp: MutableMapping[str, struc_chem_comps.ChemCompEntry], ligand_id: str, ligand_smiles: str)` — [`L151`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L151) — Adds a ligand to chemical components. Raises ValueError on mismatch.
- `_create_bond_lookup(bonded_atom_pairs: Sequence[tuple[BondAtomId, BondAtomId]])` — [`L79`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L79) — Creates maps to help find bonds during a loop over residues.
- `_first_model_string_array(col: str)` — [`L1354`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L1354)
- `_generate_required_tables_if_missing(cif: mmcif.Mmcif)` — [`L1019`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L1019) — Generates all required tables and columns if missing.
- `_get_atom_element(ccd: chemical_components.Ccd, res_name: str, atom_name: str)` — [`L102`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L102)
- `_get_branch_scheme_col(col: str)` — [`L1498`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L1498)
- `_get_chain_key_by_chain_id(resolved_chain_ids: np.ndarray, struct_asym_chain_ids: np.ndarray)` — [`L1281`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L1281) — Returns chain key for each chain ID respecting resolved chain ordering.
- `_get_first_model_id(cif: mmcif.Mmcif)` — [`L171`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L171) — Returns cheaply the first model ID from the mmCIF.
- `_get_first_non_leaving_atom(ccd: chemical_components.Ccd, res_name: str)` — [`L138`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L138) — Returns first definitely non-leaving atom if exists, as a stand-in.
- `_get_mmcif_header(cif: mmcif.Mmcif, fix_mse: bool, fix_unknown_dna: bool)` — [`L275`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L275) — Extract header fields from an mmCIF object.
- `_get_nonpoly_scheme_col(col: str)` — [`L1483`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L1483)
- `_get_poly_seq_scheme_col(col: str)` — [`L1439`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L1439)
- `_get_representative_atom(ccd: chemical_components.Ccd, res_name: str, chain_type: str, sequence_format: SequenceFormat)` — [`L111`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L111)
- `_get_str_model_id(cif: mmcif.Mmcif, model_id: ModelID | int)` — [`L178`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L178) — Converts a user-specified model_id argument into a string.
- `_get_string_array_default(cif: mmcif.Mmcif, key: str, default: list[str])` — [`L1012`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L1012)
- `_guess_entity_type(chain_residues: Collection[str], atom_types: Collection[str])` — [`L1781`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L1781) — Guess the entity type (polymer/non-polymer/water) based on residues/atoms.
- `_ligand_residue_ids(chain_ids: np.ndarray)` — [`L1551`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L1551) — Computes internal residue ID for ligand residues that don't have it.
- `_maybe_add_missing_scheme_tables(cif: mmcif.Mmcif, res_starts: Sequence[int], label_asym_ids: np.ndarray, label_seq_ids: np.ndarray, label_comp_ids: np.ndarray, auth_seq_ids: np.ndarray, pdb_ins_codes: np.ndarray)` — [`L1114`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L1114) — If missing, infers the scheme tables from the _atom_site table.
- `_parse_bonds(cif: mmcif.Mmcif, atom_key: np.ndarray, model_id: str)` — [`L211`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L211) — Returns the bonds table extracted from the mmCIF.
- `_requested_models_float_array(col: str)` — [`L1357`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L1357)
- `expand_sequence(sequence: str, chain_type: str, sequence_format: SequenceFormat)` — [`L600`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L600) — Returns full residue names based on a sequence string. — documented in [alphafold3-structure-parsing](../../../../concepts/alphafold3-structure-parsing.md)
- `from_atom_arrays(*, res_id: np.ndarray, name: str = 'unset', release_date: datetime.date | None = None, resolution: float | None = None, structure_method: str | None = None, all_residues: Mapping[str, Sequence[tuple[str, int]]] | None = None, bioassembly_data: bioassemblies.BioassemblyData | None = None, chemical_components_data: (struc_chem_comps.ChemicalComponentsData | None) = None, bond_table: structure_tables.Bonds | None = None, chain_id: np.ndarray | None = None, chain_type: np.ndarray | None = None, res_name: np.ndarray | None = None, atom_key: np.ndarray | None = None, atom_name: np.ndarray | None = None, atom_element: np.ndarray | None = None, atom_x: np.ndarray | None = None, atom_y: np.ndarray | None = None, atom_z: np.ndarray | None = None, atom_b_factor: np.ndarray | None = None, atom_occupancy: np.ndarray | None = None)` — [`L1652`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L1652) — Returns a Structure constructed from atom array level data.
- `from_mmcif(mmcif_string: str | bytes, *, name: str | None = None, fix_mse_residues: bool = False, fix_arginines: bool = False, fix_unknown_dna: bool = False, include_water: bool = False, include_other: bool = False, include_bonds: bool = False, model_id: int | ModelID = ModelID.FIRST)` — [`L408`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L408) — Construct a Structure from a mmCIF string.
- `from_parsed_mmcif(mmcif_object: mmcif.Mmcif, *, name: str | None = None, fix_mse_residues: bool = False, fix_arginines: bool = False, fix_unknown_dna: bool = False, include_water: bool = False, include_other: bool = False, include_bonds: bool = False, model_id: int | ModelID = ModelID.FIRST)` — [`L316`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L316) — Construct a Structure from a parsed mmCIF object. — documented in [alphafold3-structure-parsing](../../../../concepts/alphafold3-structure-parsing.md)
- `from_res_arrays(atom_mask: np.ndarray, **kwargs)` — [`L465`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L465) — Returns Structure created from from arrays with a residue dimension. — documented in [alphafold3-structure-parsing](../../../../concepts/alphafold3-structure-parsing.md)
- `from_sequences_and_bonds(*, sequences: Sequence[str], chain_types: Sequence[str], sequence_formats: Sequence[SequenceFormat], bonded_atom_pairs: Sequence[tuple[BondAtomId, BondAtomId]] | None, ccd: chemical_components.Ccd, chain_ids: Sequence[str] | None = None, name: str = 'from_sequences_and_bonds', bond_type: str | None = None, **constructor_args)` — [`L637`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L637) — Returns a minimal structure for the input sequences and bonds. — documented in [alphafold3-structure-bonds](../../../../concepts/alphafold3-structure-bonds.md)
- `get_tables(cif: mmcif.Mmcif, fix_mse_residues: bool, fix_arginines: bool, fix_unknown_dna: bool, include_water: bool, include_other: bool, model_id: str)` — [`L1310`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L1310) — Returns chain, residue, and atom tables from a parsed mmcif. — documented in [alphafold3-structure-parsing](../../../../concepts/alphafold3-structure-parsing.md)

## Module values
- `AtomName` — [`L46`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L46)
- `BondAtomId` — [`L47`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L47)
- `ChainIndex` — [`L44`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L44)
- `ResIndex` — [`L45`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L45)
- `_INSERTION_CODE_REMAP` — [`L49`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/parsing.py#L49)

