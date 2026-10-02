---
title: 'Module: src/alphafold3/structure/bioassemblies.py'
type: catalog
provenance: extracted
module: src/alphafold3/structure/bioassemblies.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.structure.bioassemblies`/
symbols:
  BioassemblyData.get_transforms: BioassemblyData#get_transforms().
  BioassemblyData.rename_label_asym_ids: BioassemblyData#rename_label_asym_ids().
  BioassemblyData.to_mmcif_dict: BioassemblyData#to_mmcif_dict().
  BioassemblyData: BioassemblyData#
  _get_operation: _get_operation().
  BioassemblyData.from_mmcif: BioassemblyData#from_mmcif().
  Operation.apply_to_coords: Operation#apply_to_coords().
  Transform.apply_to_coords: Transform#apply_to_coords().
  BioassemblyData._operations: BioassemblyData#_operations.
  BioassemblyData.asym_id_by_assembly_chain_id: BioassemblyData#asym_id_by_assembly_chain_id().
  BioassemblyData.assembly_chain_ids_by_asym_id: BioassemblyData#assembly_chain_ids_by_asym_id().
  Transform.operations: Transform#operations.
  Transform.chain_id_rename_map: Transform#chain_id_rename_map.
  BioassemblyData._assembly_ids: BioassemblyData#_assembly_ids.
  BioassemblyData.get_default_assembly_id: BioassemblyData#get_default_assembly_id().
  Operation: Operation#
  Transform: Transform#
  BioassemblyData._pdbx_struct_assembly: BioassemblyData#_pdbx_struct_assembly.
  BioassemblyData._pdbx_struct_assembly_gen: BioassemblyData#_pdbx_struct_assembly_gen.
  BioassemblyData._pdbx_struct_oper_list: BioassemblyData#_pdbx_struct_oper_list.
  BioassemblyData.assembly_ids: BioassemblyData#assembly_ids().
  BioassemblyData.get_assembly_info: BioassemblyData#get_assembly_info().
  Operation.trans: Operation#trans.
  Operation.rot: Operation#rot.
  Transform.chain_ids: Transform#chain_ids.
  MissingBioassemblyDataError: MissingBioassemblyDataError#
  BioassemblyData._oper_ids: BioassemblyData#_oper_ids.
  BioassemblyData._REQUIRED_COLUMNS: BioassemblyData#_REQUIRED_COLUMNS.
  BioassemblyData.__init__: BioassemblyData#__init__().
---
# Module: [`src/alphafold3/structure/bioassemblies.py`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py)

## Classes
### `BioassemblyData`
- def: [`src/alphafold3/structure/bioassemblies.py:84`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L84) — documented in [alphafold3-structure-bioassemblies](../../../../concepts/alphafold3-structure-bioassemblies.md)
- doc: Stores and processes bioassembly data from mmCIF tables.
- signature: `class BioassemblyData:`
- members:
  - `assembly_chain_ids_by_asym_id(self, assembly_id: str)` — [`L207`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L207)
  - `assembly_ids(self)` — [`L197`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L197)
  - `asym_id_by_assembly_chain_id(self, assembly_id: str)` — [`L200`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L200)
  - `from_mmcif(cls, cif: mmcif.Mmcif)` — [`L154`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L154) — Constructs an instance of `BioassemblyData` from an `Mmcif` object.
  - `get_assembly_info(self, assembly_id: str)` — [`L225`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L225)
  - `get_default_assembly_id(self)` — [`L218`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L218) — Gets a default assembly ID.
  - `get_transforms(self, assembly_id: str)` — [`L231`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L231) — Returns the transforms required to generate the given assembly. — documented in [alphafold3-structure-bioassemblies](../../../../concepts/alphafold3-structure-bioassemblies.md)
  - `rename_label_asym_ids(self, mapping: Mapping[str, str], present_chains: set[str])` — [`L292`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L292) — Returns a new BioassemblyData with renamed label_asym_ids. — documented in [alphafold3-structure-bioassemblies](../../../../concepts/alphafold3-structure-bioassemblies.md)
  - `to_mmcif_dict(self)` — [`L278`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L278) — Returns the bioassembly data as a dict suitable for `mmcif.Mmcif`. — documented in [alphafold3-structure-bioassemblies](../../../../concepts/alphafold3-structure-bioassemblies.md)
- protocol/private: `_REQUIRED_COLUMNS`[`L90`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L90), `__init__`[`L117`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L117), `_assembly_ids`[`L150`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L150), `_oper_ids`[`L151`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L151), `_operations`[`L146`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L146), `_pdbx_struct_assembly`[`L143`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L143), `_pdbx_struct_assembly_gen`[`L144`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L144), `_pdbx_struct_oper_list`[`L145`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L145)
- uses (calls/refs, reference-scoped): [`Mmcif`](mmcif.md#Mmcif), [`_get_operation`](bioassemblies.md#_get_operation), [`operations`](bioassemblies.md#Transform.operations), [`chain_id_rename_map`](bioassemblies.md#Transform.chain_id_rename_map), [`int_id_to_str_id`](mmcif.md#int_id_to_str_id), [`str_id_to_int_id`](mmcif.md#str_id_to_int_id), [`parse_oper_expr`](mmcif.md#parse_oper_expr), [`Transform`](bioassemblies.md#Transform), [`MissingBioassemblyDataError`](bioassemblies.md#MissingBioassemblyDataError), [`chain_ids`](bioassemblies.md#Transform.chain_ids)
- used by: [`copy_and_update`](structure.md#Structure.copy_and_update), [`copy_and_update_globals`](structure.md#Structure.copy_and_update_globals), [`_get_mmcif_header`](parsing.md#_get_mmcif_header), [`generate_bioassembly`](structure.md#Structure.generate_bioassembly), [`_to_mmcif_header`](structure.md#Structure._to_mmcif_header), [`rename_chain_ids`](structure.md#Structure.rename_chain_ids), [`from_atom_arrays`](parsing.md#from_atom_arrays), [`__init__`](structure.md#Structure.__init__), [`bioassembly_data`](structure.md#Structure.bioassembly_data), [`bioassembly_data`](parsing.md#_MmcifHeader.bioassembly_data)

### `MissingBioassemblyDataError`  ·  implements/extends Exception
- def: [`src/alphafold3/structure/bioassemblies.py:80`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L80)
- doc: Raised when bioassembly data is missing from an mmCIF.
- signature: `class MissingBioassemblyDataError(Exception):`
- used by: [`_get_mmcif_header`](parsing.md#_get_mmcif_header), [`from_mmcif`](bioassemblies.md#BioassemblyData.from_mmcif)

### `Operation`
- def: [`src/alphafold3/structure/bioassemblies.py:32`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L32) — documented in [alphafold3-structure-bioassemblies](../../../../concepts/alphafold3-structure-bioassemblies.md)
- doc: A rigid transformation operation.
- signature: `class Operation:`
- members:
  - `apply_to_coords(self, coords: np.ndarray)` — [`L38`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L38) — Applies the rotation followed by the translation to `coords`.
  - `rot` — [`L36`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L36) — documented in [alphafold3-structure-bioassemblies](../../../../concepts/alphafold3-structure-bioassemblies.md)
  - `trans` — [`L35`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L35) — documented in [alphafold3-structure-bioassemblies](../../../../concepts/alphafold3-structure-bioassemblies.md)
- used by: [`_get_operation`](bioassemblies.md#_get_operation), [`apply_to_coords`](bioassemblies.md#Transform.apply_to_coords), [`operations`](bioassemblies.md#Transform.operations)

### `Transform`
- def: [`src/alphafold3/structure/bioassemblies.py:44`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L44) — documented in [alphafold3-structure-bioassemblies](../../../../concepts/alphafold3-structure-bioassemblies.md)
- doc: A rigid transformation composed of a sequence of `Operation`s.
- signature: `class Transform:`
- members:
  - `apply_to_coords(self, coords: np.ndarray)` — [`L59`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L59) — Applies the `operations` in right-to-left order.
  - `chain_id_rename_map` — [`L57`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L57) — documented in [alphafold3-structure-bioassemblies](../../../../concepts/alphafold3-structure-bioassemblies.md)
  - `chain_ids` — [`L53`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L53) — documented in [alphafold3-structure-bioassemblies](../../../../concepts/alphafold3-structure-bioassemblies.md)
  - `operations` — [`L49`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L49) — documented in [alphafold3-structure-bioassemblies](../../../../concepts/alphafold3-structure-bioassemblies.md)
- uses (calls/refs, reference-scoped): [`apply_to_coords`](bioassemblies.md#Operation.apply_to_coords), [`Operation`](bioassemblies.md#Operation)
- used by: [`_apply_bioassembly_transform`](structure.md#Structure._apply_bioassembly_transform), [`get_transforms`](bioassemblies.md#BioassemblyData.get_transforms), [`assembly_chain_ids_by_asym_id`](bioassemblies.md#BioassemblyData.assembly_chain_ids_by_asym_id), [`asym_id_by_assembly_chain_id`](bioassemblies.md#BioassemblyData.asym_id_by_assembly_chain_id)

## Functions
- `_get_operation(oper_data: Mapping[str, str])` — [`L66`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/bioassemblies.py#L66) — Parses an `Operation` from a mmCIF _pdbx_struct_oper_list row.

