---
title: 'Module: src/alphafold3/structure/mmcif.py'
type: catalog
provenance: extracted
module: src/alphafold3/structure/mmcif.py
status: fresh
symbol_base: scip-python python alphafold3 0.0.0 `src.alphafold3.structure.mmcif`/
symbols:
  Mmcif: Mmcif.
  get_bond_atom_indices: get_bond_atom_indices().
  format_float_array: format_float_array().
  get_internal_to_author_chain_id_map: get_internal_to_author_chain_id_map().
  _WappedFn._WappedFn: _WappedFn._WappedFn.
  int_id_to_str_id: int_id_to_str_id().
  str_id_to_int_id: str_id_to_int_id().
  from_string: from_string().
  get_or_infer_type_symbol: get_or_infer_type_symbol().
  get_chain_type_by_entity_id: get_chain_type_by_entity_id().
  get_release_date: get_release_date().
  get_resolution: get_resolution().
  parse_oper_expr: parse_oper_expr().
  parse_multi_data_cif: parse_multi_data_cif().
  get_experimental_method: get_experimental_method().
  BondParsingError: BondParsingError#
  _P: _P.
  _T: _T.
  parse_oper_expr.range_expander: parse_oper_expr().range_expander().
---
# Module: [`src/alphafold3/structure/mmcif.py`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/mmcif.py)

## Classes
### `BondParsingError`  ·  implements/extends Exception
- def: [`src/alphafold3/structure/mmcif.py:127`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/mmcif.py#L127) — documented in [alphafold3-structure-mmcif](../../../../concepts/alphafold3-structure-mmcif.md)
- doc: Exception raised by errors when getting bond atom indices.
- signature: `class BondParsingError(Exception):`
- used by: [`get_bond_atom_indices`](mmcif.md#get_bond_atom_indices)

## Functions
- `format_float_array(values: np.ndarray, num_decimal_places: int)` — [`L308`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/mmcif.py#L308) — Converts 1D array to a list of strings with the given number of decimals.
- `from_string(mmcif_string: str | bytes)` — [`L95`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/mmcif.py#L95) — documented in [alphafold3-structure-mmcif](../../../../concepts/alphafold3-structure-mmcif.md)
- `get_bond_atom_indices(mmcif: Mmcif, model_id: str = '1')` — [`L131`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/mmcif.py#L131) — Extracts the indices of the atoms that participate in bonds. — documented in [alphafold3-structure-mmcif](../../../../concepts/alphafold3-structure-mmcif.md)
- `get_chain_type_by_entity_id(mmcif: Mmcif)` — [`L184`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/mmcif.py#L184) — Returns mapping from entity ID to its type or polymer type if available. — documented in [alphafold3-structure-mmcif](../../../../concepts/alphafold3-structure-mmcif.md)
- `get_experimental_method(mmcif: Mmcif)` — [`L221`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/mmcif.py#L221)
- `get_internal_to_author_chain_id_map(mmcif: Mmcif)` — [`L207`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/mmcif.py#L207) — Returns a mapping from internal chain ID to the author chain ID. — documented in [alphafold3-structure-mmcif](../../../../concepts/alphafold3-structure-mmcif.md)
- `get_or_infer_type_symbol(mmcif: Mmcif, ccd: chemical_components.Ccd | None = None)` — [`L163`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/mmcif.py#L163) — Returns the type symbol (element) for all of the atoms. — documented in [alphafold3-structure-mmcif](../../../../concepts/alphafold3-structure-mmcif.md)
- `get_release_date(mmcif: Mmcif)` — [`L226`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/mmcif.py#L226) — Returns the oldest revision date.
- `get_resolution(mmcif: Mmcif)` — [`L235`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/mmcif.py#L235) — Returns the resolution of the structure.
- `int_id_to_str_id(num: int)` — [`L49`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/mmcif.py#L49) — Encodes a number as a string, using reverse spreadsheet style naming.
- `parse_multi_data_cif(cif_string: str)` — [`L99`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/mmcif.py#L99) — Parses a CIF string with multiple data records. — documented in [alphafold3-structure-mmcif](../../../../concepts/alphafold3-structure-mmcif.md)
- `parse_oper_expr(oper_expression: str)` — [`L261`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/mmcif.py#L261) — Determines which transforms to apply based on an MMCIF oper_expression str.
- `range_expander(match)` — [`L287`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/mmcif.py#L287)
- `str_id_to_int_id(str_id: str)` — [`L72`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/mmcif.py#L72) — Encodes an mmCIF-style string chain ID as an integer.

## Module values
- `Mmcif` — [`L40`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/mmcif.py#L40) — documented in [alphafold3-structure-chemical_components](../../../../concepts/alphafold3-structure-chemical_components.md)
- `_P` — [`L43`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/mmcif.py#L43)
- `_T` — [`L44`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/mmcif.py#L44)
- `_WappedFn` — [`L45`](../../../../../../../raw/code/alphafold3/src/alphafold3/structure/mmcif.py#L45)

