---
title: 'Module: jaxite/jaxite_bool/lut.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_bool/lut.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_bool.lut`/
symbols:
  LookUpTable.as_cleartext_list: LookUpTable#as_cleartext_list().
  LookUpTable.as_rlwe_test_polynomial: LookUpTable#as_rlwe_test_polynomial().
  from_callable: from_callable().
  LutCache.lut_by_name: LutCache#lut_by_name().
  LutCache.lut_poly: LutCache#lut_poly().
  LutCache.lut_poly_by_name: LutCache#lut_poly_by_name().
  LookUpTable: LookUpTable#
  LookUpTable.truth_table: LookUpTable#truth_table.
  LutCache.lut: LutCache#lut().
  LookUpTable._values_from_lsb_to_msb: LookUpTable#_values_from_lsb_to_msb().
  FUNC_NAME_TO_LUT2: FUNC_NAME_TO_LUT2.
  LookUpTable.__str__: LookUpTable#__str__().
  LookUpTable.__repr__: LookUpTable#__repr__().
  LookUpTable.__hash__: LookUpTable#__hash__().
  LutCache.__init__: LutCache#__init__().
  LookUpTable.num_inputs: LookUpTable#num_inputs.
  LutCache: LutCache#
  LutCache.scheme_params: LutCache#scheme_params.
  LutCache.test_poly_cache: LutCache#test_poly_cache.
  LookUpTable.__init__: LookUpTable#__init__().
---
# Module: [`jaxite/jaxite_bool/lut.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut.py)

## Classes
### `LookUpTable`
- def: [`jaxite/jaxite_bool/lut.py:14`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut.py#L14)
- doc: A representation of a truth table.
- signature: `class LookUpTable:`
- members:
  - `as_cleartext_list(self)` — [`L36`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut.py#L36) — Return the truth table as a list of LWE cleartexts.
  - `as_rlwe_test_polynomial(self, scheme_params: parameters.SchemeParameters)` — [`L44`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut.py#L44) — Return the truth table as an encrypted RLWE test polynomial.
  - `num_inputs` — [`L25`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut.py#L25)
  - `truth_table` — [`L26`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut.py#L26)
- protocol/private: `__hash__`[`L85`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut.py#L85), `__init__`[`L24`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut.py#L24), `__repr__`[`L77`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut.py#L77), `__str__`[`L68`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut.py#L68), `_values_from_lsb_to_msb`[`L29`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut.py#L29)
- uses (calls/refs, reference-scoped): [`message_bit_length`](../jaxite_cggi/encoding.md#EncodingParameters.message_bit_length), [`SchemeParameters`](../jaxite_cggi/parameters.md#SchemeParameters), [`RlweCiphertext`](../jaxite_cggi/rlwe.md#RlweCiphertext), [`LweCleartext`](../jaxite_cggi/types.md#LweCleartext), [`ENCODING_PARAMS`](bool_encoding.md#ENCODING_PARAMS), [`CLEARTEXT_FALSE`](bool_encoding.md#CLEARTEXT_FALSE), [`CLEARTEXT_TRUE`](bool_encoding.md#CLEARTEXT_TRUE), [`CLEARTEXT_UNUSED`](bool_encoding.md#CLEARTEXT_UNUSED)  (1 test-only)
- used by: [`and_`](jaxite_bool.md#and_), [`or_`](jaxite_bool.md#or_), [`xor_`](jaxite_bool.md#xor_), [`lut2`](jaxite_bool.md#lut2), [`lut3`](jaxite_bool.md#lut3), [`andny_`](jaxite_bool.md#andny_), [`andyn_`](jaxite_bool.md#andyn_), [`nand_`](jaxite_bool.md#nand_), [`nor_`](jaxite_bool.md#nor_), [`orny_`](jaxite_bool.md#orny_), [`oryn_`](jaxite_bool.md#oryn_), [`xnor_`](jaxite_bool.md#xnor_), [`lut_by_name`](bool_params.md#Parameters.lut_by_name), [`cmux_`](jaxite_bool.md#cmux_), [`from_callable`](lut.md#from_callable), [`lut_by_name`](lut.md#LutCache.lut_by_name), [`lut_poly`](lut.md#LutCache.lut_poly), [`lut_poly_by_name`](lut.md#LutCache.lut_poly_by_name), [`test_as_cleartext_list`](lut_test.md#LutTest.test_as_cleartext_list), [`test_lut_cache_and`](lut_test.md#LutTest.test_lut_cache_and), [`test_lut_cache_generate_lut`](lut_test.md#LutTest.test_lut_cache_generate_lut), [`lut`](bool_params.md#Parameters.lut), [`lut`](lut.md#LutCache.lut), [`test_from_callable`](lut_test.md#LutTest.test_from_callable), [`__init__`](lut.md#LutCache.__init__), [`test_repr`](lut_test.md#LutTest.test_repr), [`test_str`](lut_test.md#LutTest.test_str)

### `LutCache`
- def: [`jaxite/jaxite_bool/lut.py:119`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut.py#L119)
- doc: A cache for LUT test polynomials for chosen scheme parameters.
- signature: `class LutCache:`
- members:
  - `lut(self, num_inputs: int, lut_as_int: int)` — [`L127`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut.py#L127)
  - `lut_by_name(self, name: str)` — [`L133`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut.py#L133)
  - `lut_poly(self, num_inputs: int, lut_as_int: int)` — [`L142`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut.py#L142)
  - `lut_poly_by_name(self, name: str)` — [`L147`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut.py#L147) — A helper for users that want to query the gate by name.
  - `scheme_params` — [`L123`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut.py#L123)
  - `test_poly_cache` — [`L124`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut.py#L124)
- protocol/private: `__init__`[`L122`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut.py#L122)
- uses (calls/refs, reference-scoped): [`SchemeParameters`](../jaxite_cggi/parameters.md#SchemeParameters), [`RlweCiphertext`](../jaxite_cggi/rlwe.md#RlweCiphertext), [`as_rlwe_test_polynomial`](lut.md#LookUpTable.as_rlwe_test_polynomial), [`LookUpTable`](lut.md#LookUpTable), [`truth_table`](lut.md#LookUpTable.truth_table), [`FUNC_NAME_TO_LUT2`](lut.md#FUNC_NAME_TO_LUT2)
- used by: [`lut_by_name`](bool_params.md#Parameters.lut_by_name), [`lut_poly_by_name`](bool_params.md#Parameters.lut_poly_by_name), [`lut_poly`](bool_params.md#Parameters.lut_poly), [`test_lut_cache_and`](lut_test.md#LutTest.test_lut_cache_and), [`test_lut_cache_generate_lut`](lut_test.md#LutTest.test_lut_cache_generate_lut), [`lut`](bool_params.md#Parameters.lut), [`_lut_cache`](bool_params.md#Parameters._lut_cache), [`test_lut_cache_key_error`](lut_test.md#LutTest.test_lut_cache_key_error)

## Functions
- `from_callable(num_inputs: int, fn: Callable[..., bool])` — [`L89`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut.py#L89) — Construct a LookUpTable from a callable of bools.

## Module values
- `FUNC_NAME_TO_LUT2` — [`L104`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut.py#L104)

