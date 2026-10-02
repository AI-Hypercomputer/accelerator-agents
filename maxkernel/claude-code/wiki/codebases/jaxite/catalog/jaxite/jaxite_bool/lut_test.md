---
title: 'Module: jaxite/jaxite_bool/lut_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_bool/lut_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_bool.lut_test`/
symbols:
  TEST_SCHEME_PARAMS: TEST_SCHEME_PARAMS.
  LutTest.test_as_cleartext_list: LutTest#test_as_cleartext_list().
  LutTest.test_lut_cache_and: LutTest#test_lut_cache_and().
  LutTest.test_lut_cache_generate_lut: LutTest#test_lut_cache_generate_lut().
  LutTest.test_from_callable: LutTest#test_from_callable().
  LutTest.test_lut_cache_key_error: LutTest#test_lut_cache_key_error().
  LutTest.test_str: LutTest#test_str().
  LutTest.test_repr: LutTest#test_repr().
  LutTest.lut_fn: LutTest#lut_fn().
  LutTest: LutTest#
---
# Module: [`jaxite/jaxite_bool/lut_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut_test.py)

## Classes
### `LutTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_bool/lut_test.py:14`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut_test.py#L14)
- signature: `class LutTest(absltest.TestCase):`
- members:
  - `lut_fn(x: bool, y: bool, z: bool)` — [`L34`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut_test.py#L34)
  - `test_as_cleartext_list(self)` — [`L44`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut_test.py#L44)
  - `test_from_callable(self)` — [`L33`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut_test.py#L33)
  - `test_lut_cache_and(self)` — [`L55`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut_test.py#L55)
  - `test_lut_cache_generate_lut(self)` — [`L59`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut_test.py#L59)
  - `test_lut_cache_key_error(self)` — [`L63`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut_test.py#L63)
  - `test_repr(self)` — [`L28`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut_test.py#L28)
  - `test_str(self)` — [`L16`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut_test.py#L16)
- uses (calls/refs, reference-scoped): [`as_cleartext_list`](lut.md#LookUpTable.as_cleartext_list), [`TEST_SCHEME_PARAMS`](lut_test.md#TEST_SCHEME_PARAMS), [`from_callable`](lut.md#from_callable), [`lut_by_name`](lut.md#LutCache.lut_by_name), [`LookUpTable`](lut.md#LookUpTable), [`lut`](lut.md#LutCache.lut), [`truth_table`](lut.md#LookUpTable.truth_table), [`CLEARTEXT_FALSE`](bool_encoding.md#CLEARTEXT_FALSE), [`CLEARTEXT_TRUE`](bool_encoding.md#CLEARTEXT_TRUE), [`LutCache`](lut.md#LutCache)

## Module values
- `TEST_SCHEME_PARAMS` — [`L6`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/lut_test.py#L6)

