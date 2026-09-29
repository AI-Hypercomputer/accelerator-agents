---
title: 'Module: jaxite/jaxite_ckks/add_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/add_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.add_test`/
symbols:
  AddTest.test_full_pipeline_add: AddTest#test_full_pipeline_add().
  _get_kernel: _get_kernel().
  AddTest.test_add_simple: AddTest#test_add_simple().
  AddTest.test_add_modular_barrett: AddTest#test_add_modular_barrett().
  AddTest.test_add_modular_subtract: AddTest#test_add_modular_subtract().
  AddTest.test_add_modular_subtract_hypothesis: AddTest#test_add_modular_subtract_hypothesis().
  AddTest: AddTest#
---
# Module: [`jaxite/jaxite_ckks/add_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add_test.py)

## Classes
### `AddTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_ckks/add_test.py:35`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add_test.py#L35)
- signature: `class AddTest(parameterized.TestCase):`
- members:
  - `test_add_modular_barrett(self, kernel_name)` — [`L55`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add_test.py#L55)
  - `test_add_modular_subtract(self, kernel_name)` — [`L76`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add_test.py#L76)
  - `test_add_modular_subtract_hypothesis(self, a_list, b_list)` — [`L97`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add_test.py#L97)
  - `test_add_simple(self, kernel_name)` — [`L42`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add_test.py#L42)
  - `test_full_pipeline_add(self, kernel_name)` — [`L119`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add_test.py#L119)
- uses (calls/refs, reference-scoped): [`data`](types.md#Ciphertext.data), [`encrypt`](encrypt.md#Encrypt.encrypt), [`moduli`](types.md#Ciphertext.moduli), [`encode`](encode.md#Encode.encode), [`keygen`](key_gen.md#keygen), [`Ciphertext`](types.md#Ciphertext), [`decrypt`](encrypt.md#Decrypt.decrypt), [`decode`](encode.md#Decode.decode), [`Encode`](encode.md#Encode), [`Encrypt`](encrypt.md#Encrypt), [`Decrypt`](encrypt.md#Decrypt), [`Decode`](encode.md#Decode), [`_get_kernel`](add_test.md#_get_kernel), [`add`](add.md#AddSimple.add)

## Functions
- `_get_kernel(kernel_name, moduli=None)` — [`L20`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/add_test.py#L20)

