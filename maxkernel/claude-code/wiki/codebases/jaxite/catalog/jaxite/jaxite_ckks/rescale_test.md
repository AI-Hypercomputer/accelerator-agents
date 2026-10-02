---
title: 'Module: jaxite/jaxite_ckks/rescale_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/rescale_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.rescale_test`/
symbols:
  RescaleTest.rescale_kernel: RescaleTest#rescale_kernel.
  RescaleTest.test_rescale_e2e: RescaleTest#test_rescale_e2e().
  RescaleTest.test_rescale_different_scale: RescaleTest#test_rescale_different_scale().
  RescaleTest.test_double_rescale: RescaleTest#test_double_rescale().
  RescaleTest.test_encrypt_decrypt_3_moduli: RescaleTest#test_encrypt_decrypt_3_moduli().
  RescaleTest.test_rescale_more_moduli: RescaleTest#test_rescale_more_moduli().
  RescaleTest.test_rescale_different_degree: RescaleTest#test_rescale_different_degree().
  RescaleHypothesisTest.test_rescale_hypothesis: RescaleHypothesisTest#test_rescale_hypothesis().
  RescaleHypothesisTest.rescale_kernel: RescaleHypothesisTest#rescale_kernel.
  RescaleTest.test_rescale_equivalence: RescaleTest#test_rescale_equivalence().
  RescaleHypothesisTest.sk: RescaleHypothesisTest#sk.
  RescaleTest.sk: RescaleTest#sk.
  RescaleTest.moduli: RescaleTest#moduli.
  RescaleHypothesisTest.MODULI: RescaleHypothesisTest#MODULI.
  RescaleHypothesisTest.encryptor: RescaleHypothesisTest#encryptor.
  RescaleHypothesisTest.decryptor: RescaleHypothesisTest#decryptor.
  RescaleHypothesisTest.random_source: RescaleHypothesisTest#random_source.
  RescaleTest.degree: RescaleTest#degree.
  RescaleTest.pk: RescaleTest#pk.
  RescaleHypothesisTest.DEGREE: RescaleHypothesisTest#DEGREE.
  MODULI_1: MODULI_1.
  IN_DATA_1: IN_DATA_1.
  OUT_DATA_1: OUT_DATA_1.
  MODULI_2: MODULI_2.
  IN_DATA_2: IN_DATA_2.
  OUT_DATA_2: OUT_DATA_2.
  RescaleHypothesisTest.R: RescaleHypothesisTest#R.
  RescaleHypothesisTest.C: RescaleHypothesisTest#C.
  RescaleHypothesisTest.pk: RescaleHypothesisTest#pk.
  RescaleTest: RescaleTest#
  RescaleTest.setUpClass: RescaleTest#setUpClass().
  RescaleHypothesisTest: RescaleHypothesisTest#
  RescaleHypothesisTest.setUpClass: RescaleHypothesisTest#setUpClass().
---
# Module: [`jaxite/jaxite_ckks/rescale_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py)

## Classes
### `RescaleHypothesisTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_ckks/rescale_test.py:392`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L392)
- signature: `class RescaleHypothesisTest(absltest.TestCase):`
- members:
  - `setUpClass(cls)` — [`L400`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L400)
  - `test_rescale_hypothesis(self, slots, scale_power)` — [`L423`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L423) — Test rescale with randomized inputs using Hypothesis.
  - `C` — [`L397`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L397)
  - `DEGREE` — [`L395`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L395)
  - `MODULI` — [`L394`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L394)
  - `R` — [`L396`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L396)
  - `decryptor` — [`L412`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L412)
  - `encryptor` — [`L411`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L411)
  - `pk` — [`L410`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L410)
  - `random_source` — [`L409`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L409)
  - `rescale_kernel` — [`L402`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L402)
  - `sk` — [`L410`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L410)
- uses (calls/refs, reference-scoped): [`data`](types.md#Ciphertext.data), [`encrypt`](encrypt.md#Encrypt.encrypt), [`moduli`](types.md#Ciphertext.moduli), [`encode`](encode.md#Encode.encode), [`keygen`](key_gen.md#keygen), [`Ciphertext`](types.md#Ciphertext), [`rescale`](rescale.md#Rescale.rescale), [`precompute_constants`](rescale.md#Rescale.precompute_constants), [`decrypt`](encrypt.md#Decrypt.decrypt), [`decode`](encode.md#Decode.decode), [`Encode`](encode.md#Encode), [`Encrypt`](encrypt.md#Encrypt), [`Decrypt`](encrypt.md#Decrypt), [`Decode`](encode.md#Decode), [`TestRandomSource`](random.md#TestRandomSource), [`Rescale`](rescale.md#Rescale)

### `RescaleTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_ckks/rescale_test.py:189`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L189)
- signature: `class RescaleTest(parameterized.TestCase):`
- members:
  - `setUpClass(cls)` — [`L192`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L192)
  - `test_double_rescale(self)` — [`L325`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L325) — Test double-rescale.
  - `test_encrypt_decrypt_3_moduli(self)` — [`L224`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L224)
  - `test_rescale_different_degree(self)` — [`L361`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L361) — Test rescale with a smaller polynomial degree.
  - `test_rescale_different_scale(self)` — [`L270`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L270) — Test rescale with a larger scale.
  - `test_rescale_e2e(self)` — [`L246`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L246)
  - `test_rescale_equivalence(self, moduli, in_data, expected_out)` — [`L209`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L209)
  - `test_rescale_more_moduli(self)` — [`L295`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L295) — Test rescale with a larger number of moduli.
  - `degree` — [`L194`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L194)
  - `moduli` — [`L195`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L195)
  - `pk` — [`L196`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L196)
  - `rescale_kernel` — [`L197`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L197)
  - `sk` — [`L196`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L196)
- uses (calls/refs, reference-scoped): [`data`](types.md#Ciphertext.data), [`encrypt`](encrypt.md#Encrypt.encrypt), [`moduli`](types.md#Ciphertext.moduli), [`encode`](encode.md#Encode.encode), [`keygen`](key_gen.md#keygen), [`Ciphertext`](types.md#Ciphertext), [`rescale`](rescale.md#Rescale.rescale), [`precompute_constants`](rescale.md#Rescale.precompute_constants), [`decrypt`](encrypt.md#Decrypt.decrypt), [`decode`](encode.md#Decode.decode), [`Encode`](encode.md#Encode), [`Encrypt`](encrypt.md#Encrypt), [`Decrypt`](encrypt.md#Decrypt), [`Decode`](encode.md#Decode), [`TestRandomSource`](random.md#TestRandomSource), [`Rescale`](rescale.md#Rescale), [`IN_DATA_1`](rescale_test.md#IN_DATA_1), [`IN_DATA_2`](rescale_test.md#IN_DATA_2), [`MODULI_1`](rescale_test.md#MODULI_1), [`MODULI_2`](rescale_test.md#MODULI_2), [`OUT_DATA_1`](rescale_test.md#OUT_DATA_1), [`OUT_DATA_2`](rescale_test.md#OUT_DATA_2)

## Module values
- `IN_DATA_1` — [`L21`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L21)
- `IN_DATA_2` — [`L105`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L105)
- `MODULI_1` — [`L20`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L20)
- `MODULI_2` — [`L104`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L104)
- `OUT_DATA_1` — [`L62`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L62)
- `OUT_DATA_2` — [`L146`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rescale_test.py#L146)

