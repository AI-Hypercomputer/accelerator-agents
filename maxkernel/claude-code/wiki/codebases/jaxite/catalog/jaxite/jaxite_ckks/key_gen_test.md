---
title: 'Module: jaxite/jaxite_ckks/key_gen_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/key_gen_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.key_gen_test`/
symbols:
  KeyGenTest.test_column_key: KeyGenTest#test_column_key().
  KeyGenTest.test_keygen_with_hamming_weight: KeyGenTest#test_keygen_with_hamming_weight().
  TEST_PRIMES: TEST_PRIMES.
  KeyGenTest: KeyGenTest#
---
# Module: [`jaxite/jaxite_ckks/key_gen_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_gen_test.py)

## Classes
### `KeyGenTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_ckks/key_gen_test.py:19`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_gen_test.py#L19)
- signature: `class KeyGenTest(absltest.TestCase):`
- members:
  - `test_column_key(self)` — [`L39`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_gen_test.py#L39)
  - `test_keygen_with_hamming_weight(self)` — [`L21`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_gen_test.py#L21)
- uses (calls/refs, reference-scoped): [`keygen`](key_gen.md#keygen), [`decrypt`](encrypt.md#Decrypt.decrypt), [`decode`](encode.md#Decode.decode), [`Decrypt`](encrypt.md#Decrypt), [`Decode`](encode.md#Decode), [`data`](types.md#SecretKey.data), [`gen_cm_keys`](key_gen.md#gen_cm_keys), [`intt_negacyclic_poly`](ntt_cpu.md#intt_negacyclic_poly), [`TEST_PRIMES`](key_gen_test.md#TEST_PRIMES)

## Module values
- `TEST_PRIMES` — [`L11`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_gen_test.py#L11)

