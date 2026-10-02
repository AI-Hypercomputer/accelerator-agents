---
title: 'Module: jaxite/jaxite_ckks/encode_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/encode_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.encode_test`/
symbols:
  CrossDiffTest.test_encode_diff: CrossDiffTest#test_encode_diff().
  EncodeTest.test_encode_decode_loop: EncodeTest#test_encode_decode_loop().
  EncodeTest: EncodeTest#
  CrossDiffTest: CrossDiffTest#
---
# Module: [`jaxite/jaxite_ckks/encode_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode_test.py)

## Classes
### `CrossDiffTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_ckks/encode_test.py:36`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode_test.py#L36)
- doc: Tests ensuring the exact same behavior as the CROSS reference code.
- signature: `class CrossDiffTest(absltest.TestCase):`
- members:
  - `test_encode_diff(self)` — [`L39`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode_test.py#L39)
- uses (calls/refs, reference-scoped): [`encode`](encode.md#Encode.encode), [`decode`](encode.md#Decode.decode), [`Encode`](encode.md#Encode), [`data`](types.md#Plaintext.data), [`Decode`](encode.md#Decode)

### `EncodeTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_ckks/encode_test.py:10`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode_test.py#L10)
- signature: `class EncodeTest(absltest.TestCase):`
- members:
  - `test_encode_decode_loop(self, slots, scale)` — [`L21`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode_test.py#L21)
- uses (calls/refs, reference-scoped): [`encode`](encode.md#Encode.encode), [`decode`](encode.md#Decode.decode), [`Encode`](encode.md#Encode), [`Decode`](encode.md#Decode)

