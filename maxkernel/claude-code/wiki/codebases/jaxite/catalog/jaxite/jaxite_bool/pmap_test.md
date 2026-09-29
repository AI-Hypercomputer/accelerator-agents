---
title: 'Module: jaxite/jaxite_bool/pmap_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_bool/pmap_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_bool.pmap_test`/PmapTest#
symbols:
  PmapTest.test_pmap_lut3: test_pmap_lut3().
  PmapTest.test_pmap_lut2: test_pmap_lut2().
  PmapTest.client_key_set: client_key_set.
  PmapTest.server_key_set: server_key_set.
  PmapTest.lwe_rng: lwe_rng.
  PmapTest.boolean_params: boolean_params.
  PmapTest.rlwe_rng: rlwe_rng.
  PmapTest: ''
  PmapTest.setUpClass: setUpClass().
---
# Module: [`jaxite/jaxite_bool/pmap_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/pmap_test.py)

## Classes
### `PmapTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_bool/pmap_test.py:12`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/pmap_test.py#L12)
- doc: Tests of jaxite_bool API that use pmap to parallelize across gates.
- signature: `class PmapTest(parameterized.TestCase):`
- members:
  - `setUpClass(cls)` — [`L16`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/pmap_test.py#L16)
  - `test_pmap_lut2(self)` — [`L56`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/pmap_test.py#L56)
  - `test_pmap_lut3(self)` — [`L33`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/pmap_test.py#L33)
  - `boolean_params` — [`L20`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/pmap_test.py#L20)
  - `client_key_set` — [`L21`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/pmap_test.py#L21)
  - `lwe_rng` — [`L18`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/pmap_test.py#L18)
  - `rlwe_rng` — [`L19`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/pmap_test.py#L19)
  - `server_key_set` — [`L26`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/pmap_test.py#L26)
- uses (calls/refs, reference-scoped): [`encrypt`](jaxite_bool.md#encrypt), [`decrypt`](jaxite_bool.md#decrypt), [`ServerKeySet`](jaxite_bool.md#ServerKeySet), [`pmap_lut2`](jaxite_bool.md#pmap_lut2), [`pmap_lut3`](jaxite_bool.md#pmap_lut3), [`get_params_for_128_bit_security`](bool_params.md#get_params_for_128_bit_security), [`get_lwe_rng_for_128_bit_security`](bool_params.md#get_lwe_rng_for_128_bit_security), [`get_rlwe_rng_for_128_bit_security`](bool_params.md#get_rlwe_rng_for_128_bit_security), [`ClientKeySet`](jaxite_bool.md#ClientKeySet)

