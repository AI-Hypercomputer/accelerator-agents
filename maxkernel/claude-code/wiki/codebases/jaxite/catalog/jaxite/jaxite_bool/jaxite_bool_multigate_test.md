---
title: 'Module: jaxite/jaxite_bool/jaxite_bool_multigate_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_bool/jaxite_bool_multigate_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_bool.jaxite_bool_multigate_test`/BoolMultigateTest#
symbols:
  BoolMultigateTest.test_seeds: test_seeds().
  BoolMultigateTest.client_key_set: client_key_set.
  BoolMultigateTest.test_boolean_gate_chained: test_boolean_gate_chained().
  BoolMultigateTest.callback: callback.
  BoolMultigateTest.server_key_set: server_key_set.
  BoolMultigateTest.boolean_params: boolean_params.
  BoolMultigateTest.test_chained_and_succeeds: test_chained_and_succeeds().
  BoolMultigateTest.lwe_rng: lwe_rng.
  BoolMultigateTest.rlwe_rng: rlwe_rng.
  BoolMultigateTest: ''
  BoolMultigateTest.setUpClass: setUpClass().
---
# Module: [`jaxite/jaxite_bool/jaxite_bool_multigate_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_multigate_test.py)

## Classes
### `BoolMultigateTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_bool/jaxite_bool_multigate_test.py:11`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_multigate_test.py#L11)
- doc: Boolean gate API tests that chain multiple gates together.
- signature: `class BoolMultigateTest(parameterized.TestCase):`
- members:
  - `setUpClass(cls)` — [`L15`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_multigate_test.py#L15)
  - `test_boolean_gate_chained(self)` — [`L39`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_multigate_test.py#L39)
  - `test_chained_and_succeeds(self, i0: bool, i1: bool, i2: bool)` — [`L67`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_multigate_test.py#L67)
  - `test_seeds(self, seed: int)` — [`L102`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_multigate_test.py#L102)
  - `boolean_params` — [`L19`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_multigate_test.py#L19)
  - `callback` — [`L25`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_multigate_test.py#L25) — ---
  - `client_key_set` — [`L20`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_multigate_test.py#L20)
  - `lwe_rng` — [`L17`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_multigate_test.py#L17)
  - `rlwe_rng` — [`L18`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_multigate_test.py#L18)
  - `server_key_set` — [`L31`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_multigate_test.py#L31)
- uses (calls/refs, reference-scoped): [`encrypt`](jaxite_bool.md#encrypt), [`and_`](jaxite_bool.md#and_), [`or_`](jaxite_bool.md#or_), [`xor_`](jaxite_bool.md#xor_), [`decrypt`](jaxite_bool.md#decrypt), [`scheme_params`](bool_params.md#Parameters.scheme_params), [`ServerKeySet`](jaxite_bool.md#ServerKeySet), [`ENCODING_PARAMS`](bool_encoding.md#ENCODING_PARAMS), [`get_params_for_128_bit_security`](bool_params.md#get_params_for_128_bit_security), [`not_`](jaxite_bool.md#not_), [`lwe_sk`](jaxite_bool.md#ClientKeySet.lwe_sk), [`get_lwe_rng_for_128_bit_security`](bool_params.md#get_lwe_rng_for_128_bit_security), [`get_rlwe_rng_for_128_bit_security`](bool_params.md#get_rlwe_rng_for_128_bit_security), [`rlwe_sk`](jaxite_bool.md#ClientKeySet.rlwe_sk), [`ClientKeySet`](jaxite_bool.md#ClientKeySet)  (2 test-only)

