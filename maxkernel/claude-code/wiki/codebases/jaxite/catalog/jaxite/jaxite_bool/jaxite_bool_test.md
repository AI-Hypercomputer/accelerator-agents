---
title: 'Module: jaxite/jaxite_bool/jaxite_bool_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_bool/jaxite_bool_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_bool.jaxite_bool_test`/
symbols:
  BoolBasicOperationsTest.client_key_set: BoolBasicOperationsTest#client_key_set.
  BoolBasicOperationsTest.test_ksk_decomposition_params: BoolBasicOperationsTest#test_ksk_decomposition_params().
  BoolBasicOperationsTest.test_bsk_decomposition_params: BoolBasicOperationsTest#test_bsk_decomposition_params().
  BoolBasicOperationsTest.callback: BoolBasicOperationsTest#callback.
  BoolBasicOperationsTest.test_boolean_gate_constant_succeeds: BoolBasicOperationsTest#test_boolean_gate_constant_succeeds().
  BoolBasicOperationsTest.server_key_set: BoolBasicOperationsTest#server_key_set.
  BoolBasicOperationsTest.test_two_bit_gates: BoolBasicOperationsTest#test_two_bit_gates().
  BoolBasicOperationsTest.test_boolean_gate_cmux_succeeds: BoolBasicOperationsTest#test_boolean_gate_cmux_succeeds().
  BoolBasicOperationsTest.test_lut2_asymmetric: BoolBasicOperationsTest#test_lut2_asymmetric().
  BoolBasicOperationsTest.test_boolean_gate_not_succeeds: BoolBasicOperationsTest#test_boolean_gate_not_succeeds().
  BoolBasicOperationsTest.boolean_params: BoolBasicOperationsTest#boolean_params.
  BoolBasicOperationsTest.lwe_rng: BoolBasicOperationsTest#lwe_rng.
  BoolBasicOperationsTest.rlwe_rng: BoolBasicOperationsTest#rlwe_rng.
  FUNC_NAME_TO_LAMBDA: FUNC_NAME_TO_LAMBDA.
  BoolBasicOperationsTest: BoolBasicOperationsTest#
  BoolBasicOperationsTest.setUpClass: BoolBasicOperationsTest#setUpClass().
---
# Module: [`jaxite/jaxite_bool/jaxite_bool_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_test.py)

## Classes
### `BoolBasicOperationsTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_bool/jaxite_bool_test.py:25`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_test.py#L25)
- doc: A suite of unit tests using a real server_key_set and client_key_set.
- signature: `class BoolBasicOperationsTest(parameterized.TestCase):`
- members:
  - `setUpClass(cls)` — [`L29`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_test.py#L29)
  - `test_boolean_gate_cmux_succeeds(self, v1: bool, v0: bool, c: bool)` — [`L126`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_test.py#L126)
  - `test_boolean_gate_constant_succeeds(self, value: bool)` — [`L57`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_test.py#L57)
  - `test_boolean_gate_not_succeeds(self, value: bool)` — [`L76`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_test.py#L76)
  - `test_bsk_decomposition_params(self, decomp_log_base: int, l: int)` — [`L210`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_test.py#L210)
  - `test_ksk_decomposition_params(self, decomp_log_base: int, l: int)` — [`L164`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_test.py#L164)
  - `test_lut2_asymmetric(self)` — [`L138`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_test.py#L138)
  - `test_two_bit_gates(self, func_name: str, inputs: tuple[bool, bool])` — [`L102`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_test.py#L102)
  - `boolean_params` — [`L33`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_test.py#L33)
  - `callback` — [`L39`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_test.py#L39) — ---
  - `client_key_set` — [`L34`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_test.py#L34)
  - `lwe_rng` — [`L31`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_test.py#L31)
  - `rlwe_rng` — [`L32`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_test.py#L32)
  - `server_key_set` — [`L45`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_test.py#L45)
- uses (calls/refs, reference-scoped): [`encrypt`](jaxite_bool.md#encrypt), [`encode`](../jaxite_cggi/encoding.md#encode), [`and_`](jaxite_bool.md#and_), [`DecompositionParameters`](../jaxite_cggi/decomposition.md#DecompositionParameters), [`lut2`](jaxite_bool.md#lut2), [`decrypt`](jaxite_bool.md#decrypt), [`lwe_dimension`](../jaxite_cggi/parameters.md#SchemeParameters.lwe_dimension), [`level_count`](../jaxite_cggi/decomposition.md#DecompositionParameters.level_count), [`scheme_params`](bool_params.md#Parameters.scheme_params), [`ServerKeySet`](jaxite_bool.md#ServerKeySet), [`log_base`](../jaxite_cggi/decomposition.md#DecompositionParameters.log_base), [`ENCODING_PARAMS`](bool_encoding.md#ENCODING_PARAMS), [`cmux_`](jaxite_bool.md#cmux_), [`get_params_for_128_bit_security`](bool_params.md#get_params_for_128_bit_security), [`constant`](jaxite_bool.md#constant), [`not_`](jaxite_bool.md#not_), [`lwe_sk`](jaxite_bool.md#ClientKeySet.lwe_sk), [`get_lwe_rng_for_128_bit_security`](bool_params.md#get_lwe_rng_for_128_bit_security), [`get_rlwe_rng_for_128_bit_security`](bool_params.md#get_rlwe_rng_for_128_bit_security), [`rlwe_sk`](jaxite_bool.md#ClientKeySet.rlwe_sk), [`CLEARTEXT_FALSE`](bool_encoding.md#CLEARTEXT_FALSE), [`CLEARTEXT_TRUE`](bool_encoding.md#CLEARTEXT_TRUE), [`ClientKeySet`](jaxite_bool.md#ClientKeySet), [`_bs_decomp_params`](bool_params.md#Parameters._bs_decomp_params), [`_ks_decomp_params`](bool_params.md#Parameters._ks_decomp_params), [`FUNC_NAME_TO_LAMBDA`](jaxite_bool_test.md#FUNC_NAME_TO_LAMBDA)  (2 test-only)

## Module values
- `FUNC_NAME_TO_LAMBDA` — [`L11`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool_test.py#L11)

