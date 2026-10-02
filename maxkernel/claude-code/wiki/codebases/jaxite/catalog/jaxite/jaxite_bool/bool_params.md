---
title: 'Module: jaxite/jaxite_bool/bool_params.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_bool/bool_params.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_bool.bool_params`/
symbols:
  Parameters.scheme_params: Parameters#scheme_params().
  Parameters.ks_decomp_params: Parameters#ks_decomp_params().
  Parameters.bs_decomp_params: Parameters#bs_decomp_params().
  Parameters._noiseless_embedding: Parameters#_noiseless_embedding().
  Parameters.lut_by_name: Parameters#lut_by_name().
  Parameters.lut_poly_by_name: Parameters#lut_poly_by_name().
  SCHEME_PARAMS_128_BIT_SECURITY: SCHEME_PARAMS_128_BIT_SECURITY.
  TEST_SCHEME_PARAMS: TEST_SCHEME_PARAMS.
  get_params_for_128_bit_security: get_params_for_128_bit_security().
  Parameters.lut_poly: Parameters#lut_poly().
  get_params_for_test: get_params_for_test().
  Parameters.noiseless_true: Parameters#noiseless_true().
  Parameters.lut: Parameters#lut().
  get_lwe_rng_for_128_bit_security: get_lwe_rng_for_128_bit_security().
  get_rlwe_rng_for_128_bit_security: get_rlwe_rng_for_128_bit_security().
  BSK_DECOMP_PARAMS_128_BIT_SECURITY: BSK_DECOMP_PARAMS_128_BIT_SECURITY.
  KSK_DECOMP_PARAMS_128_BIT_SECURITY: KSK_DECOMP_PARAMS_128_BIT_SECURITY.
  TEST_BSK_DECOMP_PARAMS: TEST_BSK_DECOMP_PARAMS.
  Parameters._lut_cache: Parameters#_lut_cache.
  Parameters.noiseless_false: Parameters#noiseless_false().
  Parameters: Parameters#
  Parameters.__init__: Parameters#__init__().
  ENCODING_PARAMS: ENCODING_PARAMS.
  TEST_KSK_DECOMP_PARAMS: TEST_KSK_DECOMP_PARAMS.
  get_rng_for_test: get_rng_for_test().
  Parameters._scheme_params: Parameters#_scheme_params.
  Parameters._ks_decomp_params: Parameters#_ks_decomp_params.
  Parameters._bs_decomp_params: Parameters#_bs_decomp_params.
---
# Module: [`jaxite/jaxite_bool/bool_params.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py)

## Classes
### `Parameters`
- def: [`jaxite/jaxite_bool/bool_params.py:17`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L17)
- doc: TFHE Boolean gate shared parameters.
- signature: `class Parameters:`
- members:
  - `bs_decomp_params(self)` — [`L40`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L40) — documented in [jaxite-jaxite_bool-bool_params](../../../concepts/jaxite-jaxite_bool-bool_params.md)
  - `ks_decomp_params(self)` — [`L36`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L36) — documented in [jaxite-jaxite_bool-bool_params](../../../concepts/jaxite-jaxite_bool-bool_params.md)
  - `lut(self, num_inputs: int, truth_table: int)` — [`L67`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L67)
  - `lut_by_name(self, name: str)` — [`L70`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L70)
  - `lut_poly(self, num_inputs: int, truth_table: int)` — [`L59`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L59)
  - `lut_poly_by_name(self, name: str)` — [`L64`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L64)
  - `noiseless_false(self)` — [`L56`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L56) — documented in [jaxite-jaxite_bool-bool_params](../../../concepts/jaxite-jaxite_bool-bool_params.md)
  - `noiseless_true(self)` — [`L52`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L52) — documented in [jaxite-jaxite_bool-bool_params](../../../concepts/jaxite-jaxite_bool-bool_params.md)
  - `scheme_params(self)` — [`L32`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L32) — documented in [jaxite-jaxite_bool-bool_params](../../../concepts/jaxite-jaxite_bool-bool_params.md)
- protocol/private: `__init__`[`L20`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L20), `_bs_decomp_params`[`L28`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L28), `_ks_decomp_params`[`L27`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L27), `_lut_cache`[`L29`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L29), `_noiseless_embedding`[`L43`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L43), `_scheme_params`[`L26`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L26)
- uses (calls/refs, reference-scoped): [`LweCiphertext`](../jaxite_cggi/types.md#LweCiphertext), [`encode`](../jaxite_cggi/encoding.md#encode), [`SchemeParameters`](../jaxite_cggi/parameters.md#SchemeParameters), [`DecompositionParameters`](../jaxite_cggi/decomposition.md#DecompositionParameters), [`RlweCiphertext`](../jaxite_cggi/rlwe.md#RlweCiphertext), [`lwe_dimension`](../jaxite_cggi/parameters.md#SchemeParameters.lwe_dimension), [`LweCleartext`](../jaxite_cggi/types.md#LweCleartext), [`lut_by_name`](lut.md#LutCache.lut_by_name), [`lut_poly`](lut.md#LutCache.lut_poly), [`lut_poly_by_name`](lut.md#LutCache.lut_poly_by_name), [`LookUpTable`](lut.md#LookUpTable), [`lut`](lut.md#LutCache.lut), [`CLEARTEXT_FALSE`](bool_encoding.md#CLEARTEXT_FALSE), [`CLEARTEXT_TRUE`](bool_encoding.md#CLEARTEXT_TRUE), [`noiseless_embedding`](../jaxite_cggi/lwe.md#noiseless_embedding), [`LutCache`](lut.md#LutCache), [`ENCODING_PARAMS`](bool_params.md#ENCODING_PARAMS)
- used by: [`and_`](jaxite_bool.md#and_), [`or_`](jaxite_bool.md#or_), [`xor_`](jaxite_bool.md#xor_), [`lut2`](jaxite_bool.md#lut2), [`lut3`](jaxite_bool.md#lut3), [`andny_`](jaxite_bool.md#andny_), [`andyn_`](jaxite_bool.md#andyn_), [`nand_`](jaxite_bool.md#nand_), [`nor_`](jaxite_bool.md#nor_), [`orny_`](jaxite_bool.md#orny_), [`oryn_`](jaxite_bool.md#oryn_), [`test_bsk_decomposition_params`](jaxite_bool_test.md#BoolBasicOperationsTest.test_bsk_decomposition_params), [`test_ksk_decomposition_params`](jaxite_bool_test.md#BoolBasicOperationsTest.test_ksk_decomposition_params), [`xnor_`](jaxite_bool.md#xnor_), [`pmap_lut2_impl`](jaxite_bool.md#pmap_lut2_impl), [`pmap_lut3_impl`](jaxite_bool.md#pmap_lut3_impl), [`Parameters`](jaxite_bool.md#Parameters), [`callback`](jaxite_bool_test.md#BoolBasicOperationsTest.callback), [`callback`](jaxite_bool_multigate_test.md#BoolMultigateTest.callback), [`test_boolean_gate_constant_succeeds`](jaxite_bool_test.md#BoolBasicOperationsTest.test_boolean_gate_constant_succeeds), [`cmux_`](jaxite_bool.md#cmux_), [`pmap_lut2`](jaxite_bool.md#pmap_lut2), [`pmap_lut3`](jaxite_bool.md#pmap_lut3), [`_bsk`](jaxite_bool.md#ServerKeySet._bsk), [`get_params_for_128_bit_security`](bool_params.md#get_params_for_128_bit_security), [`constant`](jaxite_bool.md#constant), [`not_`](jaxite_bool.md#not_), [`get_params_for_test`](bool_params.md#get_params_for_test), [`_ksk`](jaxite_bool.md#ServerKeySet._ksk), [`_lwe_sk`](jaxite_bool.md#ClientKeySet._lwe_sk), [`_rlwe_sk`](jaxite_bool.md#ClientKeySet._rlwe_sk)

## Functions
- `get_lwe_rng_for_128_bit_security(seed: int)` — [`L83`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L83) — Returns lwe rng for 128 bit security.
- `get_params_for_128_bit_security()` — [`L74`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L74) — Returns boolean scheme params for 128 bit security.
- `get_params_for_test()` — [`L127`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L127) — Returns boolean scheme params for tests.
- `get_rlwe_rng_for_128_bit_security(seed: int)` — [`L95`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L95) — Returns rlwe rng for 128 bit security.
- `get_rng_for_test(seed: int)` — [`L134`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L134) — Returns rng for tests.

## Module values
- `BSK_DECOMP_PARAMS_128_BIT_SECURITY` — [`L118`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L118)
- `ENCODING_PARAMS` — [`L14`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L14) — documented in [jaxite-jaxite_bool-bool_params](../../../concepts/jaxite-jaxite_bool-bool_params.md)
- `KSK_DECOMP_PARAMS_128_BIT_SECURITY` — [`L122`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L122)
- `SCHEME_PARAMS_128_BIT_SECURITY` — [`L108`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L108) — documented in [jaxite-jaxite_cggi-parameters](../../../concepts/jaxite-jaxite_cggi-parameters.md)
- `TEST_BSK_DECOMP_PARAMS` — [`L150`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L150)
- `TEST_KSK_DECOMP_PARAMS` — [`L153`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L153)
- `TEST_SCHEME_PARAMS` — [`L142`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/bool_params.py#L142)

