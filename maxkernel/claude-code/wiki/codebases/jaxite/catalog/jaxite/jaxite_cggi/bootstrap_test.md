---
title: 'Module: jaxite/jaxite_cggi/bootstrap_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_cggi/bootstrap_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_cggi.bootstrap_test`/
symbols:
  BootstrapBaseTest.run_bootstrap_test: BootstrapBaseTest#run_bootstrap_test().
  ProdSecurityTest.test_1_bit_bootstrap_prod_params: ProdSecurityTest#test_1_bit_bootstrap_prod_params().
  ProdSecurityTest.test_3_bit_bootstrap_prod_params: ProdSecurityTest#test_3_bit_bootstrap_prod_params().
  ConsistencyTest.test_3_bit_consistency: ConsistencyTest#test_3_bit_consistency().
  BootstrapTest: BootstrapTest#
  BootstrapBaseTest: BootstrapBaseTest#
  ZERO_RNG: ZERO_RNG.
  BootstrapBaseTest.callback: BootstrapBaseTest#callback.
  BootstrapBaseTest.__init__: BootstrapBaseTest#__init__().
  BootstrapTest.test_3_bit_bootstrap: BootstrapTest#test_3_bit_bootstrap().
  BootstrapTest.test_3_bit_bootstrap_larger_lwe_dimension: BootstrapTest#test_3_bit_bootstrap_larger_lwe_dimension().
  BootstrapTest.test_3_bit_bootstrap_prod_decomp_params: BootstrapTest#test_3_bit_bootstrap_prod_decomp_params().
  ProdSecurityTest: ProdSecurityTest#
  ConsistencyTest: ConsistencyTest#
  _LOG_AI_BOUNDS: _LOG_AI_BOUNDS.
  _SEEDS: _SEEDS.
  ConsistencyTest.callback: ConsistencyTest#callback.
---
# Module: [`jaxite/jaxite_cggi/bootstrap_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap_test.py)

## Classes
### `BootstrapBaseTest`
- def: [`jaxite/jaxite_cggi/bootstrap_test.py:28`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap_test.py#L28)
- doc: A base class for running bootstrap tests.
- signature: `class BootstrapBaseTest(parameterized.TestCase):`
- members:
  - `run_bootstrap_test(self, *, injected_noise: int, lwe_dimension: int, lwe_rng: random_source.RandomSource, message_bits: int, mod_degree: int, padding_bits: int, rlwe_rng: random_source.RandomSource, skip_assert: bool = False, use_bmmp: bool = True, use_bat: bool = False)` — [`L35`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap_test.py#L35)
  - `callback` — [`L33`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap_test.py#L33)
- protocol/private: `__init__`[`L31`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap_test.py#L31)
- uses (calls/refs, reference-scoped): [`bootstrap`](bootstrap.md#bootstrap), [`gen_bootstrapping_key`](bootstrap.md#gen_bootstrapping_key), [`gen_key`](key_switch.md#gen_key), [`message_bit_length`](encoding.md#EncodingParameters.message_bit_length), [`encode`](encoding.md#encode), [`encrypt`](lwe.md#encrypt), [`EncodingParameters`](encoding.md#EncodingParameters), [`RandomSource`](random_source.md#RandomSource), [`gen_key`](lwe.md#gen_key), [`padding_bit_length`](encoding.md#EncodingParameters.padding_bit_length), [`gen_key`](rlwe.md#gen_key), [`SchemeParameters`](parameters.md#SchemeParameters), [`total_bit_length`](encoding.md#EncodingParameters.total_bit_length), [`decode`](encoding.md#decode), [`polynomial_modulus_degree`](parameters.md#SchemeParameters.polynomial_modulus_degree), [`jit_bootstrap`](bootstrap.md#jit_bootstrap), [`decrypt`](lwe.md#decrypt), [`lwe_dimension`](parameters.md#SchemeParameters.lwe_dimension), [`flatten_key`](rlwe.md#flatten_key), [`message`](rlwe.md#RlweCiphertext.message), [`decrypt_without_denoising`](lwe.md#decrypt_without_denoising), [`rlwe_dimension`](parameters.md#SchemeParameters.rlwe_dimension), [`plaintext_modulus`](parameters.md#SchemeParameters.plaintext_modulus), [`extract_noise`](encoding.md#extract_noise), [`lwe_dimension`](lwe.md#LweSecretKey.lwe_dimension), [`key_from_rlwe`](rgsw.md#key_from_rlwe), [`key_data`](key_switch.md#LweKeySwitchingKey.key_data), [`encrypted_lwe_sk_bits`](bootstrap.md#BootstrappingKey.encrypted_lwe_sk_bits), [`use_bat`](bootstrap.md#BootstrappingKey.use_bat), [`use_bmmp`](bootstrap.md#BootstrappingKey.use_bmmp), [`ZERO_RNG`](bootstrap_test.md#ZERO_RNG)  (7 test-only)
- used by: [`BootstrapTest`](bootstrap_test.md#BootstrapTest), [`ConsistencyTest`](bootstrap_test.md#ConsistencyTest), [`ProdSecurityTest`](bootstrap_test.md#ProdSecurityTest)

### `BootstrapTest`
- def: [`jaxite/jaxite_cggi/bootstrap_test.py:169`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap_test.py#L169)
- signature: `class BootstrapTest(BootstrapBaseTest):`
- members:
  - `test_3_bit_bootstrap(self, log_ai_bound, seed, use_bmmp, use_bat)` — [`L171`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap_test.py#L171)
  - `test_3_bit_bootstrap_larger_lwe_dimension(self, log_ai_bound: int, seed: int, use_bmmp: bool, use_bat: bool)` — [`L202`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap_test.py#L202)
  - `test_3_bit_bootstrap_prod_decomp_params(self, log_ai_bound: int, seed: int, use_bmmp: bool, use_bat: bool)` — [`L238`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap_test.py#L238)
- uses (calls/refs, reference-scoped): [`PseudorandomSource`](random_source.md#PseudorandomSource), [`BootstrapBaseTest`](bootstrap_test.md#BootstrapBaseTest), [`_LOG_AI_BOUNDS`](bootstrap_test.md#_LOG_AI_BOUNDS), [`_SEEDS`](bootstrap_test.md#_SEEDS)

### `ConsistencyTest`
- def: [`jaxite/jaxite_cggi/bootstrap_test.py:329`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap_test.py#L329)
- doc: A test suite to check for race conditions or other nondeterminism.
- signature: `class ConsistencyTest(BootstrapBaseTest):`
- members:
  - `test_3_bit_consistency(self)` — [`L332`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap_test.py#L332)
  - `callback` — [`L344`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap_test.py#L344)
- uses (calls/refs, reference-scoped): [`PseudorandomSource`](random_source.md#PseudorandomSource), [`BootstrapBaseTest`](bootstrap_test.md#BootstrapBaseTest)  (3 test-only)

### `ProdSecurityTest`
- def: [`jaxite/jaxite_cggi/bootstrap_test.py:271`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap_test.py#L271)
- doc: Test 128-bit security parameters for bootstrap.
- signature: `class ProdSecurityTest(BootstrapBaseTest):`
- members:
  - `test_1_bit_bootstrap_prod_params(self)` — [`L274`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap_test.py#L274)
  - `test_3_bit_bootstrap_prod_params(self)` — [`L301`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap_test.py#L301)
- uses (calls/refs, reference-scoped): [`polynomial_modulus_degree`](parameters.md#SchemeParameters.polynomial_modulus_degree), [`lwe_dimension`](parameters.md#SchemeParameters.lwe_dimension), [`BootstrapBaseTest`](bootstrap_test.md#BootstrapBaseTest)  (4 test-only)

## Module values
- `ZERO_RNG` — [`L20`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap_test.py#L20)
- `_LOG_AI_BOUNDS` — [`L21`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap_test.py#L21)
- `_SEEDS` — [`L22`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap_test.py#L22)

