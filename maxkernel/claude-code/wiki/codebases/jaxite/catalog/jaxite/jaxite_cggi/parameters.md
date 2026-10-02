---
title: 'Module: jaxite/jaxite_cggi/parameters.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_cggi/parameters.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_cggi.parameters`/SchemeParameters#
symbols:
  SchemeParameters: ''
  SchemeParameters.polynomial_modulus_degree: polynomial_modulus_degree.
  SchemeParameters.lwe_dimension: lwe_dimension.
  SchemeParameters.rlwe_dimension: rlwe_dimension.
  SchemeParameters.plaintext_modulus: plaintext_modulus.
  SchemeParameters.log_plaintext_modulus: log_plaintext_modulus.
  SchemeParameters.__post_init__: __post_init__().
  SchemeParameters.log_mod_degree: log_mod_degree.
---
# Module: [`jaxite/jaxite_cggi/parameters.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/parameters.py)

## Classes
### `SchemeParameters`
- def: [`jaxite/jaxite_cggi/parameters.py:8`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/parameters.py#L8) — documented in [jaxite-jaxite_bool-bool_params](../../../concepts/jaxite-jaxite_bool-bool_params.md)
- doc: Scheme parameters for TFHE.
- signature: `class SchemeParameters:`
- members:
  - `log_mod_degree` — [`L28`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/parameters.py#L28)
  - `log_plaintext_modulus` — [`L31`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/parameters.py#L31)
  - `lwe_dimension` — [`L14`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/parameters.py#L14) — documented in [jaxite-jaxite_bool-bool_params](../../../concepts/jaxite-jaxite_bool-bool_params.md)
  - `plaintext_modulus` — [`L17`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/parameters.py#L17)
  - `polynomial_modulus_degree` — [`L25`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/parameters.py#L25) — documented in [jaxite-jaxite_cggi-parameters](../../../concepts/jaxite-jaxite_cggi-parameters.md)
  - `rlwe_dimension` — [`L22`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/parameters.py#L22) — documented in [jaxite-jaxite_cggi-parameters](../../../concepts/jaxite-jaxite_cggi-parameters.md)
- protocol/private: `__post_init__`[`L33`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/parameters.py#L33)
- used by: [`run_bootstrap_test`](bootstrap_test.md#BootstrapBaseTest.run_bootstrap_test), [`test_key_switch_128_bit_security`](key_switch_test.md#KeySwitchTest.test_key_switch_128_bit_security), [`bootstrap`](bootstrap.md#bootstrap), [`test_blind_rotate_with_noise`](blind_rotate_test.md#BlindRotateTest.test_blind_rotate_with_noise), [`test_blind_rotate_doesnt_set_padding_bit`](blind_rotate_test.md#BlindRotateTest.test_blind_rotate_doesnt_set_padding_bit), [`test_modulus_switch_preserves_message`](lwe_test.md#LweModulusSwitchingTest.test_modulus_switch_preserves_message), [`test_encrypt_add_decrypt_prod_security_params`](rlwe_test.md#RlweTest.test_encrypt_add_decrypt_prod_security_params), [`test_encrypt_decrypt_with_varying_rngs`](rgsw_test.md#RgswTest.test_encrypt_decrypt_with_varying_rngs), [`test_encrypt_decrypt_with_varying_nonzero_sk_entry`](rgsw_test.md#RgswTest.test_encrypt_decrypt_with_varying_nonzero_sk_entry), [`default_key`](lwe_test.md#LweEncryptDecryptTest.default_key), [`gen_key`](lwe.md#gen_key), [`gen_key`](rlwe.md#gen_key), [`jit_bootstrap`](bootstrap.md#jit_bootstrap), [`test_encrypt_decrypt_prod_security_params`](rgsw_test.md#RgswTest.test_encrypt_decrypt_prod_security_params), [`setUp`](rgsw_test.md#RgswTest.setUp), [`setUp`](rlwe_test.md#RlweTest.setUp), [`test_deterministic_noise_encrypt`](lwe_test.md#LweEncryptDecryptTest.test_deterministic_noise_encrypt), [`test_gen_key`](lwe_test.md#LweEncryptDecryptTest.test_gen_key), [`test_gen_key`](rlwe_test.md#RlweTest.test_gen_key), [`scheme_params`](../jaxite_bool/bool_params.md#Parameters.scheme_params), [`as_rlwe_test_polynomial`](../jaxite_bool/lut.md#LookUpTable.as_rlwe_test_polynomial), [`scheme_params`](blind_rotate_test.md#BlindRotateTest.scheme_params), [`test_boolean_gate_constant_succeeds`](../jaxite_bool/jaxite_bool_test.md#BoolBasicOperationsTest.test_boolean_gate_constant_succeeds), [`_noiseless_embedding`](../jaxite_bool/bool_params.md#Parameters._noiseless_embedding), [`in_params`](key_switch_test.md#KeySwitchTest.in_params), [`out_params`](key_switch_test.md#KeySwitchTest.out_params), [`gen_key`](rgsw.md#gen_key), [`TEST_SCHEME_PARAMS`](../jaxite_bool/lut_test.md#TEST_SCHEME_PARAMS), [`test_1_bit_bootstrap_prod_params`](bootstrap_test.md#ProdSecurityTest.test_1_bit_bootstrap_prod_params), [`test_3_bit_bootstrap_prod_params`](bootstrap_test.md#ProdSecurityTest.test_3_bit_bootstrap_prod_params), [`SCHEME_PARAMS_128_BIT_SECURITY`](../jaxite_bool/bool_params.md#SCHEME_PARAMS_128_BIT_SECURITY), [`TEST_SCHEME_PARAMS`](../jaxite_bool/bool_params.md#TEST_SCHEME_PARAMS), [`__init__`](../jaxite_bool/bool_params.md#Parameters.__init__), [`__init__`](../jaxite_bool/lut.md#LutCache.__init__)  (9 test-only)

