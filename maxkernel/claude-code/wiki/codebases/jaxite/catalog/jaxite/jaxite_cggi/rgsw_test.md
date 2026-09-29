---
title: 'Module: jaxite/jaxite_cggi/rgsw_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_cggi/rgsw_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_cggi.rgsw_test`/RgswTest#
symbols:
  RgswTest.test_encrypt_decrypt_with_varying_rngs: test_encrypt_decrypt_with_varying_rngs().
  RgswTest.test_encrypt_decrypt_with_varying_nonzero_sk_entry: test_encrypt_decrypt_with_varying_nonzero_sk_entry().
  RgswTest.test_encrypt_diagonal: test_encrypt_diagonal().
  RgswTest.test_encrypt_decrypt_prod_security_params: test_encrypt_decrypt_prod_security_params().
  RgswTest.setUp: setUp().
  RgswTest.test_encrypt_dimension: test_encrypt_dimension().
  RgswTest.test_encrypt_decrypt_no_noise: test_encrypt_decrypt_no_noise().
  RgswTest.decomposition_params: decomposition_params.
  RgswTest.test_rgsw_ciphertext_str: test_rgsw_ciphertext_str().
  RgswTest.polynomial_modulus_degree: polynomial_modulus_degree.
  RgswTest.noise_free_rng: noise_free_rng.
  RgswTest.default_key: default_key.
  RgswTest.rlwe_dimension: rlwe_dimension.
  RgswTest.dim: dim.
  RgswTest.plaintext_modulus: plaintext_modulus.
  RgswTest: ''
---
# Module: [`jaxite/jaxite_cggi/rgsw_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw_test.py)

## Classes
### `RgswTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_cggi/rgsw_test.py:14`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw_test.py#L14)
- signature: `class RgswTest(parameterized.TestCase):`
- members:
  - `setUp(self)` — [`L16`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw_test.py#L16)
  - `test_encrypt_decrypt_no_noise(self, plaintext: int)` — [`L130`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw_test.py#L130) — Checks that encrypt/decrypt works using noise-free RLWE samples.
  - `test_encrypt_decrypt_prod_security_params(self)` — [`L255`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw_test.py#L255)
  - `test_encrypt_decrypt_with_varying_nonzero_sk_entry(self, sk_nonzero_index)` — [`L205`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw_test.py#L205) — Checks that an encrypt/decrypt round works with different secret keys.
  - `test_encrypt_decrypt_with_varying_rngs(self, log_ai_bound: int, normal_std: int, seed: int)` — [`L159`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw_test.py#L159) — Checks that encrypt/decrypt works with a variety of RNG parameters.
  - `test_encrypt_diagonal(self, plaintext: int)` — [`L81`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw_test.py#L81) — Checks that the "diagonals" are properly set.
  - `test_encrypt_dimension(self, plaintext: rgsw.RgswPlaintext)` — [`L59`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw_test.py#L59)
  - `test_rgsw_ciphertext_str(self, coeffs: list[int], expected: str)` — [`L48`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw_test.py#L48)
  - `decomposition_params` — [`L20`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw_test.py#L20)
  - `default_key` — [`L26`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw_test.py#L26)
  - `dim` — [`L18`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw_test.py#L18)
  - `noise_free_rng` — [`L25`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw_test.py#L25)
  - `plaintext_modulus` — [`L19`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw_test.py#L19)
  - `polynomial_modulus_degree` — [`L23`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw_test.py#L23)
  - `rlwe_dimension` — [`L24`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/rgsw_test.py#L24)
- uses (calls/refs, reference-scoped): [`encrypt`](rgsw.md#encrypt), [`decrypt`](rgsw.md#decrypt), [`SchemeParameters`](parameters.md#SchemeParameters), [`DecompositionParameters`](decomposition.md#DecompositionParameters), [`polynomial_modulus_degree`](parameters.md#SchemeParameters.polynomial_modulus_degree), [`PseudorandomSource`](random_source.md#PseudorandomSource), [`lwe_dimension`](parameters.md#SchemeParameters.lwe_dimension), [`data`](rlwe.md#RlweSecretKey.data), [`level_count`](decomposition.md#DecompositionParameters.level_count), [`log_base`](decomposition.md#DecompositionParameters.log_base), [`rlwe_dimension`](parameters.md#SchemeParameters.rlwe_dimension), [`plaintext_modulus`](parameters.md#SchemeParameters.plaintext_modulus), [`key`](rgsw.md#RgswSecretKey.key), [`message`](rgsw.md#RgswPlaintext.message), [`gen_key`](rgsw.md#gen_key), [`RgswPlaintext`](rgsw.md#RgswPlaintext), [`modulus_degree`](rgsw.md#RgswPlaintext.modulus_degree), [`CycleRng`](random_source.md#CycleRng), [`message`](rgsw.md#RgswCiphertext.message), [`rlwe_dimension`](rlwe.md#RlweSecretKey.rlwe_dimension), [`RgswCiphertext`](rgsw.md#RgswCiphertext), [`log_coefficient_modulus`](rgsw.md#RgswCiphertext.log_coefficient_modulus), [`modulus_degree`](rgsw.md#RgswCiphertext.modulus_degree)  (2 test-only)

