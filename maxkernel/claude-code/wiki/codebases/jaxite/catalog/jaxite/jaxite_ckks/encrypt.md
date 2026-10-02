---
title: 'Module: jaxite/jaxite_ckks/encrypt.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/encrypt.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.encrypt`/
symbols:
  Encrypt.encrypt: Encrypt#encrypt().
  Decrypt.decrypt: Decrypt#decrypt().
  Encrypt: Encrypt#
  Decrypt: Decrypt#
  EncryptBase.encrypt: EncryptBase#encrypt().
  DecryptBase.decrypt: DecryptBase#decrypt().
  Plaintext: Plaintext.
  Ciphertext: Ciphertext.
  EncryptBase: EncryptBase#
  DecryptBase: DecryptBase#
  PublicKey: PublicKey.
  SecretKey: SecretKey.
  Encrypt.public_key: Encrypt#public_key.
  Encrypt.__init__: Encrypt#__init__().
  Decrypt.__init__: Decrypt#__init__().
  ABC: ABC.
  abstractmethod: abstractmethod.
  Decrypt.secret_key: Decrypt#secret_key.
---
# Module: [`jaxite/jaxite_ckks/encrypt.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt.py)

## Classes
### `Decrypt`  ·  implements/extends DecryptBase
- def: [`jaxite/jaxite_ckks/encrypt.py:92`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt.py#L92)
- doc: Kernel for CKKS decryption.
- signature: `class Decrypt(DecryptBase):`
- members:
  - `decrypt(self, ciphertext: Ciphertext)` — [`L98`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt.py#L98) — documented in [jaxite-jaxite_ckks-encrypt](../../../concepts/jaxite-jaxite_ckks-encrypt.md)
  - `secret_key` — [`L96`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt.py#L96)
- protocol/private: `__init__`[`L95`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt.py#L95)
- uses (calls/refs, reference-scoped): [`data`](types.md#Ciphertext.data), [`moduli`](types.md#Ciphertext.moduli), [`data`](types.md#SecretKey.data), [`intt_negacyclic_poly`](ntt_cpu.md#intt_negacyclic_poly), [`moduli`](types.md#SecretKey.moduli), [`Ciphertext`](encrypt.md#Ciphertext), [`Plaintext`](encrypt.md#Plaintext), [`DecryptBase`](encrypt.md#DecryptBase), [`SecretKey`](encrypt.md#SecretKey)
- used by: [`test_encrypt_multiply_decrypt`](mul_test.md#CiphertextCiphertextMulTest.test_encrypt_multiply_decrypt), [`_run_conjugate_test`](conjugate_test.md#ConjugateTest._run_conjugate_test), [`test_conjugate_of_conjugate`](conjugate_test.md#ConjugateTest.test_conjugate_of_conjugate), [`test_key_switch`](key_switching_key_test.md#KeySwitchingTest.test_key_switch), [`test_blind_rotate_cm`](blind_rotate_test.md#BlindRotateTest.test_blind_rotate_cm), [`test_brot_mux_hypothesis`](blind_rotate_test.md#BlindRotationHypothesisTest.test_brot_mux_hypothesis), [`test_brot_mux`](blind_rotate_test.md#BlindRotateTest.test_brot_mux), [`test_rescale_different_scale`](rescale_test.md#RescaleTest.test_rescale_different_scale), [`test_rescale_e2e`](rescale_test.md#RescaleTest.test_rescale_e2e), [`test_brot_mux_cyclic_identity`](blind_rotate_test.md#BlindRotateTest.test_brot_mux_cyclic_identity), [`test_double_rescale`](rescale_test.md#RescaleTest.test_double_rescale), [`test_full_pipeline_add`](add_test.md#AddTest.test_full_pipeline_add), [`test_full_pipeline_mul`](mul_test.md#PlaintextCiphertextMulTest.test_full_pipeline_mul), [`test_gen_key_switching_key`](key_switching_key_test.md#KeySwitchingTest.test_gen_key_switching_key), [`test_brot_cm_hypothesis`](blind_rotate_test.md#BlindRotationHypothesisTest.test_brot_cm_hypothesis), [`test_encrypt_decrypt_3_moduli`](rescale_test.md#RescaleTest.test_encrypt_decrypt_3_moduli), [`test_rescale_different_degree`](rescale_test.md#RescaleTest.test_rescale_different_degree), [`test_rescale_hypothesis`](rescale_test.md#RescaleHypothesisTest.test_rescale_hypothesis), [`test_rescale_more_moduli`](rescale_test.md#RescaleTest.test_rescale_more_moduli), [`test_decrypt_equivalence`](cross_equivalence_test.md#CrossEquivalenceTest.test_decrypt_equivalence), [`test_encrypt_decrypt_loop`](encrypt_test.md#EncryptTest.test_encrypt_decrypt_loop), [`test_encrypt_decrypt_loop_with_hamming_weight`](encrypt_test.md#EncryptTest.test_encrypt_decrypt_loop_with_hamming_weight), [`test_decrypt_moduli_mismatch_raises_error`](encrypt_test.md#EncryptTest.test_decrypt_moduli_mismatch_raises_error), [`test_exact_encrypt_decrypt`](encrypt_test.md#EncryptTest.test_exact_encrypt_decrypt), [`test_column_key`](key_gen_test.md#KeyGenTest.test_column_key), [`decrypt`](encrypt.md#DecryptBase.decrypt), [`decryptor_q`](blind_rotate_test.md#BlindRotationHypothesisTest.decryptor_q), [`DecryptBase`](encrypt.md#DecryptBase), [`decryptor`](rescale_test.md#RescaleHypothesisTest.decryptor)

### `DecryptBase`  ·  implements/extends ABC
- def: [`jaxite/jaxite_ckks/encrypt.py:32`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt.py#L32)
- doc: Abstract base class for decryption kernels.
- signature: `class DecryptBase(ABC):`
- members:
  - `decrypt(self, ciphertext: Ciphertext)` — [`L36`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt.py#L36) — Decrypts a CKKS plaintext. — documented in [jaxite-jaxite_ckks-encrypt](../../../concepts/jaxite-jaxite_ckks-encrypt.md)
- uses (calls/refs, reference-scoped): [`decrypt`](encrypt.md#Decrypt.decrypt), [`Decrypt`](encrypt.md#Decrypt), [`Ciphertext`](encrypt.md#Ciphertext), [`Plaintext`](encrypt.md#Plaintext), [`ABC`](encrypt.md#ABC), [`abstractmethod`](encrypt.md#abstractmethod)
- used by: [`Decrypt`](encrypt.md#Decrypt)

### `Encrypt`  ·  implements/extends EncryptBase
- def: [`jaxite/jaxite_ckks/encrypt.py:40`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt.py#L40)
- doc: Kernel for CKKS encryption.
- signature: `class Encrypt(EncryptBase):`
- members:
  - `encrypt(self, plaintext: Plaintext, random_source: random.RandomSource | None = None)` — [`L46`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt.py#L46) — documented in [jaxite-jaxite_ckks-encrypt](../../../concepts/jaxite-jaxite_ckks-encrypt.md)
  - `public_key` — [`L44`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt.py#L44)
- protocol/private: `__init__`[`L43`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt.py#L43)
- uses (calls/refs, reference-scoped): [`data`](types.md#Plaintext.data), [`ntt_negacyclic_poly`](ntt_cpu.md#ntt_negacyclic_poly), [`RandomSource`](random.md#RandomSource), [`moduli`](types.md#Plaintext.moduli), [`SecureRandomSource`](random.md#SecureRandomSource), [`gen_gaussian_poly`](random.md#RandomSource.gen_gaussian_poly), [`gen_ternary_poly`](random.md#RandomSource.gen_ternary_poly), [`Ciphertext`](encrypt.md#Ciphertext), [`Plaintext`](encrypt.md#Plaintext), [`EncryptBase`](encrypt.md#EncryptBase), [`data`](types.md#PublicKey.data), [`moduli`](types.md#PublicKey.moduli), [`PublicKey`](encrypt.md#PublicKey)
- used by: [`test_encrypt_multiply_decrypt`](mul_test.md#CiphertextCiphertextMulTest.test_encrypt_multiply_decrypt), [`test_relinearize_batched`](mul_test.md#CiphertextCiphertextMulTest.test_relinearize_batched), [`_run_conjugate_test`](conjugate_test.md#ConjugateTest._run_conjugate_test), [`test_conjugate_of_conjugate`](conjugate_test.md#ConjugateTest.test_conjugate_of_conjugate), [`test_key_switch`](key_switching_key_test.md#KeySwitchingTest.test_key_switch), [`test_blind_rotate_cm`](blind_rotate_test.md#BlindRotateTest.test_blind_rotate_cm), [`test_brot_mux_hypothesis`](blind_rotate_test.md#BlindRotationHypothesisTest.test_brot_mux_hypothesis), [`test_brot_mux`](blind_rotate_test.md#BlindRotateTest.test_brot_mux), [`test_rescale_different_scale`](rescale_test.md#RescaleTest.test_rescale_different_scale), [`test_rescale_e2e`](rescale_test.md#RescaleTest.test_rescale_e2e), [`test_brot_mux_cyclic_identity`](blind_rotate_test.md#BlindRotateTest.test_brot_mux_cyclic_identity), [`test_double_rescale`](rescale_test.md#RescaleTest.test_double_rescale), [`test_full_pipeline_add`](add_test.md#AddTest.test_full_pipeline_add), [`test_full_pipeline_mul`](mul_test.md#PlaintextCiphertextMulTest.test_full_pipeline_mul), [`test_encrypt_decrypt_3_moduli`](rescale_test.md#RescaleTest.test_encrypt_decrypt_3_moduli), [`test_rescale_different_degree`](rescale_test.md#RescaleTest.test_rescale_different_degree), [`test_rescale_hypothesis`](rescale_test.md#RescaleHypothesisTest.test_rescale_hypothesis), [`test_rescale_more_moduli`](rescale_test.md#RescaleTest.test_rescale_more_moduli), [`sk_q`](blind_rotate_test.md#BlindRotationHypothesisTest.sk_q), [`test_encrypt_equivalence`](cross_equivalence_test.md#CrossEquivalenceTest.test_encrypt_equivalence), [`test_composition_equivalence`](cross_equivalence_test.md#CrossEquivalenceTest.test_composition_equivalence), [`test_encrypt_decrypt_loop`](encrypt_test.md#EncryptTest.test_encrypt_decrypt_loop), [`test_encrypt_decrypt_loop_with_hamming_weight`](encrypt_test.md#EncryptTest.test_encrypt_decrypt_loop_with_hamming_weight), [`test_decrypt_moduli_mismatch_raises_error`](encrypt_test.md#EncryptTest.test_decrypt_moduli_mismatch_raises_error), [`test_exact_encrypt_decrypt`](encrypt_test.md#EncryptTest.test_exact_encrypt_decrypt), [`gen_cm_keys`](key_gen.md#gen_cm_keys), [`test_jax_compatibility`](encrypt_test.md#EncryptTest.test_jax_compatibility), [`cmkey_j`](blind_rotate_test.md#BlindRotationHypothesisTest.cmkey_j), [`encrypt`](encrypt.md#EncryptBase.encrypt), [`test_encrypt_moduli_mismatch_raises_error`](encrypt_test.md#EncryptTest.test_encrypt_moduli_mismatch_raises_error), [`EncryptBase`](encrypt.md#EncryptBase), [`encryptor`](rescale_test.md#RescaleHypothesisTest.encryptor), [`encryptor_q`](blind_rotate_test.md#BlindRotationHypothesisTest.encryptor_q)

### `EncryptBase`  ·  implements/extends ABC
- def: [`jaxite/jaxite_ckks/encrypt.py:20`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt.py#L20)
- doc: Abstract base class for encryption kernels.
- signature: `class EncryptBase(ABC):`
- members:
  - `encrypt(self, plaintext: Plaintext, random_source: random.RandomSource | None = None)` — [`L24`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt.py#L24) — Encrypts a CKKS plaintext. — documented in [jaxite-jaxite_ckks-encrypt](../../../concepts/jaxite-jaxite_ckks-encrypt.md)
- uses (calls/refs, reference-scoped): [`encrypt`](encrypt.md#Encrypt.encrypt), [`Encrypt`](encrypt.md#Encrypt), [`RandomSource`](random.md#RandomSource), [`Ciphertext`](encrypt.md#Ciphertext), [`Plaintext`](encrypt.md#Plaintext), [`ABC`](encrypt.md#ABC), [`abstractmethod`](encrypt.md#abstractmethod)
- used by: [`Encrypt`](encrypt.md#Encrypt)

## Module values
- `ABC` — [`L16`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt.py#L16)
- `Ciphertext` — [`L13`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt.py#L13) — documented in [jaxite-jaxite_ckks-encrypt](../../../concepts/jaxite-jaxite_ckks-encrypt.md)
- `Plaintext` — [`L10`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt.py#L10)
- `PublicKey` — [`L11`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt.py#L11)
- `SecretKey` — [`L12`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt.py#L12)
- `abstractmethod` — [`L17`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encrypt.py#L17)

