---
title: 'Module: jaxite/jaxite_ckks/types.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/types.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.types`/
symbols:
  Ciphertext.data: Ciphertext#data.
  Ciphertext.moduli: Ciphertext#moduli.
  Ciphertext: Ciphertext#
  Plaintext.data: Plaintext#data.
  SecretKey.data: SecretKey#data.
  Plaintext.moduli: Plaintext#moduli.
  Plaintext: Plaintext#
  SecretKey: SecretKey#
  SecretKey.moduli: SecretKey#moduli.
  EvaluationKeys: EvaluationKeys#
  HMuxRotKey.key0: HMuxRotKey#key0.
  HMuxRotKey.key1: HMuxRotKey#key1.
  PublicKey.data: PublicKey#data.
  Plaintext.tree_flatten: Plaintext#tree_flatten().
  Ciphertext.tree_flatten: Ciphertext#tree_flatten().
  PublicKey: PublicKey#
  EvaluationKeys.a: EvaluationKeys#a.
  EvaluationKeys.b: EvaluationKeys#b.
  HMuxRotKey: HMuxRotKey#
  HMuxRotKey.tree_flatten: HMuxRotKey#tree_flatten().
  MuxRotationKey.keys: MuxRotationKey#keys.
  PublicKey.moduli: PublicKey#moduli.
  EvaluationKeys.moduli: EvaluationKeys#moduli.
  MuxRotationKey: MuxRotationKey#
  MuxRotationKey.tree_flatten: MuxRotationKey#tree_flatten().
  Plaintext.tree_unflatten: Plaintext#tree_unflatten().
  Ciphertext.tree_unflatten: Ciphertext#tree_unflatten().
  HMuxRotKey.tree_unflatten: HMuxRotKey#tree_unflatten().
  MuxRotationKey.tree_unflatten: MuxRotationKey#tree_unflatten().
---
# Module: [`jaxite/jaxite_ckks/types.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py)

## Classes
### `Ciphertext`
- def: [`jaxite/jaxite_ckks/types.py:26`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L26) — documented in [jaxite-jaxite_ckks-types](../../../concepts/jaxite-jaxite_ckks-types.md)
- doc: A CKKS Ciphertext.
- signature: `class Ciphertext:`
- members:
  - `tree_flatten(self)` — [`L32`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L32) — documented in [jaxite-jaxite_ckks-types](../../../concepts/jaxite-jaxite_ckks-types.md)
  - `tree_unflatten(cls, _, children)` — [`L36`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L36)
  - `data` — [`L29`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L29) — documented in [jaxite-jaxite_ckks-types](../../../concepts/jaxite-jaxite_ckks-types.md)
  - `moduli` — [`L30`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L30) — documented in [jaxite-jaxite_ckks-types](../../../concepts/jaxite-jaxite_ckks-types.md)
- used by: [`relinearize`](mul.md#Mul.relinearize), [`test_relinearize_batched`](mul_test.md#CiphertextCiphertextMulTest.test_relinearize_batched), [`key_switch`](key_switching.md#KeySwitcher.key_switch), [`rescale`](rescale.md#Rescale.rescale), [`gen_hmuxrot_key`](key_gen.md#gen_hmuxrot_key), [`decrypt`](encrypt.md#Decrypt.decrypt), [`hmuxrot`](blind_rotate.md#BlindRotation.hmuxrot), [`rescale_kernel`](rescale_test.md#RescaleTest.rescale_kernel), [`test_double_rescale`](rescale_test.md#RescaleTest.test_double_rescale), [`test_full_pipeline_add`](add_test.md#AddTest.test_full_pipeline_add), [`test_gen_key_switching_key`](key_switching_key_test.md#KeySwitchingTest.test_gen_key_switching_key), [`brot_cm`](blind_rotate.md#BlindRotation.brot_cm), [`rescale_kernel`](rescale_test.md#RescaleHypothesisTest.rescale_kernel), [`test_encrypt_equivalence`](cross_equivalence_test.md#CrossEquivalenceTest.test_encrypt_equivalence), [`conjugate`](conjugate.md#Conjugation.conjugate), [`test_mul_equivalence`](cross_equivalence_test.md#CrossEquivalenceTest.test_mul_equivalence), [`test_composition_equivalence`](cross_equivalence_test.md#CrossEquivalenceTest.test_composition_equivalence), [`test_decrypt_equivalence`](cross_equivalence_test.md#CrossEquivalenceTest.test_decrypt_equivalence), [`mul`](mul.md#MulPlaintextCiphertextBarrett.mul), [`test_decrypt_moduli_mismatch_raises_error`](encrypt_test.md#EncryptTest.test_decrypt_moduli_mismatch_raises_error), [`lift_ciphertext`](blind_rotate_utils.md#lift_ciphertext), [`test_lift_ciphertext`](blind_rotate_utils_test.md#BlindRotateUtilsTest.test_lift_ciphertext), [`test_mul_modular_barrett_hypothesis`](mul_test.md#PlaintextCiphertextMulTest.test_mul_modular_barrett_hypothesis), [`test_mul_simple`](mul_test.md#PlaintextCiphertextMulTest.test_mul_simple), [`brot_mux`](blind_rotate.md#BlindRotation.brot_mux), [`test_jax_compatibility`](encrypt_test.md#EncryptTest.test_jax_compatibility), [`mul`](mul.md#MulPlaintextCiphertextSimple.mul), [`test_rescale_equivalence`](rescale_test.md#RescaleTest.test_rescale_equivalence), [`tensor_multiply`](mul.md#Mul.tensor_multiply), [`mul`](mul.md#MulPlaintextCiphertextBase.mul), [`Ciphertext`](mul.md#Ciphertext), [`Ciphertext`](encrypt.md#Ciphertext), [`key0`](types.md#HMuxRotKey.key0), [`key1`](types.md#HMuxRotKey.key1), [`Ciphertext`](rescale.md#Ciphertext)

### `EvaluationKeys`
- def: [`jaxite/jaxite_ckks/types.py:57`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L57)
- doc: CKKS Evaluation Keys.
- signature: `class EvaluationKeys:`
- members:
  - `a` — [`L60`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L60)
  - `b` — [`L61`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L61)
  - `moduli` — [`L62`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L62)
- used by: [`key_switch`](key_switching.md#KeySwitcher.key_switch), [`gen_hmuxrot_key`](key_gen.md#gen_hmuxrot_key), [`hmuxrot`](blind_rotate.md#BlindRotation.hmuxrot), [`test_gen_key_switching_key`](key_switching_key_test.md#KeySwitchingTest.test_gen_key_switching_key), [`gen_key_switching_key`](key_gen.md#gen_key_switching_key), [`conjugate`](conjugate.md#Conjugation.conjugate), [`gen_conjugate_key`](key_gen.md#gen_conjugate_key), [`gen_evaluation_key`](key_gen.md#gen_evaluation_key)

### `HMuxRotKey`
- def: [`jaxite/jaxite_ckks/types.py:67`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L67)
- doc: A key used in a single HMuxRot step.
- signature: `class HMuxRotKey:`
- members:
  - `tree_flatten(self)` — [`L79`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L79)
  - `tree_unflatten(cls, _, children)` — [`L83`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L83)
  - `key0` — [`L76`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L76) — documented in [jaxite-jaxite_ckks-types](../../../concepts/jaxite-jaxite_ckks-types.md)
  - `key1` — [`L77`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L77) — documented in [jaxite-jaxite_ckks-types](../../../concepts/jaxite-jaxite_ckks-types.md)
- uses (calls/refs, reference-scoped): [`Ciphertext`](types.md#Ciphertext)
- used by: [`gen_hmuxrot_key`](key_gen.md#gen_hmuxrot_key), [`hmuxrot`](blind_rotate.md#BlindRotation.hmuxrot), [`keys`](types.md#MuxRotationKey.keys)

### `MuxRotationKey`
- def: [`jaxite/jaxite_ckks/types.py:89`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L89)
- doc: A set of HMuxRot keys for all bits of a secret rotation index.
- signature: `class MuxRotationKey:`
- members:
  - `tree_flatten(self)` — [`L98`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L98)
  - `tree_unflatten(cls, _, children)` — [`L102`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L102)
  - `keys` — [`L96`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L96)
- uses (calls/refs, reference-scoped): [`HMuxRotKey`](types.md#HMuxRotKey)
- used by: [`brot_mux`](blind_rotate.md#BlindRotation.brot_mux), [`gen_mux_rotation_key`](key_gen.md#gen_mux_rotation_key)

### `Plaintext`
- def: [`jaxite/jaxite_ckks/types.py:10`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L10)
- doc: A CKKS Plaintext.
- signature: `class Plaintext:`
- members:
  - `tree_flatten(self)` — [`L16`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L16)
  - `tree_unflatten(cls, _, children)` — [`L20`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L20)
  - `data` — [`L13`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L13) — documented in [jaxite-jaxite_ckks-types](../../../concepts/jaxite-jaxite_ckks-types.md)
  - `moduli` — [`L14`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L14)
- used by: [`test_encrypt_multiply_decrypt`](mul_test.md#CiphertextCiphertextMulTest.test_encrypt_multiply_decrypt), [`encrypt`](encrypt.md#Encrypt.encrypt), [`key_switch`](key_switching.md#KeySwitcher.key_switch), [`test_key_switch`](key_switching_key_test.md#KeySwitchingTest.test_key_switch), [`decode`](encode.md#Decode.decode), [`test_gen_key_switching_key`](key_switching_key_test.md#KeySwitchingTest.test_gen_key_switching_key), [`brot_cm`](blind_rotate.md#BlindRotation.brot_cm), [`test_encrypt_equivalence`](cross_equivalence_test.md#CrossEquivalenceTest.test_encrypt_equivalence), [`test_decrypt_equivalence`](cross_equivalence_test.md#CrossEquivalenceTest.test_decrypt_equivalence), [`mul`](mul.md#MulPlaintextCiphertextBarrett.mul), [`test_exact_encrypt_decrypt`](encrypt_test.md#EncryptTest.test_exact_encrypt_decrypt), [`test_mul_modular_barrett_hypothesis`](mul_test.md#PlaintextCiphertextMulTest.test_mul_modular_barrett_hypothesis), [`test_mul_simple`](mul_test.md#PlaintextCiphertextMulTest.test_mul_simple), [`mul`](mul.md#MulPlaintextCiphertextSimple.mul), [`mul`](mul.md#MulPlaintextCiphertextBase.mul), [`test_encode_diff`](encode_test.md#CrossDiffTest.test_encode_diff), [`test_encode_equivalence`](cross_equivalence_test.md#CrossEquivalenceTest.test_encode_equivalence), [`test_encrypt_moduli_mismatch_raises_error`](encrypt_test.md#EncryptTest.test_encrypt_moduli_mismatch_raises_error), [`Plaintext`](encode.md#Plaintext), [`Plaintext`](encrypt.md#Plaintext)

### `PublicKey`
- def: [`jaxite/jaxite_ckks/types.py:41`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L41)
- doc: A CKKS Public Key.
- signature: `class PublicKey:`
- members:
  - `data` — [`L44`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L44)
  - `moduli` — [`L45`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L45)
- used by: [`encrypt`](encrypt.md#Encrypt.encrypt), [`test_encrypt_equivalence`](cross_equivalence_test.md#CrossEquivalenceTest.test_encrypt_equivalence), [`test_composition_equivalence`](cross_equivalence_test.md#CrossEquivalenceTest.test_composition_equivalence), [`gen_cm_keys`](key_gen.md#gen_cm_keys), [`pk_q`](blind_rotate_test.md#BlindRotationHypothesisTest.pk_q), [`PublicKey`](key_gen.md#PublicKey), [`PublicKey`](encrypt.md#PublicKey)

### `SecretKey`
- def: [`jaxite/jaxite_ckks/types.py:49`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L49)
- doc: A CKKS Secret Key.
- signature: `class SecretKey:`
- members:
  - `data` — [`L52`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L52) — documented in [jaxite-jaxite_ckks-encrypt](../../../concepts/jaxite-jaxite_ckks-encrypt.md)
  - `moduli` — [`L53`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/types.py#L53)
- used by: [`test_encrypt_multiply_decrypt`](mul_test.md#CiphertextCiphertextMulTest.test_encrypt_multiply_decrypt), [`test_relinearize_batched`](mul_test.md#CiphertextCiphertextMulTest.test_relinearize_batched), [`gen_hmuxrot_key`](key_gen.md#gen_hmuxrot_key), [`decrypt`](encrypt.md#Decrypt.decrypt), [`test_blind_rotate_cm`](blind_rotate_test.md#BlindRotateTest.test_blind_rotate_cm), [`sk_q`](blind_rotate_test.md#BlindRotationHypothesisTest.sk_q), [`gen_key_switching_key`](key_gen.md#gen_key_switching_key), [`test_decrypt_equivalence`](cross_equivalence_test.md#CrossEquivalenceTest.test_decrypt_equivalence), [`test_column_key`](key_gen_test.md#KeyGenTest.test_column_key), [`extend_secret_key`](key_gen.md#extend_secret_key), [`gen_conjugate_key`](key_gen.md#gen_conjugate_key), [`gen_evaluation_key`](key_gen.md#gen_evaluation_key), [`gen_mux_rotation_key`](key_gen.md#gen_mux_rotation_key), [`SecretKey`](key_gen.md#SecretKey), [`test_keygen_with_hamming_weight`](key_gen_test.md#KeyGenTest.test_keygen_with_hamming_weight), [`compute_scaled_source_key_partition`](key_gen.md#compute_scaled_source_key_partition), [`SecretKey`](encrypt.md#SecretKey)

