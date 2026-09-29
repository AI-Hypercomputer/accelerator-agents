---
title: 'Module: jaxite/jaxite_ckks/encode.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/encode.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.encode`/
symbols:
  Encode.encode: Encode#encode().
  Decode.decode: Decode#decode().
  Encode: Encode#
  Decode: Decode#
  fft_special_inv: fft_special_inv().
  fft_special: fft_special().
  Plaintext: Plaintext.
  EncodeBase.encode: EncodeBase#encode().
  DecodeBase.decode: DecodeBase#decode().
  EncodeBase: EncodeBase#
  DecodeBase: DecodeBase#
  Encode.moduli: Encode#moduli.
  _roots: _roots().
  _rot_group: _rot_group().
  ABC: ABC.
  abstractmethod: abstractmethod.
  Encode.degree: Encode#degree.
  Decode.scale: Decode#scale.
  Decode.num_slots: Decode#num_slots.
  Encode.scale: Encode#scale.
  Encode.__init__: Encode#__init__().
  Decode.__init__: Decode#__init__().
---
# Module: [`jaxite/jaxite_ckks/encode.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode.py)

## Classes
### `Decode`  ·  implements/extends DecodeBase
- def: [`jaxite/jaxite_ckks/encode.py:162`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode.py#L162) — documented in [jaxite-jaxite_ckks-encode](../../../concepts/jaxite-jaxite_ckks-encode.md)
- doc: Kernel for CKKS decoding.
- signature: `class Decode(DecodeBase):`
- members:
  - `decode(self, plaintext: Plaintext, is_slot_form: bool = False)` — [`L169`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode.py#L169) — documented in [jaxite-jaxite_ckks-encode](../../../concepts/jaxite-jaxite_ckks-encode.md)
  - `num_slots` — [`L167`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode.py#L167)
  - `scale` — [`L166`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode.py#L166)
- protocol/private: `__init__`[`L165`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode.py#L165)
- uses (calls/refs, reference-scoped): [`data`](types.md#Plaintext.data), [`moduli`](types.md#Plaintext.moduli), [`intt_negacyclic_poly`](ntt_cpu.md#intt_negacyclic_poly), [`fft_special`](encode.md#fft_special), [`Plaintext`](encode.md#Plaintext), [`reconstruct_crt`](rns_utils.md#reconstruct_crt), [`DecodeBase`](encode.md#DecodeBase)
- used by: [`test_encrypt_multiply_decrypt`](mul_test.md#CiphertextCiphertextMulTest.test_encrypt_multiply_decrypt), [`_run_conjugate_test`](conjugate_test.md#ConjugateTest._run_conjugate_test), [`test_conjugate_of_conjugate`](conjugate_test.md#ConjugateTest.test_conjugate_of_conjugate), [`test_blind_rotate_cm`](blind_rotate_test.md#BlindRotateTest.test_blind_rotate_cm), [`test_brot_mux_hypothesis`](blind_rotate_test.md#BlindRotationHypothesisTest.test_brot_mux_hypothesis), [`test_brot_mux`](blind_rotate_test.md#BlindRotateTest.test_brot_mux), [`test_rescale_different_scale`](rescale_test.md#RescaleTest.test_rescale_different_scale), [`test_rescale_e2e`](rescale_test.md#RescaleTest.test_rescale_e2e), [`test_brot_mux_cyclic_identity`](blind_rotate_test.md#BlindRotateTest.test_brot_mux_cyclic_identity), [`test_double_rescale`](rescale_test.md#RescaleTest.test_double_rescale), [`test_full_pipeline_add`](add_test.md#AddTest.test_full_pipeline_add), [`test_full_pipeline_mul`](mul_test.md#PlaintextCiphertextMulTest.test_full_pipeline_mul), [`test_brot_cm_hypothesis`](blind_rotate_test.md#BlindRotationHypothesisTest.test_brot_cm_hypothesis), [`test_encrypt_decrypt_3_moduli`](rescale_test.md#RescaleTest.test_encrypt_decrypt_3_moduli), [`test_rescale_different_degree`](rescale_test.md#RescaleTest.test_rescale_different_degree), [`test_rescale_hypothesis`](rescale_test.md#RescaleHypothesisTest.test_rescale_hypothesis), [`test_rescale_more_moduli`](rescale_test.md#RescaleTest.test_rescale_more_moduli), [`test_encrypt_decrypt_loop`](encrypt_test.md#EncryptTest.test_encrypt_decrypt_loop), [`test_encrypt_decrypt_loop_with_hamming_weight`](encrypt_test.md#EncryptTest.test_encrypt_decrypt_loop_with_hamming_weight), [`test_column_key`](key_gen_test.md#KeyGenTest.test_column_key), [`test_encode_diff`](encode_test.md#CrossDiffTest.test_encode_diff), [`decoder`](blind_rotate_test.md#BlindRotationHypothesisTest.decoder), [`test_encode_decode_loop`](encode_test.md#EncodeTest.test_encode_decode_loop), [`decode`](encode.md#DecodeBase.decode), [`DecodeBase`](encode.md#DecodeBase), [`decoder_cm`](blind_rotate_test.md#BlindRotationHypothesisTest.decoder_cm)

### `DecodeBase`  ·  implements/extends ABC
- def: [`jaxite/jaxite_ckks/encode.py:116`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode.py#L116) — documented in [jaxite-jaxite_ckks-encode](../../../concepts/jaxite-jaxite_ckks-encode.md)
- doc: Abstract base class for decoding kernels.
- signature: `class DecodeBase(ABC):`
- members:
  - `decode(self, plaintext: Plaintext, is_slot_form: bool = False)` — [`L120`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode.py#L120) — Decode an RNS-CKKS plaintext into a cleartext vector. — documented in [jaxite-jaxite_ckks-encode](../../../concepts/jaxite-jaxite_ckks-encode.md)
- uses (calls/refs, reference-scoped): [`decode`](encode.md#Decode.decode), [`Decode`](encode.md#Decode), [`Plaintext`](encode.md#Plaintext), [`ABC`](encode.md#ABC), [`abstractmethod`](encode.md#abstractmethod)
- used by: [`Decode`](encode.md#Decode)

### `Encode`  ·  implements/extends EncodeBase
- def: [`jaxite/jaxite_ckks/encode.py:126`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode.py#L126) — documented in [jaxite-jaxite_ckks-encode](../../../concepts/jaxite-jaxite_ckks-encode.md)
- doc: Kernel for CKKS encoding.
- signature: `class Encode(EncodeBase):`
- members:
  - `encode(self, slots: list[complex])` — [`L134`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode.py#L134) — Encode a cleartext list of slots into a plaintext. — documented in [jaxite-jaxite_ckks-encode](../../../concepts/jaxite-jaxite_ckks-encode.md)
  - `degree` — [`L130`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode.py#L130)
  - `moduli` — [`L131`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode.py#L131)
  - `scale` — [`L132`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode.py#L132)
- protocol/private: `__init__`[`L129`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode.py#L129)
- uses (calls/refs, reference-scoped): [`ntt_negacyclic_poly`](ntt_cpu.md#ntt_negacyclic_poly), [`fft_special_inv`](encode.md#fft_special_inv), [`Plaintext`](encode.md#Plaintext), [`EncodeBase`](encode.md#EncodeBase)
- used by: [`test_encrypt_multiply_decrypt`](mul_test.md#CiphertextCiphertextMulTest.test_encrypt_multiply_decrypt), [`test_relinearize_batched`](mul_test.md#CiphertextCiphertextMulTest.test_relinearize_batched), [`_run_conjugate_test`](conjugate_test.md#ConjugateTest._run_conjugate_test), [`test_conjugate_of_conjugate`](conjugate_test.md#ConjugateTest.test_conjugate_of_conjugate), [`test_key_switch`](key_switching_key_test.md#KeySwitchingTest.test_key_switch), [`test_blind_rotate_cm`](blind_rotate_test.md#BlindRotateTest.test_blind_rotate_cm), [`test_brot_mux_hypothesis`](blind_rotate_test.md#BlindRotationHypothesisTest.test_brot_mux_hypothesis), [`test_brot_mux`](blind_rotate_test.md#BlindRotateTest.test_brot_mux), [`test_rescale_different_scale`](rescale_test.md#RescaleTest.test_rescale_different_scale), [`test_rescale_e2e`](rescale_test.md#RescaleTest.test_rescale_e2e), [`test_brot_mux_cyclic_identity`](blind_rotate_test.md#BlindRotateTest.test_brot_mux_cyclic_identity), [`test_double_rescale`](rescale_test.md#RescaleTest.test_double_rescale), [`test_full_pipeline_add`](add_test.md#AddTest.test_full_pipeline_add), [`test_full_pipeline_mul`](mul_test.md#PlaintextCiphertextMulTest.test_full_pipeline_mul), [`test_brot_cm_hypothesis`](blind_rotate_test.md#BlindRotationHypothesisTest.test_brot_cm_hypothesis), [`test_encrypt_decrypt_3_moduli`](rescale_test.md#RescaleTest.test_encrypt_decrypt_3_moduli), [`test_rescale_different_degree`](rescale_test.md#RescaleTest.test_rescale_different_degree), [`test_rescale_hypothesis`](rescale_test.md#RescaleHypothesisTest.test_rescale_hypothesis), [`test_rescale_more_moduli`](rescale_test.md#RescaleTest.test_rescale_more_moduli), [`sk_q`](blind_rotate_test.md#BlindRotationHypothesisTest.sk_q), [`test_composition_equivalence`](cross_equivalence_test.md#CrossEquivalenceTest.test_composition_equivalence), [`test_encrypt_decrypt_loop`](encrypt_test.md#EncryptTest.test_encrypt_decrypt_loop), [`test_encrypt_decrypt_loop_with_hamming_weight`](encrypt_test.md#EncryptTest.test_encrypt_decrypt_loop_with_hamming_weight), [`test_decrypt_moduli_mismatch_raises_error`](encrypt_test.md#EncryptTest.test_decrypt_moduli_mismatch_raises_error), [`gen_cm_keys`](key_gen.md#gen_cm_keys), [`test_jax_compatibility`](encrypt_test.md#EncryptTest.test_jax_compatibility), [`test_encode_diff`](encode_test.md#CrossDiffTest.test_encode_diff), [`encoder_pq`](blind_rotate_test.md#BlindRotationHypothesisTest.encoder_pq), [`encoder_q`](blind_rotate_test.md#BlindRotationHypothesisTest.encoder_q), [`test_encode_decode_loop`](encode_test.md#EncodeTest.test_encode_decode_loop), [`test_encode_equivalence`](cross_equivalence_test.md#CrossEquivalenceTest.test_encode_equivalence), [`encode`](encode.md#EncodeBase.encode), [`EncodeBase`](encode.md#EncodeBase)

### `EncodeBase`  ·  implements/extends ABC
- def: [`jaxite/jaxite_ckks/encode.py:108`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode.py#L108) — documented in [jaxite-jaxite_ckks-encode](../../../concepts/jaxite-jaxite_ckks-encode.md)
- doc: Abstract base class for encoding kernels.
- signature: `class EncodeBase(ABC):`
- members:
  - `encode(self, slots: list[complex])` — [`L112`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode.py#L112) — Encode a cleartext vector into an RNS-CKKS plaintext. — documented in [jaxite-jaxite_ckks-encode](../../../concepts/jaxite-jaxite_ckks-encode.md)
- uses (calls/refs, reference-scoped): [`encode`](encode.md#Encode.encode), [`Encode`](encode.md#Encode), [`Plaintext`](encode.md#Plaintext), [`ABC`](encode.md#ABC), [`abstractmethod`](encode.md#abstractmethod)
- used by: [`Encode`](encode.md#Encode)

## Functions
- `_roots(m: int)` — [`L15`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode.py#L15)
- `_rot_group(m: int, nh: int, g: int = 5)` — [`L19`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode.py#L19)
- `fft_special(vals: np.ndarray, cycl_order: int)` — [`L66`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode.py#L66) — Compute a specialized FFT for CKKS decoding. — documented in [jaxite-jaxite_ckks-encode](../../../concepts/jaxite-jaxite_ckks-encode.md)
- `fft_special_inv(vals: np.ndarray, cycl_order: int)` — [`L26`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode.py#L26) — Compute a specialized iFFT for CKKS encoding. — documented in [jaxite-jaxite_ckks-encode](../../../concepts/jaxite-jaxite_ckks-encode.md)

## Module values
- `ABC` — [`L104`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode.py#L104)
- `Plaintext` — [`L12`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode.py#L12) — documented in [jaxite-jaxite_ckks-encode](../../../concepts/jaxite-jaxite_ckks-encode.md)
- `abstractmethod` — [`L105`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/encode.py#L105)

