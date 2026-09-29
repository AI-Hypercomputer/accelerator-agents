---
title: 'Module: jaxite/jaxite_ckks/blind_rotate_test.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/blind_rotate_test.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.blind_rotate_test`/
symbols:
  BlindRotateTest.test_blind_rotate_cm: BlindRotateTest#test_blind_rotate_cm().
  BlindRotationHypothesisTest.test_brot_mux_hypothesis: BlindRotationHypothesisTest#test_brot_mux_hypothesis().
  BlindRotateTest.test_brot_mux: BlindRotateTest#test_brot_mux().
  BlindRotateTest.test_brot_mux_cyclic_identity: BlindRotateTest#test_brot_mux_cyclic_identity().
  BlindRotationHypothesisTest.test_brot_cm_hypothesis: BlindRotationHypothesisTest#test_brot_cm_hypothesis().
  BlindRotationHypothesisTest.sk_q: BlindRotationHypothesisTest#sk_q.
  BlindRotationHypothesisTest.cmkey_j: BlindRotationHypothesisTest#cmkey_j.
  BlindRotationHypothesisTest.brot_kernel: BlindRotationHypothesisTest#brot_kernel.
  BlindRotationHypothesisTest.mux_key: BlindRotationHypothesisTest#mux_key.
  BlindRotationHypothesisTest.NUM_SLOTS: BlindRotationHypothesisTest#NUM_SLOTS.
  BlindRotationHypothesisTest.sk_pq: BlindRotationHypothesisTest#sk_pq.
  BlindRotationHypothesisTest.pk_q: BlindRotationHypothesisTest#pk_q.
  BlindRotationHypothesisTest.ntt_pq: BlindRotationHypothesisTest#ntt_pq.
  BlindRotationHypothesisTest.encoder_pq: BlindRotationHypothesisTest#encoder_pq.
  BlindRotationHypothesisTest.encoder_q: BlindRotationHypothesisTest#encoder_q.
  BlindRotationHypothesisTest.decoder: BlindRotationHypothesisTest#decoder.
  BlindRotationHypothesisTest.ALL_MODULI: BlindRotationHypothesisTest#ALL_MODULI.
  BlindRotationHypothesisTest.test_random_source: BlindRotationHypothesisTest#test_random_source.
  BlindRotationHypothesisTest.decryptor_q: BlindRotationHypothesisTest#decryptor_q.
  _cyclic_roll: _cyclic_roll().
  BlindRotationHypothesisTest.Q_LIMBS: BlindRotationHypothesisTest#Q_LIMBS.
  BlindRotationHypothesisTest.DEGREE: BlindRotationHypothesisTest#DEGREE.
  BlindRotationHypothesisTest.SCALE: BlindRotationHypothesisTest#SCALE.
  BlindRotationHypothesisTest.encryptor_q: BlindRotationHypothesisTest#encryptor_q.
  BlindRotationHypothesisTest.decoder_cm: BlindRotationHypothesisTest#decoder_cm.
  BlindRotationHypothesisTest.P_LIMBS: BlindRotationHypothesisTest#P_LIMBS.
  BlindRotationHypothesisTest.SECRET_IDX: BlindRotationHypothesisTest#SECRET_IDX.
  BlindRotationHypothesisTest.pk_pq: BlindRotationHypothesisTest#pk_pq.
  BlindRotateTest: BlindRotateTest#
  BlindRotationHypothesisTest: BlindRotationHypothesisTest#
  BlindRotationHypothesisTest.THETA: BlindRotationHypothesisTest#THETA.
  BlindRotationHypothesisTest.setUpClass: BlindRotationHypothesisTest#setUpClass().
---
# Module: [`jaxite/jaxite_ckks/blind_rotate_test.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py)

## Classes
### `BlindRotateTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_ckks/blind_rotate_test.py:39`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L39)
- signature: `class BlindRotateTest(parameterized.TestCase):`
- members:
  - `test_blind_rotate_cm(self, num_slots, secret_idx)` — [`L48`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L48)
  - `test_brot_mux(self, num_slots, secret_idx)` — [`L134`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L134)
  - `test_brot_mux_cyclic_identity(self, num_slots, r)` — [`L205`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L205)
- uses (calls/refs, reference-scoped): [`encrypt`](encrypt.md#Encrypt.encrypt), [`encode`](encode.md#Encode.encode), [`keygen`](key_gen.md#keygen), [`decrypt`](encrypt.md#Decrypt.decrypt), [`decode`](encode.md#Decode.decode), [`precompute_constants`](blind_rotate.md#BlindRotation.precompute_constants), [`brot_cm`](blind_rotate.md#BlindRotation.brot_cm), [`Encode`](encode.md#Encode), [`Encrypt`](encrypt.md#Encrypt), [`Decrypt`](encrypt.md#Decrypt), [`Decode`](encode.md#Decode), [`data`](types.md#SecretKey.data), [`brot_mux`](blind_rotate.md#BlindRotation.brot_mux), [`gen_mux_rotation_key`](key_gen.md#gen_mux_rotation_key), [`SecretKey`](types.md#SecretKey), [`moduli`](types.md#SecretKey.moduli), [`ZeroNoiseRandomSource`](random.md#ZeroNoiseRandomSource), [`_cyclic_roll`](blind_rotate_test.md#_cyclic_roll), [`BlindRotation`](blind_rotate.md#BlindRotation)

### `BlindRotationHypothesisTest`  ·  implements/extends TestCase
- def: [`jaxite/jaxite_ckks/blind_rotate_test.py:290`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L290)
- signature: `class BlindRotationHypothesisTest(absltest.TestCase):`
- members:
  - `setUpClass(cls)` — [`L302`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L302)
  - `test_brot_cm_hypothesis(self, slots)` — [`L373`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L373)
  - `test_brot_mux_hypothesis(self, slots)` — [`L401`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L401)
  - `ALL_MODULI` — [`L294`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L294)
  - `DEGREE` — [`L295`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L295)
  - `NUM_SLOTS` — [`L296`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L296)
  - `P_LIMBS` — [`L293`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L293)
  - `Q_LIMBS` — [`L292`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L292)
  - `SCALE` — [`L297`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L297)
  - `SECRET_IDX` — [`L299`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L299)
  - `THETA` — [`L298`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L298)
  - `brot_kernel` — [`L343`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L343)
  - `cmkey_j` — [`L320`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L320)
  - `decoder` — [`L361`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L361)
  - `decoder_cm` — [`L363`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L363)
  - `decryptor_q` — [`L360`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L360)
  - `encoder_pq` — [`L357`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L357)
  - `encoder_q` — [`L358`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L358)
  - `encryptor_q` — [`L359`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L359)
  - `mux_key` — [`L334`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L334)
  - `ntt_pq` — [`L354`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L354)
  - `pk_pq` — [`L305`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L305)
  - `pk_q` — [`L308`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L308)
  - `sk_pq` — [`L305`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L305)
  - `sk_q` — [`L311`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L311)
  - `test_random_source` — [`L304`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L304)
- uses (calls/refs, reference-scoped): [`encrypt`](encrypt.md#Encrypt.encrypt), [`precompute_constants`](ntt.md#NTTBarrett.precompute_constants), [`encode`](encode.md#Encode.encode), [`keygen`](key_gen.md#keygen), [`decrypt`](encrypt.md#Decrypt.decrypt), [`decode`](encode.md#Decode.decode), [`NTTBarrett`](ntt.md#NTTBarrett), [`precompute_constants`](blind_rotate.md#BlindRotation.precompute_constants), [`brot_cm`](blind_rotate.md#BlindRotation.brot_cm), [`Encode`](encode.md#Encode), [`Encrypt`](encrypt.md#Encrypt), [`Decrypt`](encrypt.md#Decrypt), [`Decode`](encode.md#Decode), [`data`](types.md#SecretKey.data), [`brot_mux`](blind_rotate.md#BlindRotation.brot_mux), [`gen_mux_rotation_key`](key_gen.md#gen_mux_rotation_key), [`SecretKey`](types.md#SecretKey), [`ZeroNoiseRandomSource`](random.md#ZeroNoiseRandomSource), [`_cyclic_roll`](blind_rotate_test.md#_cyclic_roll), [`data`](types.md#PublicKey.data), [`PublicKey`](types.md#PublicKey), [`BlindRotation`](blind_rotate.md#BlindRotation)

## Functions
- `_cyclic_roll(arr: np.ndarray, shift: int)` — [`L35`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_test.py#L35)

