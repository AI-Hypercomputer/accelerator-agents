---
title: 'Module: jaxite/jaxite_ckks/key_gen.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/key_gen.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.key_gen`/
symbols:
  keygen: keygen().
  gen_hmuxrot_key: gen_hmuxrot_key().
  gen_key_switching_key: gen_key_switching_key().
  gen_cm_keys: gen_cm_keys().
  extend_secret_key: extend_secret_key().
  gen_evaluation_key: gen_evaluation_key().
  gen_conjugate_key: gen_conjugate_key().
  gen_mux_rotation_key: gen_mux_rotation_key().
  SecretKey: SecretKey.
  compute_scaled_source_key_partition: compute_scaled_source_key_partition().
  PublicKey: PublicKey.
---
# Module: [`jaxite/jaxite_ckks/key_gen.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_gen.py)

## Functions
- `compute_scaled_source_key_partition(source_key: SecretKey, q_limbs: list[int], p_val: int, all_moduli_len: int, start_idx: int, end_idx: int)` — [`L77`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_gen.py#L77) — Computes the scaled source key (P * s_src) for a partition of limbs.
- `extend_secret_key(secret_key: SecretKey, target_moduli: list[int])` — [`L48`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_gen.py#L48) — Extends a secret key to a larger set of moduli.
- `gen_cm_keys(indices: list[int], public_key: PublicKey, scale: float)` — [`L195`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_gen.py#L195) — Generates column keys.
- `gen_conjugate_key(sk: types.SecretKey, q_limbs: list[int], p_limbs: list[int], dnum: int, random_source: random.RandomSource | None = None)` — [`L223`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_gen.py#L223) — Generates the key switching key for the conjugate automorphism.
- `gen_evaluation_key(secret_key: SecretKey, q_towers: list[int], p_towers: list[int], dnum: int, random_source: random.RandomSource | None = None)` — [`L169`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_gen.py#L169) — Generate evaluation keys for relinearization.
- `gen_hmuxrot_key(sk: types.SecretKey, beta: int, j: int, q_limbs: list[int], p_limbs: list[int], random_source: random.RandomSource | None = None, sk_ext: types.SecretKey | None = None)` — [`L245`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_gen.py#L245) — Generates an HMuxRotKey symmetrically. — documented in [jaxite-jaxite_ckks-types](../../../concepts/jaxite-jaxite_ckks-types.md)
- `gen_key_switching_key(source_key: SecretKey, dest_key: SecretKey, q_limbs: list[int], p_limbs: list[int], dnum: int, random_source: random.RandomSource | None = None, dest_key_ext: SecretKey | None = None)` — [`L111`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_gen.py#L111) — Generate key switching keys to switch from source_key to dest_key.
- `gen_mux_rotation_key(sk: types.SecretKey, secret_bits: list[int], q_limbs: list[int], p_limbs: list[int], stride: int = 1, random_source: random.RandomSource | None = None)` — [`L329`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_gen.py#L329) — Generates a MuxRotationKey for the bits of the rotation index.
- `keygen(degree: int, moduli: list[int], random_source: random.RandomSource | None = None, hamming_weight: int | None = None)` — [`L17`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_gen.py#L17) — Generate a public, secret key pair.

## Module values
- `PublicKey` — [`L13`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_gen.py#L13)
- `SecretKey` — [`L14`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/key_gen.py#L14)

