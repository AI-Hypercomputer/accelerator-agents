---
title: 'Module: jaxite/jaxite_cggi/key_switch.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_cggi/key_switch.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_cggi.key_switch`/
symbols:
  gen_key: gen_key().
  switch_key: switch_key().
  gen_key.decompose_and_encrypt: gen_key().decompose_and_encrypt().
  jit_switch_key: jit_switch_key().
  LweKeySwitchingKey.key_data: LweKeySwitchingKey#key_data.
  LweKeySwitchingKey: LweKeySwitchingKey#
  jit_switch_key.decomp_dot: jit_switch_key().decomp_dot().
  LweKeySwitchingKey.decomposition_log_base: LweKeySwitchingKey#decomposition_log_base.
  LweKeySwitchingKey.decomposition_level_count: LweKeySwitchingKey#decomposition_level_count.
  LweKeySwitchingKey.modulus: LweKeySwitchingKey#modulus.
  LweKeySwitchingKey.lwe_dimension: LweKeySwitchingKey#lwe_dimension.
  LweKeySwitchingKey.lwe_size: LweKeySwitchingKey#lwe_size.
---
# Module: [`jaxite/jaxite_cggi/key_switch.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch.py)

## Classes
### `LweKeySwitchingKey`
- def: [`jaxite/jaxite_cggi/key_switch.py:15`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch.py#L15)
- doc: A public key used to switch keys encrypted an LWE ciphertext.
- signature: `class LweKeySwitchingKey:`
- members:
  - `decomposition_level_count` — [`L31`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch.py#L31)
  - `decomposition_log_base` — [`L28`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch.py#L28)
  - `key_data` — [`L40`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch.py#L40)
  - `lwe_dimension` — [`L25`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch.py#L25)
  - `lwe_size` — [`L34`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch.py#L34)
  - `modulus` — [`L19`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch.py#L19)
- used by: [`run_bootstrap_test`](bootstrap_test.md#BootstrapBaseTest.run_bootstrap_test), [`bootstrap`](bootstrap.md#bootstrap), [`gen_key`](key_switch.md#gen_key), [`pmap_lut2_impl`](../jaxite_bool/jaxite_bool.md#pmap_lut2_impl), [`pmap_lut3_impl`](../jaxite_bool/jaxite_bool.md#pmap_lut3_impl), [`test_key_switch_gen`](key_switch_test.md#KeySwitchTest.test_key_switch_gen), [`ksk`](../jaxite_bool/jaxite_bool.md#ServerKeySet.ksk), [`switch_key`](key_switch.md#switch_key)

## Functions
- `decomp_dot(ksk_i, inp_i)` — [`L137`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch.py#L137)
- `decompose_and_encrypt(in_key_bit, lwe_ai_samples, lwe_error_samples)` — [`L79`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch.py#L79) — Decompose one bit of the input key, then encrypt each term.
- `gen_key(decomposition_params: decomposition.DecompositionParameters, prg: random_source.RandomSource, in_key: lwe.LweSecretKey, out_key: lwe.LweSecretKey)` — [`L43`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch.py#L43) — Generate an LWE key switching key.
- `jit_switch_key(ksk: jnp.ndarray, inp: types.LweCiphertext, num_levels: int, base_log: int)` — [`L124`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch.py#L124) — Perform the key switch operation on an LWE ciphertext. — documented in [jaxite-jaxite_cggi-bootstrap](../../../concepts/jaxite-jaxite_cggi-bootstrap.md)
- `switch_key(ksk: LweKeySwitchingKey, inp: types.LweCiphertext)` — [`L110`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/key_switch.py#L110) — Perform the key switch operation on an LWE ciphertext.

