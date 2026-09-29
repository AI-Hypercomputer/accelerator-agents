---
title: 'Module: jaxite/jaxite_cggi/bootstrap.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_cggi/bootstrap.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_cggi.bootstrap`/
symbols:
  bootstrap: bootstrap().
  gen_bootstrapping_key: gen_bootstrapping_key().
  jit_bootstrap: jit_bootstrap().
  blind_rotate: blind_rotate().
  cmux: cmux().
  external_product: external_product().
  jit_external_product: jit_external_product().
  jit_blind_rotate: jit_blind_rotate().
  sample_extract: sample_extract().
  jit_blind_rotate.one_external_product: jit_blind_rotate().one_external_product().
  jit_cmux: jit_cmux().
  gen_bootstrapping_key.process_one_batch: gen_bootstrapping_key().process_one_batch().
  BootstrappingKey.encrypted_lwe_sk_bits: BootstrappingKey#encrypted_lwe_sk_bits.
  jit_sample_extract: jit_sample_extract().
  BootstrappingKey: BootstrappingKey#
  BootstrappingKey.use_bmmp: BootstrappingKey#use_bmmp.
  BootstrappingKey.use_bat: BootstrappingKey#use_bat.
  GEN_BSK_NUM_BATCHES: GEN_BSK_NUM_BATCHES.
  gen_bootstrapping_key.bat_offline_compile_cggi: gen_bootstrapping_key().bat_offline_compile_cggi().
  NON_DIVISIBLE_BATCH_SIZE_WARNING: NON_DIVISIBLE_BATCH_SIZE_WARNING.
---
# Module: [`jaxite/jaxite_cggi/bootstrap.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap.py)

## Classes
### `BootstrappingKey`
- def: [`jaxite/jaxite_cggi/bootstrap.py:30`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap.py#L30)
- doc: An array with row j an RGSW encryption of bit j of an LWE secret key.
- signature: `class BootstrappingKey:`
- members:
  - `encrypted_lwe_sk_bits` — [`L33`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap.py#L33)
  - `use_bat` — [`L35`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap.py#L35)
  - `use_bmmp` — [`L34`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap.py#L34)
- used by: [`run_bootstrap_test`](bootstrap_test.md#BootstrapBaseTest.run_bootstrap_test), [`bootstrap`](bootstrap.md#bootstrap), [`gen_bootstrapping_key`](bootstrap.md#gen_bootstrapping_key), [`blind_rotate`](bootstrap.md#blind_rotate), [`pmap_lut2_impl`](../jaxite_bool/jaxite_bool.md#pmap_lut2_impl), [`pmap_lut3_impl`](../jaxite_bool/jaxite_bool.md#pmap_lut3_impl), [`bsk`](../jaxite_bool/jaxite_bool.md#ServerKeySet.bsk)

## Functions
- `bat_offline_compile_cggi(mat_a)` — [`L197`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap.py#L197) — Convert the input matrix with 32 bit elements into u8(*matrix.shape,4,4).
- `blind_rotate(rot_polynomial: rlwe.RlweCiphertext, coefficient_index: types.LweCiphertext, bsk: BootstrappingKey, decomposition_params: decomposition.DecompositionParameters)` — [`L488`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap.py#L488) — Rotate an encrypted polynomial `coefficient_index` times. — documented in [jaxite-jaxite_cggi-bootstrap](../../../concepts/jaxite-jaxite_cggi-bootstrap.md)
- `bootstrap(ciphertext: types.LweCiphertext, test_poly_ciphertext: rlwe.RlweCiphertext, bsk: BootstrappingKey, ksk: key_switch.LweKeySwitchingKey, ks_decomposition_params: decomposition.DecompositionParameters, bs_decomposition_params: decomposition.DecompositionParameters, scheme_params: parameters.SchemeParameters, callback: Optional[Callable[[str, Any], None]] = None, **kwargs: Any)` — [`L301`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap.py#L301) — Apply functional bootstrap to reduce noise in the input ciphertext. — documented in [jaxite-jaxite_bool](../../../concepts/jaxite-jaxite_bool.md)
- `cmux(control: rgsw.RgswCiphertext, eq_zero: rlwe.RlweCiphertext, neq_zero: rlwe.RlweCiphertext, decomposition_params: decomposition.DecompositionParameters, use_bat: bool = False)` — [`L415`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap.py#L415) — Compute CMUX: controlled multiplexer.
- `external_product(rgsw_ct: rgsw.RgswCiphertext, rlwe_ct: rlwe.RlweCiphertext, decomposition_params: decomposition.DecompositionParameters)` — [`L378`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap.py#L378) — Compute the external product of an RSGW and RLWE ciphertext. — documented in [jaxite-jaxite_cggi-decomposition](../../../concepts/jaxite-jaxite_cggi-decomposition.md)
- `gen_bootstrapping_key(lwe_sk: lwe.LweSecretKey, rgsw_sk: rgsw.RgswSecretKey, decomposition_params: decomposition.DecompositionParameters, prg: random_source.RandomSource, use_bmmp: bool = True, use_bat: bool = False)` — [`L38`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap.py#L38) — Generate a bootstrapping key for the given LWE secret key. — documented in [jaxite-jaxite_bool](../../../concepts/jaxite-jaxite_bool.md)
- `jit_blind_rotate(rot_polynomial: jnp.ndarray, coefficient_index: types.LweCiphertext, bsk: jnp.ndarray, log_coefficient_modulus: int, decomposition_params: decomposition.DecompositionParameters, use_bmmp: bool, use_bat: bool)` — [`L530`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap.py#L530) — Rotate an encrypted polynomial `coefficient_index` times. — documented in [jaxite-jaxite_cggi-bootstrap](../../../concepts/jaxite-jaxite_cggi-bootstrap.md)
- `jit_bootstrap(ciphertext: types.LweCiphertext, test_poly_ciphertext_message: jnp.ndarray, bsk_encrypted_lwe_sk_bits: jnp.ndarray, ksk_key_data: jnp.ndarray, ks_decomposition_params: decomposition.DecompositionParameters, bs_decomposition_params: decomposition.DecompositionParameters, scheme_params: parameters.SchemeParameters, bsk_use_bmmp: bool = True, bsk_use_bat: bool = False)` — [`L234`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap.py#L234) — Apply functional bootstrap to reduce noise in the input ciphertext.
- `jit_cmux(control: jnp.ndarray, eq_zero: jnp.ndarray, neq_zero: jnp.ndarray, decomposition_params: decomposition.DecompositionParameters, use_bat: bool)` — [`L469`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap.py#L469) — A jitted cmux.
- `jit_external_product(rgsw_ct: jnp.ndarray, rlwe_ct: jnp.ndarray, decomposition_params: decomposition.DecompositionParameters, use_bat: bool = False)` — [`L395`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap.py#L395) — Compute the external product of an RSGW and RLWE ciphertext. — documented in [jaxite-jaxite_cggi-bootstrap](../../../concepts/jaxite-jaxite_cggi-bootstrap.md)
- `jit_sample_extract(rlwe_ciphertext: jnp.ndarray, poly_deg: jnp.uint32)` — [`L616`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap.py#L616) — Extracts an LWE encryption of the constant term encrypted by the input.
- `one_external_product(j, c_prime_accum)` — [`L556`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap.py#L556) — documented in [jaxite-jaxite_cggi-bootstrap](../../../concepts/jaxite-jaxite_cggi-bootstrap.md)
- `process_one_batch(i)` — [`L169`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap.py#L169) — documented in [jaxite-jaxite_cggi-bootstrap](../../../concepts/jaxite-jaxite_cggi-bootstrap.md)
- `sample_extract(ciphertext: rlwe.RlweCiphertext)` — [`L609`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap.py#L609) — Extracts an LWE encryption of the constant term encrypted by the input.

## Module values
- `GEN_BSK_NUM_BATCHES` — [`L20`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap.py#L20)
- `NON_DIVISIBLE_BATCH_SIZE_WARNING` — [`L21`](../../../../../../raw/code/jaxite/jaxite/jaxite_cggi/bootstrap.py#L21)

