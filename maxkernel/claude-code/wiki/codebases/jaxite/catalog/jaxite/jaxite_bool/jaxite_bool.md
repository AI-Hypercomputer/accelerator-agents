---
title: 'Module: jaxite/jaxite_bool/jaxite_bool.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_bool/jaxite_bool.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_bool.jaxite_bool`/
symbols:
  encrypt: encrypt().
  and_: and_().
  or_: or_().
  xor_: xor_().
  lut2: lut2().
  lut3: lut3().
  andny_: andny_().
  andyn_: andyn_().
  orny_: orny_().
  oryn_: oryn_().
  nand_: nand_().
  nor_: nor_().
  xnor_: xnor_().
  decrypt: decrypt().
  pmap_lut3_impl: pmap_lut3_impl().
  pmap_lut2_impl: pmap_lut2_impl().
  Parameters: Parameters.
  ServerKeySet: ServerKeySet#
  ServerKeySet.bsk: ServerKeySet#bsk().
  ServerKeySet.ksk: ServerKeySet#ksk().
  ServerKeySet.bootstrap_callback: ServerKeySet#bootstrap_callback.
  cmux_: cmux_().
  pmap_lut3: pmap_lut3().
  pmap_lut2: pmap_lut2().
  ServerKeySet._bsk: ServerKeySet#_bsk.
  lut1: lut1().
  constant: constant().
  not_: not_().
  ClientKeySet.lwe_sk: ClientKeySet#lwe_sk().
  ClientKeySet.rlwe_sk: ClientKeySet#rlwe_sk().
  ServerKeySet._ksk: ServerKeySet#_ksk.
  ServerKeySet.__init__: ServerKeySet#__init__().
  ClientKeySet: ClientKeySet#
  ClientKeySet._lwe_sk: ClientKeySet#_lwe_sk.
  ClientKeySet._rlwe_sk: ClientKeySet#_rlwe_sk.
  ENCODING_PARAMS: ENCODING_PARAMS.
  ClientKeySet.__init__: ClientKeySet#__init__().
  CLEARTEXT_TRUE: CLEARTEXT_TRUE.
  CLEARTEXT_FALSE: CLEARTEXT_FALSE.
  CLEARTEXT_UNUSED: CLEARTEXT_UNUSED.
  Lut3Args: Lut3Args.
  Lut2Args: Lut2Args.
---
# Module: [`jaxite/jaxite_bool/jaxite_bool.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py)

## Classes
### `ClientKeySet`
- def: [`jaxite/jaxite_bool/jaxite_bool.py:26`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L26)
- doc: A set of secret keys for client use.
- signature: `class ClientKeySet:`
- members:
  - `lwe_sk(self)` — [`L30`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L30) — documented in [jaxite-jaxite_bool](../../../concepts/jaxite-jaxite_bool.md)
  - `rlwe_sk(self)` — [`L34`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L34)
- protocol/private: `__init__`[`L37`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L37), `_lwe_sk`[`L43`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L43), `_rlwe_sk`[`L44`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L44)
- uses (calls/refs, reference-scoped): [`RandomSource`](../jaxite_cggi/random_source.md#RandomSource), [`gen_key`](../jaxite_cggi/lwe.md#gen_key), [`gen_key`](../jaxite_cggi/rlwe.md#gen_key), [`scheme_params`](bool_params.md#Parameters.scheme_params), [`Parameters`](jaxite_bool.md#Parameters), [`LweSecretKey`](../jaxite_cggi/lwe.md#LweSecretKey), [`RlweSecretKey`](../jaxite_cggi/rlwe.md#RlweSecretKey)
- used by: [`encrypt`](jaxite_bool.md#encrypt), [`client_key_set`](jaxite_bool_test.md#BoolBasicOperationsTest.client_key_set), [`decrypt`](jaxite_bool.md#decrypt), [`client_key_set`](jaxite_bool_multigate_test.md#BoolMultigateTest.client_key_set), [`callback`](jaxite_bool_test.md#BoolBasicOperationsTest.callback), [`callback`](jaxite_bool_multigate_test.md#BoolMultigateTest.callback), [`bootstrap_callback`](jaxite_bool.md#ServerKeySet.bootstrap_callback), [`client_key_set`](pmap_test.md#PmapTest.client_key_set), [`_bsk`](jaxite_bool.md#ServerKeySet._bsk), [`_ksk`](jaxite_bool.md#ServerKeySet._ksk), [`__init__`](jaxite_bool.md#ServerKeySet.__init__)

### `ServerKeySet`
- def: [`jaxite/jaxite_bool/jaxite_bool.py:47`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L47) — documented in [jaxite-jaxite_bool](../../../concepts/jaxite-jaxite_bool.md)
- doc: A set of public keys and relevant parameters for server/cloud use.
- signature: `class ServerKeySet:`
- members:
  - `bsk(self)` — [`L51`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L51) — documented in [jaxite-jaxite_bool](../../../concepts/jaxite-jaxite_bool.md)
  - `ksk(self)` — [`L55`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L55) — documented in [jaxite-jaxite_bool](../../../concepts/jaxite-jaxite_bool.md)
  - `bootstrap_callback` — [`L66`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L66) — documented in [jaxite-jaxite_bool](../../../concepts/jaxite-jaxite_bool.md)
- protocol/private: `__init__`[`L58`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L58), `_bsk`[`L68`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L68), `_ksk`[`L76`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L76)
- uses (calls/refs, reference-scoped): [`gen_bootstrapping_key`](../jaxite_cggi/bootstrap.md#gen_bootstrapping_key), [`gen_key`](../jaxite_cggi/key_switch.md#gen_key), [`RandomSource`](../jaxite_cggi/random_source.md#RandomSource), [`flatten_key`](../jaxite_cggi/rlwe.md#flatten_key), [`Parameters`](jaxite_bool.md#Parameters), [`bs_decomp_params`](bool_params.md#Parameters.bs_decomp_params), [`ks_decomp_params`](bool_params.md#Parameters.ks_decomp_params), [`key_from_rlwe`](../jaxite_cggi/rgsw.md#key_from_rlwe), [`lwe_sk`](jaxite_bool.md#ClientKeySet.lwe_sk), [`rlwe_sk`](jaxite_bool.md#ClientKeySet.rlwe_sk), [`ClientKeySet`](jaxite_bool.md#ClientKeySet), [`BootstrappingKey`](../jaxite_cggi/bootstrap.md#BootstrappingKey), [`LweKeySwitchingKey`](../jaxite_cggi/key_switch.md#LweKeySwitchingKey)
- used by: [`and_`](jaxite_bool.md#and_), [`or_`](jaxite_bool.md#or_), [`xor_`](jaxite_bool.md#xor_), [`lut2`](jaxite_bool.md#lut2), [`lut3`](jaxite_bool.md#lut3), [`andny_`](jaxite_bool.md#andny_), [`andyn_`](jaxite_bool.md#andyn_), [`nand_`](jaxite_bool.md#nand_), [`nor_`](jaxite_bool.md#nor_), [`orny_`](jaxite_bool.md#orny_), [`oryn_`](jaxite_bool.md#oryn_), [`test_bsk_decomposition_params`](jaxite_bool_test.md#BoolBasicOperationsTest.test_bsk_decomposition_params), [`test_ksk_decomposition_params`](jaxite_bool_test.md#BoolBasicOperationsTest.test_ksk_decomposition_params), [`xnor_`](jaxite_bool.md#xnor_), [`test_seeds`](jaxite_bool_multigate_test.md#BoolMultigateTest.test_seeds), [`pmap_lut2_impl`](jaxite_bool.md#pmap_lut2_impl), [`pmap_lut3_impl`](jaxite_bool.md#pmap_lut3_impl), [`server_key_set`](jaxite_bool_multigate_test.md#BoolMultigateTest.server_key_set), [`server_key_set`](jaxite_bool_test.md#BoolBasicOperationsTest.server_key_set), [`cmux_`](jaxite_bool.md#cmux_), [`pmap_lut2`](jaxite_bool.md#pmap_lut2), [`pmap_lut3`](jaxite_bool.md#pmap_lut3), [`server_key_set`](pmap_test.md#PmapTest.server_key_set), [`lut1`](jaxite_bool.md#lut1)

## Functions
- `and_(lhs: types.LweCiphertext, rhs: types.LweCiphertext, server_key_set: ServerKeySet, params: Parameters)` — [`L114`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L114) — Computes AND of lhs and rhs. — documented in [jaxite-jaxite_bool-bool_params](../../../concepts/jaxite-jaxite_bool-bool_params.md)
- `andny_(lhs: types.LweCiphertext, rhs: types.LweCiphertext, server_key_set: ServerKeySet, params: Parameters)` — [`L134`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L134) — Computes ANDNY of lhs andny rhs.
- `andyn_(lhs: types.LweCiphertext, rhs: types.LweCiphertext, server_key_set: ServerKeySet, params: Parameters)` — [`L154`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L154) — Computes ANDYN of lhs andyn rhs.
- `cmux_(v1: types.LweCiphertext, v0: types.LweCiphertext, ctrl: types.LweCiphertext, server_key_set: ServerKeySet, params: Parameters)` — [`L418`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L418)
- `constant(value: bool, params: Parameters)` — [`L103`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L103)
- `decrypt(ciphertext: types.LweCiphertext, client_key_set: ClientKeySet)` — [`L93`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L93) — Decrypts an `LweCiphertext` encryption of `True`/`False`.
- `encrypt(value: bool, client_key_set: ClientKeySet, prg: random_source.RandomSource)` — [`L84`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L84) — Encrypts a Boolean value under a given LWE secret key. — documented in [jaxite-jaxite_bool](../../../concepts/jaxite-jaxite_bool.md)
- `lut1(a: types.LweCiphertext, truth_table: int, server_key_set: ServerKeySet, params: Parameters)` — [`L314`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L314) — Computes a 1-LUT.
- `lut2(a: types.LweCiphertext, b: types.LweCiphertext, truth_table: int, server_key_set: ServerKeySet, params: Parameters)` — [`L332`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L332) — Computes a 2-LUT. — documented in [jaxite-jaxite_bool-bool_params](../../../concepts/jaxite-jaxite_bool-bool_params.md)
- `lut3(a: types.LweCiphertext, b: types.LweCiphertext, c: types.LweCiphertext, truth_table: int, server_key_set: ServerKeySet, params: Parameters)` — [`L374`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L374) — Computes a 3-LUT. — documented in [jaxite-jaxite_bool-bool_params](../../../concepts/jaxite-jaxite_bool-bool_params.md)
- `nand_(lhs: types.LweCiphertext, rhs: types.LweCiphertext, server_key_set: ServerKeySet, params: Parameters)` — [`L234`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L234) — Computes NAND of lhs nand rhs.
- `nor_(lhs: types.LweCiphertext, rhs: types.LweCiphertext, server_key_set: ServerKeySet, params: Parameters)` — [`L254`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L254) — Computes NOR of lhs nor rhs.
- `not_(ciphertext: types.LweCiphertext, params: Parameters)` — [`L107`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L107) — Computes NOT of the input ciphertext.
- `or_(lhs: types.LweCiphertext, rhs: types.LweCiphertext, server_key_set: ServerKeySet, params: Parameters)` — [`L174`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L174) — Computes OR of lhs or rhs. — documented in [jaxite-jaxite_bool](../../../concepts/jaxite-jaxite_bool.md)
- `orny_(lhs: types.LweCiphertext, rhs: types.LweCiphertext, server_key_set: ServerKeySet, params: Parameters)` — [`L194`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L194) — Computes ORNY of lhs orny rhs.
- `oryn_(lhs: types.LweCiphertext, rhs: types.LweCiphertext, server_key_set: ServerKeySet, params: Parameters)` — [`L214`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L214) — Computes ORYN of lhs oryn rhs.
- `pmap_lut2(inputs: list[Lut2Args], sks: ServerKeySet, params: Parameters)` — [`L519`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L519) — Apply lut2 in parallel across all inputs.
- `pmap_lut2_impl(a: types.LweCiphertext, b: types.LweCiphertext, truth_table: jnp.ndarray, server_key_set: ServerKeySet, params: Parameters)` — [`L497`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L497) — A version of lut3 suitable for pmap.
- `pmap_lut3(inputs: list[Lut3Args], sks: ServerKeySet, params: Parameters)` — [`L458`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L458) — Apply lut3 in parallel across all inputs.
- `pmap_lut3_impl(a: types.LweCiphertext, b: types.LweCiphertext, c: types.LweCiphertext, truth_table: jnp.ndarray, server_key_set: ServerKeySet, params: Parameters)` — [`L435`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L435) — A version of lut3 suitable for pmap.
- `xnor_(lhs: types.LweCiphertext, rhs: types.LweCiphertext, server_key_set: ServerKeySet, params: Parameters)` — [`L294`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L294) — Computes XNOR of lhs xnor rhs.
- `xor_(lhs: types.LweCiphertext, rhs: types.LweCiphertext, server_key_set: ServerKeySet, params: Parameters)` — [`L274`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L274) — Computes XOR of lhs xor rhs. — documented in [jaxite-jaxite_bool](../../../concepts/jaxite-jaxite_bool.md)

## Module values
- `CLEARTEXT_FALSE` — [`L22`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L22)
- `CLEARTEXT_TRUE` — [`L21`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L21)
- `CLEARTEXT_UNUSED` — [`L23`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L23)
- `ENCODING_PARAMS` — [`L20`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L20)
- `Lut2Args` — [`L516`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L516)
- `Lut3Args` — [`L455`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L455)
- `Parameters` — [`L19`](../../../../../../raw/code/jaxite/jaxite/jaxite_bool/jaxite_bool.py#L19) — documented in [jaxite-jaxite_bool](../../../concepts/jaxite-jaxite_bool.md)

