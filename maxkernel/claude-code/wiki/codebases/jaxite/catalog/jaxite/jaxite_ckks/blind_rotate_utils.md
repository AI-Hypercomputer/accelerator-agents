---
title: 'Module: jaxite/jaxite_ckks/blind_rotate_utils.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/blind_rotate_utils.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.blind_rotate_utils`/
symbols:
  lift_ciphertext: lift_ciphertext().
  apply_automorphism_ntt: apply_automorphism_ntt().
  apply_automorphism_ntt.bit_reverse: apply_automorphism_ntt().bit_reverse().
---
# Module: [`jaxite/jaxite_ckks/blind_rotate_utils.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_utils.py)

## Functions
- `apply_automorphism_ntt(data: jax.Array, g: int)` — [`L11`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_utils.py#L11) — Applies the automorphism X -> X^g to a polynomial in the NTT domain.
- `bit_reverse(x)` — [`L28`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_utils.py#L28)
- `lift_ciphertext(ct: types.Ciphertext, bc_kernel: basis_conversion.BasisConversionBarrett, control_index: int, p_limbs: jax.Array, ntt_q: ntt.NTTBarrett, ntt_p: ntt.NTTBarrett, r: int, c: int)` — [`L44`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/blind_rotate_utils.py#L44) — Lifts a ciphertext from Q to PQ using basis conversion.

