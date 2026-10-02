---
title: 'Module: jaxite/jaxite_ckks/rns_utils.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/rns_utils.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.rns_utils`/
symbols:
  bit_reversal_array: bit_reversal_array().
  reconstruct_crt: reconstruct_crt().
  num_bits: num_bits().
  compute_q_hat_inv_mod_q: compute_q_hat_inv_mod_q().
  inverse_mod: inverse_mod().
  is_power_of_two: is_power_of_two().
  bit_reversal: bit_reversal().
  compute_q_hat_mod_p: compute_q_hat_mod_p().
---
# Module: [`jaxite/jaxite_ckks/rns_utils.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_utils.py)

## Functions
- `bit_reversal(x: int, num_bits: int)` — [`L21`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_utils.py#L21) — Returns the bit-reversal of x with num_bits representation.
- `bit_reversal_array(xs: list[Any])` — [`L31`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_utils.py#L31) — Rearrange the given array in bit-reversal order in place.
- `compute_q_hat_inv_mod_q(moduli: list[int])` — [`L40`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_utils.py#L40) — Computes QHatInvModq = (Q/qi)^-1 mod qi for each qi in moduli.
- `compute_q_hat_mod_p(original_moduli: list[int], target_moduli: list[int])` — [`L62`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_utils.py#L62) — Computes QHatModp = (Q/qi) mod pj for each qi in original and pj in target.
- `inverse_mod(x: int, q: int)` — [`L6`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_utils.py#L6) — Returns the inverse of x mod q.
- `is_power_of_two(x: int)` — [`L11`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_utils.py#L11) — Returns True if x is a power of two.
- `num_bits(x: int)` — [`L16`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_utils.py#L16) — Returns the number of bits in x.
- `reconstruct_crt(residues: list[list[int]], moduli: list[int])` — [`L89`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/rns_utils.py#L89) — Reconstructs integers from their RNS residues. — documented in [jaxite-jaxite_ckks-encode](../../../concepts/jaxite-jaxite_ckks-encode.md)

