---
title: 'Module: jaxite/jaxite_ckks/ntt_cpu.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/ntt_cpu.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.ntt_cpu`/
symbols:
  ntt_negacyclic_poly: ntt_negacyclic_poly().
  intt_negacyclic_poly: intt_negacyclic_poly().
  _get_bit_reversal_perm: _get_bit_reversal_perm().
  ntt_negacyclic_bit_reverse: ntt_negacyclic_bit_reverse().
  intt_negacyclic_bit_reverse: intt_negacyclic_bit_reverse().
  ntt_cyclic_bit_reverse: ntt_cyclic_bit_reverse().
  intt_cyclic_bit_reverse: intt_cyclic_bit_reverse().
  _get_powers: _get_powers().
  _PERM_CACHE: _PERM_CACHE.
---
# Module: [`jaxite/jaxite_ckks/ntt_cpu.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt_cpu.py)

## Functions
- `_get_bit_reversal_perm(n)` — [`L16`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt_cpu.py#L16) — Returns a bit-reversal permutation array of size n.
- `_get_powers(base, n, q)` — [`L29`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt_cpu.py#L29) — Returns an array of powers of base modulo q: [base^0, ..., base^{n-1}].
- `intt_cyclic_bit_reverse(a, q, omega)` — [`L86`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt_cpu.py#L86) — Iterative inverse NTT with bit-reversal.
- `intt_negacyclic_bit_reverse(a, q, psi)` — [`L48`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt_cpu.py#L48) — Inverse negacyclic NTT.
- `intt_negacyclic_poly(poly: np.ndarray, moduli: list[int])` — [`L132`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt_cpu.py#L132) — CPU-only Inverse NTT of a polynomial in RNS form. — documented in [jaxite-jaxite_ckks-encode](../../../concepts/jaxite-jaxite_ckks-encode.md)
- `ntt_cyclic_bit_reverse(a, q, omega)` — [`L58`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt_cpu.py#L58) — Iterative forward NTT with bit-reversal.
- `ntt_negacyclic_bit_reverse(a, q, psi)` — [`L39`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt_cpu.py#L39) — Forward negacyclic NTT.
- `ntt_negacyclic_poly(poly: np.ndarray, moduli: list[int])` — [`L118`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt_cpu.py#L118) — CPU-only NTT of a polynomial in RNS form. — documented in [jaxite-jaxite_ckks-encode](../../../concepts/jaxite-jaxite_ckks-encode.md)

## Module values
- `_PERM_CACHE` — [`L13`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/ntt_cpu.py#L13)

