---
title: 'Module: jaxite/jaxite_ckks/math.py'
type: catalog
provenance: extracted
module: jaxite/jaxite_ckks/math.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite.jaxite_ckks.math`/
symbols:
  gen_twiddle_matrix: gen_twiddle_matrix().
  root_of_unity: root_of_unity().
  find_generator: find_generator().
  gen_twiddle_matrix_inv: gen_twiddle_matrix_inv().
  get_bit_reverse_perm: get_bit_reverse_perm().
  prime_factors: prime_factors().
---
# Module: [`jaxite/jaxite_ckks/math.py`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/math.py)

## Functions
- `find_generator(q: int)` — [`L24`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/math.py#L24) — Finds a primitive root modulo q.
- `gen_twiddle_matrix(rows: int, cols: int, q: int, omega: int)` — [`L63`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/math.py#L63) — Precomputes the twiddle matrix T where T[r, c] = omega^(r*c) mod q. — documented in [jaxite-jaxite_ckks-ntt](../../../concepts/jaxite-jaxite_ckks-ntt.md)
- `gen_twiddle_matrix_inv(rows: int, cols: int, q: int, omega: int)` — [`L85`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/math.py#L85) — Precomputes the inverse twiddle matrix T_inv where T_inv[r, c] = omega^(-r*c) mod q.
- `get_bit_reverse_perm(n: int)` — [`L93`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/math.py#L93) — Generates a list of indices for bit-reversal permutation of size n.
- `prime_factors(n: int)` — [`L7`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/math.py#L7) — Returns the set of prime factors of n.
- `root_of_unity(m: int, q: int)` — [`L36`](../../../../../../raw/code/jaxite/jaxite/jaxite_ckks/math.py#L36) — Returns the canonical primitive m-th root of unity modulo q.

