---
title: 'Module: jaxite_ec/algorithm/rns_mod_reduce.py'
type: catalog
provenance: extracted
module: jaxite_ec/algorithm/rns_mod_reduce.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite_ec.algorithm.rns_mod_reduce`/
symbols:
  cor_mat: cor_mat.
  overall_modulus: overall_modulus.
  result_rns: result_rns.
  moduli: moduli.
  rns_mod_reduce: rns_mod_reduce().
  result_mod_q: result_mod_q.
  constant_offset_list: constant_offset_list.
  icrt_factors: icrt_factors.
  random_data: random_data.
  data_rns: data_rns.
  q: q.
  moduli_precision: moduli_precision.
  overall_moduli: overall_moduli.
  rns_coefficients_precompute: rns_coefficients_precompute().
  num_residues_for_q: num_residues_for_q.
  moduli_t: moduli_t.
  result_ref: result_ref.
  to_tuple: to_tuple().
  mul_res_dtype: mul_res_dtype.
  num_bytes: num_bytes.
  to_rns: to_rns().
  rns_stack_mat_u8: rns_stack_mat_u8.
  hardware_friendly_mod_reduce: hardware_friendly_mod_reduce().
  randint: randint.
  chunk_dtype: chunk_dtype.
  find_moduli: find_moduli().
  rns_reconstruct: rns_reconstruct().
  rns_icrt_factors_compute: rns_icrt_factors_compute().
  extra_bit_to_avoid_addition_overflow: extra_bit_to_avoid_addition_overflow.
  num_elements: num_elements.
---
# Module: [`jaxite_ec/algorithm/rns_mod_reduce.py`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py)

## Functions
- `find_moduli(total_modulus, precision)` — [`L25`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L25) — Find moduli for RNS.
- `hardware_friendly_mod_reduce(x, moduli_t)` — [`L55`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L55) — Convert input value x into the RNS form.
- `rns_coefficients_precompute(icrt_factors, overall_moduli, num_bytes, moduli_precision, overall_modulus, q)` — [`L104`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L104) — Precompute RNS coefficients.
- `rns_icrt_factors_compute(modulus, moduli)` — [`L94`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L94)
- `rns_mod_reduce(data_a_rns, data_b_rns, moduli, moduli_t, rns_stack_mat_u8, cor_mat, icrt_factors, overall_modulus)` — [`L176`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L176) — Performs RNS modular reduction.
- `rns_reconstruct(x, overall_moduli, icrt_factors)` — [`L84`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L84)
- `to_rns(x, moduli)` — [`L80`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L80)
- `to_tuple(a)` — [`L17`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L17) — Create to convert numpy array into tuple.

## Module values
- `chunk_dtype` — [`L13`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L13)
- `constant_offset_list` — [`L258`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L258)
- `cor_mat` — [`L325`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L325)
- `data_rns` — [`L399`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L399)
- `extra_bit_to_avoid_addition_overflow` — [`L241`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L241)
- `icrt_factors` — [`L320`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L320)
- `moduli` — [`L314`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L314)
- `moduli_precision` — [`L240`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L240)
- `moduli_t` — [`L391`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L391)
- `mul_res_dtype` — [`L14`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L14)
- `num_bytes` — [`L242`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L242)
- `num_elements` — [`L252`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L252)
- `num_residues_for_q` — [`L243`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L243)
- `overall_moduli` — [`L258`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L258)
- `overall_modulus` — [`L246`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L246)
- `q` — [`L239`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L239)
- `randint` — [`L9`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L9)
- `random_data` — [`L397`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L397)
- `result_mod_q` — [`L418`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L418)
- `result_ref` — [`L398`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L398)
- `result_rns` — [`L408`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L408)
- `rns_stack_mat_u8` — [`L325`](../../../../../../raw/code/jaxite/jaxite_ec/algorithm/rns_mod_reduce.py#L325)

