---
title: 'Module: jaxite_ec/finite_field.py'
type: catalog
provenance: extracted
module: jaxite_ec/finite_field.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite_ec.finite_field`/
symbols:
  mul_2u16: mul_2u16().
  mul_shift_2u16x2x1: mul_shift_2u16x2x1().
  mod_mul_barrett_2u16: mod_mul_barrett_2u16().
  barrett_reduction_u16x2: barrett_reduction_u16x2().
  mod_mul_lazy_2u16: mod_mul_lazy_2u16().
  sub_2u16: sub_2u16().
  cond_sub_mod_u16: cond_sub_mod_u16().
  moduli_rns_red_internal_2u16: moduli_rns_red_internal_2u16().
  check_any_chunk_with_carry: check_any_chunk_with_carry().
  carry_propagation: carry_propagation().
  add_2u16: add_2u16().
  add_3u16: add_3u16().
  cond_sub_2u16: cond_sub_2u16().
  mod_mul_rns_2u16: mod_mul_rns_2u16().
  add_sub_rns_var: add_sub_rns_var().
  split_view_32_to_16_8: split_view_32_to_16_8().
  mod_red_rns_2u16: mod_red_rns_2u16().
  mul_unreduced_rns_2u16: mul_unreduced_rns_2u16().
  rechunkify: rechunkify().
  add_rns_2u16: add_rns_2u16().
  add_rns_3u16: add_rns_3u16().
  conv_1d_2u16xn: conv_1d_2u16xn().
  compare_u16: compare_u16().
  split_view_32_to_16: split_view_32_to_16().
  total_modulus: total_modulus.
  to_rns: to_rns.
  carry_add: carry_add().
  negate_rns_for_var_add: negate_rns_for_var_add().
  negate_rns_for_var_add_zero_check: negate_rns_for_var_add_zero_check().
  rns_constant: rns_constant().
---
# Module: [`jaxite_ec/finite_field.py`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py)

## Functions
- `add_2u16(value_a: jax.Array, value_b: jax.Array)` — [`L223`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L223) — Add two u16 values.
- `add_3u16(value_a: jax.Array, value_b: jax.Array, value_d: jax.Array)` — [`L243`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L243)
- `add_rns_2u16(value_a: jax.Array, value_b: jax.Array, moduli_t=util.RNS_MODULI_T)` — [`L814`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L814) — Add two u16 values with RNS reduction.
- `add_rns_3u16(value_a: jax.Array, value_b: jax.Array, value_c: jax.Array, moduli_t=util.RNS_MODULI_T)` — [`L837`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L837) — Add three u16 values with RNS reduction.
- `add_sub_rns_var(*values, moduli_t=util.RNS_MODULI_T)` — [`L936`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L936) — Evaluate an static set of additions and subtractions.
- `barrett_reduction_u16x2(value_x: jax.Array, mask=util.U16_MASK, modulus_377_int_chunk=util.MODULUS_377_INT_CHUNK, mu_377_int_chunk=util.MU_377_INT_CHUNK, chunk_num_u16=util.U16_CHUNK_NUM, vmap_axes=(0, None))` — [`L552`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L552) — Performs Barrett reduction on a u16x2 value.
- `carry_add(value_c: jax.Array, iter_num=util.U16_CHUNK_NUM, mask=util.U16_MASK, chunk_shift_bits=util.U16_CHUNK_SHIFT_BITS)` — [`L49`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L49) — The purpose of this API is to enable general-purposed carry add, where the following knobs are known before runtime.
- `carry_propagation(value_c: jax.Array, mask=util.U16_MASK, chunk_shift_bits=util.U16_CHUNK_SHIFT_BITS)` — [`L103`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L103) — The purpose of this API is to enable carry propagation.
- `check_any_chunk_with_carry(value_c: jax.Array, chunk_shift_bits=util.U16_CHUNK_SHIFT_BITS)` — [`L80`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L80) — This function check whether any chunk of input vector 'value_c' has carry.
- `compare_u16(value_a: jax.Array, value_b: jax.Array, chunk_num_u16=util.U16_CHUNK_NUM)` — [`L197`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L197) — Compare two u16 values.
- `cond_sub_2u16(value_a: jax.Array, value_b: jax.Array, modulus_377_int_chunk=util.MODULUS_377_INT_CHUNK, chunk_num_u16=util.U16_CHUNK_NUM)` — [`L336`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L336) — Perform conditional subtraction: value_a - value_b.
- `cond_sub_mod_u16(value_a: jax.Array, modulus_377_int_chunk=util.MODULUS_377_INT_CHUNK, chunk_num_u16=util.U16_CHUNK_NUM)` — [`L304`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L304) — Perform conditional subtraction: value_a - modulus_377_int.
- `conv_1d_2u16xn(value_a: jax.Array, value_b: jax.Array)` — [`L140`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L140) — This function performs a 1D convolution of two u16 arrays.
- `mod_mul_barrett_2u16(value_a: jax.Array, value_b: jax.Array, mask=util.U16_MASK, modulus_377_int_chunk=util.MODULUS_377_INT_CHUNK, mu_377_int_chunk=util.MU_377_INT_CHUNK, chunk_num_u16=util.U16_CHUNK_NUM, vmap_axes=(0, None))` — [`L503`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L503) — Multiply two u16 values with Barrett reduction.
- `mod_mul_lazy_2u16(value_a, value_b, modulus_lazy_mat=util.MODULUS_377_LAZY_MAT, mask=util.U32_MASK, chunk_num_u8=util.U8_CHUNK_NUM, chunk_shift_bits=util.U32_CHUNK_SHIFT_BITS)` — [`L594`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L594) — Multiply two u16 values with lazy matrix reduction.
- `mod_mul_rns_2u16(value_a, value_b, rns_mat=util.RNS_MAT, moduli_t=util.RNS_MODULI_T)` — [`L788`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L788) — Multiply two u16 values with RNS reduction.
- `mod_red_rns_2u16(c_rns_reduced, rns_mat=util.RNS_MAT, moduli_t=util.RNS_MODULI_T, num_moduli=util.NUM_MODULI, precision=util.RNS_PRECISION)` — [`L722`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L722) — Reduce via RNS modulus.
- `moduli_rns_red_internal_2u16(vals, moduli_t=util.RNS_MODULI_T)` — [`L687`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L687) — Reduce via RNS modulus.
- `mul_2u16(value_a: jax.Array, value_b: jax.Array, mask=util.U32_MASK, chunk_num_u16=util.U16_CHUNK_NUM, chunk_shift_bits=util.U32_CHUNK_SHIFT_BITS, output_dtype=jnp.uint16, vmap_axes=(0, 0))` — [`L382`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L382) — Multiply two u16 values.
- `mul_shift_2u16x2x1(value_a: jax.Array, value_b: jax.Array, mask=util.U32_MASK, barrett_shift_u8=util.BARRETT_SHIFT_U8, chunk_num_u16=util.U16_CHUNK_NUM, chunk_num_u32=util.U32_CHUNK_NUM, chunk_shift_bits=util.U32_CHUNK_SHIFT_BITS, vmap_axes=(0, None))` — [`L441`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L441) — Multiply and shift two u16 values.
- `mul_unreduced_rns_2u16(value_a, value_b, moduli_t=util.RNS_MODULI_T)` — [`L773`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L773)
- `negate_rns_for_var_add(value_a: jax.Array, moduli_sub=util.MODULI_SUB)` — [`L862`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L862) — Negate a value for use in subtraction.
- `negate_rns_for_var_add_zero_check(value_a: jax.Array, moduli_sub=util.MODULI_SUB)` — [`L895`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L895) — Negate a value for use in subtraction.
- `rechunkify(mul_result: jax.Array, chunk_num_u16, chunk_num_u32)` — [`L162`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L162) — Given the carry add takes O(C) algorithm complexity, where C is the number of chunks.
- `rns_constant(c, num_moduli=util.NUM_MODULI)` — [`L973`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L973)
- `split_view_32_to_16(a: jnp.ndarray)` — [`L657`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L657)
- `split_view_32_to_16_8(a: jnp.ndarray)` — [`L668`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L668)
- `sub_2u16(value_a: jax.Array, value_b: jax.Array, mask=util.U16_MASK, chunk_num_u16=util.U16_CHUNK_NUM)` — [`L259`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L259) — Subtract two u16 values.

## Module values
- `to_rns` — [`L39`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L39)
- `total_modulus` — [`L38`](../../../../../raw/code/jaxite/jaxite_ec/finite_field.py#L38)

