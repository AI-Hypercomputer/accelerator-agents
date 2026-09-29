---
title: 'Module: jaxite_ec/util.py'
type: catalog
provenance: extracted
module: jaxite_ec/util.py
status: fresh
symbol_base: scip-python python jaxite 0.0.0 `jaxite_ec.util`/
symbols:
  COR_MAT_NEW: COR_MAT_NEW.
  construct_rns_matrix: construct_rns_matrix().
  to_tuple: to_tuple().
  int_to_precomputed_array: int_to_precomputed_array().
  M: M.
  BASE: BASE.
  moduli: moduli.
  MODULI_SUB: MODULI_SUB.
  MODULUS_377_INT_CHUNK: MODULUS_377_INT_CHUNK.
  MU_377_INT_CHUNK: MU_377_INT_CHUNK.
  TWIST_D_INT_CHUNK: TWIST_D_INT_CHUNK.
  MODULUS_377_S16_INT_CHUNK: MODULUS_377_S16_INT_CHUNK.
  MODULI: MODULI.
  array_rns_to_int: array_rns_to_int().
  icrt_factors: icrt_factors.
  int_point_to_jax_point_pack: int_point_to_jax_point_pack().
  construct_lazy_matrix: construct_lazy_matrix().
  overall_constant_offset: overall_constant_offset.
  int_list_to_array: int_list_to_array().
  int_point_batch_to_jax_point_pack: int_point_batch_to_jax_point_pack().
  MODULUS_377_INT: MODULUS_377_INT.
  U16_CHUNK_NUM: U16_CHUNK_NUM.
  int_to_array: int_to_array().
  rns_coefficients_precompute: rns_coefficients_precompute().
  moduli_precision: moduli_precision.
  overall_moduli: overall_moduli.
  TWIST_D_RNS: TWIST_D_RNS.
  jax_point_pack_to_int_point: jax_point_pack_to_int_point().
  jax_point_pack_to_int_point_batch: jax_point_pack_to_int_point_batch().
  MODULUS_377_LAZY_MAT: MODULUS_377_LAZY_MAT.
  RNS_MAT: RNS_MAT.
  int_to_array_rns: int_to_array_rns().
  to_rns: to_rns().
  num_bytes: num_bytes.
  MODULUS_377_S16_INT: MODULUS_377_S16_INT.
  NUM_MODULI: NUM_MODULI.
  array_to_int_list: array_to_int_list().
  array_rns_to_int_list: array_rns_to_int_list().
  int_point_to_jax_rns_point_pack: int_point_to_jax_rns_point_pack().
  rns_precompute: rns_precompute().
  rns_reconstruct: rns_reconstruct().
  find_moduli: find_moduli().
  RNS_MODULI_T: RNS_MODULI_T.
  total_modulus: total_modulus().
  int_list_to_array_rns: int_list_to_array_rns().
  jax_rns_point_pack_to_int_point: jax_rns_point_pack_to_int_point().
  int_point_batch_to_jax_rns_point_pack: int_point_batch_to_jax_rns_point_pack().
  jax_rns_point_pack_to_int_point_batch: jax_rns_point_pack_to_int_point_batch().
  TWIST_D_INT: TWIST_D_INT.
  U16_EXT_CHUNK_NUM: U16_EXT_CHUNK_NUM.
  array_to_int: array_to_int().
  MU_377_INT: MU_377_INT.
  U8_CHUNK_NUM: U8_CHUNK_NUM.
  ADDITION_BOUND: ADDITION_BOUND.
  rns_icrt_factors_compute: rns_icrt_factors_compute().
  RNS_STACK_MAT_NEW: RNS_STACK_MAT_NEW.
  gcd: gcd.
  BASE_TYPE: BASE_TYPE.
  U16_MASK: U16_MASK.
  U32_MASK: U32_MASK.
  U32_CHUNK_NUM: U32_CHUNK_NUM.
  U16_CHUNK_SHIFT_BITS: U16_CHUNK_SHIFT_BITS.
  U32_CHUNK_SHIFT_BITS: U32_CHUNK_SHIFT_BITS.
  BARRETT_SHIFT_U8: BARRETT_SHIFT_U8.
  CHUNK_PRECISION: CHUNK_PRECISION.
  COORDINATE_NUM: COORDINATE_NUM.
  RNS_PRECISION: RNS_PRECISION.
  print_hex_values: print_hex_values().
  profile_jax_functions: profile_jax_functions().
  get_parts: get_parts().
---
# Module: [`jaxite_ec/util.py`](../../../../../raw/code/jaxite/jaxite_ec/util.py)

## Functions
- `array_rns_to_int(residues)` — [`L212`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L212)
- `array_rns_to_int_list(jax_array)` — [`L225`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L225) — Converts JAX array to single integer.
- `array_to_int(jax_array: jax.Array, base)` — [`L142`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L142) — Converts a JAX array to a single Python integer.
- `array_to_int_list(jax_array, base)` — [`L171`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L171) — Converts JAX array to single integer.
- `construct_lazy_matrix(p, chunk_precision=8, chunk_num_u8=U8_CHUNK_NUM)` — [`L413`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L413) — Construct the lazy matrix. — documented in [jaxite_ec-util](../../concepts/jaxite_ec-util.md)
- `construct_rns_matrix(q)` — [`L603`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L603) — documented in [jaxite_ec-util](../../concepts/jaxite_ec-util.md)
- `find_moduli(total_modulus, precision)` — [`L443`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L443) — Finds a list of moduli close to the given precision. — documented in [jaxite_ec-util](../../concepts/jaxite_ec-util.md)
- `get_parts(u16mat)` — [`L570`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L570)
- `int_list_to_array(int_list, base=BASE, array_size=U16_CHUNK_NUM)` — [`L181`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L181) — Converts a list of integers to a JAX array.
- `int_list_to_array_rns(int_list)` — [`L217`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L217) — Converts a list of integers to a JAX array.
- `int_point_batch_to_jax_point_pack(points: List[List[int]], base=BASE, chunk_num=U16_CHUNK_NUM)` — [`L251`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L251)
- `int_point_batch_to_jax_rns_point_pack(points: List[List[int]])` — [`L268`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L268)
- `int_point_to_jax_point_pack(coordinates: List[int], base=BASE, chunk_num=U16_CHUNK_NUM)` — [`L189`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L189)
- `int_point_to_jax_rns_point_pack(coordinates: List[int])` — [`L235`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L235)
- `int_to_array(python_int, base=BASE, dtype=jnp.uint16, array_size=U16_CHUNK_NUM)` — [`L152`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L152) — Converts a Python integer to a JAX array.
- `int_to_array_rns(x)` — [`L208`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L208)
- `int_to_precomputed_array(python_int, base=BASE, dtype=jnp.uint16, array_size=U16_CHUNK_NUM)` — [`L326`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L326) — Converts a Python integer to a JAX array. — documented in [jaxite_ec-util](../../concepts/jaxite_ec-util.md)
- `jax_point_pack_to_int_point(point: jax.Array)` — [`L198`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L198)
- `jax_point_pack_to_int_point_batch(point_pack: jnp.ndarray, base=BASE)` — [`L260`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L260)
- `jax_rns_point_pack_to_int_point(point: jax.Array)` — [`L242`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L242)
- `jax_rns_point_pack_to_int_point_batch(point_pack: jnp.ndarray)` — [`L275`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L275)
- `print_hex_values(int_list)` — [`L137`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L137)
- `profile_jax_functions(tasks: List[Tuple[Callable[..., Any], Tuple[Any, ...]]], profile_name: str = "jax_profile")` — [`L350`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L350) — Profiles a list of JAX functions.
- `rns_coefficients_precompute(icrt_factors, overall_moduli, num_bytes, moduli_precision, overall_modulus, q)` — [`L493`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L493) — Precompute RNS coefficients. — documented in [jaxite_ec-util](../../concepts/jaxite_ec-util.md)
- `rns_icrt_factors_compute(modulus, moduli)` — [`L483`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L483) — documented in [jaxite_ec-util](../../concepts/jaxite_ec-util.md)
- `rns_precompute(moduli)` — [`L291`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L291)
- `rns_reconstruct(residues, moduli, precomputed)` — [`L302`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L302)
- `to_rns(x, moduli)` — [`L311`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L311) — documented in [jaxite_ec-util](../../concepts/jaxite_ec-util.md)
- `to_tuple(a)` — [`L316`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L316) — Create to convert numpy array into tuple. — documented in [jaxite_ec-util](../../concepts/jaxite_ec-util.md)
- `total_modulus(moduli)` — [`L284`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L284) — documented in [jaxite_ec-util](../../concepts/jaxite_ec-util.md)

## Module values
- `ADDITION_BOUND` — [`L122`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L122)
- `BARRETT_SHIFT_U8` — [`L49`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L49)
- `BASE` — [`L35`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L35) — documented in [jaxite_ec-util](../../concepts/jaxite_ec-util.md)
- `BASE_TYPE` — [`L36`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L36)
- `CHUNK_PRECISION` — [`L50`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L50)
- `COORDINATE_NUM` — [`L56`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L56)
- `COR_MAT_NEW` — [`L593`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L593) — documented in [jaxite_ec-util](../../concepts/jaxite_ec-util.md)
- `M` — [`L576`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L576) — documented in [jaxite_ec-util](../../concepts/jaxite_ec-util.md)
- `MODULI` — [`L113`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L113)
- `MODULI_SUB` — [`L125`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L125)
- `MODULUS_377_INT` — [`L26`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L26) — documented in [jaxite_ec-util](../../concepts/jaxite_ec-util.md)
- `MODULUS_377_INT_CHUNK` — [`L617`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L617) — documented in [jaxite_ec-util](../../concepts/jaxite_ec-util.md)
- `MODULUS_377_LAZY_MAT` — [`L435`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L435)
- `MODULUS_377_S16_INT` — [`L53`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L53)
- `MODULUS_377_S16_INT_CHUNK` — [`L626`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L626)
- `MU_377_INT` — [`L27`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L27)
- `MU_377_INT_CHUNK` — [`L620`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L620) — documented in [jaxite_ec-util](../../concepts/jaxite_ec-util.md)
- `NUM_MODULI` — [`L120`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L120)
- `RNS_MAT` — [`L616`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L616) — documented in [jaxite_ec-util](../../concepts/jaxite_ec-util.md)
- `RNS_MODULI_T` — [`L60`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L60)
- `RNS_PRECISION` — [`L119`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L119)
- `RNS_STACK_MAT_NEW` — [`L593`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L593)
- `TWIST_D_INT` — [`L28`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L28)
- `TWIST_D_INT_CHUNK` — [`L623`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L623)
- `TWIST_D_RNS` — [`L129`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L129)
- `U16_CHUNK_NUM` — [`L41`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L41) — documented in [jaxite_ec-util](../../concepts/jaxite_ec-util.md)
- `U16_CHUNK_SHIFT_BITS` — [`L43`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L43)
- `U16_EXT_CHUNK_NUM` — [`L46`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L46)
- `U16_MASK` — [`L37`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L37)
- `U32_CHUNK_NUM` — [`L42`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L42)
- `U32_CHUNK_SHIFT_BITS` — [`L44`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L44)
- `U32_MASK` — [`L38`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L38)
- `U8_CHUNK_NUM` — [`L40`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L40)
- `gcd` — [`L19`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L19)
- `icrt_factors` — [`L591`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L591) — documented in [jaxite_ec-util](../../concepts/jaxite_ec-util.md)
- `moduli` — [`L584`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L584)
- `moduli_precision` — [`L577`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L577)
- `num_bytes` — [`L578`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L578)
- `overall_constant_offset` — [`L582`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L582)
- `overall_moduli` — [`L582`](../../../../../raw/code/jaxite/jaxite_ec/util.py#L582) — documented in [jaxite_ec-util](../../concepts/jaxite_ec-util.md)

