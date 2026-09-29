---
title: 'Module: jax/_src/dtypes.py'
type: catalog
provenance: extracted
module: jax/_src/dtypes.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.dtypes`/
symbols:
  issubdtype: issubdtype().
  check_and_canonicalize_user_dtype: check_and_canonicalize_user_dtype().
  itemsize_bits: itemsize_bits().
  dtype: dtype().
  extended: extended#
  ExtendedDType: ExtendedDType#
  finfo: finfo.
  _least_upper_bound: _least_upper_bound().
  lattice_result_type: lattice_result_type().
  _custom_float_dtypes: _custom_float_dtypes.
  _custom_float_scalar_types: _custom_float_scalar_types.
  _issubdtype_cached: _issubdtype_cached().
  _int_types._int_types: _int_types._int_types.
  float0.float0: float0.float0.
  result_type: result_type().
  _intn_dtypes: _intn_dtypes.
  default_int_dtype: default_int_dtype().
  promote_types: promote_types().
  _type_promotion_lattice: _type_promotion_lattice().
  iinfo: iinfo.
  _jax_dtype_set: _jax_dtype_set.
  is_weakly_typed: is_weakly_typed().
  JAXType: JAXType.
  _float8_dtypes: _float8_dtypes.
  to_inexact_dtype: to_inexact_dtype().
  isdtype: isdtype().
  _maybe_canonicalize_explicit_dtype: _maybe_canonicalize_explicit_dtype().
  bfloat16.bfloat16: bfloat16.bfloat16.
  _dtype_kinds._dtype_kinds: _dtype_kinds._dtype_kinds.
  canonicalize_dtype: canonicalize_dtype().
  prng_key: prng_key#
  supports_inf: supports_inf().
  jax_dtype: jax_dtype().
  default_float_dtype: default_float_dtype().
  _canonicalize_dtype: _canonicalize_dtype().
  default_types.default_types: default_types.default_types.
  to_complex_dtype: to_complex_dtype().
  float8_e4m3fn.float8_e4m3fn: float8_e4m3fn.float8_e4m3fn.
  _float_types._float_types: _float_types._float_types.
  export: export.
  canonicalize_value: canonicalize_value.
  _jax_types: _jax_types.
  _dtype_and_weaktype: _dtype_and_weaktype().
  primal_tangent_dtype: primal_tangent_dtype().
  default_uint_dtype: default_uint_dtype().
  to_floating_dtype: to_floating_dtype().
  int1.int1: int1.int1.
  uint1.uint1: uint1.uint1.
  coerce_to_array: coerce_to_array().
  _jax_type: _jax_type().
  _DEFAULT_TYPEMAP._DEFAULT_TYPEMAP: _DEFAULT_TYPEMAP._DEFAULT_TYPEMAP.
  register_canonicalize_value_handler: register_canonicalize_value_handler.
  float8_e8m0fnu.float8_e8m0fnu: float8_e8m0fnu.float8_e8m0fnu.
  scalar_type_of: scalar_type_of().
  scalar_type_to_dtype: scalar_type_to_dtype().
  _weak_types._weak_types: _weak_types._weak_types.
  _signed_types._signed_types: _signed_types._signed_types.
  _unsigned_types._unsigned_types: _unsigned_types._unsigned_types.
  _complex_types._complex_types: _complex_types._complex_types.
  ExtendedDType._rules: ExtendedDType#_rules.
  default_complex_dtype: default_complex_dtype().
  _issubclass: _issubclass().
  _make_lattice_upper_bounds: _make_lattice_upper_bounds().
  uint2.uint2: uint2.uint2.
  python_scalar_types_to_dtypes.python_scalar_types_to_dtypes: python_scalar_types_to_dtypes.python_scalar_types_to_dtypes.
  _types_for_issubdtype: _types_for_issubdtype.
  string_dtype: string_dtype.
  check_valid_dtype: check_valid_dtype().
  float8_e4m3fnuz.float8_e4m3fnuz: float8_e4m3fnuz.float8_e4m3fnuz.
  float8_e5m2.float8_e5m2: float8_e5m2.float8_e5m2.
  float8_e5m2fnuz.float8_e5m2fnuz: float8_e5m2fnuz.float8_e5m2fnuz.
  int2.int2: int2.int2.
  _bool_types._bool_types: _bool_types._bool_types.
  to_numeric_dtype: to_numeric_dtype().
  safe_to_cast: safe_to_cast().
  float8_e4m3b11fnuz.float8_e4m3b11fnuz: float8_e4m3b11fnuz.float8_e4m3b11fnuz.
  _float6_dtypes._float6_dtypes: _float6_dtypes._float6_dtypes.
  int4.int4: int4.int4.
  uint4.uint4: uint4.uint4.
  python_scalar_types.python_scalar_types: python_scalar_types.python_scalar_types.
  TypePromotionError: TypePromotionError#
  short_dtype_name: short_dtype_name().
  _float8_e3m4_dtype._float8_e3m4_dtype: _float8_e3m4_dtype._float8_e3m4_dtype.
  _float8_e4m3_dtype._float8_e4m3_dtype: _float8_e4m3_dtype._float8_e4m3_dtype.
  _float8_e8m0fnu_dtype._float8_e8m0fnu_dtype: _float8_e8m0fnu_dtype._float8_e8m0fnu_dtype.
  _float8_e4m3b11fnuz_dtype._float8_e4m3b11fnuz_dtype: _float8_e4m3b11fnuz_dtype._float8_e4m3b11fnuz_dtype.
  _float8_e4m3fn_dtype._float8_e4m3fn_dtype: _float8_e4m3fn_dtype._float8_e4m3fn_dtype.
  _float8_e4m3fnuz_dtype._float8_e4m3fnuz_dtype: _float8_e4m3fnuz_dtype._float8_e4m3fnuz_dtype.
  _float8_e5m2_dtype._float8_e5m2_dtype: _float8_e5m2_dtype._float8_e5m2_dtype.
  _float8_e5m2fnuz_dtype._float8_e5m2fnuz_dtype: _float8_e5m2fnuz_dtype._float8_e5m2fnuz_dtype.
  _float4_e2m1fn_dtype._float4_e2m1fn_dtype: _float4_e2m1fn_dtype._float4_e2m1fn_dtype.
  _types_whose_dtype_should_not_be_canonicalized._types_whose_dtype_should_not_be_canonicalized: _types_whose_dtype_should_not_be_canonicalized._types_whose_dtype_should_not_be_canonicalized.
  PrimalTangentDType._rules: PrimalTangentDType#_rules().
  float8_e3m4.float8_e3m4: float8_e3m4.float8_e3m4.
  float8_e4m3.float8_e4m3: float8_e4m3.float8_e4m3.
  ExtendedDType.type: ExtendedDType#type().
  _int4_dtype: _int4_dtype.
  _uint4_dtype: _uint4_dtype.
  is_weakly_typed_scalar: is_weakly_typed_scalar().
  _float6_e2m3fn_dtype._float6_e2m3fn_dtype: _float6_e2m3fn_dtype._float6_e2m3fn_dtype.
  _float6_e3m2fn_dtype._float6_e3m2fn_dtype: _float6_e3m2fn_dtype._float6_e3m2fn_dtype.
  _bfloat16_dtype._bfloat16_dtype: _bfloat16_dtype._bfloat16_dtype.
  _dtype_to_inexact._dtype_to_inexact: _dtype_to_inexact._dtype_to_inexact.
  _registered_weak_types._registered_weak_types: _registered_weak_types._registered_weak_types.
  InvalidInputException: InvalidInputException#
  float4_e2m1fn.float4_e2m1fn: float4_e2m1fn.float4_e2m1fn.
  _ml_dtypes_version: _ml_dtypes_version.
  _standard_x64_lattice_ubs: _standard_x64_lattice_ubs.
  _standard_x32_lattice_ubs: _standard_x32_lattice_ubs.
  _strict_lattice_ubs: _strict_lattice_ubs.
  primal_tangent_dtype_scalar: primal_tangent_dtype_scalar#
  PrimalTangentDType: PrimalTangentDType#
  _float4_dtypes._float4_dtypes: _float4_dtypes._float4_dtypes.
  _int2_dtype._int2_dtype: _int2_dtype._int2_dtype.
  _uint2_dtype._uint2_dtype: _uint2_dtype._uint2_dtype.
  _dtype_to_32bit_dtype._dtype_to_32bit_dtype: _dtype_to_32bit_dtype._dtype_to_32bit_dtype.
  bool_: bool_.
  float6_e2m3fn.float6_e2m3fn: float6_e2m3fn.float6_e2m3fn.
  float6_e3m2fn.float6_e3m2fn: float6_e3m2fn.float6_e3m2fn.
  _int1_dtype._int1_dtype: _int1_dtype._int1_dtype.
  _uint1_dtype._uint1_dtype: _uint1_dtype._uint1_dtype.
  _CanonicalizeValueHandlersDict.__getitem__: _CanonicalizeValueHandlersDict#__getitem__().
  _CanonicalizeValueHandlersDict.__setitem__: _CanonicalizeValueHandlersDict#__setitem__().
  canonicalize_value_handlers: canonicalize_value_handlers.
  register_weak_scalar_type: register_weak_scalar_type().
  register_type_whose_dtype_should_not_be_canonicalized: register_type_whose_dtype_should_not_be_canonicalized().
  PrimalTangentDType.type: PrimalTangentDType#type.
  PrimalTangentDType.__repr__: PrimalTangentDType#__repr__().
  _CanonicalizeValueHandlersDict: _CanonicalizeValueHandlersDict#
  can_cast: can_cast.
  PrimalTangentDType.primal_dtype: PrimalTangentDType#primal_dtype.
  PrimalTangentDType.tangent_dtype: PrimalTangentDType#tangent_dtype.
  PrimalTangentDType.name: PrimalTangentDType#name.
  int_.int_: int_.int_.
  uint.uint: uint.uint.
  float_.float_: float_.float_.
  complex_.complex_: complex_.complex_.
---
# Module: [`jax/_src/dtypes.py`](../../../../../../raw/code/jax/jax/_src/dtypes.py)

## Classes
### `ExtendedDType`  ·  implements/extends StrictABC
- def: [`jax/_src/dtypes.py:86`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L86)
- doc: Abstract Base Class for extended dtypes
- signature: `class ExtendedDType(StrictABC):`
- members:
  - `type(self)` — [`L90`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L90)
- protocol/private: `_rules`[`L92`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L92)
- uses (calls/refs, reference-scoped): [`StrictABC`](util.md#StrictABC), [`AbstractSemaphoreTy`](pallas/core.md#AbstractSemaphoreTy), [`BarrierType`](pallas/mosaic_gpu/core.md#BarrierType), [`ClusterBarrierType`](pallas/mosaic_gpu/core.md#ClusterBarrierType), [`FusionDType`](pallas/fuser/fusible_dtype.md#FusionDType), [`PrimalTangentDType`](dtypes.md#PrimalTangentDType), [`bint`](core.md#bint), [`type`](core.md#bint.type)
- used by: [`issubdtype`](dtypes.md#issubdtype), [`full_like`](lax/lax.md#full_like), [`convert_element_type`](lax/lax.md#convert_element_type), [`_convert_element_type`](lax/lax.md#_convert_element_type), [`full`](lax/lax.md#full), [`_issubdtype_cached`](dtypes.md#_issubdtype_cached), [`_to_edtype_abstract_eval`](lax/lax.md#_to_edtype_abstract_eval), [`_from_edtype_abstract_eval`](lax/lax.md#_from_edtype_abstract_eval), [`physical_element_aval`](core.md#physical_element_aval), [`primal_dtype_to_tangent_dtype`](core.md#primal_dtype_to_tangent_dtype), [`get_array_aval`](pallas/core.md#MemoryRef.get_array_aval), [`naryop_dtype_rule`](lax/lax.md#naryop_dtype_rule), [`StrictABC`](util.md#StrictABC), [`_canonicalize_dtype`](dtypes.md#_canonicalize_dtype), [`AbstractSemaphoreTy`](pallas/core.md#AbstractSemaphoreTy), [`BarrierType`](pallas/mosaic_gpu/core.md#BarrierType), [`ClusterBarrierType`](pallas/mosaic_gpu/core.md#ClusterBarrierType), [`FusionDType`](pallas/fuser/fusible_dtype.md#FusionDType), [`primal_tangent_dtype`](dtypes.md#primal_tangent_dtype), [`_types_for_issubdtype`](dtypes.md#_types_for_issubdtype), [`short_dtype_name`](dtypes.md#short_dtype_name), [`PrimalTangentDType`](dtypes.md#PrimalTangentDType), [`bint`](core.md#bint), [`_dtype_object_types`](core.md#_dtype_object_types)

### `InvalidInputException`  ·  implements/extends TypeError
- def: [`jax/_src/dtypes.py:388`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L388)
- signature: `class InvalidInputException(TypeError):`
- used by: [`_run_python_pjit`](pjit.md#_run_python_pjit)

### `PrimalTangentDType`  ·  implements/extends ExtendedDType
- def: [`jax/_src/dtypes.py:1168`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L1168)
- signature: `class PrimalTangentDType(ExtendedDType):`
- members:
  - `name` — [`L1171`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L1171)
  - `primal_dtype` — [`L1169`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L1169)
  - `tangent_dtype` — [`L1170`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L1170)
  - `type` — [`L1172`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L1172)
- protocol/private: `__repr__`[`L1173`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L1173), `_rules`[`L1175`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L1175)
- uses (calls/refs, reference-scoped): [`ExtendedDType`](dtypes.md#ExtendedDType), [`primal_tangent_dtype_scalar`](dtypes.md#primal_tangent_dtype_scalar)
- used by: [`ExtendedDType`](dtypes.md#ExtendedDType), [`primal_tangent_dtype`](dtypes.md#primal_tangent_dtype)

### `TypePromotionError`  ·  implements/extends ValueError
- def: [`jax/_src/dtypes.py:781`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L781)
- doc: Raised when JAX type promotion fails.
- signature: `class TypePromotionError(ValueError):`
- uses (calls/refs, reference-scoped): [`export`](dtypes.md#export)
- used by: [`_least_upper_bound`](dtypes.md#_least_upper_bound), [`_check_all_safe_to_cast`](random/core.md#_check_all_safe_to_cast)

### `_CanonicalizeValueHandlersDict`
- def: [`jax/_src/dtypes.py:397`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L397)
- signature: `class _CanonicalizeValueHandlersDict:`
- protocol/private: `__getitem__`[`L399`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L399), `__setitem__`[`L402`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L402)
- uses (calls/refs, reference-scoped): [`canonicalize_value`](dtypes.md#canonicalize_value), [`register_canonicalize_value_handler`](dtypes.md#register_canonicalize_value_handler)
- used by: [`canonicalize_value_handlers`](dtypes.md#canonicalize_value_handlers)

### `extended`  ·  implements/extends generic
- def: [`jax/_src/dtypes.py:55`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L55)
- doc: Scalar class for extended dtypes.
- signature: `class extended(np.generic):`
- uses (calls/refs, reference-scoped): [`prng_key`](dtypes.md#prng_key), [`semaphore_dtype`](pallas/core.md#semaphore_dtype), [`barrier_dtype`](pallas/mosaic_gpu/core.md#barrier_dtype), [`primal_tangent_dtype_scalar`](dtypes.md#primal_tangent_dtype_scalar)
- used by: [`check_and_canonicalize_user_dtype`](dtypes.md#check_and_canonicalize_user_dtype), [`_call_exported_lowering`](export/_export.md#_call_exported_lowering), [`make_array_from_callback`](array.md#make_array_from_callback), [`dtype`](dtypes.md#dtype), [`_dtype_to_ir_type`](pallas/mosaic/lowering.md#_dtype_to_ir_type), [`isclose`](numpy/lax_numpy.md#isclose), [`_scatter_lower`](lax/slicing.md#_scatter_lower), [`_sharding_constraint_hlo_lowering`](pjit.md#_sharding_constraint_hlo_lowering), [`zeros_like_shaped_array`](lax/lax.md#zeros_like_shaped_array), [`make_array_from_single_device_arrays`](array.md#make_array_from_single_device_arrays), [`_issubdtype_cached`](dtypes.md#_issubdtype_cached), [`_device_put_replicated`](api.md#device_put_replicated._device_put_replicated), [`_gather_lower`](lax/slicing.md#_gather_lower), [`__init__`](core.md#ShapeDtypeStruct.__init__), [`_empty_custom_call_lower`](lax/lax.md#_empty_custom_call_lower), [`_get_fastpath_data`](pjit.md#_get_fastpath_data), [`_device_get`](api.md#_device_get), [`_shardy_shard_map_sharding`](shard_map.md#_shardy_shard_map_sharding), [`_add_arrays`](lax/lax.md#_add_arrays), [`_device_put_sharded`](api.md#device_put_sharded._device_put_sharded), [`check_same_dtypes`](lax/lax.md#check_same_dtypes), [`prng_key`](dtypes.md#prng_key), [`_temporary_dtype_exception`](custom_derivatives.md#_temporary_dtype_exception), [`_compare_lower_hlo`](lax/lax.md#_compare_lower_hlo), [`semaphore_dtype`](pallas/core.md#semaphore_dtype), [`_empty2_lower`](lax/lax.md#_empty2_lower), [`_xla_shard`](shard_map.md#_xla_shard), [`_xla_unshard`](shard_map.md#_xla_unshard), [`jax_dtype`](dtypes.md#jax_dtype), [`should_physicalize_dtype`](pallas/mosaic/lowering.md#should_physicalize_dtype), [`_array_global_result_handler`](array.md#_array_global_result_handler), [`_check_input_dtype_revderiv`](api.md#_check_input_dtype_revderiv), [`_empty_lower`](lax/lax.md#_empty_lower), [`_stop_gradient`](lax/lax.md#_stop_gradient), [`_canonicalize_dtype`](dtypes.md#_canonicalize_dtype), [`_select_hlo_lowering`](lax/lax.md#_select_hlo_lowering), [`_convert_elt_type_folding_rule`](lax/lax.md#_convert_elt_type_folding_rule), [`_transpose_lower`](lax/lax.md#_transpose_lower), [`_check_output_dtype_revderiv`](api.md#_check_output_dtype_revderiv), [`_check_input_dtype_jacfwd`](api.md#_check_input_dtype_jacfwd)  (+5 more)

### `primal_tangent_dtype_scalar`  ·  implements/extends extended
- def: [`jax/_src/dtypes.py:1165`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L1165)
- signature: `class primal_tangent_dtype_scalar(extended):`
- uses (calls/refs, reference-scoped): [`extended`](dtypes.md#extended)
- used by: [`extended`](dtypes.md#extended), [`type`](dtypes.md#PrimalTangentDType.type)

### `prng_key`  ·  implements/extends extended
- def: [`jax/_src/dtypes.py:71`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L71)
- doc: Scalar class for PRNG Key dtypes.
- signature: `class prng_key(extended):`
- uses (calls/refs, reference-scoped): [`extended`](dtypes.md#extended), [`export`](dtypes.md#export)
- used by: [`_check_prng_key`](random/core.md#_check_prng_key), [`gather`](lax/slicing.md#gather), [`extended`](dtypes.md#extended), [`_split`](random/core.md#_split), [`to_pallas_key`](pallas/mosaic/random.md#to_pallas_key), [`_slice_signature`](../experimental/key_reuse/_core.md#_slice_signature), [`_key`](random/core.md#_key), [`_random_bits`](random/core.md#_random_bits), [`is_key`](../experimental/key_reuse/_core.md#jaxpr_type_signature.is_key), [`_key_impl`](random/core.md#_key_impl), [`check_for_prngkeys`](numpy/util.md#check_for_prngkeys), [`_key_data`](random/core.md#_key_data), [`is_key`](../experimental/key_reuse/_core.md#unknown_signature.is_key), [`_return_prng_keys`](random/core.md#_return_prng_keys)

## Functions
- `_canonicalize_dtype(x64_enabled: bool, allow_extended_dtype: bool, dtype: Any)` — [`L349`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L349)
- `_dtype_and_weaktype(value: Any)` — [`L692`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L692) — Return a (dtype, weak_type) tuple for the given input.
- `_issubclass(a: Any, b: Any)` — [`L483`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L483) — Determines if ``a`` is a subclass of ``b``.
- `_issubdtype_cached(a: type | np.dtype | ExtendedDType, b: type | np.dtype | ExtendedDType)` — [`L525`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L525)
- `_jax_type(dtype: DType, weak_type: bool)` — [`L682`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L682) — Return the jax type for a dtype and weak type.
- `_least_upper_bound(jax_numpy_dtype_promotion: config.NumpyDtypePromotion, x64: bool, *nodes: JAXType)` — [`L788`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L788) — Compute the least upper bound of a set of nodes.
- `_make_lattice_upper_bounds(strict: bool, x64: bool)` — [`L762`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L762)
- `_maybe_canonicalize_explicit_dtype(dtype: DType, fun_name: str)` — [`L957`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L957) — Canonicalizes explicitly requested dtypes, per explicit_x64_dtypes.
- `_type_promotion_lattice(strict: bool, x64: bool)` — [`L696`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L696) — Return the type promotion lattice in the form of a DAG.
- `canonicalize_dtype(dtype: Any, allow_extended_dtype: Literal[False] = False)` — [`L368`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L368)
- `check_and_canonicalize_user_dtype(dtype, fun_name=None)` — [`L1104`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L1104) — Checks validity of a user-provided dtype, and returns its canonical form. — documented in [jax-_src-dtypes](../../../concepts/jax-_src-dtypes.md)
- `check_valid_dtype(dtype: DType)` — [`L952`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L952)
- `coerce_to_array(x: Any, dtype: DTypeLike | None = None)` — [`L470`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L470) — Coerces a scalar or NumPy array to an np.array.
- `default_complex_dtype()` — [`L245`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L245)
- `default_float_dtype()` — [`L239`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L239)
- `default_int_dtype()` — [`L231`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L231)
- `default_uint_dtype()` — [`L235`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L235)
- `dtype(x: Any)` — [`L992`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L992) — Return the dtype object for a value or type. — documented in [jax-_src-dtypes](../../../concepts/jax-_src-dtypes.md)
- `is_weakly_typed(x: Any)` — [`L938`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L938)
- `is_weakly_typed_scalar(x: Any)` — [`L946`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L946)
- `isdtype(dtype: DTypeLike, kind: str | DTypeLike | tuple[str | DTypeLike, ...])` — [`L637`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L637) — Returns a boolean indicating whether a provided dtype is of a specified kind.
- `issubdtype(a: DTypeLike | ExtendedDType | None, b: DTypeLike | ExtendedDType | None)` — [`L500`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L500) — Returns True if first argument is a typecode lower/equal in type hierarchy. — documented in [jax-_src-dtypes](../../../concepts/jax-_src-dtypes.md)
- `itemsize_bits(dtype: DTypeLike)` — [`L282`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L282) — Number of bits per element for the dtype. — documented in [jax-_src-dtypes](../../../concepts/jax-_src-dtypes.md)
- `jax_dtype(obj: DTypeLike | None, *, align: bool = False, copy: bool = False)` — [`L261`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L261) — Cast an object to a dtype, respecting JAX dtype defaults.
- `lattice_result_type(*args: Any)` — [`L1049`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L1049)
- `primal_tangent_dtype(primal_dtype, tangent_dtype, name: str | None = None)` — [`L1182`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L1182)
- `promote_types(a: DTypeLike, b: DTypeLike)` — [`L881`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L881) — Returns the type to which a binary operation should cast its arguments.
- `register_type_whose_dtype_should_not_be_canonicalized(typ: type)` — [`L988`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L988)
- `register_weak_scalar_type(typ: type)` — [`L931`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L931) — Register a scalar type as a weak type.
- `result_type(*args: Any, return_weak_type_flag: Literal[True])` — [`L1077`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L1077)
- `safe_to_cast(input_dtype_or_value: Any, output_dtype_or_value: Any)` — [`L1128`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L1128) — Check if a dtype/value is safe to cast to another dtype/value
- `scalar_type_of(x: Any)` — [`L420`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L420) — Return the scalar type associated with a JAX value.
- `scalar_type_to_dtype(typ: type, value: Any = None)` — [`L439`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L439) — Return the numpy dtype for the given scalar type.
- `short_dtype_name(dtype)` — [`L1190`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L1190)
- `supports_inf(dtype: DTypeLike)` — [`L125`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L125) — Return true if the dtype supports infinity, else return False.
- `to_complex_dtype(dtype: DTypeLike)` — [`L341`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L341)
- `to_floating_dtype(dtype: DTypeLike)` — [`L335`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L335) — Promotes a dtype to a non-complex floating dtype.
- `to_inexact_dtype(dtype: DTypeLike)` — [`L329`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L329) — Promotes a dtype into an inexact dtype, if it is not already one.
- `to_numeric_dtype(dtype: DTypeLike)` — [`L323`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L323) — Promotes a dtype into an numeric dtype, if it is not already one.

## Module values
- `JAXType` — [`L560`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L560)
- `_DEFAULT_TYPEMAP` — [`L275`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L275)
- `_bfloat16_dtype` — [`L134`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L134)
- `_bool_types` — [`L564`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L564)
- `_complex_types` — [`L598`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L598)
- `_custom_float_dtypes` — [`L150`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L150)
- `_custom_float_scalar_types` — [`L136`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L136)
- `_dtype_kinds` — [`L625`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L625)
- `_dtype_to_32bit_dtype` — [`L302`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L302)
- `_dtype_to_inexact` — [`L312`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L312)
- `_float4_dtypes` — [`L180`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L180)
- `_float4_e2m1fn_dtype` — [`L123`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L123)
- `_float6_dtypes` — [`L175`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L175)
- `_float6_e2m3fn_dtype` — [`L117`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L117)
- `_float6_e3m2fn_dtype` — [`L118`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L118)
- `_float8_dtypes` — [`L164`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L164)
- `_float8_e3m4_dtype` — [`L104`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L104)
- `_float8_e4m3_dtype` — [`L105`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L105)
- `_float8_e4m3b11fnuz_dtype` — [`L107`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L107)
- `_float8_e4m3fn_dtype` — [`L108`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L108)
- `_float8_e4m3fnuz_dtype` — [`L109`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L109)
- `_float8_e5m2_dtype` — [`L110`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L110)
- `_float8_e5m2fnuz_dtype` — [`L111`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L111)
- `_float8_e8m0fnu_dtype` — [`L106`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L106)
- `_float_types` — [`L592`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L592)
- `_int1_dtype` — [`L186`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L186)
- `_int2_dtype` — [`L192`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L192)
- `_int4_dtype` — [`L198`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L198)
- `_int_types` — [`L567`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L567)
- `_intn_dtypes` — [`L201`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L201)
- `_jax_dtype_set` — [`L612`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L612)
- `_jax_types` — [`L623`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L623)
- `_ml_dtypes_version` — [`L44`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L44)
- `_registered_weak_types` — [`L935`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L935)
- `_signed_types` — [`L565`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L565)
- `_standard_x32_lattice_ubs` — [`L776`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L776)
- `_standard_x64_lattice_ubs` — [`L775`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L775)
- `_strict_lattice_ubs` — [`L777`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L777)
- `_types_for_issubdtype` — [`L495`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L495)
- `_types_whose_dtype_should_not_be_canonicalized` — [`L984`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L984)
- `_uint1_dtype` — [`L187`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L187)
- `_uint2_dtype` — [`L193`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L193)
- `_uint4_dtype` — [`L199`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L199)
- `_unsigned_types` — [`L566`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L566)
- `_weak_types` — [`L563`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L563)
- `bfloat16` — [`L133`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L133)
- `bool_` — [`L219`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L219)
- `can_cast` — [`L558`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L558)
- `canonicalize_value` — [`L394`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L394)
- `canonicalize_value_handlers` — [`L405`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L405)
- `complex_` — [`L223`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L223)
- `default_types` — [`L253`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L253)
- `export` — [`L52`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L52)
- `finfo` — [`L481`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L481)
- `float0` — [`L300`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L300)
- `float4_e2m1fn` — [`L121`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L121)
- `float6_e2m3fn` — [`L114`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L114)
- `float6_e3m2fn` — [`L115`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L115)
- `float8_e3m4` — [`L95`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L95)
- `float8_e4m3` — [`L96`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L96)
- `float8_e4m3b11fnuz` — [`L98`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L98)
- `float8_e4m3fn` — [`L99`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L99)
- `float8_e4m3fnuz` — [`L100`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L100)
- `float8_e5m2` — [`L101`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L101)
- `float8_e5m2fnuz` — [`L102`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L102)
- `float8_e8m0fnu` — [`L97`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L97)
- `float_` — [`L222`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L222)
- `iinfo` — [`L480`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L480)
- `int1` — [`L184`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L184)
- `int2` — [`L189`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L189)
- `int4` — [`L196`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L196)
- `int_` — [`L220`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L220)
- `python_scalar_types` — [`L409`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L409)
- `python_scalar_types_to_dtypes` — [`L412`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L412)
- `register_canonicalize_value_handler` — [`L393`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L393)
- `string_dtype` — [`L610`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L610)
- `uint` — [`L221`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L221)
- `uint1` — [`L185`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L185)
- `uint2` — [`L190`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L190)
- `uint4` — [`L197`](../../../../../../raw/code/jax/jax/_src/dtypes.py#L197)

