---
title: 'Module: jax/_src/pallas/triton/lowering.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/triton/lowering.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.triton.lowering`/
symbols:
  nextafter_dispatch_table: nextafter_dispatch_table.
  _Extern: _Extern#
  _dot_general_lowering: _dot_general_lowering().
  lower_jaxpr_to_triton_ir: lower_jaxpr_to_triton_ir().
  _compute_offsets_from_indices: _compute_offsets_from_indices().
  _masked_load_lowering_rule: _masked_load_lowering_rule().
  lower_jaxpr_to_triton_module: lower_jaxpr_to_triton_module().
  _Fallback: _Fallback#
  LoweringRuleContext: LoweringRuleContext#
  register_lowering: register_lowering().
  LoweringRuleContext.avals_in: LoweringRuleContext#avals_in.
  _ir_cast: _ir_cast().
  _JAX_TO_TRITON_SIGNED_BINARY: _JAX_TO_TRITON_SIGNED_BINARY.
  _eval_index_map: _eval_index_map().
  _JAX_FN_MAPPING: _JAX_FN_MAPPING.
  _Extern.lower: _Extern#lower().
  _reinterpret_int4_as_uint8: _reinterpret_int4_as_uint8().
  _make_dispatch_table: _make_dispatch_table().
  _reduction_lowering: _reduction_lowering().
  _associative_scan_lowering: _associative_scan_lowering().
  _masked_swap_lowering_rule: _masked_swap_lowering_rule().
  _JAX_TO_TRITON_BINARY: _JAX_TO_TRITON_BINARY.
  _is_contiguous_int4: _is_contiguous_int4().
  _cond_lowering_rule: _cond_lowering_rule().
  _integer_pow_rule: _integer_pow_rule().
  _ensure_ir_value: _ensure_ir_value().
  _argreduce_lowering: _argreduce_lowering().
  _reshape_lowering_rule: _reshape_lowering_rule().
  _div_lowering_rule: _div_lowering_rule().
  _scan_lowering_rule: _scan_lowering_rule().
  _element_type: _element_type().
  _max_lowering_rule: _max_lowering_rule().
  _reduce_lowering: _reduce_lowering().
  _while_lowering_rule: _while_lowering_rule().
  _bcast: _bcast().
  abs_dispatch_table: abs_dispatch_table.
  _maybe_pattern_match_fori_loop: _maybe_pattern_match_fori_loop().
  _min_lowering_rule: _min_lowering_rule().
  _convert_element_type_lowering_rule: _convert_element_type_lowering_rule().
  _addupdate_lowering_rule: _addupdate_lowering_rule().
  LoweringRuleContext.context: LoweringRuleContext#context.
  pow_dispatch_table: pow_dispatch_table.
  signed_rule: signed_rule().
  _bcast_to: _bcast_to().
  debug_print_lowering_rule: debug_print_lowering_rule().
  _iota_lowering_rule: _iota_lowering_rule().
  _concatenate_lowering_rule: _concatenate_lowering_rule().
  _cast: _cast().
  _lower_jaxpr_to_for_loop: _lower_jaxpr_to_for_loop().
  _is_triton_pointer_type: _is_triton_pointer_type().
  _process_grid_to_3d_grid: _process_grid_to_3d_grid().
  _make_dispatch_table.inner: _make_dispatch_table().inner().
  _closed_call_lowering_rule: _closed_call_lowering_rule().
  _dtype_to_ir_type: _dtype_to_ir_type().
  _compute_pointers_from_indices: _compute_pointers_from_indices().
  _Fallback.lower: _Fallback#lower().
  _full: _full().
  _broadcast_in_dim_lowering_rule: _broadcast_in_dim_lowering_rule().
  _get_lowering_rule: _get_lowering_rule().
  _swap_lowering_rule: _swap_lowering_rule().
  _axis_index_rule: _axis_index_rule().
  LoweringRuleContext.avals_out: LoweringRuleContext#avals_out.
  _eval_index_map._get_start_index: _eval_index_map()._get_start_index().
  ceil_dispatch_table: ceil_dispatch_table.
  floor_dispatch_table: floor_dispatch_table.
  exp_dispatch_table: exp_dispatch_table.
  exp2_dispatch_table: exp2_dispatch_table.
  expm1_dispatch_table: expm1_dispatch_table.
  log_dispatch_table: log_dispatch_table.
  log1p_dispatch_table: log1p_dispatch_table.
  sqrt_dispatch_table: sqrt_dispatch_table.
  cbrt_dispatch_table: cbrt_dispatch_table.
  rsqrt_dispatch_table: rsqrt_dispatch_table.
  sin_dispatch_table: sin_dispatch_table.
  cos_dispatch_table: cos_dispatch_table.
  tan_dispatch_table: tan_dispatch_table.
  asin_dispatch_table: asin_dispatch_table.
  acos_dispatch_table: acos_dispatch_table.
  atan_dispatch_table: atan_dispatch_table.
  atan2_dispatch_table: atan2_dispatch_table.
  sinh_dispatch_table: sinh_dispatch_table.
  cosh_dispatch_table: cosh_dispatch_table.
  tanh_dispatch_table: tanh_dispatch_table.
  asinh_dispatch_table: asinh_dispatch_table.
  acosh_dispatch_table: acosh_dispatch_table.
  atanh_dispatch_table: atanh_dispatch_table.
  population_count_dispatch_table: population_count_dispatch_table.
  clz_dispatch_table: clz_dispatch_table.
  _not_lowering_rule: _not_lowering_rule().
  _multiple_of_rule: _multiple_of_rule().
  _broadcast_to_rule: _broadcast_to_rule().
  select_n_lowering_rule: select_n_lowering_rule().
  _unstack_lowering_rule: _unstack_lowering_rule().
  _pjit_lowering_rule: _pjit_lowering_rule().
  _remat_lowering_rule: _remat_lowering_rule().
  _sub: _sub().
  _load: _load().
  LoweringRuleContext.block_infos: LoweringRuleContext#block_infos.
  _get_index_alignment._get_bdim_alignment: _get_index_alignment()._get_bdim_alignment().
  _program_id_lowering_rule: _program_id_lowering_rule().
  _Extern.matches: _Extern#matches().
  signless_rule: signless_rule().
  _store: _store().
  _as_bf16: _as_bf16().
  triton_lowering_rules: triton_lowering_rules.
  _add: _add().
  _cumsum_lowering_rule: _cumsum_lowering_rule().
  _stack_lowering_rule: _stack_lowering_rule().
  _split_lowering_rule: _split_lowering_rule().
  _bitcast_convert_type_lowering_rule: _bitcast_convert_type_lowering_rule().
  _minus: _minus().
  _as_f32: _as_f32().
  lower_fun.f_lowered: lower_fun().f_lowered().
  _float_int_cast: _float_int_cast().
  _ir_constant: _ir_constant().
  _mul: _mul().
  _Fallback.matches: _Fallback#matches().
  _squeeze_lowering_rule: _squeeze_lowering_rule().
  NDIndexer: NDIndexer.
  _expand_dims: _expand_dims().
  _TF32_PRECISIONS: _TF32_PRECISIONS.
  _fp_bits_type: _fp_bits_type().
  _get_index_alignment: _get_index_alignment().
  _cmp: _cmp().
  _int_int_cast: _int_int_cast().
  _reduce_argmax_combine: _reduce_argmax_combine().
  BlockInfo: BlockInfo#
  BlockInfo.full_shape_dtype: BlockInfo#full_shape_dtype.
  _num_programs_lowering_rule: _num_programs_lowering_rule().
  _zeros_like: _zeros_like().
  _reshape: _reshape().
  _transpose_lowering: _transpose_lowering().
  fn: fn.
  lower_jaxpr_to_triton_ir.read_env: lower_jaxpr_to_triton_ir().read_env().
  _truediv: _truediv().
  _floordiv: _floordiv().
  _not_equal: _not_equal.
  _reduce_argmin_combine: _reduce_argmin_combine().
  GridMapping: GridMapping.
  lower_jaxpr_to_triton_ir.write_env: lower_jaxpr_to_triton_ir().write_env().
  lower_fun: lower_fun().
  _program_id: _program_id().
  _Extern.arg_types: _Extern#arg_types.
  _mod: _mod().
  _int_float_cast: _int_float_cast().
  ModuleContext.mlir_ctx: ModuleContext#mlir_ctx.
  _reshard_lowering_rule: _reshard_lowering_rule().
  _i32_constant: _i32_constant().
  prim: prim.
  BlockMapping: BlockMapping.
  BlockInfo.block_shape: BlockInfo#block_shape.
  _Fallback.arg_classes: _Fallback#arg_classes.
  _equal: _equal.
  _less_than: _less_than.
  _greater_equal: _greater_equal.
  _zeros: _zeros().
  _ones_like: _ones_like().
  ModuleContext: ModuleContext#
  BlockInfo.start_indices: BlockInfo#start_indices.
  LoweringRuleContext.replace: LoweringRuleContext#replace.
  _atomic_rmw: _atomic_rmw().
  _make_range: _make_range().
  ModuleContext.grid_mapping: ModuleContext#grid_mapping.
  register_lowering.wrapper: register_lowering().wrapper().
  _check_tensor_size: _check_tensor_size().
  lower_jaxpr_to_triton_ir.read_block_info_env: lower_jaxpr_to_triton_ir().read_block_info_env().
  _less_equal: _less_equal.
  _greater_than: _greater_than.
  _float_float_cast: _float_float_cast().
  _UNSUPPORTED_CAST_DTYPES: _UNSUPPORTED_CAST_DTYPES.
  _split_lowering_rule.split_into_2: _split_lowering_rule().split_into_2().
  _cond_lowering_rule.to_type: _cond_lowering_rule().to_type().
  Blocked: Blocked.
  _ones: _ones().
  unsafe_map: unsafe_map.
  unsafe_zip: unsafe_zip.
  _T: _T.
  ModuleContext.platform: ModuleContext#platform.
  BlockInfo.start_indices_alignment: BlockInfo#start_indices_alignment.
  LoweringResult: LoweringResult#
  LoweringError: LoweringError#
  _set_attr: _set_attr().
  _splat: _splat().
  get_join_type: get_join_type().
  _STR_TO_EVICTION_POLICY: _STR_TO_EVICTION_POLICY.
  _STR_TO_CACHE_MODIFIER: _STR_TO_CACHE_MODIFIER.
  ModuleContext.program_ids: ModuleContext#program_ids.
  ModuleContext.compute_capability: ModuleContext#compute_capability.
  LoweringResult.module: LoweringResult#module.
  LoweringResult.grid: LoweringResult#grid.
  _new_ir_context: _new_ir_context().
  _Extern.symbol: _Extern#symbol.
  _Extern.result_type: _Extern#result_type.
  _Fallback.op: _Fallback#op.
  _is_nan: _is_nan().
  _is_float8_e4m3fn_cast_supported: _is_float8_e4m3fn_cast_supported().
  _i64_constant: _i64_constant().
  ModuleContext.name: ModuleContext#name.
  ModuleContext.traceback_caches: ModuleContext#traceback_caches.
---
# Module: [`jax/_src/pallas/triton/lowering.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py)

## Classes
### `BlockInfo`
- def: [`jax/_src/pallas/triton/lowering.py:82`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L82)
- signature: `class BlockInfo:`
- members:
  - `block_shape` — [`L86`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L86)
  - `full_shape_dtype` — [`L83`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L83)
  - `start_indices` — [`L84`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L84)
  - `start_indices_alignment` — [`L85`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L85)
- uses (calls/refs, reference-scoped): [`ShapedArray`](../../core.md#ShapedArray), [`Squeezed`](../core.md#Squeezed)
- used by: [`_compute_offsets_from_indices`](lowering.md#_compute_offsets_from_indices), [`lower_jaxpr_to_triton_ir`](lowering.md#lower_jaxpr_to_triton_ir), [`_masked_load_lowering_rule`](lowering.md#_masked_load_lowering_rule), [`lower_jaxpr_to_triton_module`](lowering.md#lower_jaxpr_to_triton_module), [`_reinterpret_int4_as_uint8`](lowering.md#_reinterpret_int4_as_uint8), [`_is_contiguous_int4`](lowering.md#_is_contiguous_int4), [`_compute_pointers_from_indices`](lowering.md#_compute_pointers_from_indices), [`block_infos`](lowering.md#LoweringRuleContext.block_infos)

### `LoweringError`  ·  implements/extends Exception
- def: [`jax/_src/pallas/triton/lowering.py:107`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L107)
- signature: `class LoweringError(Exception):`
- used by: [`lower_jaxpr_to_triton_ir`](lowering.md#lower_jaxpr_to_triton_ir)

### `LoweringResult`
- def: [`jax/_src/pallas/triton/lowering.py:100`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L100)
- doc: Keeps python objects alive.
- signature: `class LoweringResult:`
- members:
  - `grid` — [`L104`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L104)
  - `module` — [`L103`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L103)
- used by: [`pallas_call_lowering`](pallas_call_registration.md#pallas_call_lowering), [`lower_jaxpr_to_triton_module`](lowering.md#lower_jaxpr_to_triton_module)

### `LoweringRuleContext`
- def: [`jax/_src/pallas/triton/lowering.py:90`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L90)
- signature: `class LoweringRuleContext:`
- members:
  - `avals_in` — [`L92`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L92)
  - `avals_out` — [`L93`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L93)
  - `block_infos` — [`L94`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L94)
  - `context` — [`L91`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L91)
  - `replace` — [`L96`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L96)
- uses (calls/refs, reference-scoped): [`ShapedArray`](../../core.md#ShapedArray), [`BlockInfo`](lowering.md#BlockInfo), [`ModuleContext`](lowering.md#ModuleContext)
- used by: [`_dot_general_lowering`](lowering.md#_dot_general_lowering), [`lower_jaxpr_to_triton_ir`](lowering.md#lower_jaxpr_to_triton_ir), [`_masked_load_lowering_rule`](lowering.md#_masked_load_lowering_rule), [`_atomic_lowering_rule`](primitives.md#_atomic_lowering_rule), [`lower`](lowering.md#_Extern.lower), [`_reduction_lowering`](lowering.md#_reduction_lowering), [`_associative_scan_lowering`](lowering.md#_associative_scan_lowering), [`_masked_swap_lowering_rule`](lowering.md#_masked_swap_lowering_rule), [`_approx_tanh_rocm_lowering`](primitives.md#_approx_tanh_rocm_lowering), [`_cond_lowering_rule`](lowering.md#_cond_lowering_rule), [`_integer_pow_rule`](lowering.md#_integer_pow_rule), [`_argreduce_lowering`](lowering.md#_argreduce_lowering), [`_reshape_lowering_rule`](lowering.md#_reshape_lowering_rule), [`_div_lowering_rule`](lowering.md#_div_lowering_rule), [`_elementwise_inline_asm_lowering`](primitives.md#_elementwise_inline_asm_lowering), [`_scan_lowering_rule`](lowering.md#_scan_lowering_rule), [`_max_lowering_rule`](lowering.md#_max_lowering_rule), [`_reduce_lowering`](lowering.md#_reduce_lowering), [`_while_lowering_rule`](lowering.md#_while_lowering_rule), [`_maybe_pattern_match_fori_loop`](lowering.md#_maybe_pattern_match_fori_loop), [`_addupdate_lowering_rule`](lowering.md#_addupdate_lowering_rule), [`_convert_element_type_lowering_rule`](lowering.md#_convert_element_type_lowering_rule), [`_min_lowering_rule`](lowering.md#_min_lowering_rule), [`signed_rule`](lowering.md#signed_rule), [`_concatenate_lowering_rule`](lowering.md#_concatenate_lowering_rule), [`_iota_lowering_rule`](lowering.md#_iota_lowering_rule), [`debug_print_lowering_rule`](lowering.md#debug_print_lowering_rule), [`_lower_jaxpr_to_for_loop`](lowering.md#_lower_jaxpr_to_for_loop), [`_closed_call_lowering_rule`](lowering.md#_closed_call_lowering_rule), [`inner`](lowering.md#_make_dispatch_table.inner), [`_axis_index_rule`](lowering.md#_axis_index_rule), [`_broadcast_in_dim_lowering_rule`](lowering.md#_broadcast_in_dim_lowering_rule), [`_get_lowering_rule`](lowering.md#_get_lowering_rule), [`_swap_lowering_rule`](lowering.md#_swap_lowering_rule), [`lower`](lowering.md#_Fallback.lower), [`_broadcast_to_rule`](lowering.md#_broadcast_to_rule), [`_max_contiguous_rule`](primitives.md#_max_contiguous_rule), [`_multiple_of_rule`](lowering.md#_multiple_of_rule), [`_not_lowering_rule`](lowering.md#_not_lowering_rule), [`_pjit_lowering_rule`](lowering.md#_pjit_lowering_rule)  (+15 more)

### `ModuleContext`
- def: [`jax/_src/pallas/triton/lowering.py:71`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L71)
- signature: `class ModuleContext:`
- members:
  - `compute_capability` — [`L77`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L77)
  - `grid_mapping` — [`L73`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L73)
  - `mlir_ctx` — [`L78`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L78)
  - `name` — [`L72`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L72)
  - `platform` — [`L76`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L76)
  - `program_ids` — [`L74`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L74)
  - `traceback_caches` — [`L75`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L75)
- uses (calls/refs, reference-scoped): [`GridMapping`](lowering.md#GridMapping)
- used by: [`lower_jaxpr_to_triton_ir`](lowering.md#lower_jaxpr_to_triton_ir), [`lower_jaxpr_to_triton_module`](lowering.md#lower_jaxpr_to_triton_module), [`_eval_index_map`](lowering.md#_eval_index_map), [`_approx_tanh_rocm_lowering`](primitives.md#_approx_tanh_rocm_lowering), [`_elementwise_inline_asm_lowering`](primitives.md#_elementwise_inline_asm_lowering), [`_convert_element_type_lowering_rule`](lowering.md#_convert_element_type_lowering_rule), [`context`](lowering.md#LoweringRuleContext.context), [`inner`](lowering.md#_make_dispatch_table.inner), [`_axis_index_rule`](lowering.md#_axis_index_rule), [`_program_id_lowering_rule`](lowering.md#_program_id_lowering_rule)

### `_Extern`
- def: [`jax/_src/pallas/triton/lowering.py:575`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L575) — documented in [jax-_src-pallas-triton-lowering](../../../../../concepts/jax-_src-pallas-triton-lowering.md)
- signature: `class _Extern:`
- members:
  - `lower(self, ctx: LoweringRuleContext, *args: ir.Value)` — [`L589`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L589)
  - `matches(self, avals: Sequence[jax_core.ShapedArray])` — [`L580`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L580)
  - `arg_types` — [`L576`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L576)
  - `result_type` — [`L578`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L578)
  - `symbol` — [`L577`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L577)
- uses (calls/refs, reference-scoped): [`ShapedArray`](../../core.md#ShapedArray), [`DTypeLike`](../../typing.md#DTypeLike), [`shape`](../../core.md#ShapedArray.shape), [`dtype`](../../core.md#ShapedArray.dtype), [`dtype`](../../../numpy/__init__.pyi.md#dtype), [`LoweringRuleContext`](lowering.md#LoweringRuleContext), [`avals_in`](lowering.md#LoweringRuleContext.avals_in), [`_ensure_ir_value`](lowering.md#_ensure_ir_value), [`weak_type`](../../core.md#ShapedArray.weak_type), [`_bcast_to`](lowering.md#_bcast_to), [`_cast`](lowering.md#_cast), [`_dtype_to_ir_type`](lowering.md#_dtype_to_ir_type), [`avals_out`](lowering.md#LoweringRuleContext.avals_out)
- used by: [`nextafter_dispatch_table`](lowering.md#nextafter_dispatch_table), [`_make_dispatch_table`](lowering.md#_make_dispatch_table), [`abs_dispatch_table`](lowering.md#abs_dispatch_table), [`pow_dispatch_table`](lowering.md#pow_dispatch_table), [`inner`](lowering.md#_make_dispatch_table.inner), [`acos_dispatch_table`](lowering.md#acos_dispatch_table), [`acosh_dispatch_table`](lowering.md#acosh_dispatch_table), [`asin_dispatch_table`](lowering.md#asin_dispatch_table), [`asinh_dispatch_table`](lowering.md#asinh_dispatch_table), [`atan2_dispatch_table`](lowering.md#atan2_dispatch_table), [`atan_dispatch_table`](lowering.md#atan_dispatch_table), [`atanh_dispatch_table`](lowering.md#atanh_dispatch_table), [`cbrt_dispatch_table`](lowering.md#cbrt_dispatch_table), [`ceil_dispatch_table`](lowering.md#ceil_dispatch_table), [`clz_dispatch_table`](lowering.md#clz_dispatch_table), [`cos_dispatch_table`](lowering.md#cos_dispatch_table), [`cosh_dispatch_table`](lowering.md#cosh_dispatch_table), [`exp2_dispatch_table`](lowering.md#exp2_dispatch_table), [`exp_dispatch_table`](lowering.md#exp_dispatch_table), [`expm1_dispatch_table`](lowering.md#expm1_dispatch_table), [`floor_dispatch_table`](lowering.md#floor_dispatch_table), [`log1p_dispatch_table`](lowering.md#log1p_dispatch_table), [`log_dispatch_table`](lowering.md#log_dispatch_table), [`population_count_dispatch_table`](lowering.md#population_count_dispatch_table), [`rsqrt_dispatch_table`](lowering.md#rsqrt_dispatch_table), [`sin_dispatch_table`](lowering.md#sin_dispatch_table), [`sinh_dispatch_table`](lowering.md#sinh_dispatch_table), [`sqrt_dispatch_table`](lowering.md#sqrt_dispatch_table), [`tan_dispatch_table`](lowering.md#tan_dispatch_table), [`tanh_dispatch_table`](lowering.md#tanh_dispatch_table)

### `_Fallback`
- def: [`jax/_src/pallas/triton/lowering.py:612`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L612) — documented in [jax-_src-pallas-triton-lowering](../../../../../concepts/jax-_src-pallas-triton-lowering.md)
- signature: `class _Fallback:`
- members:
  - `lower(self, ctx: LoweringRuleContext, *args: ir.Value)` — [`L624`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L624)
  - `matches(self, avals: Sequence[jax_core.ShapedArray])` — [`L616`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L616)
  - `arg_classes` — [`L613`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L613)
  - `op` — [`L614`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L614)
- uses (calls/refs, reference-scoped): [`ShapedArray`](../../core.md#ShapedArray), [`DTypeLike`](../../typing.md#DTypeLike), [`shape`](../../core.md#ShapedArray.shape), [`dtype`](../../core.md#ShapedArray.dtype), [`issubdtype`](../../../numpy/__init__.pyi.md#issubdtype), [`LoweringRuleContext`](lowering.md#LoweringRuleContext), [`avals_in`](lowering.md#LoweringRuleContext.avals_in), [`_ensure_ir_value`](lowering.md#_ensure_ir_value), [`_bcast_to`](lowering.md#_bcast_to), [`avals_out`](lowering.md#LoweringRuleContext.avals_out)
- used by: [`_make_dispatch_table`](lowering.md#_make_dispatch_table), [`abs_dispatch_table`](lowering.md#abs_dispatch_table), [`pow_dispatch_table`](lowering.md#pow_dispatch_table), [`acos_dispatch_table`](lowering.md#acos_dispatch_table), [`acosh_dispatch_table`](lowering.md#acosh_dispatch_table), [`asin_dispatch_table`](lowering.md#asin_dispatch_table), [`asinh_dispatch_table`](lowering.md#asinh_dispatch_table), [`atan2_dispatch_table`](lowering.md#atan2_dispatch_table), [`atan_dispatch_table`](lowering.md#atan_dispatch_table), [`atanh_dispatch_table`](lowering.md#atanh_dispatch_table), [`cbrt_dispatch_table`](lowering.md#cbrt_dispatch_table), [`ceil_dispatch_table`](lowering.md#ceil_dispatch_table), [`clz_dispatch_table`](lowering.md#clz_dispatch_table), [`cos_dispatch_table`](lowering.md#cos_dispatch_table), [`cosh_dispatch_table`](lowering.md#cosh_dispatch_table), [`exp2_dispatch_table`](lowering.md#exp2_dispatch_table), [`exp_dispatch_table`](lowering.md#exp_dispatch_table), [`expm1_dispatch_table`](lowering.md#expm1_dispatch_table), [`floor_dispatch_table`](lowering.md#floor_dispatch_table), [`log1p_dispatch_table`](lowering.md#log1p_dispatch_table), [`log_dispatch_table`](lowering.md#log_dispatch_table), [`population_count_dispatch_table`](lowering.md#population_count_dispatch_table), [`rsqrt_dispatch_table`](lowering.md#rsqrt_dispatch_table), [`sin_dispatch_table`](lowering.md#sin_dispatch_table), [`sinh_dispatch_table`](lowering.md#sinh_dispatch_table), [`sqrt_dispatch_table`](lowering.md#sqrt_dispatch_table), [`tan_dispatch_table`](lowering.md#tan_dispatch_table), [`tanh_dispatch_table`](lowering.md#tanh_dispatch_table)

## Functions
- `_add(x: ir.Value, y: ir.Value)` — [`L1127`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1127)
- `_addupdate_lowering_rule(ctx: LoweringRuleContext, ptr, value, *idx, tree)` — [`L2293`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2293)
- `_argreduce_lowering(body, ctx: LoweringRuleContext, a, *, axes, index_dtype)` — [`L2538`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2538)
- `_as_bf16(x)` — [`L2319`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2319)
- `_as_f32(x)` — [`L2323`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2323)
- `_associative_scan_lowering(body, ctx: LoweringRuleContext, args, axes)` — [`L527`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L527)
- `_atomic_rmw(op: tt_dialect.RMWOp, ptr: ir.Value, val: ir.Value, mask: ir.Value | None = None, semantic: tt_dialect.MemSemantic = tt_dialect.MemSemantic.ACQUIRE_RELEASE, sync_scope: tt_dialect.MemSyncScope = tt_dialect.MemSyncScope.GPU)` — [`L507`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L507)
- `_axis_index_rule(ctx: LoweringRuleContext, *, axis_name: Hashable)` — [`L2625`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2625)
- `_bcast(x: ir.Value, y: ir.Value, x_aval: jax_core.ShapedArray, y_aval: jax_core.ShapedArray, out_aval: jax_core.ShapedArray)` — [`L178`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L178)
- `_bcast_to(a: ir.Value, shape: Sequence[int])` — [`L158`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L158)
- `_bitcast_convert_type_lowering_rule(ctx: LoweringRuleContext, operand: ir.Value, *, new_dtype)` — [`L2951`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2951)
- `_broadcast_in_dim_lowering_rule(ctx: LoweringRuleContext, x, *, broadcast_dimensions, shape, sharding)` — [`L1762`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1762)
- `_broadcast_to_rule(ctx: LoweringRuleContext, x, shape: Sequence[int])` — [`L1377`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1377)
- `_cast(src: ir.Value, src_type: jax.typing.DTypeLike, dst_type: jax.typing.DTypeLike, *, compute_capability: int | None = None)` — [`L1634`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1634)
- `_check_tensor_size(shape: tuple[int | pallas_core.Squeezed, ...])` — [`L291`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L291)
- `_closed_call_lowering_rule(ctx: LoweringRuleContext, *args, call_jaxpr, **_)` — [`L2607`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2607)
- `_cmp(x: ir.Value, y: ir.Value, si_pred: arith_dialect.CmpIPredicate, ui_pred: arith_dialect.CmpIPredicate, f_pred: arith_dialect.CmpFPredicate, *, signed: bool)` — [`L1209`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1209)
- `_compute_offsets_from_indices(block_info: BlockInfo, nd_indexer: NDIndexer)` — [`L1890`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1890)
- `_compute_pointers_from_indices(root_ptr: ir.Value, block_info: BlockInfo, nd_indexer: NDIndexer)` — [`L1983`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1983)
- `_concatenate_lowering_rule(ctx: LoweringRuleContext, *args, dimension)` — [`L1818`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1818)
- `_cond_lowering_rule(ctx: LoweringRuleContext, index, *args, branches)` — [`L2864`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2864)
- `_convert_element_type_lowering_rule(ctx: LoweringRuleContext, x, *, new_dtype, weak_type, sharding)` — [`L1741`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1741)
- `_cumsum_lowering_rule(ctx: LoweringRuleContext, x, *, axis: int, reverse: bool)` — [`L560`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L560)
- `_div_lowering_rule(ctx: LoweringRuleContext, x, y)` — [`L1457`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1457)
- `_dot_general_lowering(ctx: LoweringRuleContext, a, b, *, dimension_numbers, out_sharding, precision, preferred_element_type)` — [`L2328`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2328) — documented in [jax-_src-pallas-triton-lowering](../../../../../concepts/jax-_src-pallas-triton-lowering.md)
- `_dtype_to_ir_type(dtype: jax.typing.DTypeLike)` — [`L2942`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2942)
- `_element_type(t: ir.Type)` — [`L1491`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1491)
- `_ensure_ir_value(x: object, aval: jax_core.ShapedArray)` — [`L2911`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2911)
- `_eval_index_map(ctx: ModuleContext, idx, block_mapping: BlockMapping)` — [`L111`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L111)
- `_expand_dims(x: ir.Value, axis: int)` — [`L1551`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1551)
- `_float_float_cast(src: ir.Value, dst_type: ir.Type)` — [`L1559`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1559)
- `_float_int_cast(src: ir.Value, dst_type: ir.Type, *, signed: bool)` — [`L1592`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1592)
- `_floordiv(x: ir.Value, y: ir.Value, *, signed: bool)` — [`L1171`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1171)
- `_fp_bits_type(t: ir.Type)` — [`L1106`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1106)
- `_full(t: ir.Type, v: Any)` — [`L1512`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1512)
- `_get_bdim_alignment(b: pallas_core.BlockDim)` — [`L149`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L149)
- `_get_index_alignment(block_mapping: BlockMapping)` — [`L148`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L148)
- `_get_lowering_rule(ctx: LoweringRuleContext, ptr, *idx, tree)` — [`L1992`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1992)
- `_get_start_index(i, b)` — [`L134`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L134)
- `_i32_constant(v: int)` — [`L2934`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2934)
- `_i64_constant(v: int)` — [`L2938`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2938)
- `_int_float_cast(src: ir.Value, dst_type: ir.Type, *, signed: bool)` — [`L1619`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1619)
- `_int_int_cast(src: ir.Value, dst_type: ir.Type, signed: bool)` — [`L1575`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1575)
- `_integer_pow_rule(ctx: LoweringRuleContext, x, *, y: int)` — [`L1383`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1383)
- `_iota_lowering_rule(ctx: LoweringRuleContext, *, dtype, shape, dimension, sharding)` — [`L1481`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1481)
- `_ir_cast(src: ir.Value, dst_type: ir.Type, *, signed: bool, dst_signed: bool = False, compute_capability: int | None = None)` — [`L1661`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1661)
- `_ir_constant(v: object, t: ir.Type)` — [`L2921`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2921)
- `_is_contiguous_int4(block_info: BlockInfo, nd_indexer: NDIndexer)` — [`L2082`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2082) — Returns True if the block is contiguous in the last dimension.
- `_is_float8_e4m3fn_cast_supported(compute_capability: int | None)` — [`L1650`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1650)
- `_is_nan(x: ir.Value)` — [`L1266`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1266)
- `_is_triton_pointer_type(t)` — [`L1101`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1101)
- `_load(ptr: ir.Value, mask: ir.Value | None = None, other: ir.Value | None = None, *, cache_modifier: str | None = None, eviction_policy: str | None = None, is_volatile: bool = False)` — [`L2012`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2012)
- `_lower_jaxpr_to_for_loop(ctx: LoweringRuleContext, jaxpr: jax_core.Jaxpr, lower_bound, upper_bound, consts, *args, has_loop_index: bool, step: int = 1, bound_type: ir.IntegerType | None = None)` — [`L2633`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2633)
- `_make_dispatch_table(name: str, **tables: Sequence[_Extern | _Fallback])` — [`L632`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L632)
- `_make_range(start: int, end: int)` — [`L1498`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1498)
- `_masked_load_lowering_rule(ctx: LoweringRuleContext, *args_flat, args_tree, eviction_policy, cache_modifier, is_volatile)` — [`L2123`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2123)
- `_masked_swap_lowering_rule(ctx: LoweringRuleContext, *args_flat, args_tree, eviction_policy)` — [`L2264`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2264)
- `_max_lowering_rule(ctx: LoweringRuleContext, x, y)` — [`L1439`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1439)
- `_maybe_pattern_match_fori_loop(ctx: LoweringRuleContext, *args, cond_nconsts, cond_jaxpr, body_nconsts, body_jaxpr)` — [`L2711`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2711)
- `_min_lowering_rule(ctx: LoweringRuleContext, x, y)` — [`L1421`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1421)
- `_minus(x: ir.Value)` — [`L1121`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1121)
- `_mod(x: ir.Value, y: ir.Value, *, signed: bool)` — [`L1196`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1196)
- `_mul(x: ir.Value, y: ir.Value, *, out_dtype=None)` — [`L1159`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1159)
- `_multiple_of_rule(ctx: LoweringRuleContext, x, values: Sequence[int])` — [`L1365`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1365)
- `_new_ir_context()` — [`L279`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L279)
- `_not_lowering_rule(ctx: LoweringRuleContext, x)` — [`L569`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L569)
- `_num_programs_lowering_rule(ctx: LoweringRuleContext, *, axis)` — [`L502`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L502)
- `_ones(t: ir.Type)` — [`L1535`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1535)
- `_ones_like(x: ir.Value)` — [`L1539`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1539)
- `_pjit_lowering_rule(ctx: LoweringRuleContext, *args, jaxpr, **_)` — [`L2592`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2592)
- `_process_grid_to_3d_grid(grid_mapping: GridMapping)` — [`L216`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L216)
- `_program_id(axis: int, launch_grid: Sequence[int])` — [`L488`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L488)
- `_program_id_lowering_rule(ctx: LoweringRuleContext, *, axis)` — [`L497`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L497)
- `_reduce_argmax_combine(left, right)` — [`L2559`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2559)
- `_reduce_argmin_combine(left, right)` — [`L2575`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2575)
- `_reduce_lowering(body, ctx: LoweringRuleContext, a, *, axes, **kwargs)` — [`L2507`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2507)
- `_reduction_lowering(body, ctx: LoweringRuleContext, a, axes)` — [`L2476`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2476)
- `_reinterpret_int4_as_uint8(block_info: BlockInfo, nd_indexer: NDIndexer)` — [`L2098`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2098) — Returns a new block info and indexer that reads `int4` as `uint8`.
- `_remat_lowering_rule(ctx: LoweringRuleContext, *args, jaxpr, **_)` — [`L2617`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2617)
- `_reshape(a: ir.Value, shape: Sequence[int])` — [`L1798`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1798)
- `_reshape_lowering_rule(ctx: LoweringRuleContext, a, *, new_sizes, dimensions, sharding)` — [`L1782`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1782)
- `_reshard_lowering_rule(ctx, x, *, dst_sharding, concrete_mesh)` — [`L2601`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2601)
- `_scan_lowering_rule(ctx: LoweringRuleContext, *args, jaxpr, length, reverse, unroll, num_consts, num_carry)` — [`L2668`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2668)
- `_set_attr(v: ir.Value, name: str, attr: ir.Attribute)` — [`L1349`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1349)
- `_splat(x: ir.Value, shape: Sequence[int])` — [`L1543`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1543)
- `_split_lowering_rule(ctx: LoweringRuleContext, x, *, sizes, axis)` — [`L1869`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1869)
- `_squeeze_lowering_rule(ctx: LoweringRuleContext, a, *, dimensions)` — [`L1776`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1776)
- `_stack_lowering_rule(ctx: LoweringRuleContext, *args, axis)` — [`L1837`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1837)
- `_store(ptr: ir.Value, value: ir.Value, mask: ir.Value | None = None, *, cache_modifier: str | None = None, eviction_policy: str | None = None)` — [`L2211`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2211)
- `_sub(x: ir.Value, y: ir.Value)` — [`L1145`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1145)
- `_swap_lowering_rule(ctx: LoweringRuleContext, ptr, value, *idx, tree)` — [`L2198`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2198)
- `_transpose_lowering(ctx: LoweringRuleContext, x, *, permutation)` — [`L2312`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2312)
- `_truediv(x: ir.Value, y: ir.Value, *, signed: bool)` — [`L1184`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1184)
- `_unstack_lowering_rule(ctx: LoweringRuleContext, x, *, axis)` — [`L1857`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1857)
- `_while_lowering_rule(ctx: LoweringRuleContext, *args, cond_nconsts, cond_jaxpr, body_nconsts, body_jaxpr)` — [`L2793`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2793)
- `_zeros(t: ir.Type)` — [`L1527`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1527)
- `_zeros_like(x: ir.Value)` — [`L1531`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1531)
- `debug_print_lowering_rule(ctx: LoweringRuleContext, *args: ir.Value, fmt: str, ordered, partitioned, in_tree, static_args, np_printoptions, has_placeholders, logging_record)` — [`L1313`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1313)
- `f_lowered(ctx: LoweringRuleContext, *args, **params)` — [`L467`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L467)
- `get_join_type(old_type: ir.RankedTensorType)` — [`L1811`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1811)
- `inner(ctx: LoweringRuleContext, *args: ir.Value, **_)` — [`L636`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L636)
- `lower_fun(fun: Callable[..., Any], *, multiple_results: bool)` — [`L462`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L462)
- `lower_jaxpr_to_triton_ir(ctx: ModuleContext, jaxpr: jax_core.Jaxpr, block_infos: Sequence[BlockInfo | None] | None, *args)` — [`L392`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L392) — documented in [jax-_src-pallas-triton-lowering](../../../../../concepts/jax-_src-pallas-triton-lowering.md)
- `lower_jaxpr_to_triton_module(jaxpr: jax_core.Jaxpr, grid_mapping: GridMapping, platform: str, compute_capability: int | None, mlir_ctx: mlir.ModuleContext)` — [`L303`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L303) — documented in [jax-_src-pallas-triton-lowering](../../../../../concepts/jax-_src-pallas-triton-lowering.md)
- `read_block_info_env(atom: jax_core.Atom)` — [`L404`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L404)
- `read_env(atom: jax_core.Atom)` — [`L401`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L401)
- `register_lowering(primitive: jax_core.Primitive)` — [`L209`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L209) — documented in [jax-_src-pallas-triton-lowering](../../../../../concepts/jax-_src-pallas-triton-lowering.md)
- `select_n_lowering_rule(ctx: LoweringRuleContext, pred, x, y)` — [`L1753`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1753)
- `signed_rule(ctx: LoweringRuleContext, x, y, fn=fn)` — [`L1304`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1304)
- `signless_rule(ctx: LoweringRuleContext, x, y, fn=fn, **kwargs)` — [`L1285`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1285)
- `split_into_2(x)` — [`L1878`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1878)
- `to_type(out_aval)` — [`L2872`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2872)
- `wrapper(fn)` — [`L210`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L210)
- `write_env(var: jax_core.Var, val)` — [`L409`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L409)

## Module values
- `BlockMapping` — [`L65`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L65)
- `Blocked` — [`L66`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L66)
- `GridMapping` — [`L64`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L64)
- `NDIndexer` — [`L63`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L63)
- `_JAX_FN_MAPPING` — [`L1410`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1410)
- `_JAX_TO_TRITON_BINARY` — [`L1270`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1270)
- `_JAX_TO_TRITON_SIGNED_BINARY` — [`L1292`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1292)
- `_STR_TO_CACHE_MODIFIER` — [`L2009`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2009)
- `_STR_TO_EVICTION_POLICY` — [`L2008`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2008)
- `_T` — [`L58`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L58)
- `_TF32_PRECISIONS` — [`L2316`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L2316)
- `_UNSUPPORTED_CAST_DTYPES` — [`L1654`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1654)
- `_equal` — [`L1228`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1228)
- `_greater_equal` — [`L1258`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1258)
- `_greater_than` — [`L1252`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1252)
- `_less_equal` — [`L1246`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1246)
- `_less_than` — [`L1240`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1240)
- `_not_equal` — [`L1234`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1234)
- `abs_dispatch_table` — [`L651`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L651)
- `acos_dispatch_table` — [`L897`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L897)
- `acosh_dispatch_table` — [`L995`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L995)
- `asin_dispatch_table` — [`L883`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L883)
- `asinh_dispatch_table` — [`L981`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L981)
- `atan2_dispatch_table` — [`L925`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L925)
- `atan_dispatch_table` — [`L911`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L911)
- `atanh_dispatch_table` — [`L1009`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1009)
- `cbrt_dispatch_table` — [`L813`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L813)
- `ceil_dispatch_table` — [`L669`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L669)
- `clz_dispatch_table` — [`L1037`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1037)
- `cos_dispatch_table` — [`L855`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L855)
- `cosh_dispatch_table` — [`L953`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L953)
- `exp2_dispatch_table` — [`L711`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L711)
- `exp_dispatch_table` — [`L697`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L697)
- `expm1_dispatch_table` — [`L725`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L725)
- `floor_dispatch_table` — [`L683`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L683)
- `fn` — [`L1283`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1283)
- `log1p_dispatch_table` — [`L753`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L753)
- `log_dispatch_table` — [`L739`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L739)
- `nextafter_dispatch_table` — [`L1051`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1051) — documented in [jax-_src-pallas-triton-lowering](../../../../../concepts/jax-_src-pallas-triton-lowering.md)
- `population_count_dispatch_table` — [`L1023`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1023)
- `pow_dispatch_table` — [`L781`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L781)
- `prim` — [`L1283`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L1283)
- `rsqrt_dispatch_table` — [`L827`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L827)
- `sin_dispatch_table` — [`L841`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L841)
- `sinh_dispatch_table` — [`L939`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L939)
- `sqrt_dispatch_table` — [`L767`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L767)
- `tan_dispatch_table` — [`L869`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L869)
- `tanh_dispatch_table` — [`L967`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L967)
- `triton_lowering_rules` — [`L206`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L206)
- `unsafe_map` — [`L60`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L60)
- `unsafe_zip` — [`L61`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/lowering.py#L61)

