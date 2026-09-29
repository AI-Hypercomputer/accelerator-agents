---
title: 'Module: jax/_src/lax/utils.py'
type: catalog
provenance: extracted
module: jax/_src/lax/utils.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.lax.utils`/
symbols:
  standard_primitive: standard_primitive().
  call_shape_dtype_sharding_rule: call_shape_dtype_sharding_rule().
  input_dtype: input_dtype().
  dtype_to_string: dtype_to_string().
  int_dtype_for_dim: int_dtype_for_dim().
  int_dtype_for_shape: int_dtype_for_shape().
  call_sharding_rule: call_sharding_rule().
  standard_abstract_eval: standard_abstract_eval().
  standard_multi_result_abstract_eval: standard_multi_result_abstract_eval().
  ensure_shaped: ensure_shaped().
  manual_rule: manual_rule().
  _default_memory_space_rule: _default_memory_space_rule().
  _argnum_weak_type: _argnum_weak_type().
  _get_abstract_mesh_from_avals: _get_abstract_mesh_from_avals().
  index_dtype_for_axis_size: index_dtype_for_axis_size().
  multi_mem_space_rule: multi_mem_space_rule().
  unsafe_zip: unsafe_zip.
  _get_array_abstraction_level: _get_array_abstraction_level().
  call_ur_rule: call_ur_rule().
  _int32_max: _int32_max.
  _uint32_max: _uint32_max.
  _standard_weak_type_rule: _standard_weak_type_rule().
---
# Module: [`jax/_src/lax/utils.py`](../../../../../../../raw/code/jax/jax/_src/lax/utils.py)

## Functions
- `_argnum_weak_type(*argnums)` — [`L42`](../../../../../../../raw/code/jax/jax/_src/lax/utils.py#L42)
- `_default_memory_space_rule(prim, *avals, **kwargs)` — [`L135`](../../../../../../../raw/code/jax/jax/_src/lax/utils.py#L135)
- `_get_abstract_mesh_from_avals(in_avals)` — [`L59`](../../../../../../../raw/code/jax/jax/_src/lax/utils.py#L59)
- `_get_array_abstraction_level(a)` — [`L57`](../../../../../../../raw/code/jax/jax/_src/lax/utils.py#L57)
- `_standard_weak_type_rule(*avals, **kwargs)` — [`L233`](../../../../../../../raw/code/jax/jax/_src/lax/utils.py#L233)
- `call_shape_dtype_sharding_rule(prim, shape_rule, dtype_rule, sharding_rule, ur_rule, multi_out, *avals, **kwargs)` — [`L113`](../../../../../../../raw/code/jax/jax/_src/lax/utils.py#L113)
- `call_sharding_rule(prim, sh_rule, ur_rule, num_out, *avals, **kwargs)` — [`L92`](../../../../../../../raw/code/jax/jax/_src/lax/utils.py#L92)
- `call_ur_rule(prim, ur_rule, out_s, num_out, *avals, **kwargs)` — [`L75`](../../../../../../../raw/code/jax/jax/_src/lax/utils.py#L75)
- `dtype_to_string(dtype)` — [`L236`](../../../../../../../raw/code/jax/jax/_src/lax/utils.py#L236)
- `ensure_shaped(*avals: core.AbstractValue)` — [`L281`](../../../../../../../raw/code/jax/jax/_src/lax/utils.py#L281) — Cast all inputs to ShapedArray with a runtime instance check.
- `index_dtype_for_axis_size(indices_dtype: DType, axis_size: DimSize, wrap_negative_indices: bool)` — [`L288`](../../../../../../../raw/code/jax/jax/_src/lax/utils.py#L288) — Upcast indices_dtype if necessary to avoid overflow.
- `input_dtype(x, *_, out_dtype=None, **__)` — [`L37`](../../../../../../../raw/code/jax/jax/_src/lax/utils.py#L37)
- `int_dtype_for_dim(d: DimSize, *, signed: bool)` — [`L250`](../../../../../../../raw/code/jax/jax/_src/lax/utils.py#L250) — Returns a integer dtype large enough to contain indices in dimension d.
- `int_dtype_for_shape(shape: Shape, *, signed: bool)` — [`L261`](../../../../../../../raw/code/jax/jax/_src/lax/utils.py#L261) — Returns a integer dtype large enough to contain indices in `shape`.
- `manual_rule(prim, vma_rule, ur_rule, multi_out, *avals, **kwargs)` — [`L156`](../../../../../../../raw/code/jax/jax/_src/lax/utils.py#L156)
- `multi_mem_space_rule(prim, num_out, *avals, **kwargs)` — [`L152`](../../../../../../../raw/code/jax/jax/_src/lax/utils.py#L152)
- `standard_abstract_eval(prim, shape_rule, dtype_rule, weak_type_rule, sharding_rule, vma_rule, ur_rule, memory_space_rule, *avals, **kwargs)` — [`L176`](../../../../../../../raw/code/jax/jax/_src/lax/utils.py#L176)
- `standard_multi_result_abstract_eval(prim, shape_rule, dtype_rule, weak_type_rule, sharding_rule, vma_rule, ur_rule, *avals, **kwargs)` — [`L206`](../../../../../../../raw/code/jax/jax/_src/lax/utils.py#L206)
- `standard_primitive(shape_rule, dtype_rule, name, weak_type_rule=None, sharding_rule=None, vma_rule=None, ur_rule=None, memory_space_rule=None)` — [`L45`](../../../../../../../raw/code/jax/jax/_src/lax/utils.py#L45)

## Module values
- `_int32_max` — [`L247`](../../../../../../../raw/code/jax/jax/_src/lax/utils.py#L247)
- `_uint32_max` — [`L248`](../../../../../../../raw/code/jax/jax/_src/lax/utils.py#L248)
- `unsafe_zip` — [`L34`](../../../../../../../raw/code/jax/jax/_src/lax/utils.py#L34)

