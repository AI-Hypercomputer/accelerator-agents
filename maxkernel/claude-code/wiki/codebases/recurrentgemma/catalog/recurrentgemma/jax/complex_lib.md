---
title: 'Module: recurrentgemma/jax/complex_lib.py'
type: catalog
provenance: extracted
module: recurrentgemma/jax/complex_lib.py
status: fresh
symbol_base: scip-python python recurrentgemma 0.0.0 `recurrentgemma.jax.complex_lib`/
symbols:
  RealOrComplex: RealOrComplex.
  Complex: Complex#
  Complex.real: Complex#real.
  Complex.imag: Complex#imag.
  _treat_method: _treat_method().
  exp: exp().
  ones_like: ones_like().
  conjugate: conjugate().
  Complex.to_numpy: Complex#to_numpy().
  Complex._sanity_check: Complex#_sanity_check().
  Complex.__mul__: Complex#__mul__().
  Complex.__add__: Complex#__add__().
  softplus: softplus().
  log: log().
  Complex.__matmul__: Complex#__matmul__().
  Complex.__truediv__: Complex#__truediv__().
  Complex.__sub__: Complex#__sub__().
  Complex.__rsub__: Complex#__rsub__().
  concatenate: concatenate.
  sigmoid: sigmoid().
  Complex.__post_init__: Complex#__post_init__().
  Complex.astype: Complex#astype().
  Complex.reshape: Complex#reshape().
  Complex.__neg__: Complex#__neg__().
  Complex.__getitem__: Complex#__getitem__().
  Complex.__setitem__: Complex#__setitem__().
  Complex.__eq__: Complex#__eq__().
  Complex.__iter__: Complex#__iter__().
  einsum: einsum().
  to_custom_complex: to_custom_complex().
  Complex.dtype: Complex#dtype().
  zeros_like: zeros_like.
  reshape: reshape.
  _is_pytree_placeholder: _is_pytree_placeholder().
  Complex.shape: Complex#shape().
  Complex.size: Complex#size().
  Complex.ndim: Complex#ndim().
  Complex.__rmul__: Complex#__rmul__.
  Complex.__radd__: Complex#__radd__.
  broadcast_to: broadcast_to.
  split: split.
  expand_dims: expand_dims.
  flip: flip.
  squeeze: squeeze.
  stack: stack.
  tile: tile.
  transpose: transpose.
  add: add.
  pad: pad.
  slice_in_dim: slice_in_dim.
  rearrange: rearrange.
  repeat: repeat.
  abs_squared: abs_squared().
  _arg_is_pytree_placeholder: _arg_is_pytree_placeholder().
---
# Module: [`recurrentgemma/jax/complex_lib.py`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py)

## Classes
### `Complex`
- def: [`recurrentgemma/jax/complex_lib.py:70`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L70) — documented in [recurrentgemma-jax-complex_lib](../../../concepts/recurrentgemma-jax-complex_lib.md)
- doc: Custom Complex class.
- signature: `class Complex:`
- members:
  - `__matmul__(self, x: Union[jax.Array, Complex])` — [`L139`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L139) — Performs the matrix multiplication operation. — documented in [recurrentgemma-jax-complex_lib](../../../concepts/recurrentgemma-jax-complex_lib.md)
  - `__mul__(self, x: Union[jax.Array, Complex])` — [`L154`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L154) — Performs the multiplication operation. — documented in [recurrentgemma-jax-complex_lib](../../../concepts/recurrentgemma-jax-complex_lib.md)
  - `_sanity_check(self, x: Union[jax.Array, Complex])` — [`L118`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L118) — Check if the arg is not native complex and has the same dtype as this instance. — documented in [recurrentgemma-jax-complex_lib](../../../concepts/recurrentgemma-jax-complex_lib.md)
  - `astype(self, dtype: jnp.dtype | None)` — [`L105`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L105)
  - `dtype(self)` — [`L90`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L90)
  - `ndim(self)` — [`L102`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L102)
  - `reshape(self, shape: Sequence[int])` — [`L110`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L110)
  - `shape(self)` — [`L94`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L94)
  - `size(self)` — [`L98`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L98)
  - `to_numpy(self)` — [`L113`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L113)
  - `imag` — [`L82`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L82) — documented in [recurrentgemma-jax-complex_lib](../../../concepts/recurrentgemma-jax-complex_lib.md)
  - `real` — [`L81`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L81) — documented in [recurrentgemma-jax-complex_lib](../../../concepts/recurrentgemma-jax-complex_lib.md)
- protocol/private: `__add__`[`L189`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L189), `__eq__`[`L204`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L204), `__getitem__`[`L195`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L195), `__iter__`[`L214`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L214), `__neg__`[`L178`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L178), `__post_init__`[`L84`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L84), `__radd__`[`L193`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L193), `__rmul__`[`L165`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L165), `__rsub__`[`L185`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L185), `__setitem__`[`L198`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L198), `__sub__`[`L181`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L181), `__truediv__`[`L167`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L167)
- uses (calls/refs, reference-scoped): [`_is_pytree_placeholder`](complex_lib.md#_is_pytree_placeholder)
- used by: [`RealOrComplex`](complex_lib.md#RealOrComplex), [`_treat_method`](complex_lib.md#_treat_method), [`_lru_bwd`](pallas.md#_lru_bwd), [`lru_pallas_scan`](pallas.md#lru_pallas_scan), [`pallas_lru`](pallas.md#pallas_lru), [`exp`](complex_lib.md#exp), [`reverse_block_spec`](pallas.md#reverse_block_spec), [`ones_like`](complex_lib.md#ones_like), [`real_imag_complex`](layers.md#RGLRU.real_imag_complex), [`pad_array_to_divisible`](pallas.md#pad_array_to_divisible), [`complex_to_merged`](layers.md#RGLRU.complex_to_merged), [`conjugate`](complex_lib.md#conjugate), [`get_acc_dtype`](pallas.md#get_acc_dtype), [`log`](complex_lib.md#log), [`softplus`](complex_lib.md#softplus), [`test_change_dtype`](complex_lib_test.md#ComplexTest.test_change_dtype), [`sigmoid`](complex_lib.md#sigmoid), [`einsum`](complex_lib.md#einsum), [`test_complex_wrapper_ops_with_different_dtype_exception`](complex_lib_test.md#ComplexTest.test_complex_wrapper_ops_with_different_dtype_exception), [`test_complex_wrapper_ops_with_jax_complex_exception`](complex_lib_test.md#ComplexTest.test_complex_wrapper_ops_with_jax_complex_exception), [`test_complex_wrapper_with_complex_wrapper_ops`](complex_lib_test.md#ComplexTest.test_complex_wrapper_with_complex_wrapper_ops), [`test_complex_wrapper_with_jax_array_ops`](complex_lib_test.md#ComplexTest.test_complex_wrapper_with_jax_array_ops), [`test_complex_wrapper_with_jax_array_ops_commutes`](complex_lib_test.md#ComplexTest.test_complex_wrapper_with_jax_array_ops_commutes), [`to_custom_complex`](complex_lib.md#to_custom_complex), [`to_complex_wrapper`](complex_lib_test.md#ComplexTest.to_complex_wrapper), [`maybe_wrap_in_complex`](pallas.md#maybe_wrap_in_complex), [`Spec`](pallas.md#Spec), [`convert_to_complex_wrapper`](complex_lib_test.md#ComplexTest.convert_to_complex_wrapper)

## Functions
- `_arg_is_pytree_placeholder(arg: Any)` — [`L29`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L29) — Check if argument is consistent with being a placeholder for pytree validation.
- `_is_pytree_placeholder(*args: Sequence[Any])` — [`L64`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L64) — Check if arguments are consistent with being a placeholder for pytree validation.
- `_treat_method(method_name: str, module: types.ModuleType, x: list[RealOrComplex] | RealOrComplex, *args: Any, **kwargs: Any)` — [`L222`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L222) — Calls the appropriate method depending on the parameters type. — documented in [recurrentgemma-jax-complex_lib](../../../concepts/recurrentgemma-jax-complex_lib.md)
- `abs_squared(x: RealOrComplex)` — [`L334`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L334)
- `conjugate(x: RealOrComplex)` — [`L327`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L327) — documented in [recurrentgemma-jax-complex_lib](../../../concepts/recurrentgemma-jax-complex_lib.md)
- `einsum(sum_str: str, *args: jax.Array | Complex)` — [`L338`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L338) — Computes the equivalent of jnp.einsum. — documented in [recurrentgemma-jax-complex_lib](../../../concepts/recurrentgemma-jax-complex_lib.md)
- `exp(x: RealOrComplex)` — [`L309`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L309) — documented in [recurrentgemma-jax-complex_lib](../../../concepts/recurrentgemma-jax-complex_lib.md)
- `log(x: RealOrComplex)` — [`L318`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L318) — documented in [recurrentgemma-jax-complex_lib](../../../concepts/recurrentgemma-jax-complex_lib.md)
- `ones_like(x: RealOrComplex)` — [`L302`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L302) — documented in [recurrentgemma-jax-complex_lib](../../../concepts/recurrentgemma-jax-complex_lib.md)
- `sigmoid(x: RealOrComplex)` — [`L287`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L287) — documented in [recurrentgemma-jax-layers](../../../concepts/recurrentgemma-jax-layers.md)
- `softplus(x: RealOrComplex)` — [`L294`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L294) — documented in [recurrentgemma-jax-layers](../../../concepts/recurrentgemma-jax-layers.md)
- `to_custom_complex(x: RealOrComplex)` — [`L260`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L260) — documented in [recurrentgemma-jax-complex_lib](../../../concepts/recurrentgemma-jax-complex_lib.md)

## Module values
- `RealOrComplex` — [`L219`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L219) — documented in [recurrentgemma-jax-complex_lib](../../../concepts/recurrentgemma-jax-complex_lib.md)
- `add` — [`L278`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L278)
- `broadcast_to` — [`L265`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L265)
- `concatenate` — [`L266`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L266)
- `expand_dims` — [`L268`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L268)
- `flip` — [`L269`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L269)
- `pad` — [`L279`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L279)
- `rearrange` — [`L283`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L283)
- `repeat` — [`L284`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L284)
- `reshape` — [`L270`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L270)
- `slice_in_dim` — [`L280`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L280)
- `split` — [`L267`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L267)
- `squeeze` — [`L271`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L271)
- `stack` — [`L272`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L272)
- `tile` — [`L273`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L273)
- `transpose` — [`L274`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L274)
- `zeros_like` — [`L275`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/complex_lib.py#L275)

