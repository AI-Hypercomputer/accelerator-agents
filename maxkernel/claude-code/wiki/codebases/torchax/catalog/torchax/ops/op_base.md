---
title: 'Module: torchax/ops/op_base.py'
type: catalog
provenance: extracted
module: torchax/ops/op_base.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `torchax.ops.op_base`/
symbols:
  promote_int_input: promote_int_input().
  InplaceOp: InplaceOp#
  OutVariant: OutVariant#
  foreach_loop: foreach_loop().
  convert_dtype: convert_dtype().
  InplaceOp.__call__: InplaceOp#__call__().
  OutVariant.__call__: OutVariant#__call__().
  maybe_convert_constant_dtype: maybe_convert_constant_dtype().
  convert_dtype.decorator: convert_dtype().decorator().
  convert_dtype.decorator.wrapper: convert_dtype().decorator().wrapper().
  promote_int_input.wrapper: promote_int_input().wrapper().
  P: P.
  InplaceOp.functional: InplaceOp#functional.
  InplaceOp.replace: InplaceOp#replace.
  InplaceOp.position_to_mutate: InplaceOp#position_to_mutate.
  InplaceOp.is_jax_func: InplaceOp#is_jax_func.
  OutVariant.functional: OutVariant#functional.
  InplaceOp.__init__: InplaceOp#__init__().
  OutVariant.__init__: OutVariant#__init__().
---
# Module: [`torchax/ops/op_base.py`](../../../../../../raw/code/torchax/torchax/ops/op_base.py)

## Classes
### `InplaceOp`
- def: [`torchax/ops/op_base.py:30`](../../../../../../raw/code/torchax/torchax/ops/op_base.py#L30) — documented in [torchax-ops-op_base](../../../concepts/torchax-ops-op_base.md)
- signature: `class InplaceOp:`
- members:
  - `functional` — [`L34`](../../../../../../raw/code/torchax/torchax/ops/op_base.py#L34)
  - `is_jax_func` — [`L37`](../../../../../../raw/code/torchax/torchax/ops/op_base.py#L37)
  - `position_to_mutate` — [`L36`](../../../../../../raw/code/torchax/torchax/ops/op_base.py#L36)
  - `replace` — [`L35`](../../../../../../raw/code/torchax/torchax/ops/op_base.py#L35)
- protocol/private: `__call__`[`L39`](../../../../../../raw/code/torchax/torchax/ops/op_base.py#L39), `__init__`[`L31`](../../../../../../raw/code/torchax/torchax/ops/op_base.py#L31)
- uses (calls/refs, reference-scoped): [`View`](../view.md#View), [`update`](../view.md#View.update), [`torch`](../view.md#View.torch)
- used by: [`mutation_ops_to_functional`](jaten.md#mutation_ops_to_functional)

### `OutVariant`
- def: [`torchax/ops/op_base.py:66`](../../../../../../raw/code/torchax/torchax/ops/op_base.py#L66) — documented in [torchax-ops-op_base](../../../concepts/torchax-ops-op_base.md)
- signature: `class OutVariant:`
- members:
  - `functional` — [`L68`](../../../../../../raw/code/torchax/torchax/ops/op_base.py#L68)
- protocol/private: `__call__`[`L70`](../../../../../../raw/code/torchax/torchax/ops/op_base.py#L70), `__init__`[`L67`](../../../../../../raw/code/torchax/torchax/ops/op_base.py#L67)
- uses (calls/refs, reference-scoped): [`Tensor`](../tensor.md#Tensor), [`View`](../view.md#View), [`_elem`](../tensor.md#Tensor._elem), [`update`](../view.md#View.update)
- used by: [`_out_variant_to_functional`](jaten.md#_out_variant_to_functional)

## Functions
- `convert_dtype(use_default_dtype: bool = True)` — [`L84`](../../../../../../raw/code/torchax/torchax/ops/op_base.py#L84) — Converts `dtype` kwarg of function from torch to JAX.
- `decorator(func: types.TorchCallable)` — [`L94`](../../../../../../raw/code/torchax/torchax/ops/op_base.py#L94)
- `foreach_loop(seq: jax.Array, fn: Callable[[jax.Array, jax.Array], jax.Array], init_val=0)` — [`L138`](../../../../../../raw/code/torchax/torchax/ops/op_base.py#L138) — Run `fn` for each element of 1D array `seq`. — documented in [torchax-ops-op_base](../../../concepts/torchax-ops-op_base.md)
- `maybe_convert_constant_dtype(val: types.JaxValue | None, dtype: jnp.dtype | None)` — [`L111`](../../../../../../raw/code/torchax/torchax/ops/op_base.py#L111) — Optionally converts scalar constant's dtype using `numpy`
- `promote_int_input(f: Callable[Concatenate[jax.Array, P], types.JaxValue])` — [`L125`](../../../../../../raw/code/torchax/torchax/ops/op_base.py#L125) — If the first argument is an int array, promote it to float32. — documented in [torchax-ops-op_base](../../../concepts/torchax-ops-op_base.md)
- `wrapper(*args: P.args, dtype: torch.dtype | None = None, **kwargs: P.kwargs)` — [`L96`](../../../../../../raw/code/torchax/torchax/ops/op_base.py#L96)
- `wrapper(x: jax.Array, *args: P.args, **kwargs: P.kwargs)` — [`L129`](../../../../../../raw/code/torchax/torchax/ops/op_base.py#L129) — documented in [torchax-ops-op_base](../../../concepts/torchax-ops-op_base.md)

## Module values
- `P` — [`L81`](../../../../../../raw/code/torchax/torchax/ops/op_base.py#L81)

