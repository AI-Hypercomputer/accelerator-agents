---
title: 'Module: graphcast/sparse_transformer_utils.py'
type: catalog
provenance: extracted
module: graphcast/sparse_transformer_utils.py
status: fresh
symbol_base: scip-python python graphcast 0.0.0 `graphcast.sparse_transformer_utils`/
symbols:
  wrap_fn_for_upcast_downcast: wrap_fn_for_upcast_downcast().
  reduce_precision: reduce_precision().
  reduce_precision_fwd: reduce_precision_fwd().
  reduce_precision_bwd: reduce_precision_bwd().
---
# Module: [`graphcast/sparse_transformer_utils.py`](../../../../../raw/code/graphcast/graphcast/sparse_transformer_utils.py)

## Functions
- `reduce_precision(x, exponent_bits, mantissa_bits)` — [`L25`](../../../../../raw/code/graphcast/graphcast/sparse_transformer_utils.py#L25) — documented in [graphcast-sparse_transformer](../../concepts/graphcast-sparse_transformer.md)
- `reduce_precision_bwd(exponent_bits, mantissa_bits, res, dout)` — [`L34`](../../../../../raw/code/graphcast/graphcast/sparse_transformer_utils.py#L34)
- `reduce_precision_fwd(x, exponent_bits, mantissa_bits)` — [`L30`](../../../../../raw/code/graphcast/graphcast/sparse_transformer_utils.py#L30)
- `wrap_fn_for_upcast_downcast(inputs: Union[jnp.ndarray, Tuple[jnp.ndarray, ...]], fn: Callable[[Union[jnp.ndarray, Tuple[jnp.ndarray, ...]]], Union[jnp.ndarray, Tuple[jnp.ndarray, ...]]], f32_upcast: bool = True, guard_against_excess_precision: bool = True)` — [`L42`](../../../../../raw/code/graphcast/graphcast/sparse_transformer_utils.py#L42) — Wraps `fn` to  upcast to float32 and then downcast, for use with BF16. — documented in [graphcast-sparse_transformer](../../concepts/graphcast-sparse_transformer.md)

