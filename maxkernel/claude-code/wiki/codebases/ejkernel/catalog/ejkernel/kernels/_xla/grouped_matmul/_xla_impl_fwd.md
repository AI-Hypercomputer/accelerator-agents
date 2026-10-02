---
title: 'Module: ejkernel/kernels/_xla/grouped_matmul/_xla_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_xla/grouped_matmul/_xla_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._xla.grouped_matmul._xla_impl_fwd`/
symbols:
  grouped_matmul: grouped_matmul().
  set_xla_metadata: set_xla_metadata.
---
# Module: [`ejkernel/kernels/_xla/grouped_matmul/_xla_impl_fwd.py`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/grouped_matmul/_xla_impl_fwd.py)

## Functions
- `grouped_matmul(lhs: Float[Array, "m k"], rhs: Float[Array, "num_groups k n"] | Float[Array, "num_groups n k"], group_sizes: Int[Array, num_groups_or_shards], preferred_element_type: DTypeLike = jnp.float32, tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128), group_offset: Int[Array, ...] | None = None, existing_out: Float[Array, "m n"] | None = None, transpose_rhs: bool = False, interpret: bool = False, precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT)` — [`L43`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/grouped_matmul/_xla_impl_fwd.py#L43) — Grouped Matrix Multiplication via ``jax.lax.ragged_dot_general``.

## Module values
- `set_xla_metadata` — [`L40`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/grouped_matmul/_xla_impl_fwd.py#L40)

