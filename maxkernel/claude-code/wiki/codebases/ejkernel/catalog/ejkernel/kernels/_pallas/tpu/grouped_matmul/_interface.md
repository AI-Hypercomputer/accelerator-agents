---
title: 'Module: ejkernel/kernels/_pallas/tpu/grouped_matmul/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/grouped_matmul/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.grouped_matmul._interface`/
symbols:
  grouped_matmul: grouped_matmul().
  _grouped_matmul_bwd: _grouped_matmul_bwd().
  _back_grouped_matmul: _back_grouped_matmul.
  _grouped_matmul_fwd: _grouped_matmul_fwd().
---
# Module: [`ejkernel/kernels/_pallas/tpu/grouped_matmul/_interface.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_interface.py)

## Functions
- `_grouped_matmul_bwd(preferred_element_type: DTypeLike, tiling: tuple[int, int, int], transpose_rhs: bool, interpret: bool, residual: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None, int], grad: jnp.ndarray)` — [`L141`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_interface.py#L141) — Backward pass for grouped matrix multiplication with custom VJP.
- `_grouped_matmul_fwd(lhs: jnp.ndarray, rhs: jnp.ndarray, group_sizes: jnp.ndarray, preferred_element_type: DTypeLike = jnp.float32, tiling: tuple[int, int, int] = (128, 128, 128), group_offset: jnp.ndarray | None = None, existing_out: jnp.ndarray | None = None, transpose_rhs: bool = False, interpret: bool = False)` — [`L88`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_interface.py#L88) — Forward pass for grouped matrix multiplication with custom VJP.
- `grouped_matmul(lhs: Float[Array, "m k"], rhs: Float[Array, "num_groups k n"] | Float[Array, "num_groups n k"], group_sizes: Int[Array, num_groups_or_shards], preferred_element_type: DTypeLike = jnp.float32, tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128), group_offset: Int[Array, ...] | None = None, existing_out: Float[Array, "m n"] | None = None, transpose_rhs: bool = False, interpret: bool = False, precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT)` — [`L205`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_interface.py#L205) — Grouped Matrix Multiplication: Compute separate matrix products for each group.

## Module values
- `_back_grouped_matmul` — [`L85`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmul/_interface.py#L85)

