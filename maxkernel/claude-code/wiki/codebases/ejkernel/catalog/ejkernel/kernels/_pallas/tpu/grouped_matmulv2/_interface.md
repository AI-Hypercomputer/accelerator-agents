---
title: 'Module: ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.grouped_matmulv2._interface`/
symbols:
  grouped_matmulv2: grouped_matmulv2().
  _back_grouped_matmul: _back_grouped_matmul.
  _grouped_matmul_bwd: _grouped_matmul_bwd().
  _grouped_matmul_fwd: _grouped_matmul_fwd().
---
# Module: [`ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_interface.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_interface.py)

## Functions
- `_grouped_matmul_bwd(preferred_element_type: jnp.dtype, tiling: tuple[int, int, int] | None, input_buffer_count: int, transpose_rhs: bool, interpret: bool, residual: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None, int], grad: jnp.ndarray)` — [`L153`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_interface.py#L153) — Backward pass for grouped matrix multiplication with custom VJP.
- `_grouped_matmul_fwd(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, preferred_element_type: jnp.dtype, tiling: tuple[int, int, int] | None = (128, 128, 128), input_buffer_count: int = 2, group_offset: jax.Array | None = None, transpose_rhs: bool = False, interpret: bool = False)` — [`L98`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_interface.py#L98) — Forward pass for grouped matrix multiplication with custom VJP.
- `grouped_matmulv2(lhs: Float[Array, "m k"], rhs: Float[Array, "num_groups k n"] | Float[Array, "num_groups n k"], group_sizes: Int[Array, num_groups_or_shards], preferred_element_type: DTypeLike = jnp.float32, tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128), group_offset: Int[Array, ...] | None = None, existing_out: Float[Array, "m n"] | None = None, transpose_rhs: bool = False, interpret: bool = False, precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT)` — [`L222`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_interface.py#L222) — Grouped Matrix Multiplication: Compute separate matrix products for each group.

## Module values
- `_back_grouped_matmul` — [`L92`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv2/_interface.py#L92)

