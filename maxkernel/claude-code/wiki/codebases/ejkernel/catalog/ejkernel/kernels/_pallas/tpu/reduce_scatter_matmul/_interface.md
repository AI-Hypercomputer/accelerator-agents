---
title: 'Module: ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.reduce_scatter_matmul._interface`/
symbols:
  reduce_scatter_matmul: reduce_scatter_matmul().
  _reduce_scatter_matmul_core: _reduce_scatter_matmul_core().
  _reduce_scatter_matmul_core_fwd: _reduce_scatter_matmul_core_fwd().
  _reduce_scatter_matmul_core_bwd: _reduce_scatter_matmul_core_bwd().
  __all__: __all__.
---
# Module: [`ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_interface.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_interface.py)

## Functions
- `_reduce_scatter_matmul_core(x: Float[Array, "m k_shard"], y: Float[Array, "n k_shard"], axis_name: str, bm: int, bn: int, bk: int, tp_size: int | None, collective_id: int | None, precision: jax.lax.PrecisionLike)` — [`L33`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_interface.py#L33) — Differentiable core of reduce-scatter matmul with custom VJP.
- `_reduce_scatter_matmul_core_bwd(axis_name: str, bm: int, bn: int, bk: int, tp_size: int | None, collective_id: int | None, precision: jax.lax.PrecisionLike, residual, dy: Float[Array, "m_local n"])` — [`L128`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_interface.py#L128) — Custom VJP backward rule: reconstruct full dy via all-gather and compute grads.
- `_reduce_scatter_matmul_core_fwd(x: Float[Array, "m k_shard"], y: Float[Array, "n k_shard"], axis_name: str, bm: int, bn: int, bk: int, tp_size: int | None, collective_id: int | None, precision: jax.lax.PrecisionLike)` — [`L83`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_interface.py#L83) — Custom VJP forward rule: run the Pallas kernel and save x, y as residuals.
- `reduce_scatter_matmul(x: Float[Array, "m k_shard"], y: Float[Array, "n k_shard"], axis_name: str, bm: int = 128, bn: int = 128, bk: int = 128, tp_size: int | None = None, collective_id: int | None = 0, precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT)` — [`L176`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_interface.py#L176) — Bidirectional M-split reduce-scatter matmul for TPU with differentiable backward.

## Module values
- `__all__` — [`L226`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/reduce_scatter_matmul/_interface.py#L226)

