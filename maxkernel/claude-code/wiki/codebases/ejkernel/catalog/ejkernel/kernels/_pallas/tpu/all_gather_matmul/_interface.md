---
title: 'Module: ejkernel/kernels/_pallas/tpu/all_gather_matmul/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/all_gather_matmul/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.all_gather_matmul._interface`/
symbols:
  all_gather_matmul: all_gather_matmul().
  _all_gather_matmul_core_bwd: _all_gather_matmul_core_bwd().
  _all_gather_matmul_core: _all_gather_matmul_core().
  _resolve_tp_size: _resolve_tp_size().
  _all_gather_matmul_core_fwd: _all_gather_matmul_core_fwd().
  _largest_divisor_leq: _largest_divisor_leq().
  _infer_axis_size: _infer_axis_size().
  __all__: __all__.
---
# Module: [`ejkernel/kernels/_pallas/tpu/all_gather_matmul/_interface.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_interface.py)

## Functions
- `_all_gather_matmul_core(x: Float[Array, "m_local k"], y: Float[Array, "k n_local"] | Float[Array, "n_local k"], axis_name: str, rhs_transpose: bool, bn: int | None, bk: int | None, tp_size: int | None, collective_id: int | None, precision: jax.lax.PrecisionLike)` — [`L74`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_interface.py#L74) — Forward-only dispatch to the Pallas all-gather matmul kernel.
- `_all_gather_matmul_core_bwd(axis_name: str, rhs_transpose: bool, bn: int | None, bk: int | None, tp_size: int | None, collective_id: int | None, precision: jax.lax.PrecisionLike, residual, dy: Float[Array, "m n_local"])` — [`L148`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_interface.py#L148) — Custom VJP backward pass for all-gather matmul.
- `_all_gather_matmul_core_fwd(x: Float[Array, "m_local k"], y: Float[Array, "k n_local"] | Float[Array, "n_local k"], axis_name: str, rhs_transpose: bool, bn: int | None, bk: int | None, tp_size: int | None, collective_id: int | None, precision: jax.lax.PrecisionLike)` — [`L122`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_interface.py#L122) — Custom VJP forward pass: compute output and save residuals ``(x, y)``.
- `_infer_axis_size(axis_name: str)` — [`L45`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_interface.py#L45) — Infer collective axis size from the active mapped context when available.
- `_largest_divisor_leq(x: int, candidates: tuple[int, ...] = (512, 256, 128, 64, 32, 16, 8, 4, 2, 1))` — [`L65`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_interface.py#L65)
- `_resolve_tp_size(tp_size: int | None, axis_name: str)` — [`L57`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_interface.py#L57) — Resolve tensor-parallel world size using explicit value, axis context, then global device count.
- `all_gather_matmul(x: Float[Array, "m_local k"], y: Float[Array, "k n_local"] | Float[Array, "n_local k"], axis_name: str, rhs_transpose: bool = False, bn: int | None = None, bk: int | None = None, tp_size: int | None = None, collective_id: int | None = 0, precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT)` — [`L233`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_interface.py#L233) — Bidirectional ring all-gather fused with matmul on TPU.

## Module values
- `__all__` — [`L293`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/all_gather_matmul/_interface.py#L293)

