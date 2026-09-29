---
title: 'Module: ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.grouped_matmulv3._interface`/
symbols:
  grouped_matmulv3: grouped_matmulv3().
  _normalize_tiling: _normalize_tiling().
  _normalize_tiling._wrapped_tile_fn: _normalize_tiling()._wrapped_tile_fn().
  _grouped_matmulv3_bwd: _grouped_matmulv3_bwd().
  _call_grouped_matmulv3: _call_grouped_matmulv3().
  _grouped_matmulv3_core: _grouped_matmulv3_core().
  _grouped_matmulv3_fwd: _grouped_matmulv3_fwd().
  __all__: __all__.
---
# Module: [`ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_interface.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_interface.py)

## Functions
- `_call_grouped_matmulv3(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, preferred_element_type: DTypeLike, tiling: tuple[int, int, int] | LutFn | None, group_offset: jax.Array | None, rhs_scale: jax.Array | None, rhs_bias: jax.Array | None, transpose_rhs: bool, interpret: bool, precision: jax.lax.PrecisionLike)` — [`L73`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_interface.py#L73) — Run the v3 Pallas kernel and return the output and the pre-processed RHS.
- `_grouped_matmulv3_bwd(preferred_element_type: DTypeLike, tiling: tuple[int, int, int] | LutFn | None, transpose_rhs: bool, interpret: bool, precision: jax.lax.PrecisionLike, residual, grad: jax.Array)` — [`L232`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_interface.py#L232) — Backward rule for the v3 grouped matmul custom VJP.
- `_grouped_matmulv3_core(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, preferred_element_type: DTypeLike, tiling: tuple[int, int, int] | LutFn | None, group_offset: jax.Array | None, existing_out: jax.Array | None, rhs_scale: jax.Array | None, rhs_bias: jax.Array | None, transpose_rhs: bool, interpret: bool, precision: jax.lax.PrecisionLike)` — [`L125`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_interface.py#L125) — Custom-VJP wrapper for the v3 grouped matmul.
- `_grouped_matmulv3_fwd(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, preferred_element_type: DTypeLike, tiling: tuple[int, int, int] | LutFn | None, group_offset: jax.Array | None, existing_out: jax.Array | None, rhs_scale: jax.Array | None, rhs_bias: jax.Array | None, transpose_rhs: bool, interpret: bool, precision: jax.lax.PrecisionLike)` — [`L180`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_interface.py#L180) — Forward rule for the v3 grouped matmul custom VJP.
- `_normalize_tiling(tiling: tuple[int, int, int] | LutFn | None, lhs: jax.Array, rhs: jax.Array)` — [`L36`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_interface.py#L36) — Normalize ejkernel-style tiling spec into the v3 ``TileSizes | TileFn`` format.
- `_wrapped_tile_fn(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act)` — [`L64`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_interface.py#L64)
- `grouped_matmulv3(lhs: Float[Array, "m k"], rhs: Float[Array, "num_groups k n"] | Float[Array, "num_groups n k"], group_sizes: Int[Array, num_groups_or_shards], preferred_element_type: DTypeLike = jnp.float32, tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128), group_offset: Int[Array, ...] | None = None, existing_out: Float[Array, "m n"] | None = None, rhs_scale: Float[Array, "num_groups num_blocks 1 n"] | None = None, rhs_bias: Float[Array, "num_groups 1 n"] | None = None, transpose_rhs: bool = False, interpret: bool = False, precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT)` — [`L360`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_interface.py#L360) — Grouped Matrix Multiplication v3 on TPU using the upstream emit_pipeline kernel.

## Module values
- `__all__` — [`L444`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_interface.py#L444)

