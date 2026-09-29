---
title: 'Module: ejkernel/kernels/_xla/grouped_matmulv3/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_xla/grouped_matmulv3/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._xla.grouped_matmulv3._interface`/
symbols:
  grouped_matmulv3: grouped_matmulv3().
  grouped_matmulv3_reference: grouped_matmulv3_reference().
  grouped_matmulv3_autodiff_reference: grouped_matmulv3_autodiff_reference().
  _grouped_matmulv3_core: _grouped_matmulv3_core().
  _grouped_matmulv3_fwd: _grouped_matmulv3_fwd().
  _grouped_matmulv3_bwd: _grouped_matmulv3_bwd().
  _apply_rhs_scale_bias: _apply_rhs_scale_bias().
  _active_group_ids: _active_group_ids().
  __all__: __all__.
---
# Module: [`ejkernel/kernels/_xla/grouped_matmulv3/_interface.py`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/grouped_matmulv3/_interface.py)

## Functions
- `_active_group_ids(group_sizes: jax.Array, num_groups: int, total_rows: int, group_offset: jax.Array | None)` — [`L111`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/grouped_matmulv3/_interface.py#L111) — Build a per-row group-index vector for use with ``jax.vmap`` dispatch.
- `_apply_rhs_scale_bias(rhs: jax.Array, rhs_scale: jax.Array | None, rhs_bias: jax.Array | None, *, transpose_rhs: bool)` — [`L49`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/grouped_matmulv3/_interface.py#L49) — Pre-process ``rhs`` by applying optional block-wise scale and extracting bias.
- `_grouped_matmulv3_bwd(preferred_element_type: DTypeLike, tiling: tuple[int, int, int] | LutFn | None, transpose_rhs: bool, interpret: bool, precision: jax.lax.PrecisionLike, residual, grad: jax.Array)` — [`L394`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/grouped_matmulv3/_interface.py#L394) — Backward rule for ``_grouped_matmulv3_core``'s custom VJP.
- `_grouped_matmulv3_core(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, preferred_element_type: DTypeLike, tiling: tuple[int, int, int] | LutFn | None, group_offset: jax.Array | None, existing_out: jax.Array | None, rhs_scale: jax.Array | None, rhs_bias: jax.Array | None, transpose_rhs: bool, interpret: bool, precision: jax.lax.PrecisionLike)` — [`L298`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/grouped_matmulv3/_interface.py#L298) — GMM v3 core function with a custom VJP defined for stable gradient computation.
- `_grouped_matmulv3_fwd(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, preferred_element_type: DTypeLike, tiling: tuple[int, int, int] | LutFn | None, group_offset: jax.Array | None, existing_out: jax.Array | None, rhs_scale: jax.Array | None, rhs_bias: jax.Array | None, transpose_rhs: bool, interpret: bool, precision: jax.lax.PrecisionLike)` — [`L354`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/grouped_matmulv3/_interface.py#L354) — Forward rule for ``_grouped_matmulv3_core``'s custom VJP.
- `grouped_matmulv3(lhs: Float[Array, "m k"], rhs: Float[Array, "num_groups k n"] | Float[Array, "num_groups n k"], group_sizes: Int[Array, num_groups_or_shards], preferred_element_type: DTypeLike = jnp.float32, tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128), group_offset: Int[Array, ...] | None = None, existing_out: Float[Array, "m n"] | None = None, rhs_scale: Float[Array, "num_groups num_blocks 1 n"] | None = None, rhs_bias: Float[Array, "num_groups 1 n"] | None = None, transpose_rhs: bool = False, interpret: bool = False, precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT)` — [`L505`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/grouped_matmulv3/_interface.py#L505) — Grouped Matrix Multiplication v3 with optional block-float scale and bias.
- `grouped_matmulv3_autodiff_reference(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, preferred_element_type: DTypeLike = jnp.float32, tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128), group_offset: jax.Array | None = None, existing_out: jax.Array | None = None, rhs_scale: jax.Array | None = None, rhs_bias: jax.Array | None = None, transpose_rhs: bool = False, interpret: bool = False, precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT)` — [`L148`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/grouped_matmulv3/_interface.py#L148) — Pure-JAX vmap reference for GMM v3, used as the autodiff-compatible path.
- `grouped_matmulv3_reference(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, preferred_element_type: DTypeLike = jnp.float32, tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128), group_offset: jax.Array | None = None, existing_out: jax.Array | None = None, rhs_scale: jax.Array | None = None, rhs_bias: jax.Array | None = None, transpose_rhs: bool = False, interpret: bool = False, precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT)` — [`L212`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/grouped_matmulv3/_interface.py#L212) — Forward pass for GMM v3 used by both XLA execution and the TPU backward helpers.

## Module values
- `__all__` — [`L583`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/grouped_matmulv3/_interface.py#L583)

