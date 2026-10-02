---
title: 'Module: src/maxtext/kernels/megablox/ops.py'
type: catalog
provenance: extracted
module: src/maxtext/kernels/megablox/ops.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.kernels.megablox.ops`/
symbols:
  _gmm_bwd: _gmm_bwd().
  gmm: gmm().
  _gmm_fwd: _gmm_fwd().
  DLHS_RAGGED_DOT_DIM_NUMS: DLHS_RAGGED_DOT_DIM_NUMS.
  DRHS_RAGGED_DOT_DIM_NUMS: DRHS_RAGGED_DOT_DIM_NUMS.
---
# Module: [`src/maxtext/kernels/megablox/ops.py`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/ops.py)

## Functions
- `_gmm_bwd(lhs_dtype: jax.typing.DTypeLike, rhs_dtype: jax.typing.DTypeLike, preferred_element_type: jnp.dtype, tiling: tuple[int, int, int, int, int, int, int, int, int], transpose_rhs: bool, interpret: bool, quantization_rule: qwix.QtRule | None, use_tokamax_backend: bool, weight_gather_axes: List[Tuple[str, int]] | None, use_manual_quantization: bool, lhs_vma_axes: tuple, rhs_vma_axes: tuple, use_gmm_v2: bool, residual: tuple[jnp.ndarray | qpl.QArray, jnp.ndarray | qpl.QArray, jnp.ndarray, jnp.ndarray | None,], grad: jnp.ndarray)` — [`L240`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/ops.py#L240) — Backward function for throughput GMM VJP.
- `_gmm_fwd(lhs: jnp.ndarray, rhs: jnp.ndarray, group_sizes: jnp.ndarray, preferred_element_type: jnp.dtype = jnp.float32, tiling: tuple[int, int, int, int, int, int, int, int, int] = (128, 128, 128, 128, 128, 128, 128, 128, 128), group_offset: jnp.ndarray | None = None, existing_out: jnp.ndarray | None = None, transpose_rhs: bool = False, interpret: bool = False, quantization_rule: qwix.QtRule | None = None, use_tokamax_backend: bool = False, weight_gather_axes: List[Tuple[str, int]] | None = None, use_manual_quantization: bool = False, lhs_vma_axes: tuple = tuple(), rhs_vma_axes: tuple = tuple(), use_gmm_v2: bool = False)` — [`L118`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/ops.py#L118) — Forward function for GMM VJP.
- `gmm(lhs: jnp.ndarray, rhs: jnp.ndarray, group_sizes: jnp.ndarray, preferred_element_type: jnp.dtype = jnp.float32, tiling: tuple[int, int, int, int, int, int, int, int, int] = (128, 128, 128, 128, 128, 128, 128, 128, 128), group_offset: jnp.ndarray | None = None, existing_out: jnp.ndarray | None = None, transpose_rhs: bool = False, interpret: bool = False, lhs_quantize_dtype: Literal[jnp.int4, jnp.int8] | None = None, rhs_quantize_dtype: Literal[jnp.int4, jnp.int8] | None = None, use_qwix_quantization: bool = False, use_tokamax_backend: bool = False, weight_gather_axes: List[Tuple[str, int]] | None = None, lhs_vma_axes: tuple = tuple(), rhs_vma_axes: tuple = tuple(), qwix_rule: qwix.QtRule | None = None, use_manual_quantization: bool = False, use_gmm_v2: bool = False)` — [`L46`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/ops.py#L46) — Grouped matrix multiplication operation.

## Module values
- `DLHS_RAGGED_DOT_DIM_NUMS` — [`L33`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/ops.py#L33)
- `DRHS_RAGGED_DOT_DIM_NUMS` — [`L39`](../../../../../../../../raw/code/maxtext/src/maxtext/kernels/megablox/ops.py#L39)

