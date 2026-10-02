---
title: 'Module: axlearn/common/megablock/ops.py'
type: catalog
provenance: extracted
module: axlearn/common/megablock/ops.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.megablock.ops`/
symbols:
  _gmm_bwd: _gmm_bwd().
  _gmm_fwd: _gmm_fwd().
  CAST_DTYPE: CAST_DTYPE.
  gmm: gmm().
  _select_and_cache_gmm_backend: _select_and_cache_gmm_backend().
  _gmm_impl_cache: _gmm_impl_cache.
  GmmResidual: GmmResidual.
  gmm_gpu: gmm_gpu.
  select_tiling_fn: select_tiling_fn().
  select_tiling_fn.tiling_fn: select_tiling_fn().tiling_fn().
---
# Module: [`axlearn/common/megablock/ops.py`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/ops.py)

## Functions
- `_gmm_bwd(preferred_element_type: jnp.dtype, tiling: tuple[int, int, int], transpose_rhs: bool, interpret: bool, residual: GmmResidual, grad: Tensor)` — [`L59`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/ops.py#L59) — Backward function for throughput GMM VJP.
- `_gmm_fwd(lhs: Tensor, rhs: Tensor, group_sizes: Tensor, preferred_element_type: jnp.dtype = jnp.float32, tiling: tuple[int, int, int] = (128, 128, 128), group_offset: Optional[Tensor] = None, transpose_rhs: bool = False, interpret: bool = False)` — [`L27`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/ops.py#L27) — Forward function for GMM VJP.
- `_select_and_cache_gmm_backend()` — [`L106`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/ops.py#L106) — Internal function to detect backend, select implementation, and cache it.
- `gmm(*args, **kwargs)` — [`L160`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/ops.py#L160) — Universal GMM interface. Selects and caches the backend-specific
- `select_tiling_fn(tiling_larger_k: tuple[int, int, int], tiling_larger_n: tuple[int, int, int])` — [`L125`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/ops.py#L125) — Select tiling for GMM dynamically depending on the k and n values.
- `tiling_fn(m: int, k: int, n: int)` — [`L150`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/ops.py#L150)

## Module values
- `CAST_DTYPE` — [`L24`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/ops.py#L24)
- `GmmResidual` — [`L14`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/ops.py#L14)
- `_gmm_impl_cache` — [`L13`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/ops.py#L13)
- `gmm_gpu` — [`L16`](../../../../../../../raw/code/axlearn/axlearn/common/megablock/ops.py#L16)

