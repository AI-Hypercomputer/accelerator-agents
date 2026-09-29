---
title: 'Module: ejkernel/xla_utils/cumsum.py'
type: catalog
provenance: extracted
module: ejkernel/xla_utils/cumsum.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.xla_utils.cumsum`/
symbols:
  chunk_local_cumsum: chunk_local_cumsum().
  _chunk_local_cumsum_vmap_core: _chunk_local_cumsum_vmap_core().
  chunk_global_cumsum: chunk_global_cumsum().
  chunk_local_cumsum_scalar: chunk_local_cumsum_scalar().
  chunk_local_cumsum_vector: chunk_local_cumsum_vector().
  chunk_global_cumsum_scalar: chunk_global_cumsum_scalar().
  chunk_global_cumsum_vector: chunk_global_cumsum_vector().
  chunk_local_cumsum.create_padded_batch: chunk_local_cumsum().create_padded_batch().
---
# Module: [`ejkernel/xla_utils/cumsum.py`](../../../../../../raw/code/ejkernel/ejkernel/xla_utils/cumsum.py)

## Functions
- `_chunk_local_cumsum_vmap_core(g_padded_batched: jnp.ndarray, mask: jnp.ndarray, chunk_size: int, reverse: bool, softmax_scale: float | None, head_first: bool, output_dtype: DTypeLike | None, is_vector: bool)` — [`L382`](../../../../../../raw/code/ejkernel/ejkernel/xla_utils/cumsum.py#L382) — Internal vmapped implementation of chunked local cumsum for variable-length sequences.
- `chunk_global_cumsum(s: jnp.ndarray, reverse: bool = False, cu_seqlens: jnp.ndarray | None = None, softmax_scale: float | None = None, head_first: bool = False, output_dtype: DTypeLike | None = None)` — [`L513`](../../../../../../raw/code/ejkernel/ejkernel/xla_utils/cumsum.py#L513) — Compute global cumulative sum across sequences.
- `chunk_global_cumsum_scalar(s: jnp.ndarray, reverse: bool = False, cu_seqlens: jnp.ndarray | None = None, softmax_scale: float | None = None, head_first: bool = False, output_dtype: DTypeLike | None = None)` — [`L200`](../../../../../../raw/code/ejkernel/ejkernel/xla_utils/cumsum.py#L200) — Compute global cumulative sum across sequences for 3D scalar inputs.
- `chunk_global_cumsum_vector(s: jnp.ndarray, reverse: bool = False, cu_seqlens: jnp.ndarray | None = None, softmax_scale: float | None = None, head_first: bool = False, output_dtype: DTypeLike | None = None)` — [`L287`](../../../../../../raw/code/ejkernel/ejkernel/xla_utils/cumsum.py#L287) — Compute global cumulative sum across sequences for 4D vector inputs.
- `chunk_local_cumsum(g: jnp.ndarray, chunk_size: int, reverse: bool = False, softmax_scale: float | None = None, cu_seqlens: jnp.ndarray | None = None, head_first: bool = False, output_dtype: DTypeLike | None = None, **kwargs)` — [`L417`](../../../../../../raw/code/ejkernel/ejkernel/xla_utils/cumsum.py#L417) — Compute local cumulative sum within fixed-size chunks.
- `chunk_local_cumsum_scalar(g: jnp.ndarray, chunk_size: int, reverse: bool = False, softmax_scale: float | None = None, head_first: bool = False, output_dtype: DTypeLike | None = None)` — [`L57`](../../../../../../raw/code/ejkernel/ejkernel/xla_utils/cumsum.py#L57) — Compute local cumulative sum within chunks for 3D scalar inputs.
- `chunk_local_cumsum_vector(g: jnp.ndarray, chunk_size: int, reverse: bool = False, softmax_scale: float | None = None, head_first: bool = False, output_dtype: DTypeLike | None = None)` — [`L128`](../../../../../../raw/code/ejkernel/ejkernel/xla_utils/cumsum.py#L128) — Compute local cumulative sum within chunks for 4D vector inputs.
- `create_padded_batch(i, _)` — [`L489`](../../../../../../raw/code/ejkernel/ejkernel/xla_utils/cumsum.py#L489)

