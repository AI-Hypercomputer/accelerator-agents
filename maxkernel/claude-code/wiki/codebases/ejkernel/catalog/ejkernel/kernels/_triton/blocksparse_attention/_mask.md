---
title: 'Module: ejkernel/kernels/_triton/blocksparse_attention/_mask.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_triton/blocksparse_attention/_mask.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._triton.blocksparse_attention._mask`/
symbols:
  SparseMask.from_inputs: SparseMask#from_inputs().
  define_sparse_mask_fn: define_sparse_mask_fn().
  SparseMask: SparseMask#
  create_sparsity_mask: create_sparsity_mask().
  SparseMask.lower_bounds: SparseMask#lower_bounds.
  SparseMask.upper_bounds: SparseMask#upper_bounds.
  SparseMask.lower_full_bounds: SparseMask#lower_full_bounds.
  SparseMask.upper_full_bounds: SparseMask#upper_full_bounds.
  _compute_sparse_mask: _compute_sparse_mask().
---
# Module: [`ejkernel/kernels/_triton/blocksparse_attention/_mask.py`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_mask.py)

## Classes
### `SparseMask`
- def: [`ejkernel/kernels/_triton/blocksparse_attention/_mask.py:76`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_mask.py#L76)
- doc: Sparse attention mask at the block level.
- signature: `class SparseMask:`
- members:
  - `from_inputs(cls, q_positions: ArrayLike, q_segment_ids: ArrayLike, kv_positions: ArrayLike, kv_segment_ids: ArrayLike, kv_blocksize: int, q_blocksize: int, calculate_dkdv_mask: bool = False, causal: bool = True, window_left: int = -1, window_right: int = -1, mesh: Mesh | None = None)` — [`L112`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_mask.py#L112) — Create a SparseMask from query and key-value positions and segments.
  - `lower_bounds` — [`L106`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_mask.py#L106)
  - `lower_full_bounds` — [`L108`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_mask.py#L108)
  - `upper_bounds` — [`L107`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_mask.py#L107)
  - `upper_full_bounds` — [`L109`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_mask.py#L109)
- uses (calls/refs, reference-scoped): [`define_sparse_mask_fn`](_mask.md#define_sparse_mask_fn)
- used by: [`blocksparse_attention`](../../_pallas/tpu/blocksparse_attention/_kernel.md#blocksparse_attention), [`blocksparse_attention`](../../../modules/operations/blocksparse_attention.md#blocksparse_attention), [`run`](../../../modules/operations/blocksparse_attention.md#BlockSparseAttention.run), [`define_sparse_mask_fn`](_mask.md#define_sparse_mask_fn), [`create_shard_map_wrapper`](../../../modules/operations/blocksparse_attention.md#BlockSparseAttention.create_shard_map_wrapper), [`create_sparsity_mask`](_mask.md#create_sparsity_mask)

## Functions
- `_compute_sparse_mask(outer_positions_ptr, outer_segment_id_ptr, inner_positions_ptr, inner_segment_ids_ptr, lower_block_ptr, upper_block_ptr, lower_full_block_ptr, upper_full_block_ptr, INNER_BLOCK_SIZE: tl.constexpr, INNER_SEQ_LEN: tl.constexpr, OUTER_SEQ_LEN: tl.constexpr, OUTER_BLOCK_SIZE: tl.constexpr, PADDING_SEGMENT_ID: tl.constexpr, USE_SEGMENT_MASK: tl.constexpr, CAUSAL: tl.constexpr, WINDOW_LEFT: tl.constexpr, WINDOW_RIGHT: tl.constexpr, QUERY_IS_OUTER: tl.constexpr)` — [`L182`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_mask.py#L182) — Triton kernel to compute sparse attention mask boundaries.
- `create_sparsity_mask(q_positions: ArrayLike, q_segment_ids: ArrayLike, kv_positions: ArrayLike, kv_segment_ids: ArrayLike, mesh: Mesh | None = None, kv_blocksize: int = 64, q_blocksize: int = 64, causal: bool = True, window_left: int = -1, window_right: int = -1)` — [`L507`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_mask.py#L507) — Create sparse attention masks for forward and backward block-sparse attention passes.
- `define_sparse_mask_fn(q_positions: ArrayLike, q_segment_ids: ArrayLike, kv_positions: ArrayLike, kv_segment_ids: ArrayLike, kv_blocksize: int, q_blocksize: int, calculate_dkdv_mask: bool = False, causal: bool = True, window_left: int = -1, window_right: int = -1)` — [`L367`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_triton/blocksparse_attention/_mask.py#L367) — Generate sparse attention mask boundaries using Triton kernel.

