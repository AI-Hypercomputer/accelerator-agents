---
title: 'Module: ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.ragged_page_attention_v2._pallas_impl_fwd`/
symbols:
  ragged_page_attention_kernel.compute_with_cur_q_blk.compute_with_kv_blk_in_cur_seq: ragged_page_attention_kernel().compute_with_cur_q_blk().compute_with_kv_blk_in_cur_seq().
  ragged_page_attention_kernel: ragged_page_attention_kernel().
  ragged_page_attention_kernel.compute_with_cur_q_blk.flash_attention: ragged_page_attention_kernel().compute_with_cur_q_blk().flash_attention().
  ragged_page_attention_kernel.create_kv_async_copy_descriptors: ragged_page_attention_kernel().create_kv_async_copy_descriptors().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  get_min_heads_per_blk: get_min_heads_per_blk().
  MultiPageAsyncCopyDescriptor.wait: MultiPageAsyncCopyDescriptor#wait().
  ragged_page_attention_kernel.compute_with_cur_q_blk: ragged_page_attention_kernel().compute_with_cur_q_blk().
  MultiPageAsyncCopyDescriptor._async_copies: MultiPageAsyncCopyDescriptor#_async_copies.
  ref_ragged_page_attention: ref_ragged_page_attention().
  dynamic_validate_inputs: dynamic_validate_inputs().
  ragged_page_attention_kernel.prefetch_first_kv_blk: ragged_page_attention_kernel().prefetch_first_kv_blk().
  ragged_page_attention_kernel.compute_with_cur_q_blk.compute_with_kv_blk_in_cur_seq.prefetch_next_kv_blk: ragged_page_attention_kernel().compute_with_cur_q_blk().compute_with_kv_blk_in_cur_seq().prefetch_next_kv_blk().
  get_dtype_packing: get_dtype_packing().
  MultiPageAsyncCopyDescriptor.start: MultiPageAsyncCopyDescriptor#start().
  static_validate_inputs: static_validate_inputs().
  cdiv: cdiv().
  ragged_page_attention_kernel.strided_load_kv: ragged_page_attention_kernel().strided_load_kv().
  ragged_page_attention_kernel.fold_on_2nd_minor: ragged_page_attention_kernel().fold_on_2nd_minor().
  ragged_page_attention_kernel.compute_with_cur_q_blk.flash_attention.masked_store: ragged_page_attention_kernel().compute_with_cur_q_blk().flash_attention().masked_store().
  ragged_page_attention_kernel.compute_with_cur_q_blk.flash_attention.load_with_init: ragged_page_attention_kernel().compute_with_cur_q_blk().flash_attention().load_with_init().
  ragged_page_attention_kernel.compute_with_cur_q_blk.flash_attention.broadcast_to_shape: ragged_page_attention_kernel().compute_with_cur_q_blk().flash_attention().broadcast_to_shape().
  get_min_heads_per_blk.can_be_xla_fully_tiled: get_min_heads_per_blk().can_be_xla_fully_tiled().
  MultiPageAsyncCopyDescriptor: MultiPageAsyncCopyDescriptor#
  MultiPageAsyncCopyDescriptor._vmem_buf: MultiPageAsyncCopyDescriptor#_vmem_buf.
  ragged_page_attention_kernel.is_cur_q_blk_needed: ragged_page_attention_kernel().is_cur_q_blk_needed().
  ragged_page_attention_kernel.compute_with_cur_q_blk.get_next_prefetch_ids: ragged_page_attention_kernel().compute_with_cur_q_blk().get_next_prefetch_ids().
  ragged_page_attention_kernel.compute_with_cur_q_blk.is_valid_kv_blk_in_cur_seq: ragged_page_attention_kernel().compute_with_cur_q_blk().is_valid_kv_blk_in_cur_seq().
  MultiPageAsyncCopyDescriptor.__init__: MultiPageAsyncCopyDescriptor#__init__().
---
# Module: [`ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py)

## Classes
### `MultiPageAsyncCopyDescriptor`
- def: [`ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py:79`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L79)
- doc: Descriptor for asynchronous DMA copy of multiple KV cache pages from HBM to VMEM.
- signature: `class MultiPageAsyncCopyDescriptor:`
- members:
  - `start(self)` — [`L113`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L113) — Starts the async copies.
  - `wait(self)` — [`L118`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L118) — Wait for all async copies to complete and return the VMEM buffer.
- protocol/private: `__init__`[`L91`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L91), `_async_copies`[`L101`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L101), `_vmem_buf`[`L99`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L99)
- used by: [`compute_with_kv_blk_in_cur_seq`](_pallas_impl_fwd.md#ragged_page_attention_kernel.compute_with_cur_q_blk.compute_with_kv_blk_in_cur_seq), [`create_kv_async_copy_descriptors`](_pallas_impl_fwd.md#ragged_page_attention_kernel.create_kv_async_copy_descriptors), [`prefetch_first_kv_blk`](_pallas_impl_fwd.md#ragged_page_attention_kernel.prefetch_first_kv_blk), [`prefetch_next_kv_blk`](_pallas_impl_fwd.md#ragged_page_attention_kernel.compute_with_cur_q_blk.compute_with_kv_blk_in_cur_seq.prefetch_next_kv_blk)

## Functions
- `broadcast_to_shape(arr, shape)` — [`L640`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L640)
- `can_be_xla_fully_tiled(x, packing)` — [`L789`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L789)
- `cdiv(a, b)` — [`L738`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L738) — Compute ceiling division of a by b.
- `compute_with_cur_q_blk(q_states)` — [`L522`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L522)
- `compute_with_kv_blk_in_cur_seq(kv_states)` — [`L676`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L676)
- `create_kv_async_copy_descriptors(heads_blk_idx, seq_idx, kv_blk_idx, buf_idx)` — [`L457`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L457)
- `dynamic_validate_inputs(q: jax.Array, kv_pages: jax.Array, context_lens: jax.Array, block_tables: jax.Array, query_start_loc: jax.Array, num_seqs: jax.Array, *, softmax_scale: float | None = None, sliding_window: int | None = None, logits_soft_cap: float | None = None, mask_value: float | None = None, num_kv_pages_per_block: int | None = None, num_queries_per_block: int | None = None, vmem_limit_bytes: int | None = None)` — [`L228`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L228) — Validate inputs with both static shape checks and dynamic value checks.
- `flash_attention(q, k, v, head_l_ref, head_m_ref, head_acc_ref, softmax_aux_ref, *, kv_blk_idx)` — [`L557`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L557)
- `fold_on_2nd_minor(vec)` — [`L503`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L503)
- `get_dtype_packing(dtype)` — [`L752`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L752) — Compute how many elements of the given dtype fit in a 32-bit word.
- `get_min_heads_per_blk(num_q_heads, num_combined_kv_heads, q_dtype, kv_dtype)` — [`L765`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L765) — Determine the minimum number of heads per processing block for XLA tiling.
- `get_next_prefetch_ids(heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx)` — [`L529`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L529)
- `is_cur_q_blk_needed(q_states)` — [`L517`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L517)
- `is_valid_kv_blk_in_cur_seq(kv_states)` — [`L672`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L672)
- `load_with_init(ref, init_val)` — [`L580`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L580)
- `masked_store(ref, val, start, end, group=1)` — [`L575`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L575)
- `prefetch_first_kv_blk()` — [`L513`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L513)
- `prefetch_next_kv_blk()` — [`L683`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L683)
- `ragged_page_attention_kernel(context_lens_ref, page_indices_ref, cu_q_lens_ref, seq_buf_idx_ref, num_seqs_ref, q_ref, kv_pages_hbm_ref, softmax_aux_ref, o_ref, kv_bufs, sems, l_ref, m_ref, acc_ref, *, softmax_scale: float, sliding_window: int | None = None, logits_soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE)` — [`L386`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L386) — Pallas kernel for ragged paged attention with async DMA and online softmax.
- `ref_ragged_page_attention(queries: jax.Array, kv_pages: jax.Array, context_lens: jax.Array, block_tables: jax.Array, query_start_loc: jax.Array, num_seqs: jax.Array, *, softmax_scale: float = 1, sliding_window: int | None = None, logits_soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, softmax_aux: jax.Array | None = None)` — [`L129`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L129) — Reference implementation of ragged paged attention for correctness testing.
- `static_validate_inputs(q: jax.Array, kv_pages: jax.Array, context_lens: jax.Array, block_tables: jax.Array, query_start_loc: jax.Array, num_seqs: jax.Array, *, softmax_scale: float | None = None, sliding_window: int | None = None, logits_soft_cap: float | None = None, mask_value: float | None = None, num_kv_pages_per_block: int | None = None, num_queries_per_block: int | None = None, vmem_limit_bytes: int | None = None)` — [`L306`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L306) — Validate input tensor shapes, dtypes, and static parameter constraints.
- `strided_load_kv(ref, start, step)` — [`L471`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L471)

## Module values
- `DEFAULT_MASK_VALUE` — [`L76`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v2/_pallas_impl_fwd.py#L76)

