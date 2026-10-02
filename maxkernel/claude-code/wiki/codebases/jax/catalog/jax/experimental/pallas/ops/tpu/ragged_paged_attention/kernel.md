---
title: 'Module: jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.tpu.ragged_paged_attention.kernel`/
symbols:
  ragged_paged_attention: ragged_paged_attention().
  ragged_paged_attention_kernel.compute_with_cur_q_blk.flash_attention: ragged_paged_attention_kernel().compute_with_cur_q_blk().flash_attention().
  ref_ragged_paged_attention: ref_ragged_paged_attention().
  ragged_paged_attention_kernel.compute_with_cur_q_blk.compute_with_kv_blk_in_cur_seq: ragged_paged_attention_kernel().compute_with_cur_q_blk().compute_with_kv_blk_in_cur_seq().
  ragged_paged_attention_kernel: ragged_paged_attention_kernel().
  ragged_paged_attention_kernel.strided_load_kv: ragged_paged_attention_kernel().strided_load_kv().
  static_validate_inputs: static_validate_inputs().
  dynamic_validate_inputs: dynamic_validate_inputs().
  ragged_paged_attention_kernel.create_kv_async_copy_descriptors: ragged_paged_attention_kernel().create_kv_async_copy_descriptors().
  ragged_paged_attention_kernel.compute_with_cur_q_blk.flash_attention.masked_store: ragged_paged_attention_kernel().compute_with_cur_q_blk().flash_attention().masked_store().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  ragged_paged_attention_kernel.compute_with_cur_q_blk: ragged_paged_attention_kernel().compute_with_cur_q_blk().
  MultiPageAsyncCopyDescriptor._async_copies: MultiPageAsyncCopyDescriptor#_async_copies.
  ragged_paged_attention_kernel.fold_on_2nd_minor: ragged_paged_attention_kernel().fold_on_2nd_minor().
  ragged_paged_attention_kernel.compute_with_cur_q_blk.flash_attention.load_with_init: ragged_paged_attention_kernel().compute_with_cur_q_blk().flash_attention().load_with_init().
  ragged_paged_attention_kernel.prefetch_first_kv_blk: ragged_paged_attention_kernel().prefetch_first_kv_blk().
  ragged_paged_attention_kernel.compute_with_cur_q_blk.compute_with_kv_blk_in_cur_seq.prefetch_next_kv_blk: ragged_paged_attention_kernel().compute_with_cur_q_blk().compute_with_kv_blk_in_cur_seq().prefetch_next_kv_blk().
  get_dtype_packing: get_dtype_packing().
  MultiPageAsyncCopyDescriptor.wait: MultiPageAsyncCopyDescriptor#wait().
  get_min_heads_per_blk: get_min_heads_per_blk().
  ragged_paged_attention_kernel.compute_with_cur_q_blk.flash_attention.broadcast_to_shape: ragged_paged_attention_kernel().compute_with_cur_q_blk().flash_attention().broadcast_to_shape().
  MultiPageAsyncCopyDescriptor.start: MultiPageAsyncCopyDescriptor#start().
  ragged_paged_attention_kernel.is_cur_q_blk_needed: ragged_paged_attention_kernel().is_cur_q_blk_needed().
  ragged_paged_attention_kernel.compute_with_cur_q_blk.get_next_prefetch_ids: ragged_paged_attention_kernel().compute_with_cur_q_blk().get_next_prefetch_ids().
  get_min_heads_per_blk.can_be_xla_fully_tiled: get_min_heads_per_blk().can_be_xla_fully_tiled().
  MultiPageAsyncCopyDescriptor: MultiPageAsyncCopyDescriptor#
  MultiPageAsyncCopyDescriptor._vmem_buf: MultiPageAsyncCopyDescriptor#_vmem_buf.
  ragged_paged_attention_kernel.compute_with_cur_q_blk.is_valid_kv_blk_in_cur_seq: ragged_paged_attention_kernel().compute_with_cur_q_blk().is_valid_kv_blk_in_cur_seq().
  ragged_paged_attention.q_index_map: ragged_paged_attention().q_index_map().
  MultiPageAsyncCopyDescriptor.__init__: MultiPageAsyncCopyDescriptor#__init__().
---
# Module: [`jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py)

## Classes
### `MultiPageAsyncCopyDescriptor`
- def: [`jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py:35`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L35)
- doc: Descriptor for async copy of multiple K/V pages from HBM.
- signature: `class MultiPageAsyncCopyDescriptor:`
- members:
  - `start(self)` — [`L62`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L62) — Starts the async copies.
  - `wait(self)` — [`L67`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L67)
- protocol/private: `__init__`[`L38`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L38), `_async_copies`[`L48`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L48), `_vmem_buf`[`L46`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L46)
- uses (calls/refs, reference-scoped): [`select`](../../../../../_src/lax/lax.md#select), [`make_async_copy`](../../../../../_src/pallas/mosaic/primitives.md#make_async_copy)
- used by: [`compute_with_kv_blk_in_cur_seq`](kernel.md#ragged_paged_attention_kernel.compute_with_cur_q_blk.compute_with_kv_blk_in_cur_seq), [`create_kv_async_copy_descriptors`](kernel.md#ragged_paged_attention_kernel.create_kv_async_copy_descriptors), [`prefetch_first_kv_blk`](kernel.md#ragged_paged_attention_kernel.prefetch_first_kv_blk), [`prefetch_next_kv_blk`](kernel.md#ragged_paged_attention_kernel.compute_with_cur_q_blk.compute_with_kv_blk_in_cur_seq.prefetch_next_kv_blk)

## Functions
- `broadcast_to_shape(arr, shape)` — [`L565`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L565)
- `can_be_xla_fully_tiled(x, packing)` — [`L689`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L689)
- `compute_with_cur_q_blk(q_states)` — [`L417`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L417)
- `compute_with_kv_blk_in_cur_seq(kv_states)` — [`L595`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L595)
- `create_kv_async_copy_descriptors(heads_blk_idx, seq_idx, kv_blk_idx, buf_idx)` — [`L338`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L338)
- `dynamic_validate_inputs(q: jax.Array, kv_pages: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, num_seqs: jax.Array, *, sm_scale: float | None = None, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = None, k_scale: float | None = None, v_scale: float | None = None, num_kv_pages_per_block: int | None = None, num_queries_per_block: int | None = None, vmem_limit_bytes: int | None = None)` — [`L152`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L152)
- `flash_attention(q, k, v, head_l_ref, head_m_ref, head_acc_ref, *, kv_blk_idx)` — [`L454`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L454)
- `fold_on_2nd_minor(vec)` — [`L395`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L395)
- `get_dtype_packing(dtype)` — [`L678`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L678)
- `get_min_heads_per_blk(num_q_heads, num_combined_kv_heads, q_dtype, kv_dtype)` — [`L683`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L683)
- `get_next_prefetch_ids(heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx)` — [`L424`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L424)
- `is_cur_q_blk_needed(q_states)` — [`L411`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L411)
- `is_valid_kv_blk_in_cur_seq(kv_states)` — [`L591`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L591)
- `load_with_init(ref, init_val)` — [`L496`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L496)
- `masked_store(ref, val, start, end, group=1)` — [`L492`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L492)
- `prefetch_first_kv_blk()` — [`L405`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L405)
- `prefetch_next_kv_blk()` — [`L604`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L604)
- `q_index_map(heads_blk_idx, q_blk_idx, *_)` — [`L825`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L825)
- `ragged_paged_attention(q: jax.Array, kv_pages: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, num_seqs: jax.Array, *, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, k_scale: float | None = None, v_scale: float | None = None, num_kv_pages_per_block: int | None = None, num_queries_per_block: int | None = None, vmem_limit_bytes: int | None = None)` — [`L732`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L732) — Ragged paged attention that supports mixed prefill and decode.
- `ragged_paged_attention_kernel(kv_lens_ref, page_indices_ref, cu_q_lens_ref, seq_buf_idx_ref, num_seqs_ref, q_ref, kv_pages_hbm_ref, o_ref, kv_bufs, sems, l_ref, m_ref, acc_ref, *, sm_scale: float, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, k_scale: float | None = None, v_scale: float | None = None)` — [`L290`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L290)
- `ref_ragged_paged_attention(queries: jax.Array, kv_pages: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, num_seqs: jax.Array, *, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, k_scale: float | None = None, v_scale: float | None = None)` — [`L73`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L73)
- `static_validate_inputs(q: jax.Array, kv_pages: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, num_seqs: jax.Array, *, sm_scale: float | None = None, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = None, k_scale: float | None = None, v_scale: float | None = None, num_kv_pages_per_block: int | None = None, num_queries_per_block: int | None = None, vmem_limit_bytes: int | None = None)` — [`L216`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L216)
- `strided_load_kv(ref, start, step)` — [`L361`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L361)

## Module values
- `DEFAULT_MASK_VALUE` — [`L32`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py#L32)

