---
title: 'Module: axlearn/common/flash_attention/tpu_paged_attention_kernel.py'
type: catalog
provenance: extracted
module: axlearn/common/flash_attention/tpu_paged_attention_kernel.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.flash_attention.tpu_paged_attention_kernel`/
symbols:
  prepare_block_sparse_map: prepare_block_sparse_map().
  MultiPageAsyncCopyDescriptor._make_async_copy: MultiPageAsyncCopyDescriptor#_make_async_copy().
  MultiPageAsyncCopyDescriptor.wait_and_get_loaded: MultiPageAsyncCopyDescriptor#wait_and_get_loaded().
  MultiPageAsyncCopyDescriptor.start: MultiPageAsyncCopyDescriptor#start().
  _make_index_map: _make_index_map().
  _paged_flash_attention_sparse_kernel.flash_attention: _paged_flash_attention_sparse_kernel().flash_attention().
  _paged_flash_attention_kernel.flash_attention: _paged_flash_attention_kernel().flash_attention().
  MultiPageAsyncCopyDescriptor._async_copies: MultiPageAsyncCopyDescriptor#_async_copies.
  _make_index_map_for_logit_sink: _make_index_map_for_logit_sink().
  _paged_flash_attention_sparse_kernel.create_kv_async_copy_descriptors: _paged_flash_attention_sparse_kernel().create_kv_async_copy_descriptors().
  _paged_flash_attention_kernel.create_kv_async_copy_descriptors: _paged_flash_attention_kernel().create_kv_async_copy_descriptors().
  _paged_flash_attention_sparse_kernel.flash_attention.prefetch_first_block: _paged_flash_attention_sparse_kernel().flash_attention().prefetch_first_block().
  _paged_flash_attention_sparse_kernel.flash_attention.prefetch_next_block: _paged_flash_attention_sparse_kernel().flash_attention().prefetch_next_block().
  _paged_flash_attention_kernel.flash_attention.prefetch_first_block: _paged_flash_attention_kernel().flash_attention().prefetch_first_block().
  _paged_flash_attention_kernel.flash_attention.prefetch_next_block: _paged_flash_attention_kernel().flash_attention().prefetch_next_block().
  MultiPageAsyncCopyDescriptor._pages_hbm_ref: MultiPageAsyncCopyDescriptor#_pages_hbm_ref.
  _paged_flash_attention_sparse_kernel: _paged_flash_attention_sparse_kernel().
  _paged_flash_attention_kernel: _paged_flash_attention_kernel().
  MultiPageAsyncCopyDescriptor: MultiPageAsyncCopyDescriptor#
  _paged_flash_attention_sparse_kernel.compute_block_indices: _paged_flash_attention_sparse_kernel().compute_block_indices().
  _paged_flash_attention_sparse_kernel.compute_block_indices.advance_b: _paged_flash_attention_sparse_kernel().compute_block_indices().advance_b().
  _paged_flash_attention_sparse_kernel.compute_block_indices.advance_h: _paged_flash_attention_sparse_kernel().compute_block_indices().advance_h().
  _paged_flash_attention_kernel.compute_block_indices: _paged_flash_attention_kernel().compute_block_indices().
  _paged_flash_attention_kernel.compute_block_indices.advance_b: _paged_flash_attention_kernel().compute_block_indices().advance_b().
  _paged_flash_attention_kernel.compute_block_indices.advance_h: _paged_flash_attention_kernel().compute_block_indices().advance_h().
  MultiPageAsyncCopyDescriptor._vmem_buffer: MultiPageAsyncCopyDescriptor#_vmem_buffer.
  _paged_flash_attention_sparse_kernel.flash_attention.init: _paged_flash_attention_sparse_kernel().flash_attention().init().
  _paged_flash_attention_kernel.flash_attention.init: _paged_flash_attention_kernel().flash_attention().init().
  MultiPageAsyncCopyDescriptor._num_pages_to_load: MultiPageAsyncCopyDescriptor#_num_pages_to_load.
  MultiPageAsyncCopyDescriptor._sem: MultiPageAsyncCopyDescriptor#_sem.
  MultiPageAsyncCopyDescriptor._page_indices: MultiPageAsyncCopyDescriptor#_page_indices.
  MultiPageAsyncCopyDescriptor._page_indices_start_offset: MultiPageAsyncCopyDescriptor#_page_indices_start_offset.
  _make_index_map.dense_index_map: _make_index_map().dense_index_map().
  _make_index_map.sparse_index_map: _make_index_map().sparse_index_map().
  _make_index_map_for_logit_sink.index_map: _make_index_map_for_logit_sink().index_map().
  _paged_flash_attention_sparse_kernel.compute_block_indices.advance_b.advance_to_next_non_zero_length: _paged_flash_attention_sparse_kernel().compute_block_indices().advance_b().advance_to_next_non_zero_length().
  _paged_flash_attention_kernel.compute_block_indices.advance_b.advance_to_next_non_zero_length: _paged_flash_attention_kernel().compute_block_indices().advance_b().advance_to_next_non_zero_length().
  MultiPageAsyncCopyDescriptor.__init__: MultiPageAsyncCopyDescriptor#__init__().
---
# Module: [`axlearn/common/flash_attention/tpu_paged_attention_kernel.py`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py)

## Classes
### `MultiPageAsyncCopyDescriptor`
- def: [`axlearn/common/flash_attention/tpu_paged_attention_kernel.py:40`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L40)
- doc: Descriptor for async copy of multiple K/V pages from HBM.
- signature: `class MultiPageAsyncCopyDescriptor:`
- members:
  - `start(self)` — [`L76`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L76) — Starts the async copies.
  - `wait_and_get_loaded(self)` — [`L81`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L81) — Wait async copies and gets the loaded buffer as a Tensor.
- protocol/private: `__init__`[`L47`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L47), `_async_copies`[`L66`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L66), `_make_async_copy`[`L68`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L68), `_num_pages_to_load`[`L58`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L58), `_page_indices`[`L64`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L64), `_page_indices_start_offset`[`L65`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L65), `_pages_hbm_ref`[`L60`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L60), `_sem`[`L63`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L63), `_vmem_buffer`[`L57`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L57)
- uses (calls/refs, reference-scoped): [`Tensor`](../utils.md#Tensor)
- used by: [`flash_attention`](tpu_paged_attention_kernel.md#_paged_flash_attention_kernel.flash_attention), [`flash_attention`](tpu_paged_attention_kernel.md#_paged_flash_attention_sparse_kernel.flash_attention), [`create_kv_async_copy_descriptors`](tpu_paged_attention_kernel.md#_paged_flash_attention_kernel.create_kv_async_copy_descriptors), [`create_kv_async_copy_descriptors`](tpu_paged_attention_kernel.md#_paged_flash_attention_sparse_kernel.create_kv_async_copy_descriptors), [`prefetch_first_block`](tpu_paged_attention_kernel.md#_paged_flash_attention_kernel.flash_attention.prefetch_first_block), [`prefetch_first_block`](tpu_paged_attention_kernel.md#_paged_flash_attention_sparse_kernel.flash_attention.prefetch_first_block), [`prefetch_next_block`](tpu_paged_attention_kernel.md#_paged_flash_attention_kernel.flash_attention.prefetch_next_block), [`prefetch_next_block`](tpu_paged_attention_kernel.md#_paged_flash_attention_sparse_kernel.flash_attention.prefetch_next_block)

## Functions
- `_make_index_map(megacore_mode: Optional[str] = None, num_cores: int = 2, is_rearranged: bool = False, is_query: bool = False, is_sparse: bool = False)` — [`L138`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L138) — Creates an index map function for query/bias tensor.
- `_make_index_map_for_logit_sink(megacore_mode: Optional[str] = None, num_cores: int = 2, is_rearranged: bool = False)` — [`L189`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L189) — Creates an index map function for logit_sink tensor.
- `_paged_flash_attention_kernel(lengths_ref, page_indices_ref, buffer_index_ref, init_flag_ref, q_ref, k_pages_hbm_ref, v_pages_hbm_ref, bias_ref, logit_sink_ref, o_ref, m_i, l_i, o_scratch, k_vmem_buffer, v_vmem_buffer, k_sems, v_sems, *, batch_size: int, pages_per_compute_block: int, pages_per_sequence: int, softmax_scale: float, mask_fn: Optional[MaskFn] = None, megacore_mode: Optional[str] = None, program_id: Optional[Tuple[int, int, int, int]] = None)` — [`L430`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L430) — Compute paged flash attention.
- `_paged_flash_attention_sparse_kernel(kv_block_offset, kv_block_offset_size, lengths_ref, page_indices_ref, buffer_index_ref, init_flag_ref, q_ref, k_pages_hbm_ref, v_pages_hbm_ref, bias_ref, logit_sink_ref, o_ref, m_i, l_i, o_scratch, k_vmem_buffer, v_vmem_buffer, k_sems, v_sems, *, batch_size: int, pages_per_compute_block: int, pages_per_sequence: int, softmax_scale: float, mask_fn: Optional[MaskFn] = None, megacore_mode: Optional[str] = None)` — [`L211`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L211)
- `advance_b()` — [`L278`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L278)
- `advance_b()` — [`L500`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L500)
- `advance_h()` — [`L307`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L307)
- `advance_h()` — [`L525`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L525)
- `advance_to_next_non_zero_length()` — [`L281`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L281)
- `advance_to_next_non_zero_length()` — [`L503`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L503)
- `compute_block_indices(b, h, i)` — [`L266`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L266) — Given current block indices prefetch next block.
- `compute_block_indices(b, h, i)` — [`L494`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L494) — Given current block indices, get (next_b, next_h, next_i) for pre-fetching.
- `create_kv_async_copy_descriptors(b, h, i, buffer_index)` — [`L313`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L313)
- `create_kv_async_copy_descriptors(b, h, i, buffer_index)` — [`L531`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L531)
- `dense_index_map(core_index, b, h, i, *_)` — [`L147`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L147)
- `flash_attention()` — [`L339`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L339)
- `flash_attention()` — [`L555`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L555)
- `index_map(core_index, b, h, i, *_)` — [`L196`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L196)
- `init()` — [`L347`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L347)
- `init()` — [`L573`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L573)
- `prefetch_first_block()` — [`L360`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L360)
- `prefetch_first_block()` — [`L562`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L562)
- `prefetch_next_block()` — [`L371`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L371)
- `prefetch_next_block()` — [`L586`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L586)
- `prepare_block_sparse_map(mask: BaseAttentionBias, lengths: Tensor, block_size: int, seq_len: int)` — [`L90`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L90) — Computes a full block map num_kv_blocks * num_kv_blocks.
- `sparse_index_map(core_index, b, h, i, kv_block_offset, *_)` — [`L166`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_paged_attention_kernel.py#L166)

