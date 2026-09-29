---
title: 'Module: ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.page_attention._pallas_impl_fwd`/
symbols:
  MultiPageAsyncCopyDescriptor._make_async_copy: MultiPageAsyncCopyDescriptor#_make_async_copy().
  paged_flash_attention_kernel.flash_attention: paged_flash_attention_kernel().flash_attention().
  MultiPageAsyncCopyDescriptor._async_copies: MultiPageAsyncCopyDescriptor#_async_copies.
  MultiPageAsyncCopyDescriptor.wait_and_get_loaded: MultiPageAsyncCopyDescriptor#wait_and_get_loaded().
  MultiPageAsyncCopyDescriptor.start: MultiPageAsyncCopyDescriptor#start().
  paged_flash_attention_kernel.create_kv_async_copy_descriptors: paged_flash_attention_kernel().create_kv_async_copy_descriptors().
  paged_flash_attention_kernel.flash_attention.prefetch_first_block: paged_flash_attention_kernel().flash_attention().prefetch_first_block().
  paged_flash_attention_kernel.flash_attention.prefetch_next_block: paged_flash_attention_kernel().flash_attention().prefetch_next_block().
  MultiPageAsyncCopyDescriptor._pages_hbm_ref: MultiPageAsyncCopyDescriptor#_pages_hbm_ref.
  paged_flash_attention_kernel_inline_seq_dim: paged_flash_attention_kernel_inline_seq_dim().
  paged_flash_attention_kernel.compute_block_indices: paged_flash_attention_kernel().compute_block_indices().
  paged_flash_attention_kernel.compute_block_indices.advance_b: paged_flash_attention_kernel().compute_block_indices().advance_b().
  paged_flash_attention_kernel.compute_block_indices.advance_h: paged_flash_attention_kernel().compute_block_indices().advance_h().
  paged_flash_attention_kernel_inline_seq_dim.body: paged_flash_attention_kernel_inline_seq_dim().body().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  MultiPageAsyncCopyDescriptor._vmem_buffer: MultiPageAsyncCopyDescriptor#_vmem_buffer.
  paged_flash_attention_kernel: paged_flash_attention_kernel().
  ref_paged_attention: ref_paged_attention().
  MultiPageAsyncCopyDescriptor: MultiPageAsyncCopyDescriptor#
  MultiPageAsyncCopyDescriptor._num_pages_to_load: MultiPageAsyncCopyDescriptor#_num_pages_to_load.
  MultiPageAsyncCopyDescriptor._sem: MultiPageAsyncCopyDescriptor#_sem.
  MultiPageAsyncCopyDescriptor._page_indices: MultiPageAsyncCopyDescriptor#_page_indices.
  MultiPageAsyncCopyDescriptor._page_indices_start_offset: MultiPageAsyncCopyDescriptor#_page_indices_start_offset.
  paged_flash_attention_kernel.compute_block_indices.advance_b.advance_to_next_non_zero_length: paged_flash_attention_kernel().compute_block_indices().advance_b().advance_to_next_non_zero_length().
  MultiPageAsyncCopyDescriptor.__init__: MultiPageAsyncCopyDescriptor#__init__().
  paged_flash_attention_kernel.flash_attention.init: paged_flash_attention_kernel().flash_attention().init().
---
# Module: [`ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py)

## Classes
### `MultiPageAsyncCopyDescriptor`
- def: [`ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py:138`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py#L138)
- doc: Descriptor for async DMA copies of one compute block of K/V pages from HBM to VMEM.
- signature: `class MultiPageAsyncCopyDescriptor:`
- members:
  - `_make_async_copy(self, i)` — [`L180`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py#L180) — Build an ``AsyncCopy`` descriptor for the i-th page in this block.
  - `start(self)` — [`L185`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py#L185) — Starts the async copies.
  - `wait_and_get_loaded(self)` — [`L190`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py#L190) — Wait async copies and gets the loaded buffer as a jax.Array.
- protocol/private: `__init__`[`L158`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py#L158), `_async_copies`[`L178`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py#L178), `_num_pages_to_load`[`L169`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py#L169), `_page_indices`[`L176`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py#L176), `_page_indices_start_offset`[`L177`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py#L177), `_pages_hbm_ref`[`L171`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py#L171), `_sem`[`L175`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py#L175), `_vmem_buffer`[`L168`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py#L168)
- used by: [`flash_attention`](_pallas_impl_fwd.md#paged_flash_attention_kernel.flash_attention), [`create_kv_async_copy_descriptors`](_pallas_impl_fwd.md#paged_flash_attention_kernel.create_kv_async_copy_descriptors), [`prefetch_first_block`](_pallas_impl_fwd.md#paged_flash_attention_kernel.flash_attention.prefetch_first_block), [`prefetch_next_block`](_pallas_impl_fwd.md#paged_flash_attention_kernel.flash_attention.prefetch_next_block)

## Functions
- `advance_b()` — [`L297`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py#L297)
- `advance_h()` — [`L319`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py#L319)
- `advance_to_next_non_zero_length()` — [`L300`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py#L300)
- `body(i, _)` — [`L481`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py#L481)
- `compute_block_indices(b, h, i)` — [`L296`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py#L296)
- `create_kv_async_copy_descriptors(b, h, i, buffer_index)` — [`L325`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py#L325)
- `flash_attention()` — [`L349`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py#L349)
- `init()` — [`L362`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py#L362)
- `paged_flash_attention_kernel(lengths_ref, page_indices_ref, buffer_index_ref, init_flag_ref, q_ref, k_pages_hbm_ref, v_pages_hbm_ref, o_ref, m_ref, l_ref, k_vmem_buffer, v_vmem_buffer, k_sems, v_sems, *, batch_size: int, pages_per_compute_block: int, pages_per_sequence: int, mask_value: float, attn_logits_soft_cap: float | None, megacore_mode: str | None, sliding_window: int | None = None, program_ids=())` — [`L199`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py#L199) — Pallas kernel body for one tile of paged flash attention.
- `paged_flash_attention_kernel_inline_seq_dim(lengths_ref, page_indices_ref, buffer_index_ref, init_flag_ref, q_ref, k_pages_hbm_ref, v_pages_hbm_ref, o_ref, m_ref, l_ref, k_vmem_buffer, v_vmem_buffer, k_sems, v_sems, *, batch_size: int, pages_per_compute_block: int, pages_per_sequence: int, mask_value: float, attn_logits_soft_cap: float | None, megacore_mode: str | None, sliding_window: int | None = None)` — [`L411`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py#L411) — Variant of ``paged_flash_attention_kernel`` that loops over the sequence dimension inline.
- `prefetch_first_block()` — [`L356`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py#L356)
- `prefetch_next_block()` — [`L368`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py#L368)
- `ref_paged_attention(query: jax.Array, key_cache: jax.Array, value_cache: jax.Array, context_lens: jax.Array, block_tables: jax.Array, *, mask_value: float = DEFAULT_MASK_VALUE, attn_logits_soft_cap: float | None = None, sliding_window: int | None = None)` — [`L73`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py#L73) — Reference implementation of paged attention for testing.

## Module values
- `DEFAULT_MASK_VALUE` — [`L70`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/page_attention/_pallas_impl_fwd.py#L70)

