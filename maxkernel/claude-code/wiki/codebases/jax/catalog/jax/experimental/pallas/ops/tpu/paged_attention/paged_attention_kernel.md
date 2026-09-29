---
title: 'Module: jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.tpu.paged_attention.paged_attention_kernel`/
symbols:
  paged_attention: paged_attention().
  paged_flash_attention_kernel.flash_attention: paged_flash_attention_kernel().flash_attention().
  MultiPageAsyncCopyDescriptor.wait_and_get_loaded: MultiPageAsyncCopyDescriptor#wait_and_get_loaded().
  paged_flash_attention_kernel_inline_seq_dim: paged_flash_attention_kernel_inline_seq_dim().
  MultiPageAsyncCopyDescriptor._async_copies: MultiPageAsyncCopyDescriptor#_async_copies.
  MultiPageAsyncCopyDescriptor._make_async_copy: MultiPageAsyncCopyDescriptor#_make_async_copy().
  MultiPageAsyncCopyDescriptor._make_scales_async_copy: MultiPageAsyncCopyDescriptor#_make_scales_async_copy().
  paged_flash_attention_kernel.compute_block_indices.advance_b: paged_flash_attention_kernel().compute_block_indices().advance_b().
  paged_flash_attention_kernel.flash_attention.init: paged_flash_attention_kernel().flash_attention().init().
  paged_flash_attention_kernel.flash_attention.prefetch_next_block: paged_flash_attention_kernel().flash_attention().prefetch_next_block().
  MultiPageAsyncCopyDescriptor._scales_pages_hbm_ref: MultiPageAsyncCopyDescriptor#_scales_pages_hbm_ref.
  MultiPageAsyncCopyDescriptor.start: MultiPageAsyncCopyDescriptor#start().
  paged_flash_attention_kernel.compute_block_indices.advance_b.advance_to_next_non_zero_length: paged_flash_attention_kernel().compute_block_indices().advance_b().advance_to_next_non_zero_length().
  paged_flash_attention_kernel.flash_attention.prefetch_first_block: paged_flash_attention_kernel().flash_attention().prefetch_first_block().
  paged_flash_attention_kernel: paged_flash_attention_kernel().
  MultiPageAsyncCopyDescriptor._maybe_dequantize: MultiPageAsyncCopyDescriptor#_maybe_dequantize().
  paged_flash_attention_kernel.compute_block_indices: paged_flash_attention_kernel().compute_block_indices().
  paged_flash_attention_kernel.compute_block_indices.advance_h: paged_flash_attention_kernel().compute_block_indices().advance_h().
  paged_flash_attention_kernel.create_kv_async_copy_descriptors: paged_flash_attention_kernel().create_kv_async_copy_descriptors().
  MultiPageAsyncCopyDescriptor._scales_vmem_buffer: MultiPageAsyncCopyDescriptor#_scales_vmem_buffer.
  paged_flash_attention_kernel_inline_seq_dim.body: paged_flash_attention_kernel_inline_seq_dim().body().
  MultiPageAsyncCopyDescriptor._vmem_buffer: MultiPageAsyncCopyDescriptor#_vmem_buffer.
  MultiPageAsyncCopyDescriptor: MultiPageAsyncCopyDescriptor#
  MultiPageAsyncCopyDescriptor._num_pages_to_load: MultiPageAsyncCopyDescriptor#_num_pages_to_load.
  MultiPageAsyncCopyDescriptor._pages_hbm_ref: MultiPageAsyncCopyDescriptor#_pages_hbm_ref.
  MultiPageAsyncCopyDescriptor._sem: MultiPageAsyncCopyDescriptor#_sem.
  MultiPageAsyncCopyDescriptor._page_indices: MultiPageAsyncCopyDescriptor#_page_indices.
  MultiPageAsyncCopyDescriptor._page_indices_start_offset: MultiPageAsyncCopyDescriptor#_page_indices_start_offset.
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  MultiPageAsyncCopyDescriptor.__init__: MultiPageAsyncCopyDescriptor#__init__().
---
# Module: [`jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py)

## Classes
### `MultiPageAsyncCopyDescriptor`
- def: [`jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py:33`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L33)
- doc: Descriptor for async copy of multiple K/V pages from HBM.
- signature: `class MultiPageAsyncCopyDescriptor:`
- members:
  - `start(self)` — [`L89`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L89) — Starts the async copies.
  - `wait_and_get_loaded(self)` — [`L99`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L99) — Wait async copies and gets the loaded buffer as a jax.Array.
- protocol/private: `__init__`[`L36`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L36), `_async_copies`[`L63`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L63), `_make_async_copy`[`L75`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L75), `_make_scales_async_copy`[`L81`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L81), `_maybe_dequantize`[`L94`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L94), `_num_pages_to_load`[`L50`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L50), `_page_indices`[`L61`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L61), `_page_indices_start_offset`[`L62`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L62), `_pages_hbm_ref`[`L52`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L52), `_scales_pages_hbm_ref`[`L54`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L54), `_scales_vmem_buffer`[`L49`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L49), `_sem`[`L60`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L60), `_vmem_buffer`[`L48`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L48)
- uses (calls/refs, reference-scoped): [`Array`](../../../../../_src/basearray.md#Array), [`float32`](../../../../../numpy/__init__.pyi.md#float32.float32), [`reshape`](../../../../../_src/basearray.md#Array.reshape), [`bfloat16`](../../../../../numpy/__init__.pyi.md#bfloat16.bfloat16), [`start`](../../../../../_src/pallas/mosaic/primitives.md#AsyncCopyDescriptor.start), [`make_async_copy`](../../../../../_src/pallas/mosaic/primitives.md#make_async_copy), [`wait`](../../../../../_src/pallas/mosaic/primitives.md#AsyncCopyDescriptor.wait), [`from_int8`](quantization_utils.md#from_int8)
- used by: [`flash_attention`](paged_attention_kernel.md#paged_flash_attention_kernel.flash_attention), [`prefetch_next_block`](paged_attention_kernel.md#paged_flash_attention_kernel.flash_attention.prefetch_next_block), [`prefetch_first_block`](paged_attention_kernel.md#paged_flash_attention_kernel.flash_attention.prefetch_first_block), [`create_kv_async_copy_descriptors`](paged_attention_kernel.md#paged_flash_attention_kernel.create_kv_async_copy_descriptors)

## Functions
- `advance_b()` — [`L166`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L166)
- `advance_h()` — [`L190`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L190)
- `advance_to_next_non_zero_length()` — [`L169`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L169)
- `body(i, _)` — [`L325`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L325)
- `compute_block_indices(b, h, i)` — [`L164`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L164)
- `create_kv_async_copy_descriptors(b, h, i, buffer_index)` — [`L196`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L196)
- `flash_attention()` — [`L228`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L228)
- `init()` — [`L243`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L243)
- `paged_attention(q: jax.Array, k_pages: jax.Array | quantization_utils.QuantizedTensor, v_pages: jax.Array | quantization_utils.QuantizedTensor, lengths: jax.Array, page_indices: jax.Array, *, mask_value: float = DEFAULT_MASK_VALUE, attn_logits_soft_cap: float | None = None, pages_per_compute_block: int, megacore_mode: str | None = None, inline_seq_dim: bool = True)` — [`L376`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L376) — Paged grouped query attention.
- `paged_flash_attention_kernel(lengths_ref, page_indices_ref, buffer_index_ref, init_flag_ref, q_ref, k_pages_hbm_ref, k_scales_pages_hbm_ref, v_pages_hbm_ref, v_scales_pages_hbm_ref, o_ref, m_ref, l_ref, k_vmem_buffer, k_scales_vmem_buffer, v_vmem_buffer, v_scales_vmem_buffer, k_sems, v_sems, *, batch_size: int, pages_per_compute_block: int, pages_per_sequence: int, mask_value: float, attn_logits_soft_cap: float | None, megacore_mode: str | None, program_ids=())` — [`L113`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L113) — Pallas kernel for paged attention.
- `paged_flash_attention_kernel_inline_seq_dim(lengths_ref, page_indices_ref, buffer_index_ref, init_flag_ref, q_ref, k_pages_hbm_ref, k_scales_pages_hbm_ref, v_pages_hbm_ref, v_scales_pages_hbm_ref, o_ref, m_ref, l_ref, k_vmem_buffer, k_scales_vmem_buffer, v_vmem_buffer, v_scales_vmem_buffer, k_sems, v_sems, *, batch_size: int, pages_per_compute_block: int, pages_per_sequence: int, mask_value: float, attn_logits_soft_cap: float | None, megacore_mode: str | None)` — [`L290`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L290)
- `prefetch_first_block()` — [`L235`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L235)
- `prefetch_next_block()` — [`L249`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L249)

## Module values
- `DEFAULT_MASK_VALUE` — [`L30`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py#L30)

