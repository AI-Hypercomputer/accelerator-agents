---
title: 'Module: tpu_inference/kernels/ragged_paged_attention/v2/kernel.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/ragged_paged_attention/v2/kernel.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.ragged_paged_attention.v2.kernel`/
symbols:
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  MultiPageAsyncCopyDescriptor: MultiPageAsyncCopyDescriptor#
  MultiPageAsyncCopyDescriptor.__init__: MultiPageAsyncCopyDescriptor#__init__().
  MultiPageAsyncCopyDescriptor.start: MultiPageAsyncCopyDescriptor#start().
  MultiPageAsyncCopyDescriptor.wait: MultiPageAsyncCopyDescriptor#wait().
  ref_ragged_paged_attention: ref_ragged_paged_attention().
  dynamic_validate_inputs: dynamic_validate_inputs().
  static_validate_inputs: static_validate_inputs().
  ragged_paged_attention_kernel: ragged_paged_attention_kernel().
  cdiv: cdiv().
  get_dtype_packing: get_dtype_packing().
  get_min_heads_per_blk: get_min_heads_per_blk().
  ragged_paged_attention: ragged_paged_attention().
---
# Module: [`tpu_inference/kernels/ragged_paged_attention/v2/kernel.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/kernel.py)

## Classes
### `MultiPageAsyncCopyDescriptor`
- def: [`tpu_inference/kernels/ragged_paged_attention/v2/kernel.py:35`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/kernel.py#L35)
- doc: Descriptor for async copy of multiple K/V pages from HBM.
- signature: `class MultiPageAsyncCopyDescriptor:`
- members:
  - `start(self)` — [`L61`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/kernel.py#L61) — Starts the async copies.
  - `wait(self)` — [`L66`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/kernel.py#L66)
- protocol/private: `__init__`[`L38`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/kernel.py#L38)

## Functions
- `cdiv(a, b)` — [`L651`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/kernel.py#L651)
- `dynamic_validate_inputs(q: jax.Array, kv_pages: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, num_seqs: jax.Array, *, sm_scale: float | None=None, sliding_window: int | None=None, soft_cap: float | None=None, mask_value: float | None=None, k_scale: float | None=None, v_scale: float | None=None, num_kv_pages_per_block: int | None=None, num_queries_per_block: int | None=None, vmem_limit_bytes: int | None=None)` — [`L152`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/kernel.py#L152)
- `get_dtype_packing(dtype)` — [`L656`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/kernel.py#L656)
- `get_min_heads_per_blk(num_q_heads, num_combined_kv_heads, q_dtype, kv_dtype)` — [`L661`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/kernel.py#L661)
- `ragged_paged_attention(q: jax.Array, kv_pages: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, num_seqs: jax.Array, *, sm_scale: float=1.0, sliding_window: int | None=None, soft_cap: float | None=None, mask_value: float | None=DEFAULT_MASK_VALUE, k_scale: float | None=None, v_scale: float | None=None, num_kv_pages_per_block: int | None=None, num_queries_per_block: int | None=None, vmem_limit_bytes: int | None=None)` — [`L704`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/kernel.py#L704) — Ragged paged attention that supports mixed prefill and decode.
- `ragged_paged_attention_kernel(kv_lens_ref, page_indices_ref, cu_q_lens_ref, seq_buf_idx_ref, num_seqs_ref, q_ref, kv_pages_hbm_ref, o_ref, kv_bufs, sems, l_ref, m_ref, acc_ref, *, sm_scale: float, sliding_window: int | None=None, soft_cap: float | None=None, mask_value: float | None=DEFAULT_MASK_VALUE, k_scale: float | None=None, v_scale: float | None=None)` — [`L284`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/kernel.py#L284)
- `ref_ragged_paged_attention(queries: jax.Array, kv_pages: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, num_seqs: jax.Array, *, sm_scale: float=1.0, sliding_window: int | None=None, soft_cap: float | None=None, mask_value: float | None=DEFAULT_MASK_VALUE, k_scale: float | None=None, v_scale: float | None=None)` — [`L72`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/kernel.py#L72)
- `static_validate_inputs(q: jax.Array, kv_pages: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, num_seqs: jax.Array, *, sm_scale: float | None=None, sliding_window: int | None=None, soft_cap: float | None=None, mask_value: float | None=None, k_scale: float | None=None, v_scale: float | None=None, num_kv_pages_per_block: int | None=None, num_queries_per_block: int | None=None, vmem_limit_bytes: int | None=None)` — [`L216`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/kernel.py#L216)

## Module values
- `DEFAULT_MASK_VALUE` — [`L32`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v2/kernel.py#L32)

