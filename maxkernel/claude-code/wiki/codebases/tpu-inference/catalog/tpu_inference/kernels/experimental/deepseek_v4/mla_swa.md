---
title: 'Module: tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.experimental.deepseek_v4.mla_swa`/
symbols:
  mla_sliding_window_ragged_paged_attention: mla_sliding_window_ragged_paged_attention().
  _mla_sliding_window_ragged_paged_attention_kernel.process.compute_with_bq: _mla_sliding_window_ragged_paged_attention_kernel().process().compute_with_bq().
  mla_sliding_window_ragged_paged_attention.run_mla_kernel: mla_sliding_window_ragged_paged_attention().run_mla_kernel().
  _mla_sliding_window_ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv: _mla_sliding_window_ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().
  cdiv: cdiv().
  MlaCase.symbol: MlaCase#symbol().
  _mla_sliding_window_ragged_paged_attention_kernel.prologue: _mla_sliding_window_ragged_paged_attention_kernel().prologue().
  _mla_sliding_window_ragged_paged_attention_kernel._fetch_bkv: _mla_sliding_window_ragged_paged_attention_kernel()._fetch_bkv().
  _mla_sliding_window_ragged_paged_attention_kernel._update_kv_cache: _mla_sliding_window_ragged_paged_attention_kernel()._update_kv_cache().
  _mla_sliding_window_ragged_paged_attention_kernel.wait_send_bo: _mla_sliding_window_ragged_paged_attention_kernel().wait_send_bo().
  _mla_sliding_window_ragged_paged_attention_kernel._async_copy: _mla_sliding_window_ragged_paged_attention_kernel()._async_copy().
  align_to: align_to().
  _mla_sliding_window_ragged_paged_attention_kernel: _mla_sliding_window_ragged_paged_attention_kernel().
  _mla_sliding_window_ragged_paged_attention_kernel._pack_new_kv: _mla_sliding_window_ragged_paged_attention_kernel()._pack_new_kv().
  _mla_sliding_window_ragged_paged_attention_kernel.start_send_bo: _mla_sliding_window_ragged_paged_attention_kernel().start_send_bo().
  MlaCase: MlaCase#
  get_dtype_packing: get_dtype_packing().
  _mla_sliding_window_ragged_paged_attention_kernel.flash_attention: _mla_sliding_window_ragged_paged_attention_kernel().flash_attention().
  _mla_sliding_window_ragged_paged_attention_kernel.process: _mla_sliding_window_ragged_paged_attention_kernel().process().
  _mla_sliding_window_ragged_paged_attention_kernel.process.get_next_bkv_ids: _mla_sliding_window_ragged_paged_attention_kernel().process().get_next_bkv_ids().
  prepare_q_inputs: prepare_q_inputs().
  prepare_kv_inputs: prepare_kv_inputs().
  _mla_sliding_window_ragged_paged_attention_kernel.broadcast_minor: _mla_sliding_window_ragged_paged_attention_kernel().broadcast_minor().
  _mla_sliding_window_ragged_paged_attention_kernel.epilogue: _mla_sliding_window_ragged_paged_attention_kernel().epilogue().
  _mla_sliding_window_ragged_paged_attention_kernel._fetch_bq: _mla_sliding_window_ragged_paged_attention_kernel()._fetch_bq().
  _mla_sliding_window_ragged_paged_attention_kernel._send_bo: _mla_sliding_window_ragged_paged_attention_kernel()._send_bo().
  _mla_sliding_window_ragged_paged_attention_kernel._send_l: _mla_sliding_window_ragged_paged_attention_kernel()._send_l().
  _mla_sliding_window_ragged_paged_attention_kernel._send_m: _mla_sliding_window_ragged_paged_attention_kernel()._send_m().
  _mla_sliding_window_ragged_paged_attention_kernel.start_fetch_bkv: _mla_sliding_window_ragged_paged_attention_kernel().start_fetch_bkv().
  _mla_sliding_window_ragged_paged_attention_kernel.start_fetch_bq: _mla_sliding_window_ragged_paged_attention_kernel().start_fetch_bq().
  _mla_sliding_window_ragged_paged_attention_kernel.wait_update_kv_cache: _mla_sliding_window_ragged_paged_attention_kernel().wait_update_kv_cache().
  _mla_sliding_window_ragged_paged_attention_kernel._update_kv_cache.loop_body: _mla_sliding_window_ragged_paged_attention_kernel()._update_kv_cache().loop_body().
  _mla_sliding_window_ragged_paged_attention_kernel.wait_fetch_bkv: _mla_sliding_window_ragged_paged_attention_kernel().wait_fetch_bkv().
  _mla_sliding_window_ragged_paged_attention_kernel.wait_fetch_bq: _mla_sliding_window_ragged_paged_attention_kernel().wait_fetch_bq().
  _mla_sliding_window_ragged_paged_attention_kernel.start_update_kv_cache: _mla_sliding_window_ragged_paged_attention_kernel().start_update_kv_cache().
  MlaCase.MIXED: MlaCase#MIXED.
  _mla_sliding_window_ragged_paged_attention_kernel.flash_attention.load_with_init: _mla_sliding_window_ragged_paged_attention_kernel().flash_attention().load_with_init().
  _mla_sliding_window_ragged_paged_attention_kernel.process.compute_with_bq.prefetch_next_bq: _mla_sliding_window_ragged_paged_attention_kernel().process().compute_with_bq().prefetch_next_bq().
  _mla_sliding_window_ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv.prefetch_next_bkv: _mla_sliding_window_ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().prefetch_next_bkv().
  _mla_sliding_window_ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv.pack_new_kv: _mla_sliding_window_ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().pack_new_kv().
  _mla_sliding_window_ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv.update_cur_bkv_to_cache: _mla_sliding_window_ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().update_cur_bkv_to_cache().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  get_dtype_bitwidth: get_dtype_bitwidth().
  MlaCase.DECODE: MlaCase#DECODE.
  MlaCase.PREFILL: MlaCase#PREFILL.
  _mla_sliding_window_ragged_paged_attention_kernel._get_kv_len: _mla_sliding_window_ragged_paged_attention_kernel()._get_kv_len().
  _mla_sliding_window_ragged_paged_attention_kernel._get_q_len: _mla_sliding_window_ragged_paged_attention_kernel()._get_q_len().
  DEFAULT_VMEM_LIMIT_BYTES: DEFAULT_VMEM_LIMIT_BYTES.
  _mla_sliding_window_ragged_paged_attention_kernel._pack_new_kv.merge_loop_body: _mla_sliding_window_ragged_paged_attention_kernel()._pack_new_kv().merge_loop_body().
  _mla_sliding_window_ragged_paged_attention_kernel.load_bq: _mla_sliding_window_ragged_paged_attention_kernel().load_bq().
  _mla_sliding_window_ragged_paged_attention_kernel.load_bkv: _mla_sliding_window_ragged_paged_attention_kernel().load_bkv().
  _mla_sliding_window_ragged_paged_attention_kernel.process.get_next_bq_ids: _mla_sliding_window_ragged_paged_attention_kernel().process().get_next_bq_ids().
  quantize_kv_inputs: quantize_kv_inputs().
  prepare_outputs: prepare_outputs().
---
# Module: [`tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py)

## Classes
### `MlaCase`  ·  implements/extends Enum
- def: [`tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py:48`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L48)
- doc: Represents the different cases for MLA.
- signature: `class MlaCase(Enum):`
- members:
  - `symbol(self)` — [`L61`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L61)
  - `DECODE` — [`L56`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L56)
  - `MIXED` — [`L58`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L58)
  - `PREFILL` — [`L57`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L57)
- used by: [`mla_sliding_window_ragged_paged_attention`](mla_swa.md#mla_sliding_window_ragged_paged_attention), [`run_mla_kernel`](mla_swa.md#mla_sliding_window_ragged_paged_attention.run_mla_kernel)

## Functions
- `_async_copy(src, dst, sem, wait)` — [`L196`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L196)
- `_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False)` — [`L213`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L213)
- `_fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False)` — [`L516`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L516)
- `_get_kv_len(seq_idx)` — [`L203`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L203)
- `_get_q_len(seq_idx)` — [`L206`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L206)
- `_mla_sliding_window_ragged_paged_attention_kernel(kv_lens_ref, page_indices_ref, cu_q_lens_ref, start_end_seq_idx_ref, sem_ids_ref, bo_ids_ref, bkv_update_ids_ref, attention_sinks_ref, q_hbm_ref, new_kv_hbm_ref, cache_kv_hbm_ref, in_output_hbm_ref, in_l_hbm_ref, in_m_hbm_ref, o_hbm_ref, updated_cache_kv_hbm_ref, l_hbm_ref, m_hbm_ref, bkv_x2_ref, bq_x2_ref, bo_x2_ref, bl_x2_ref, bm_x2_ref, sems, l_ref, m_ref, acc_ref, *, static_q_len: int, sm_scale: float, sliding_window: int, logical_page_size: int, unnormalized_output: bool, mask_value: float = DEFAULT_MASK_VALUE, bkv_p, bq_sz)` — [`L69`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L69)
- `_pack_new_kv(bkv_sem_idx, offset, update_sz)` — [`L325`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L325) — Packs newly computed KVs into the correct sub-word alignment in VMEM.
- `_send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False)` — [`L531`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L531)
- `_send_l(seq_idx, bo_idx, bo_sem_idx, *, wait=False)` — [`L545`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L545)
- `_send_m(seq_idx, bo_idx, bo_sem_idx, *, wait=False)` — [`L559`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L559)
- `_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, *, wait=False)` — [`L441`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L441)
- `align_to(x, a)` — [`L35`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L35)
- `broadcast_minor(src, shape)` — [`L663`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L663)
- `cdiv(a, b)` — [`L30`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L30)
- `compute_with_bkv(bkv_idx, carry)` — [`L726`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L726)
- `compute_with_bq(bq_idx, _)` — [`L715`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L715)
- `epilogue()` — [`L846`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L846)
- `flash_attention(q, kv, *, bq_idx, bkv_idx, is_first_bkv)` — [`L147`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L147)
- `get_dtype_bitwidth(dtype)` — [`L39`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L39)
- `get_dtype_packing(dtype)` — [`L43`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L43)
- `get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx)` — [`L696`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L696)
- `get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx)` — [`L688`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L688)
- `load_bkv(bkv_sem_idx, bkv_idx)` — [`L631`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L631)
- `load_bq(bq_sem_idx)` — [`L622`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L622)
- `load_with_init(ref, init_val)` — [`L164`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L164)
- `loop_body(i, states)` — [`L471`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L471)
- `merge_loop_body(i, vals)` — [`L378`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L378)
- `mla_sliding_window_ragged_paged_attention(q: jax.Array, new_kv: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, attention_sinks: jax.Array, *, sm_scale: float = 1, sliding_window: int, logical_page_size: int, mask_value: float | None = DEFAULT_MASK_VALUE, chunk_prefill_size: int | None = None, num_kv_pages_per_block: tuple[int, int, int] | int | None = None, num_queries_per_block: tuple[int, int, int] | int | None = None, vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES, unnormalized_output: bool = False)` — [`L946`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L946) — MLA Ragged paged attention that supports mixed prefill and decode.
- `pack_new_kv()` — [`L748`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L748)
- `prefetch_next_bkv()` — [`L736`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L736)
- `prefetch_next_bq()` — [`L722`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L722)
- `prepare_kv_inputs(kv: jax.Array)` — [`L873`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L873)
- `prepare_outputs(out, actual_num_q_heads: int, actual_head_dim: int)` — [`L922`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L922)
- `prepare_q_inputs(q: jax.Array)` — [`L854`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L854)
- `process()` — [`L674`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L674)
- `prologue()` — [`L831`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L831)
- `quantize_kv_inputs(kv: jax.Array)` — [`L889`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L889)
- `run_mla_kernel(q: jax.Array, new_kv: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, start_seq_idx: jax.Array, end_seq_idx: jax.Array, in_output: jax.Array, in_l: jax.Array, in_m: jax.Array, attention_sinks: jax.Array, static_q_len: int | None, unnormalized_output: bool, num_kv_pages_per_block: int, num_queries_per_block: int, case: MlaCase = MlaCase.MIXED)` — [`L1053`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L1053)
- `start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)` — [`L573`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L573)
- `start_fetch_bq(seq_idx, bq_idx, bq_sem_idx)` — [`L579`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L579)
- `start_send_bo(seq_idx, bo_idx, bo_sem_idx)` — [`L585`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L585)
- `start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)` — [`L602`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L602)
- `update_cur_bkv_to_cache()` — [`L754`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L754)
- `wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)` — [`L576`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L576)
- `wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)` — [`L582`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L582)
- `wait_send_bo(bo_sem_idx)` — [`L592`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L592)
- `wait_update_kv_cache(bkv_sem_idx)` — [`L608`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L608)

## Module values
- `DEFAULT_MASK_VALUE` — [`L25`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L25)
- `DEFAULT_VMEM_LIMIT_BYTES` — [`L27`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla_swa.py#L27)

