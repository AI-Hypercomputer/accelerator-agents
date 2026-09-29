---
title: 'Module: tpu_inference/kernels/experimental/deepseek_v4/mla.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/experimental/deepseek_v4/mla.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.experimental.deepseek_v4.mla`/
symbols:
  mla_ragged_paged_attention: mla_ragged_paged_attention().
  _mla_ragged_paged_attention_kernel.process.compute_with_bq: _mla_ragged_paged_attention_kernel().process().compute_with_bq().
  _mla_ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv: _mla_ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().
  mla_ragged_paged_attention.run_mla_kernel: mla_ragged_paged_attention().run_mla_kernel().
  _mla_ragged_paged_attention_kernel._async_copy: _mla_ragged_paged_attention_kernel()._async_copy().
  align_to: align_to().
  MlaCase.symbol: MlaCase#symbol().
  _mla_ragged_paged_attention_kernel.prologue: _mla_ragged_paged_attention_kernel().prologue().
  _mla_ragged_paged_attention_kernel: _mla_ragged_paged_attention_kernel().
  MlaCase: MlaCase#
  get_dtype_packing: get_dtype_packing().
  _mla_ragged_paged_attention_kernel.process.compute_with_bq.prefetch_next_bq: _mla_ragged_paged_attention_kernel().process().compute_with_bq().prefetch_next_bq().
  _mla_ragged_paged_attention_kernel._fetch_bkv: _mla_ragged_paged_attention_kernel()._fetch_bkv().
  _mla_ragged_paged_attention_kernel.process: _mla_ragged_paged_attention_kernel().process().
  prepare_q_inputs: prepare_q_inputs().
  prepare_swa_inputs: prepare_swa_inputs().
  _mla_ragged_paged_attention_kernel.broadcast_minor: _mla_ragged_paged_attention_kernel().broadcast_minor().
  cdiv: cdiv().
  get_kv_cache_shape: get_kv_cache_shape().
  _mla_ragged_paged_attention_kernel._fetch_bq: _mla_ragged_paged_attention_kernel()._fetch_bq().
  _mla_ragged_paged_attention_kernel._send_bo: _mla_ragged_paged_attention_kernel()._send_bo().
  _mla_ragged_paged_attention_kernel._fetch_swa: _mla_ragged_paged_attention_kernel()._fetch_swa().
  _mla_ragged_paged_attention_kernel._fetch_topk_indices: _mla_ragged_paged_attention_kernel()._fetch_topk_indices().
  _mla_ragged_paged_attention_kernel.start_fetch_bkv: _mla_ragged_paged_attention_kernel().start_fetch_bkv().
  _mla_ragged_paged_attention_kernel.start_fetch_bq: _mla_ragged_paged_attention_kernel().start_fetch_bq().
  _mla_ragged_paged_attention_kernel.start_fetch_swa: _mla_ragged_paged_attention_kernel().start_fetch_swa().
  _mla_ragged_paged_attention_kernel.start_fetch_topk_indices: _mla_ragged_paged_attention_kernel().start_fetch_topk_indices().
  _mla_ragged_paged_attention_kernel.wait_send_bo: _mla_ragged_paged_attention_kernel().wait_send_bo().
  _mla_ragged_paged_attention_kernel.flash_attention: _mla_ragged_paged_attention_kernel().flash_attention().
  _mla_ragged_paged_attention_kernel.wait_fetch_bkv: _mla_ragged_paged_attention_kernel().wait_fetch_bkv().
  _mla_ragged_paged_attention_kernel.wait_fetch_bq: _mla_ragged_paged_attention_kernel().wait_fetch_bq().
  _mla_ragged_paged_attention_kernel.wait_fetch_swa: _mla_ragged_paged_attention_kernel().wait_fetch_swa().
  _mla_ragged_paged_attention_kernel.wait_fetch_topk_indices: _mla_ragged_paged_attention_kernel().wait_fetch_topk_indices().
  _mla_ragged_paged_attention_kernel.start_send_bo: _mla_ragged_paged_attention_kernel().start_send_bo().
  MlaCase.MIXED: MlaCase#MIXED.
  _mla_ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv.prefetch_next_bkv: _mla_ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().prefetch_next_bkv().
  _mla_ragged_paged_attention_kernel.epilogue: _mla_ragged_paged_attention_kernel().epilogue().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  MlaCase.DECODE: MlaCase#DECODE.
  MlaCase.PREFILL: MlaCase#PREFILL.
  DEFAULT_VMEM_LIMIT_BYTES: DEFAULT_VMEM_LIMIT_BYTES.
  get_dtype_bitwidth: get_dtype_bitwidth().
  _mla_ragged_paged_attention_kernel.load_topk_indices: _mla_ragged_paged_attention_kernel().load_topk_indices().
  _mla_ragged_paged_attention_kernel.load_bq: _mla_ragged_paged_attention_kernel().load_bq().
  _mla_ragged_paged_attention_kernel.load_bkv: _mla_ragged_paged_attention_kernel().load_bkv().
  _mla_ragged_paged_attention_kernel.process.get_next_bq_ids: _mla_ragged_paged_attention_kernel().process().get_next_bq_ids().
  _mla_ragged_paged_attention_kernel.process.get_next_bkv_ids: _mla_ragged_paged_attention_kernel().process().get_next_bkv_ids().
  prepare_outputs: prepare_outputs().
---
# Module: [`tpu_inference/kernels/experimental/deepseek_v4/mla.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py)

## Classes
### `MlaCase`  ·  implements/extends Enum
- def: [`tpu_inference/kernels/experimental/deepseek_v4/mla.py:63`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L63)
- doc: Represents the different cases for MLA.
- signature: `class MlaCase(Enum):`
- members:
  - `symbol(self)` — [`L76`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L76)
  - `DECODE` — [`L71`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L71)
  - `MIXED` — [`L73`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L73)
  - `PREFILL` — [`L72`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L72)
- used by: [`mla_ragged_paged_attention`](mla.md#mla_ragged_paged_attention), [`run_mla_kernel`](mla.md#mla_ragged_paged_attention.run_mla_kernel)

## Functions
- `_async_copy(src, dst, sem, wait)` — [`L209`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L209)
- `_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False)` — [`L216`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L216)
- `_fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False)` — [`L278`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L278)
- `_fetch_swa(seq_idx, bq_idx, bq_sem_idx, *, wait=False)` — [`L307`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L307)
- `_fetch_topk_indices(seq_idx, bq_idx, bq_sem_idx, *, wait=False)` — [`L359`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L359)
- `_mla_ragged_paged_attention_kernel(kv_lens_ref, kv_lens_to_attend_ref, page_indices_ref, cu_q_lens_ref, start_end_seq_idx_ref, sem_ids_ref, bo_ids_ref, attention_sinks_ref, q_hbm_ref, cache_kv_hbm_ref, swa_accumution_hbm_ref, swa_l_hbm_ref, swa_m_hbm_ref, topk_indices_ref, o_hbm_ref, bkv_x2_ref, bq_x2_ref, bo_x2_ref, bl_x2_ref, bm_x2_ref, swa_acc_x2_ref, topk_indices_x2_ref, sems, l_ref, m_ref, acc_ref, *, static_q_len: int, sm_scale: float, mask_value: float = DEFAULT_MASK_VALUE, bkv_p, bq_sz)` — [`L84`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L84)
- `_send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False)` — [`L293`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L293)
- `align_to(x, a)` — [`L35`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L35)
- `broadcast_minor(src, shape)` — [`L460`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L460)
- `cdiv(a, b)` — [`L30`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L30)
- `compute_with_bkv(bkv_idx, carry)` — [`L526`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L526)
- `compute_with_bq(bq_idx, _)` — [`L501`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L501)
- `epilogue()` — [`L618`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L618)
- `flash_attention(q, kv, *, bq_idx, bkv_idx, kv_lens_to_attend_segment, bq_topk_indices)` — [`L153`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L153)
- `get_dtype_bitwidth(dtype)` — [`L39`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L39)
- `get_dtype_packing(dtype)` — [`L43`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L43)
- `get_kv_cache_shape(total_num_pages, page_size, kv_dim, kv_dtype)` — [`L48`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L48)
- `get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx)` — [`L490`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L490)
- `get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx)` — [`L482`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L482)
- `load_bkv(bkv_sem_idx, bkv_idx)` — [`L428`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L428)
- `load_bq(bq_sem_idx)` — [`L419`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L419)
- `load_topk_indices(bq_sem_idx)` — [`L403`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L403)
- `mla_ragged_paged_attention(q: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, kv_lens_to_attend: jax.Array | None, topk_indices: jax.Array | None, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, attention_sinks: jax.Array, swa_accumution: jax.Array, swa_l: jax.Array, swa_m: jax.Array, *, sm_scale: float = 1, mask_value: float | None = DEFAULT_MASK_VALUE, chunk_prefill_size: int | None = None, num_kv_pages_per_block: tuple[int, int, int] | int | None = None, num_queries_per_block: tuple[int, int, int] | int | None = None, vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES)` — [`L708`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L708) — MLA Ragged paged attention that supports mixed prefill and decode.
- `prefetch_next_bkv()` — [`L536`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L536)
- `prefetch_next_bq()` — [`L519`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L519)
- `prepare_outputs(out, actual_num_q_heads: int, actual_head_dim: int)` — [`L682`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L682)
- `prepare_q_inputs(q: jax.Array)` — [`L625`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L625)
- `prepare_swa_inputs(swa_accumution: jax.Array, swa_l: jax.Array, swa_m: jax.Array)` — [`L644`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L644)
- `process()` — [`L471`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L471)
- `prologue()` — [`L609`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L609)
- `run_mla_kernel(q: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, kv_lens_to_attend: jax.Array | None, topk_indices: jax.Array | None, page_indices: jax.Array, cu_q_lens: jax.Array, attention_sinks: jax.Array, swa_accumution: jax.Array, swa_l: jax.Array, swa_m: jax.Array, start_seq_idx: jax.Array, end_seq_idx: jax.Array, static_q_len: int | None, num_kv_pages_per_block: int, num_queries_per_block: int, case: MlaCase = MlaCase.MIXED)` — [`L814`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L814)
- `start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)` — [`L379`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L379)
- `start_fetch_bq(seq_idx, bq_idx, bq_sem_idx)` — [`L385`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L385)
- `start_fetch_swa(seq_idx, bq_idx, bq_sem_idx)` — [`L391`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L391)
- `start_fetch_topk_indices(seq_idx, bq_idx, bq_sem_idx)` — [`L397`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L397)
- `start_send_bo(seq_idx, bo_idx, bo_sem_idx)` — [`L406`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L406)
- `wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)` — [`L382`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L382)
- `wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)` — [`L388`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L388)
- `wait_fetch_swa(seq_idx, bq_idx, bq_sem_idx)` — [`L394`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L394)
- `wait_fetch_topk_indices(seq_idx, bq_idx, bq_sem_idx)` — [`L400`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L400)
- `wait_send_bo(bo_sem_idx)` — [`L411`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L411)

## Module values
- `DEFAULT_MASK_VALUE` — [`L25`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L25)
- `DEFAULT_VMEM_LIMIT_BYTES` — [`L27`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/mla.py#L27)

