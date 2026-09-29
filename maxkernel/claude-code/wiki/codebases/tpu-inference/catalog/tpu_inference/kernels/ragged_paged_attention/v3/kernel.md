---
title: 'Module: tpu_inference/kernels/ragged_paged_attention/v3/kernel.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/ragged_paged_attention/v3/kernel.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.ragged_paged_attention.v3.kernel`/
symbols:
  _ragged_paged_attention_kernel_loop.debug_print: _ragged_paged_attention_kernel_loop().debug_print().
  ragged_paged_attention: ragged_paged_attention().
  RpaCase: RpaCase#
  get_default_block_sizes: get_default_block_sizes().
  ragged_paged_attention.run_rpa_kernel: ragged_paged_attention().run_rpa_kernel().
  _ragged_paged_attention_kernel_loop: _ragged_paged_attention_kernel_loop().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq: _ragged_paged_attention_kernel_loop().process().compute_with_bq().
  ref_ragged_paged_attention: ref_ragged_paged_attention().
  _ragged_paged_attention_kernel_loop._update_kv_cache: _ragged_paged_attention_kernel_loop()._update_kv_cache().
  static_validate_inputs: static_validate_inputs().
  RpaCase.symbol: RpaCase#symbol().
  RpaCase.get_range: RpaCase#get_range().
  _ragged_paged_attention_kernel_loop.flash_attention_step1_qk_softmax: _ragged_paged_attention_kernel_loop().flash_attention_step1_qk_softmax().
  get_vmem_estimate_bytes: get_vmem_estimate_bytes().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq.compute_with_bkv: _ragged_paged_attention_kernel_loop().process().compute_with_bq().compute_with_bkv().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq.compute_with_bkv.attention_loop: _ragged_paged_attention_kernel_loop().process().compute_with_bq().compute_with_bkv().attention_loop().
  _ragged_paged_attention_kernel_loop._fetch_bkv: _ragged_paged_attention_kernel_loop()._fetch_bkv().
  prepare_inputs: prepare_inputs().
  get_kv_cache_shape: get_kv_cache_shape().
  RpaCase.DECODE: RpaCase#DECODE.
  RpaCase.MIXED: RpaCase#MIXED.
  _ragged_paged_attention_kernel_loop._async_copy: _ragged_paged_attention_kernel_loop()._async_copy().
  _ragged_paged_attention_kernel_loop._fetch_bq: _ragged_paged_attention_kernel_loop()._fetch_bq().
  _ragged_paged_attention_kernel_loop._send_bo: _ragged_paged_attention_kernel_loop()._send_bo().
  merge_kv: merge_kv().
  _ragged_paged_attention_kernel_loop.strided_load: _ragged_paged_attention_kernel_loop().strided_load().
  _ragged_paged_attention_kernel_loop._update_kv_cache.loop_body: _ragged_paged_attention_kernel_loop()._update_kv_cache().loop_body().
  dynamic_validate_inputs: dynamic_validate_inputs().
  _ragged_paged_attention_kernel_loop.broadcast_minor: _ragged_paged_attention_kernel_loop().broadcast_minor().
  ragged_paged_attention._prepare_block_sizes: ragged_paged_attention()._prepare_block_sizes().
  get_smem_estimate_bytes: get_smem_estimate_bytes().
  _ragged_paged_attention_kernel_loop.prologue: _ragged_paged_attention_kernel_loop().prologue().
  _ragged_paged_attention_kernel_loop.epilogue: _ragged_paged_attention_kernel_loop().epilogue().
  _ragged_paged_attention_kernel_loop.flash_attention_step2_pv: _ragged_paged_attention_kernel_loop().flash_attention_step2_pv().
  _ragged_paged_attention_kernel_loop.start_fetch_bkv: _ragged_paged_attention_kernel_loop().start_fetch_bkv().
  _ragged_paged_attention_kernel_loop.start_fetch_bq: _ragged_paged_attention_kernel_loop().start_fetch_bq().
  _ragged_paged_attention_kernel_loop.wait_send_bo: _ragged_paged_attention_kernel_loop().wait_send_bo().
  _ragged_paged_attention_kernel_loop.wait_update_kv_cache: _ragged_paged_attention_kernel_loop().wait_update_kv_cache().
  RpaCase.PREFILL: RpaCase#PREFILL.
  _ragged_paged_attention_kernel: _ragged_paged_attention_kernel().
  _ragged_paged_attention_kernel_loop.wait_fetch_bkv: _ragged_paged_attention_kernel_loop().wait_fetch_bkv().
  _ragged_paged_attention_kernel_loop.wait_fetch_bq: _ragged_paged_attention_kernel_loop().wait_fetch_bq().
  _ragged_paged_attention_kernel_loop.start_send_bo: _ragged_paged_attention_kernel_loop().start_send_bo().
  _ragged_paged_attention_kernel_loop.start_update_kv_cache: _ragged_paged_attention_kernel_loop().start_update_kv_cache().
  _ragged_paged_attention_kernel_loop.strided_store: _ragged_paged_attention_kernel_loop().strided_store().
  _ragged_paged_attention_kernel_loop.load_bq: _ragged_paged_attention_kernel_loop().load_bq().
  _ragged_paged_attention_kernel_loop.load_bkv: _ragged_paged_attention_kernel_loop().load_bkv().
  _ragged_paged_attention_kernel_loop.process: _ragged_paged_attention_kernel_loop().process().
  _ragged_paged_attention_kernel_loop.mask_and: _ragged_paged_attention_kernel_loop().mask_and().
  static_validate_inputs._validate_block_sizes: static_validate_inputs()._validate_block_sizes().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq.prefetch_next_bq: _ragged_paged_attention_kernel_loop().process().compute_with_bq().prefetch_next_bq().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq.compute_with_bkv.prefetch_next_bkv: _ragged_paged_attention_kernel_loop().process().compute_with_bq().compute_with_bkv().prefetch_next_bkv().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq.compute_with_bkv.wait_cur_bq: _ragged_paged_attention_kernel_loop().process().compute_with_bq().compute_with_bkv().wait_cur_bq().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq.compute_with_bkv.update_cur_bkv_to_cache: _ragged_paged_attention_kernel_loop().process().compute_with_bq().compute_with_bkv().update_cur_bkv_to_cache().
  _ragged_paged_attention_kernel_loop.pipeline: _ragged_paged_attention_kernel_loop().pipeline().
  has_bank_conflicts: has_bank_conflicts().
  ragged_paged_attention.run_rpa_kernel.run: ragged_paged_attention().run_rpa_kernel().run().
  _ragged_paged_attention_kernel_loop.process.get_next_bq_ids: _ragged_paged_attention_kernel_loop().process().get_next_bq_ids().
  _ragged_paged_attention_kernel_loop.process.get_next_bkv_ids: _ragged_paged_attention_kernel_loop().process().get_next_bkv_ids().
  prepare_outputs: prepare_outputs().
---
# Module: [`tpu_inference/kernels/ragged_paged_attention/v3/kernel.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py)

## Classes
### `RpaCase`  ·  implements/extends Enum
- def: [`tpu_inference/kernels/ragged_paged_attention/v3/kernel.py:35`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L35)
- doc: Represents the different cases for Ragged Paged Attention.
- signature: `class RpaCase(Enum):`
- members:
  - `get_range(self, distribution)` — [`L54`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L54)
  - `symbol(self)` — [`L47`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L47)
  - `DECODE` — [`L42`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L42)
  - `MIXED` — [`L44`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L44)
  - `PREFILL` — [`L43`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L43)
- used by: [`ragged_paged_attention`](kernel.md#ragged_paged_attention), [`get_default_block_sizes`](kernel.md#get_default_block_sizes), [`run_rpa_kernel`](kernel.md#ragged_paged_attention.run_rpa_kernel), [`_ragged_paged_attention_kernel_loop`](kernel.md#_ragged_paged_attention_kernel_loop)

## Functions
- `_async_copy(src, dst, sem, wait)` — [`L539`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L539)
- `_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False)` — [`L549`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L549)
- `_fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False)` — [`L706`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L706)
- `_prepare_block_sizes(block_sizes, case)` — [`L1883`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L1883)
- `_ragged_paged_attention_kernel(*args, **kwargs)` — [`L278`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L278)
- `_ragged_paged_attention_kernel_loop(seq_idx, kv_lens_ref, page_indices_ref, cu_q_lens_ref, distribution_ref, sem_ids_ref, bo_ids_ref, bkv_update_ids_ref, q_hbm_ref, kv_hbm_ref, kv_cache_hbm_ref, o_hbm_ref, updated_kv_cache_hbm_ref, bkv_x2_ref, bq_x2_ref, bo_x2_ref, sems, l_ref, m_ref, acc_ref, *, use_causal_mask: bool = True, update_kv_cache: bool = True, skip_kv_mask: bool = False, sm_scale: float, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, static_q_len: int | None = None, bq_sz, bkv_sz, bq_csz, bkv_csz, case: RpaCase = RpaCase.MIXED, debug_mode: bool = False)` — [`L291`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L291)
- `_send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False)` — [`L730`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L730)
- `_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, *, wait=False)` — [`L637`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L637)
- `_validate_block_sizes(block_sizes, prefix)` — [`L1461`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L1461)
- `attention_loop(idx)` — [`L1011`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L1011)
- `broadcast_minor(src, shape)` — [`L871`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L871)
- `compute_with_bkv(bkv_idx)` — [`L957`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L957)
- `compute_with_bq(bq_idx)` — [`L927`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L927)
- `debug_print(msg, *args)` — [`L401`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L401)
- `dynamic_validate_inputs(queries: jax.Array, keys: jax.Array, values: jax.Array, kv_cache: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, *, use_causal_mask: bool = True, skip_kv_mask: bool = False, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, out_dtype: Any = None, mask_value: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, d_block_sizes: tuple[int, int, int, int] | None = None, p_block_sizes: tuple[int, int, int, int] | None = None, m_block_sizes: tuple[int, int, int, int] | None = None, vmem_limit_bytes: int | None = None)` — [`L1226`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L1226)
- `epilogue()` — [`L1108`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L1108)
- `flash_attention_step1_qk_softmax(q, k, v, l_ref, m_ref, *, processed_q_len, processed_kv_len, effective_kv_len)` — [`L425`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L425)
- `flash_attention_step2_pv(p, v, exp_m_diff, o_ref)` — [`L515`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L515)
- `get_default_block_sizes(q_dtype, kv_dtype, actual_num_q_heads, actual_num_kv_heads, head_dim, page_size, max_num_tokens, max_num_seqs, pages_per_seq, *, case: RpaCase = RpaCase.MIXED)` — [`L1496`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L1496) — Get (bq, bkv_sz, bq_csz, bkv_csz) by some heuristic formulas.
- `get_kv_cache_shape(total_num_pages, page_size, actual_num_kv_heads, actual_head_dim, kv_dtype)` — [`L261`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L261)
- `get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx, *, num_bkv)` — [`L905`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L905)
- `get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx)` — [`L897`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L897)
- `get_smem_estimate_bytes(max_num_seqs, pages_per_seq)` — [`L208`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L208)
- `get_vmem_estimate_bytes(actual_num_kv_heads, actual_num_q_heads_per_kv_head, actual_head_dim, bq_sz, bkv_sz, q_dtype, kv_dtype)` — [`L227`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L227)
- `has_bank_conflicts(stride, distance=24, num_banks=32)` — [`L1116`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L1116)
- `load_bkv(bkv_sem_idx, kv_head_idx, start, sz)` — [`L844`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L844)
- `load_bq(bq_sem_idx, kv_head_idx, start, sz)` — [`L836`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L836)
- `loop_body(i, states)` — [`L672`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L672)
- `mask_and(mask, new_mask)` — [`L882`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L882)
- `merge_kv(k: jax.Array, v: jax.Array)` — [`L1126`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L1126)
- `pipeline()` — [`L1104`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L1104)
- `prefetch_next_bkv()` — [`L968`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L968)
- `prefetch_next_bq()` — [`L952`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L952)
- `prepare_inputs(q: jax.Array, k: jax.Array, v: jax.Array)` — [`L1159`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L1159)
- `prepare_outputs(out, actual_num_q_heads_per_kv_head: int, actual_head_dim: int)` — [`L1203`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L1203)
- `process(static_q_len=None)` — [`L887`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L887)
- `prologue()` — [`L1097`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L1097)
- `ragged_paged_attention(queries: jax.Array, keys: jax.Array, values: jax.Array, kv_cache: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, *, use_causal_mask: bool = True, update_kv_cache: bool = True, skip_kv_mask: bool = False, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, out_dtype: Any = None, mask_value: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, d_block_sizes: tuple[int, int, int, int] | None = None, p_block_sizes: tuple[int, int, int, int] | None = None, m_block_sizes: tuple[int, int, int, int] | None = None, vmem_limit_bytes: int | None = None, debug_mode: bool = False, disable_bounds_checks: bool = True, disable_semaphore_checks: bool = True)` — [`L1586`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L1586) — Ragged paged attention that supports mixed prefill and decode.
- `ref_ragged_paged_attention(queries: jax.Array, keys: jax.Array, values: jax.Array, kv_cache: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, *, use_causal_mask: bool = True, skip_kv_mask: bool = False, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, out_dtype: Any = None, mask_value: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None)` — [`L66`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L66)
- `run(scalar_prefetches, q, kv, kv_cache)` — [`L1869`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L1869)
- `run_rpa_kernel(q, kv_cache, *, bq_sz, bkv_sz, bq_csz, bkv_csz, static_q_len=None, case: RpaCase = RpaCase.MIXED)` — [`L1736`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L1736)
- `start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)` — [`L754`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L754)
- `start_fetch_bq(seq_idx, bq_idx, bq_sem_idx)` — [`L760`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L760)
- `start_send_bo(seq_idx, bo_idx, bo_sem_idx)` — [`L766`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L766)
- `start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)` — [`L781`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L781)
- `static_validate_inputs(queries: jax.Array, keys: jax.Array, values: jax.Array, kv_cache: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, *, use_causal_mask: bool = True, skip_kv_mask: bool = False, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, out_dtype: Any = None, mask_value: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, d_block_sizes: tuple[int, int, int, int] | None = None, p_block_sizes: tuple[int, int, int, int] | None = None, m_block_sizes: tuple[int, int, int, int] | None = None, vmem_limit_bytes: int | None = None)` — [`L1321`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L1321) — Validate inputs to the RPA kernel statically.
- `strided_load(ref, start, sz, step, *, dtype=None)` — [`L801`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L801)
- `strided_store(ref, start, sz, step, val)` — [`L819`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L819)
- `update_cur_bkv_to_cache()` — [`L990`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L990)
- `wait_cur_bq()` — [`L975`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L975)
- `wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)` — [`L757`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L757)
- `wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)` — [`L763`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L763)
- `wait_send_bo(bo_sem_idx)` — [`L771`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L771)
- `wait_update_kv_cache(bkv_sem_idx)` — [`L787`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L787)

