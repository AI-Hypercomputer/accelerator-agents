---
title: 'Module: simply/kernels/ragged_paged_attention.py'
type: catalog
provenance: extracted
module: simply/kernels/ragged_paged_attention.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.kernels.ragged_paged_attention`/
symbols:
  _ragged_paged_attention_kernel_loop.debug_print: _ragged_paged_attention_kernel_loop().debug_print().
  ragged_paged_attention: ragged_paged_attention().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq: _ragged_paged_attention_kernel_loop().process().compute_with_bq().
  cdiv: cdiv().
  RpaCase: RpaCase#
  _ragged_paged_attention_kernel_loop: _ragged_paged_attention_kernel_loop().
  get_default_block_sizes: get_default_block_sizes().
  align_to: align_to().
  get_dtype_packing: get_dtype_packing().
  ragged_paged_attention.run_rpa_kernel: ragged_paged_attention().run_rpa_kernel().
  ref_ragged_paged_attention: ref_ragged_paged_attention().
  _ragged_paged_attention_kernel_loop._fetch_bkv: _ragged_paged_attention_kernel_loop()._fetch_bkv().
  _ragged_paged_attention_kernel_loop._update_kv_cache: _ragged_paged_attention_kernel_loop()._update_kv_cache().
  static_validate_inputs: static_validate_inputs().
  RpaCase.symbol: RpaCase#symbol().
  _ragged_paged_attention_kernel: _ragged_paged_attention_kernel().
  RpaCase.get_range: RpaCase#get_range().
  get_vmem_estimate_bytes: get_vmem_estimate_bytes().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq.compute_with_bkv: _ragged_paged_attention_kernel_loop().process().compute_with_bq().compute_with_bkv().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq.compute_with_bkv.attention_loop: _ragged_paged_attention_kernel_loop().process().compute_with_bq().compute_with_bkv().attention_loop().
  prepare_inputs: prepare_inputs().
  _ragged_paged_attention_kernel_loop._async_copy: _ragged_paged_attention_kernel_loop()._async_copy().
  _ragged_paged_attention_kernel_loop.epilogue: _ragged_paged_attention_kernel_loop().epilogue().
  RpaCase.DECODE: RpaCase#DECODE.
  RpaCase.MIXED: RpaCase#MIXED.
  _find_first_valid_seq: _find_first_valid_seq().
  _ragged_paged_attention_kernel_loop._fetch_bq: _ragged_paged_attention_kernel_loop()._fetch_bq().
  _ragged_paged_attention_kernel_loop._send_bo: _ragged_paged_attention_kernel_loop()._send_bo().
  merge_kv: merge_kv().
  _ragged_paged_attention_kernel_loop._global_pos_to_local_bkv_idx: _ragged_paged_attention_kernel_loop()._global_pos_to_local_bkv_idx().
  _ragged_paged_attention_kernel_loop.strided_load: _ragged_paged_attention_kernel_loop().strided_load().
  get_kv_cache_shape: get_kv_cache_shape().
  compute_local_kv_len: compute_local_kv_len().
  _ragged_paged_attention_kernel._body: _ragged_paged_attention_kernel()._body().
  _ragged_paged_attention_kernel_loop.flash_attention_step1_qk_softmax: _ragged_paged_attention_kernel_loop().flash_attention_step1_qk_softmax().
  _ragged_paged_attention_kernel_loop._update_kv_cache.loop_body: _ragged_paged_attention_kernel_loop()._update_kv_cache().loop_body().
  dynamic_validate_inputs: dynamic_validate_inputs().
  _ragged_paged_attention_kernel_loop.broadcast_minor: _ragged_paged_attention_kernel_loop().broadcast_minor().
  ragged_paged_attention._prepare_block_sizes: ragged_paged_attention()._prepare_block_sizes().
  get_smem_estimate_bytes: get_smem_estimate_bytes().
  _ragged_paged_attention_kernel_loop.prologue: _ragged_paged_attention_kernel_loop().prologue().
  _ragged_paged_attention_kernel_loop.flash_attention_step2_pv: _ragged_paged_attention_kernel_loop().flash_attention_step2_pv().
  _ragged_paged_attention_kernel_loop.start_fetch_bkv: _ragged_paged_attention_kernel_loop().start_fetch_bkv().
  _ragged_paged_attention_kernel_loop.start_fetch_bq: _ragged_paged_attention_kernel_loop().start_fetch_bq().
  _ragged_paged_attention_kernel_loop.wait_send_bo: _ragged_paged_attention_kernel_loop().wait_send_bo().
  _ragged_paged_attention_kernel_loop._send_lse: _ragged_paged_attention_kernel_loop()._send_lse().
  _ragged_paged_attention_kernel_loop.wait_send_lse: _ragged_paged_attention_kernel_loop().wait_send_lse().
  _ragged_paged_attention_kernel_loop.wait_update_kv_cache: _ragged_paged_attention_kernel_loop().wait_update_kv_cache().
  RpaCase.PREFILL: RpaCase#PREFILL.
  _find_first_valid_seq.cond: _find_first_valid_seq().cond().
  _ragged_paged_attention_kernel_loop.wait_fetch_bkv: _ragged_paged_attention_kernel_loop().wait_fetch_bkv().
  _ragged_paged_attention_kernel_loop.wait_fetch_bq: _ragged_paged_attention_kernel_loop().wait_fetch_bq().
  _ragged_paged_attention_kernel_loop.start_send_bo: _ragged_paged_attention_kernel_loop().start_send_bo().
  _ragged_paged_attention_kernel_loop.start_send_lse: _ragged_paged_attention_kernel_loop().start_send_lse().
  _ragged_paged_attention_kernel_loop.start_update_kv_cache: _ragged_paged_attention_kernel_loop().start_update_kv_cache().
  _ragged_paged_attention_kernel_loop.strided_store: _ragged_paged_attention_kernel_loop().strided_store().
  _ragged_paged_attention_kernel_loop.load_bq: _ragged_paged_attention_kernel_loop().load_bq().
  _ragged_paged_attention_kernel_loop.load_bkv: _ragged_paged_attention_kernel_loop().load_bkv().
  _ragged_paged_attention_kernel_loop.process: _ragged_paged_attention_kernel_loop().process().
  _ragged_paged_attention_kernel_loop.process.get_next_bkv_ids: _ragged_paged_attention_kernel_loop().process().get_next_bkv_ids().
  next_power_of_2: next_power_of_2().
  get_tpu_version: get_tpu_version().
  _ragged_paged_attention_kernel_loop.mask_and: _ragged_paged_attention_kernel_loop().mask_and().
  static_validate_inputs._validate_block_sizes: static_validate_inputs()._validate_block_sizes().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq.prefetch_next_bq: _ragged_paged_attention_kernel_loop().process().compute_with_bq().prefetch_next_bq().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq.compute_with_bkv.prefetch_next_bkv: _ragged_paged_attention_kernel_loop().process().compute_with_bq().compute_with_bkv().prefetch_next_bkv().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq.compute_with_bkv.wait_cur_bq: _ragged_paged_attention_kernel_loop().process().compute_with_bq().compute_with_bkv().wait_cur_bq().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq.compute_with_bkv.update_cur_bkv_to_cache: _ragged_paged_attention_kernel_loop().process().compute_with_bq().compute_with_bkv().update_cur_bkv_to_cache().
  has_bank_conflicts: has_bank_conflicts().
  get_dtype_bitwidth: get_dtype_bitwidth().
  DEFAULT_VMEM_LIMIT_BYTES: DEFAULT_VMEM_LIMIT_BYTES.
  _find_first_valid_seq.body: _find_first_valid_seq().body().
  _ragged_paged_attention_kernel._cond: _ragged_paged_attention_kernel()._cond().
  _ragged_paged_attention_kernel_loop.process.get_next_bq_ids: _ragged_paged_attention_kernel_loop().process().get_next_bq_ids().
  prepare_outputs: prepare_outputs().
  prepare_logsumexp_output: prepare_logsumexp_output().
  ragged_paged_attention.run_rpa_kernel.run: ragged_paged_attention().run_rpa_kernel().run().
---
# Module: [`simply/kernels/ragged_paged_attention.py`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py)

## Classes
### `RpaCase`  ·  implements/extends Enum
- def: [`simply/kernels/ragged_paged_attention.py:86`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L86) — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
- doc: Represents the different cases for Ragged Paged Attention.
- signature: `class RpaCase(Enum):`
- members:
  - `get_range(self, distribution)` — [`L105`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L105)
  - `symbol(self)` — [`L98`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L98) — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
  - `DECODE` — [`L93`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L93) — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
  - `MIXED` — [`L95`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L95) — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
  - `PREFILL` — [`L94`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L94) — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
- used by: [`ragged_paged_attention`](ragged_paged_attention.md#ragged_paged_attention), [`_ragged_paged_attention_kernel_loop`](ragged_paged_attention.md#_ragged_paged_attention_kernel_loop), [`get_default_block_sizes`](ragged_paged_attention.md#get_default_block_sizes), [`run_rpa_kernel`](ragged_paged_attention.md#ragged_paged_attention.run_rpa_kernel)

## Functions
- `_async_copy(src, dst, sem, wait)` — [`L689`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L689)
- `_body(seq_idx)` — [`L406`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L406)
- `_cond(seq_idx)` — [`L421`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L421)
- `_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False)` — [`L699`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L699) — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
- `_fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False)` — [`L867`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L867) — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
- `_find_first_valid_seq(start_idx, end_idx, shard_info, kv_lens_ref, cu_q_lens_ref, page_size)` — [`L358`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L358) — Find the first seq idx >= start_idx with local_kv_len > 0 and q_len > 0. — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
- `_global_pos_to_local_bkv_idx(global_pos)` — [`L528`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L528) — Convert a global KV position to the local BKV block index. — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
- `_prepare_block_sizes(block_sizes, case)` — [`L2258`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L2258)
- `_ragged_paged_attention_kernel(*args, **kwargs)` — [`L382`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L382) — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
- `_ragged_paged_attention_kernel_loop(seq_idx, next_seq_idx, start_seq_idx, end_seq_idx, kv_lens_ref, page_indices_ref, cu_q_lens_ref, distribution_ref, sem_ids_ref, bo_ids_ref, bkv_update_ids_ref, shard_info_ref, q_hbm_ref, kv_hbm_ref, kv_cache_hbm_ref, o_hbm_ref, updated_kv_cache_hbm_ref, lse_hbm_ref, bkv_x2_ref, bq_x2_ref, bo_x2_ref, sems, l_ref, m_ref, acc_ref, blse_x2_ref, *, use_causal_mask: bool = True, skip_kv_mask: bool = False, sm_scale: float, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, static_q_len: int | None = None, bq_sz, bkv_sz, bq_csz, bkv_csz, case: RpaCase = RpaCase.MIXED, debug_mode: bool = False)` — [`L427`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L427) — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
- `_send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False)` — [`L892`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L892) — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
- `_send_lse(seq_idx, bo_idx, bo_sem_idx, *, wait=False)` — [`L944`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L944) — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
- `_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, *, wait=False)` — [`L801`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L801)
- `_validate_block_sizes(block_sizes, prefix)` — [`L1711`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1711)
- `align_to(x, a)` — [`L51`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L51)
- `attention_loop(idx)` — [`L1213`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1213)
- `body(idx)` — [`L377`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L377)
- `broadcast_minor(src, shape)` — [`L1060`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1060)
- `cdiv(a, b)` — [`L47`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L47)
- `compute_local_kv_len(global_kv_len, shard_id, num_shards, page_size)` — [`L346`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L346) — Compute local KV length on this shard for a given global KV length. — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
- `compute_with_bkv(bkv_idx)` — [`L1165`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1165)
- `compute_with_bq(bq_idx)` — [`L1119`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1119)
- `cond(idx)` — [`L365`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L365)
- `debug_print(msg, *args)` — [`L555`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L555)
- `dynamic_validate_inputs(queries: jax.Array, keys: jax.Array, values: jax.Array, kv_cache: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, *, use_causal_mask: bool = True, skip_kv_mask: bool = False, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, out_dtype: Any = None, mask_value: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, d_block_sizes: tuple[int, int, int, int] | None = None, p_block_sizes: tuple[int, int, int, int] | None = None, m_block_sizes: tuple[int, int, int, int] | None = None, vmem_limit_bytes: int | None = None)` — [`L1465`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1465) — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
- `epilogue()` — [`L1330`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1330)
- `flash_attention_step1_qk_softmax(q, k, v, l_ref, m_ref, *, processed_q_len, processed_kv_len, effective_kv_len)` — [`L578`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L578)
- `flash_attention_step2_pv(p, v, exp_m_diff, o_ref)` — [`L670`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L670)
- `get_default_block_sizes(q_dtype, kv_dtype, actual_num_q_heads, actual_num_kv_heads, head_dim, page_size, max_num_tokens, max_num_seqs, pages_per_seq, *, case: RpaCase = RpaCase.MIXED)` — [`L1748`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1748) — Get (bq, bkv_sz, bq_csz, bkv_csz) by some heuristic formulas. — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
- `get_dtype_bitwidth(dtype)` — [`L55`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L55)
- `get_dtype_packing(dtype)` — [`L59`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L59)
- `get_kv_cache_shape(total_num_pages, page_size, actual_num_kv_heads, actual_head_dim, kv_dtype)` — [`L329`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L329) — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
- `get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx, *, num_bkv)` — [`L1094`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1094)
- `get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx)` — [`L1086`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1086)
- `get_smem_estimate_bytes(max_num_seqs, pages_per_seq)` — [`L262`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L262) — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
- `get_tpu_version()` — [`L71`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L71)
- `get_vmem_estimate_bytes(actual_num_kv_heads, actual_num_q_heads_per_kv_head, actual_head_dim, bq_sz, bkv_sz, q_dtype, kv_dtype)` — [`L291`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L291) — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
- `has_bank_conflicts(stride, distance=24, num_banks=32)` — [`L1341`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1341)
- `load_bkv(bkv_sem_idx, kv_head_idx, start, sz)` — [`L1030`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1030)
- `load_bq(bq_sem_idx, kv_head_idx, start, sz)` — [`L1020`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1020)
- `loop_body(i, states)` — [`L831`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L831)
- `mask_and(mask, new_mask)` — [`L1071`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1071)
- `merge_kv(k: jax.Array, v: jax.Array)` — [`L1351`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1351) — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
- `next_power_of_2(x: int)` — [`L64`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L64)
- `prefetch_next_bkv()` — [`L1177`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1177)
- `prefetch_next_bq()` — [`L1160`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1160)
- `prepare_inputs(q: jax.Array, k: jax.Array, v: jax.Array)` — [`L1382`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1382) — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
- `prepare_logsumexp_output(lse, actual_num_q_heads_per_kv_head: int, num_q_heads_per_kv_head: int)` — [`L1449`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1449)
- `prepare_outputs(out, actual_num_q_heads_per_kv_head: int, actual_head_dim: int)` — [`L1424`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1424)
- `process(static_q_len=None)` — [`L1076`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1076) — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
- `prologue()` — [`L1321`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1321)
- `ragged_paged_attention(queries: jax.Array, keys: jax.Array, values: jax.Array, kv_cache: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, *, shard_info: jax.Array | None = None, use_causal_mask: bool = True, skip_kv_mask: bool = False, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, out_dtype: Any = None, mask_value: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, d_block_sizes: tuple[int, int, int, int] | None = None, p_block_sizes: tuple[int, int, int, int] | None = None, m_block_sizes: tuple[int, int, int, int] | None = None, vmem_limit_bytes: int | None = None, debug_mode: bool = False, disable_bounds_checks: bool = True, disable_semaphore_checks: bool = True, update_kv_cache: bool = True, save_residuals: bool = False)` — [`L1931`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1931) — Ragged paged attention that supports mixed prefill and decode. — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
- `ref_ragged_paged_attention(queries: jax.Array, keys: jax.Array, values: jax.Array, kv_cache: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, *, use_causal_mask: bool = True, skip_kv_mask: bool = False, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, out_dtype: Any = None, mask_value: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, update_kv_cache: bool = True, save_residuals: bool = False)` — [`L117`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L117) — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
- `run(scalar_prefetches, q, kv, kv_cache)` — [`L2248`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L2248)
- `run_rpa_kernel(q, kv_cache, *, bq_sz, bkv_sz, bq_csz, bkv_csz, static_q_len=None, case: RpaCase = RpaCase.MIXED)` — [`L2100`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L2100) — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
- `start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)` — [`L917`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L917)
- `start_fetch_bq(seq_idx, bq_idx, bq_sem_idx)` — [`L923`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L923)
- `start_send_bo(seq_idx, bo_idx, bo_sem_idx)` — [`L929`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L929)
- `start_send_lse(seq_idx, bo_idx, bo_sem_idx)` — [`L957`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L957)
- `start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)` — [`L970`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L970)
- `static_validate_inputs(queries: jax.Array, keys: jax.Array, values: jax.Array, kv_cache: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, *, shard_info: jax.Array | None = None, use_causal_mask: bool = True, skip_kv_mask: bool = False, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, out_dtype: Any = None, mask_value: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, d_block_sizes: tuple[int, int, int, int] | None = None, p_block_sizes: tuple[int, int, int, int] | None = None, m_block_sizes: tuple[int, int, int, int] | None = None, vmem_limit_bytes: int | None = None)` — [`L1558`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1558) — Validate inputs to the RPA kernel statically. — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
- `strided_load(ref, start, sz, step, *, dtype=None)` — [`L986`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L986)
- `strided_store(ref, start, sz, step, val)` — [`L1004`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1004)
- `update_cur_bkv_to_cache()` — [`L1193`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1193)
- `wait_cur_bq()` — [`L1183`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L1183)
- `wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)` — [`L920`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L920)
- `wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)` — [`L926`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L926)
- `wait_send_bo(bo_sem_idx)` — [`L934`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L934)
- `wait_send_lse(bo_sem_idx)` — [`L960`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L960)
- `wait_update_kv_cache(bkv_sem_idx)` — [`L976`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L976)

## Module values
- `DEFAULT_VMEM_LIMIT_BYTES` — [`L83`](../../../../../../raw/code/simply/simply/kernels/ragged_paged_attention.py#L83)

