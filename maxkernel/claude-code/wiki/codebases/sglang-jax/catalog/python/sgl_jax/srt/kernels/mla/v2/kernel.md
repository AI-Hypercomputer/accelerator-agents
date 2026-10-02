---
title: 'Module: python/sgl_jax/srt/kernels/mla/v2/kernel.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/mla/v2/kernel.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.mla.v2.kernel`/
symbols:
  _mla_ragged_paged_attention_kernel.debug_print: _mla_ragged_paged_attention_kernel().debug_print().
  mla_ragged_paged_attention: mla_ragged_paged_attention().
  align_to: align_to().
  _mla_ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv: _mla_ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().
  _mla_ragged_paged_attention_kernel._fetch_bkv: _mla_ragged_paged_attention_kernel()._fetch_bkv().
  _mla_ragged_paged_attention_kernel._update_kv_cache: _mla_ragged_paged_attention_kernel()._update_kv_cache().
  MlaCase.symbol: MlaCase#symbol().
  _mla_ragged_paged_attention_kernel.process.compute_with_bq: _mla_ragged_paged_attention_kernel().process().compute_with_bq().
  _mla_ragged_paged_attention_kernel._async_copy: _mla_ragged_paged_attention_kernel()._async_copy().
  mla_ragged_paged_attention.run_mla_kernel: mla_ragged_paged_attention().run_mla_kernel().
  get_dtype_packing: get_dtype_packing().
  cdiv: cdiv().
  _mla_ragged_paged_attention_kernel: _mla_ragged_paged_attention_kernel().
  _mla_ragged_paged_attention_kernel._pack_new_kv._update: _mla_ragged_paged_attention_kernel()._pack_new_kv()._update().
  get_kv_cache_shape: get_kv_cache_shape().
  MlaCase: MlaCase#
  static_validate_inputs: static_validate_inputs().
  _mla_ragged_paged_attention_kernel.process: _mla_ragged_paged_attention_kernel().process().
  cdiv_on_kv_packing: cdiv_on_kv_packing().
  prepare_q_inputs: prepare_q_inputs().
  prepare_kv_inputs: prepare_kv_inputs().
  _mla_ragged_paged_attention_kernel.flash_attention: _mla_ragged_paged_attention_kernel().flash_attention().
  _mla_ragged_paged_attention_kernel.broadcast_minor: _mla_ragged_paged_attention_kernel().broadcast_minor().
  floor_div_on_kv_packing: floor_div_on_kv_packing().
  _mla_ragged_paged_attention_kernel.prologue: _mla_ragged_paged_attention_kernel().prologue().
  _mla_ragged_paged_attention_kernel.epilogue: _mla_ragged_paged_attention_kernel().epilogue().
  _mla_ragged_paged_attention_kernel._fetch_bq: _mla_ragged_paged_attention_kernel()._fetch_bq().
  _mla_ragged_paged_attention_kernel._send_bo: _mla_ragged_paged_attention_kernel()._send_bo().
  _mla_ragged_paged_attention_kernel.start_fetch_bkv: _mla_ragged_paged_attention_kernel().start_fetch_bkv().
  _mla_ragged_paged_attention_kernel.start_fetch_bq: _mla_ragged_paged_attention_kernel().start_fetch_bq().
  _mla_ragged_paged_attention_kernel.wait_send_bo: _mla_ragged_paged_attention_kernel().wait_send_bo().
  _mla_ragged_paged_attention_kernel.wait_update_kv_cache: _mla_ragged_paged_attention_kernel().wait_update_kv_cache().
  _mla_ragged_paged_attention_kernel._update_kv_cache.loop_body: _mla_ragged_paged_attention_kernel()._update_kv_cache().loop_body().
  _mla_ragged_paged_attention_kernel.wait_fetch_bkv: _mla_ragged_paged_attention_kernel().wait_fetch_bkv().
  _mla_ragged_paged_attention_kernel.wait_fetch_bq: _mla_ragged_paged_attention_kernel().wait_fetch_bq().
  _mla_ragged_paged_attention_kernel.start_send_bo: _mla_ragged_paged_attention_kernel().start_send_bo().
  _mla_ragged_paged_attention_kernel.start_update_kv_cache: _mla_ragged_paged_attention_kernel().start_update_kv_cache().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  MlaCase.MIXED: MlaCase#MIXED.
  _mla_ragged_paged_attention_kernel.flash_attention.load_with_init: _mla_ragged_paged_attention_kernel().flash_attention().load_with_init().
  _mla_ragged_paged_attention_kernel._fetch_bq._copy: _mla_ragged_paged_attention_kernel()._fetch_bq()._copy().
  _mla_ragged_paged_attention_kernel._send_bo._copy: _mla_ragged_paged_attention_kernel()._send_bo()._copy().
  _mla_ragged_paged_attention_kernel.process.compute_with_bq.prefetch_next_bq: _mla_ragged_paged_attention_kernel().process().compute_with_bq().prefetch_next_bq().
  _mla_ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv.prefetch_next_bkv: _mla_ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().prefetch_next_bkv().
  _mla_ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv.wait_cur_bq: _mla_ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().wait_cur_bq().
  _mla_ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv.update_cur_bkv_to_cache: _mla_ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().update_cur_bkv_to_cache().
  get_dtype_bitwidth: get_dtype_bitwidth().
  MlaCase.DECODE: MlaCase#DECODE.
  MlaCase.BATCHED_DECODE: MlaCase#BATCHED_DECODE.
  MlaCase.PREFILL: MlaCase#PREFILL.
  _mla_ragged_paged_attention_kernel._pack_new_kv: _mla_ragged_paged_attention_kernel()._pack_new_kv().
  _mla_ragged_paged_attention_kernel._pack_new_kv._update.merge_loop_body: _mla_ragged_paged_attention_kernel()._pack_new_kv()._update().merge_loop_body().
  _mla_ragged_paged_attention_kernel.load_bq: _mla_ragged_paged_attention_kernel().load_bq().
  _mla_ragged_paged_attention_kernel.load_bkv: _mla_ragged_paged_attention_kernel().load_bkv().
  _mla_ragged_paged_attention_kernel.process.get_next_bq_ids: _mla_ragged_paged_attention_kernel().process().get_next_bq_ids().
  _mla_ragged_paged_attention_kernel.process.get_next_bkv_ids: _mla_ragged_paged_attention_kernel().process().get_next_bkv_ids().
  prepare_outputs: prepare_outputs().
  DEFAULT_VMEM_LIMIT_BYTES: DEFAULT_VMEM_LIMIT_BYTES.
---
# Module: [`python/sgl_jax/srt/kernels/mla/v2/kernel.py`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py)

## Classes
### `MlaCase`  ·  implements/extends Enum
- def: [`python/sgl_jax/srt/kernels/mla/v2/kernel.py:89`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L89)
- doc: Represents the different cases for MLA.
- signature: `class MlaCase(Enum):`
- members:
  - `symbol(self)` — [`L103`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L103)
  - `BATCHED_DECODE` — [`L100`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L100)
  - `DECODE` — [`L97`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L97)
  - `MIXED` — [`L99`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L99)
  - `PREFILL` — [`L98`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L98)
- used by: [`mla_ragged_paged_attention`](kernel.md#mla_ragged_paged_attention), [`run_mla_kernel`](kernel.md#mla_ragged_paged_attention.run_mla_kernel)

## Functions
- `_async_copy(src, dst, sem, wait)` — [`L420`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L420)
- `_copy(q_len_start=q_len_start, sz=sz, bq_nope_vmem_ref=bq_nope_vmem_ref, bq_rope_vmem_ref=bq_rope_vmem_ref, sem=sem)` — [`L920`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L920)
- `_copy(vmem_ref=vmem_ref, sz=sz, q_len_start=q_len_start, sem=sem)` — [`L962`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L962)
- `_fetch_bkv(batch_start_seq_idx, bkv_idx, bkv_sem_idx, *, wait=False)` — [`L430`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L430)
- `_fetch_bq(batch_start_seq_idx, bq_idx, bq_sem_idx, *, wait=False)` — [`L898`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L898)
- `_mla_ragged_paged_attention_kernel(kv_lens_ref, page_indices_ref, cu_q_lens_ref, cu_kv_lens_ref, start_end_seq_idx_ref, sem_ids_ref, bo_ids_ref, bkv_update_ids_ref, ql_nope_hbm_ref, q_pe_hbm_ref, new_kv_c_hbm_ref, new_k_pe_hbm_ref, cache_kv_hbm_ref, o_hbm_ref, updated_cache_kv_hbm_ref, bkvc_x2_ref, bkpe_x2_ref, bq_nope_x2_ref, bq_rope_x2_ref, bo_x2_ref, sems, l_ref, m_ref, acc_ref, *, static_q_len: int, sm_scale: float, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, bkv_p, bq_sz, batch_size: int = 1, debug_mode: bool = False)` — [`L253`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L253)
- `_pack_new_kv(bkv_sem_idx, offsets, update_szs)` — [`L633`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L633) — Packs newly computed KVs into the correct sub-word alignment in VMEM.
- `_send_bo(batch_start_seq_idx, bo_idx, bo_sem_idx, *, wait=False)` — [`L941`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L941)
- `_update(b=b)` — [`L663`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L663)
- `_update_kv_cache(batch_start_seq_idx, b, bkv_sem_idx, offset, update_sz, *, wait=False)` — [`L794`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L794)
- `align_to(x, a)` — [`L61`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L61)
- `broadcast_minor(src, shape)` — [`L1054`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L1054)
- `cdiv(a, b)` — [`L56`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L56)
- `cdiv_on_kv_packing(a, kv_packing)` — [`L42`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L42)
- `compute_with_bkv(bkv_idx, _)` — [`L1127`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L1127)
- `compute_with_bq(bq_idx, _)` — [`L1115`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L1115)
- `debug_print(msg, *args)` — [`L326`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L326)
- `epilogue()` — [`L1246`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L1246)
- `flash_attention(ql_nope, q_pe, kv_c, k_pe, *, bq_idx, bkv_idx)` — [`L340`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L340)
- `floor_div_on_kv_packing(a, kv_packing)` — [`L49`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L49)
- `get_dtype_bitwidth(dtype)` — [`L65`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L65)
- `get_dtype_packing(dtype)` — [`L69`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L69)
- `get_kv_cache_shape(total_num_pages, page_size, kv_dim, kv_dtype)` — [`L74`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L74)
- `get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx)` — [`L1104`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L1104)
- `get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx)` — [`L1096`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L1096)
- `load_bkv(bkv_sem_idx)` — [`L1041`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L1041)
- `load_bq(bq_sem_idx, *, actual_bq_sz=bq_sz)` — [`L1022`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L1022)
- `load_with_init(ref, init_val)` — [`L364`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L364)
- `loop_body(i, states)` — [`L831`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L831)
- `merge_loop_body(i, vals)` — [`L703`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L703)
- `mla_ragged_paged_attention(ql_nope: jax.Array, q_pe: jax.Array, new_kv_c: jax.Array, new_k_pe: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, cu_kv_lens: jax.Array, distribution: jax.Array, *, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, num_kv_pages_per_block: tuple[int, int, int] | int | None = None, num_queries_per_block: tuple[int, int, int] | int | None = None, vmem_limit_bytes: int | None = None, decode_batch_size: int = 1, debug_mode: bool = False)` — [`L1333`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L1333) — MLA Ragged paged attention that supports mixed prefill and decode.
- `prefetch_next_bkv()` — [`L1137`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L1137)
- `prefetch_next_bq()` — [`L1123`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L1123)
- `prepare_kv_inputs(kv: jax.Array)` — [`L1282`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L1282)
- `prepare_outputs(out, actual_num_q_heads: int, actual_head_dim: int)` — [`L1296`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L1296)
- `prepare_q_inputs(q: jax.Array)` — [`L1254`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L1254)
- `process()` — [`L1080`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L1080)
- `prologue()` — [`L1226`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L1226)
- `run_mla_kernel(ql_nope: jax.Array, q_pe: jax.Array, new_kv_c: jax.Array, new_k_pe: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, cu_kv_lens: jax.Array, start_seq_idx: jax.Array, end_seq_idx: jax.Array, static_q_len: int | None, num_kv_pages_per_block: int, num_queries_per_block: int, batch_size: int = 1, case: MlaCase = MlaCase.MIXED)` — [`L1508`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L1508)
- `start_fetch_bkv(batch_start_seq_idx, bkv_idx, bkv_sem_idx)` — [`L970`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L970)
- `start_fetch_bq(batch_start_seq_idx, bq_idx, bq_sem_idx)` — [`L976`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L976)
- `start_send_bo(batch_start_seq_idx, bo_idx, bo_sem_idx)` — [`L982`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L982)
- `start_update_kv_cache(start_seq_idx, bkv_sem_idx, offsets, update_szs)` — [`L999`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L999)
- `static_validate_inputs(ql_nope: jax.Array, q_pe: jax.Array, new_kv_c: jax.Array, new_k_pe: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, cu_kv_lens: jax.Array, distribution: jax.Array, *, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, num_kv_pages_per_blocks: tuple[int, int, int] | None = None, num_queries_per_blocks: tuple[int, int, int] | None = None, vmem_limit_bytes: int | None = None, decode_batch_size: int = 1, debug_mode: bool = False)` — [`L113`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L113) — Validate inputs to the MLA RPA kernel statically.
- `update_cur_bkv_to_cache()` — [`L1156`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L1156)
- `wait_cur_bq()` — [`L1143`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L1143)
- `wait_fetch_bkv(batch_start_seq_idx, bkv_idx, bkv_sem_idx)` — [`L973`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L973)
- `wait_fetch_bq(batch_start_seq_idx, bq_idx, bq_sem_idx)` — [`L979`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L979)
- `wait_send_bo(bo_sem_idx)` — [`L987`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L987)
- `wait_update_kv_cache(bkv_sem_idx)` — [`L1011`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L1011)

## Module values
- `DEFAULT_MASK_VALUE` — [`L37`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L37)
- `DEFAULT_VMEM_LIMIT_BYTES` — [`L39`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/mla/v2/kernel.py#L39)

