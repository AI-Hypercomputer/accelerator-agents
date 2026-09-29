---
title: 'Module: tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.experimental.mla.pallas_mosaic_tpu_kernel`/
symbols:
  _mla_ragged_paged_attention_kernel.debug_print: _mla_ragged_paged_attention_kernel().debug_print().
  _mla_ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv: _mla_ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().
  mla_ragged_paged_attention: mla_ragged_paged_attention().
  _mla_ragged_paged_attention_kernel._async_copy: _mla_ragged_paged_attention_kernel()._async_copy().
  _mla_ragged_paged_attention_kernel._update_transposed_kv_cache: _mla_ragged_paged_attention_kernel()._update_transposed_kv_cache().
  unsigned_floor_div: unsigned_floor_div().
  _mla_ragged_paged_attention_kernel._fetch_transposed_bkv: _mla_ragged_paged_attention_kernel()._fetch_transposed_bkv().
  unsigned_align_to: unsigned_align_to().
  mla_ragged_paged_attention.run_mla_kernel: mla_ragged_paged_attention().run_mla_kernel().
  _mla_ragged_paged_attention_kernel._pack_new_transposed_kv._update: _mla_ragged_paged_attention_kernel()._pack_new_transposed_kv()._update().
  _mla_ragged_paged_attention_kernel._update_kv_cache: _mla_ragged_paged_attention_kernel()._update_kv_cache().
  unsigned_mod: unsigned_mod().
  unsigned_cdiv: unsigned_cdiv().
  MlaCase.symbol: MlaCase#symbol().
  _mla_ragged_paged_attention_kernel: _mla_ragged_paged_attention_kernel().
  _mla_ragged_paged_attention_kernel._fetch_bkv: _mla_ragged_paged_attention_kernel()._fetch_bkv().
  _mla_ragged_paged_attention_kernel.process.compute_with_bq: _mla_ragged_paged_attention_kernel().process().compute_with_bq().
  _mla_ragged_paged_attention_kernel._pack_new_kv._update: _mla_ragged_paged_attention_kernel()._pack_new_kv()._update().
  prepare_q_inputs: prepare_q_inputs().
  MlaCase: MlaCase#
  get_dtype_packing: get_dtype_packing().
  _mla_ragged_paged_attention_kernel.batch_flash_attention: _mla_ragged_paged_attention_kernel().batch_flash_attention().
  _mla_ragged_paged_attention_kernel.flash_attention_step1_qk_softmax: _mla_ragged_paged_attention_kernel().flash_attention_step1_qk_softmax().
  _mla_ragged_paged_attention_kernel._pack_new_transposed_kv._update.update_body: _mla_ragged_paged_attention_kernel()._pack_new_transposed_kv()._update().update_body().
  _mla_ragged_paged_attention_kernel.process: _mla_ragged_paged_attention_kernel().process().
  _mla_ragged_paged_attention_kernel.wait_update_kv_cache: _mla_ragged_paged_attention_kernel().wait_update_kv_cache().
  _mla_ragged_paged_attention_kernel.broadcast_minor: _mla_ragged_paged_attention_kernel().broadcast_minor().
  _mla_ragged_paged_attention_kernel.flash_attention_step2_pv: _mla_ragged_paged_attention_kernel().flash_attention_step2_pv().
  prepare_kv_inputs: prepare_kv_inputs().
  _mla_ragged_paged_attention_kernel._update_transposed_kv_cache.loop_body: _mla_ragged_paged_attention_kernel()._update_transposed_kv_cache().loop_body().
  _mla_ragged_paged_attention_kernel.start_update_kv_cache: _mla_ragged_paged_attention_kernel().start_update_kv_cache().
  get_kv_cache_shape: get_kv_cache_shape().
  static_validate_inputs: static_validate_inputs().
  _mla_ragged_paged_attention_kernel.prologue: _mla_ragged_paged_attention_kernel().prologue().
  _mla_ragged_paged_attention_kernel.epilogue: _mla_ragged_paged_attention_kernel().epilogue().
  _mla_ragged_paged_attention_kernel._pack_new_transposed_kv._update.copy_partial: _mla_ragged_paged_attention_kernel()._pack_new_transposed_kv()._update().copy_partial().
  _mla_ragged_paged_attention_kernel._fetch_bq: _mla_ragged_paged_attention_kernel()._fetch_bq().
  _mla_ragged_paged_attention_kernel._send_bo: _mla_ragged_paged_attention_kernel()._send_bo().
  _mla_ragged_paged_attention_kernel.start_fetch_bq: _mla_ragged_paged_attention_kernel().start_fetch_bq().
  _mla_ragged_paged_attention_kernel.wait_send_bo: _mla_ragged_paged_attention_kernel().wait_send_bo().
  align_to: align_to().
  _mla_ragged_paged_attention_kernel._pack_new_transposed_kv: _mla_ragged_paged_attention_kernel()._pack_new_transposed_kv().
  _mla_ragged_paged_attention_kernel._pack_new_kv: _mla_ragged_paged_attention_kernel()._pack_new_kv().
  _mla_ragged_paged_attention_kernel._update_kv_cache.loop_body: _mla_ragged_paged_attention_kernel()._update_kv_cache().loop_body().
  _mla_ragged_paged_attention_kernel.wait_fetch_bq: _mla_ragged_paged_attention_kernel().wait_fetch_bq().
  _mla_ragged_paged_attention_kernel.start_send_bo: _mla_ragged_paged_attention_kernel().start_send_bo().
  MlaCase.MIXED: MlaCase#MIXED.
  _mla_ragged_paged_attention_kernel.batch_flash_attention.load_with_init: _mla_ragged_paged_attention_kernel().batch_flash_attention().load_with_init().
  _mla_ragged_paged_attention_kernel.start_fetch_bkv: _mla_ragged_paged_attention_kernel().start_fetch_bkv().
  _mla_ragged_paged_attention_kernel._fetch_bq._copy: _mla_ragged_paged_attention_kernel()._fetch_bq()._copy().
  _mla_ragged_paged_attention_kernel._send_bo._copy: _mla_ragged_paged_attention_kernel()._send_bo()._copy().
  _mla_ragged_paged_attention_kernel.process.compute_with_bq.prefetch_next_bq: _mla_ragged_paged_attention_kernel().process().compute_with_bq().prefetch_next_bq().
  _mla_ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv.prefetch_next_bkv: _mla_ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().prefetch_next_bkv().
  _mla_ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv.wait_cur_bq: _mla_ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().wait_cur_bq().
  _mla_ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv.update_cur_bkv_to_cache: _mla_ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().update_cur_bkv_to_cache().
  get_dtype_bitwidth: get_dtype_bitwidth().
  MlaCase.DECODE: MlaCase#DECODE.
  MlaCase.BATCHED_DECODE: MlaCase#BATCHED_DECODE.
  _mla_ragged_paged_attention_kernel.flash_attention_step1_qk_softmax.load_with_init: _mla_ragged_paged_attention_kernel().flash_attention_step1_qk_softmax().load_with_init().
  prepare_kv_inputs_for_transposed_kv_cache: prepare_kv_inputs_for_transposed_kv_cache().
  MlaCase.PREFILL: MlaCase#PREFILL.
  _mla_ragged_paged_attention_kernel.flash_attention_step2_pv.load_with_init: _mla_ragged_paged_attention_kernel().flash_attention_step2_pv().load_with_init().
  _mla_ragged_paged_attention_kernel._pack_new_kv._update.merge_loop_body: _mla_ragged_paged_attention_kernel()._pack_new_kv()._update().merge_loop_body().
  _mla_ragged_paged_attention_kernel.wait_fetch_bkv: _mla_ragged_paged_attention_kernel().wait_fetch_bkv().
  _mla_ragged_paged_attention_kernel.load_batch_bq: _mla_ragged_paged_attention_kernel().load_batch_bq().
  _mla_ragged_paged_attention_kernel.load_bq: _mla_ragged_paged_attention_kernel().load_bq().
  _mla_ragged_paged_attention_kernel.load_bkv: _mla_ragged_paged_attention_kernel().load_bkv().
  _mla_ragged_paged_attention_kernel.load_transposed_bkv: _mla_ragged_paged_attention_kernel().load_transposed_bkv().
  _mla_ragged_paged_attention_kernel.load_batch_bkv: _mla_ragged_paged_attention_kernel().load_batch_bkv().
  _mla_ragged_paged_attention_kernel.process.get_next_bq_ids: _mla_ragged_paged_attention_kernel().process().get_next_bq_ids().
  _mla_ragged_paged_attention_kernel.process.get_next_bkv_ids: _mla_ragged_paged_attention_kernel().process().get_next_bkv_ids().
  prepare_outputs: prepare_outputs().
---
# Module: [`tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py)

## Classes
### `MlaCase`  ·  implements/extends Enum
- def: [`tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py:90`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L90) — documented in [tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel](../../../../../../concepts/tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel.md)
- doc: Represents the different cases for MLA.
- signature: `class MlaCase(Enum):`
- members:
  - `symbol(self)` — [`L104`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L104) — documented in [tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel](../../../../../../concepts/tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel.md)
  - `BATCHED_DECODE` — [`L101`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L101) — documented in [tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel](../../../../../../concepts/tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel.md)
  - `DECODE` — [`L98`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L98) — documented in [tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel](../../../../../../concepts/tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel.md)
  - `MIXED` — [`L100`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L100) — documented in [tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel](../../../../../../concepts/tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel.md)
  - `PREFILL` — [`L99`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L99) — documented in [tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel](../../../../../../concepts/tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel.md)
- used by: [`mla_ragged_paged_attention`](pallas_mosaic_tpu_kernel.md#mla_ragged_paged_attention), [`run_mla_kernel`](pallas_mosaic_tpu_kernel.md#mla_ragged_paged_attention.run_mla_kernel)

## Functions
- `_async_copy(src, dst, sem, wait)` — [`L570`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L570)
- `_copy(q_len_start=q_len_start, sz=sz, bq_nope_vmem_ref=bq_nope_vmem_ref, bq_rope_vmem_ref=bq_rope_vmem_ref, sem=sem)` — [`L1561`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1561)
- `_copy(vmem_ref=vmem_ref, sz=sz, q_len_start=q_len_start, sem=sem)` — [`L1604`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1604)
- `_fetch_bkv(batch_start_seq_idx, bkv_idx, bkv_sem_idx, *, wait=False)` — [`L783`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L783) — documented in [tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel](../../../../../../concepts/tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel.md)
- `_fetch_bq(batch_start_seq_idx, bq_idx, bq_sem_idx, *, wait=False)` — [`L1538`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1538) — documented in [tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel](../../../../../../concepts/tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel.md)
- `_fetch_transposed_bkv(batch_start_seq_idx, bkv_idx, bkv_sem_idx, *, wait=False)` — [`L580`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L580) — documented in [tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel](../../../../../../concepts/tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel.md)
- `_mla_ragged_paged_attention_kernel(kv_lens_ref, page_indices_ref, cu_q_lens_ref, start_end_seq_idx_ref, sem_ids_ref, bo_ids_ref, bkv_update_ids_ref, ql_nope_hbm_ref, q_pe_hbm_ref, new_kv_c_hbm_ref, new_k_pe_hbm_ref, cache_kv_hbm_ref, o_hbm_ref, updated_cache_kv_hbm_ref, bkvc_x2_ref, bkpe_x2_ref, bq_nope_x2_ref, bq_rope_x2_ref, bo_x2_ref, sems, l_ref, m_ref, acc_ref, *, static_q_len: int, sm_scale: float, mask_value: float, s_dtype: jnp.dtype, transpose_kv_cache: bool, two_step_flash_attention: bool, p_same_dtype_as_v: bool, sliding_window: int | None = None, soft_cap: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, bkv_p, bq_sz, batch_size: int = 1, debug_mode: bool = False)` — [`L257`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L257)
- `_pack_new_kv(bkv_sem_idx, offsets, update_szs)` — [`L1134`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1134) — Packs newly computed KVs into the correct sub-word alignment in VMEM. — documented in [tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel](../../../../../../concepts/tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel.md)
- `_pack_new_transposed_kv(bkv_sem_idx, bkv_idx, offsets, update_szs)` — [`L998`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L998) — Packs newly computed KVs into the correct sub-word alignment in VMEM. — documented in [tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel](../../../../../../concepts/tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel.md)
- `_send_bo(batch_start_seq_idx, bo_idx, bo_sem_idx, *, wait=False)` — [`L1582`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1582)
- `_update(b=b, update_sz=update_sz)` — [`L1014`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1014)
- `_update(b=b, offset=offset, update_sz=update_sz)` — [`L1165`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1165)
- `_update_kv_cache(batch_start_seq_idx, b, bkv_sem_idx, offset, update_sz, *, wait=False)` — [`L1429`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1429)
- `_update_transposed_kv_cache(batch_start_seq_idx, b, bkv_sem_idx, offset, update_sz, *, wait=False)` — [`L1304`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1304)
- `align_to(a, b)` — [`L62`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L62)
- `batch_flash_attention(ql_nope, q_pe, kv_c, k_pe, *, bq_idx, bkv_idx)` — [`L363`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L363)
- `broadcast_minor(src, shape)` — [`L1757`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1757)
- `compute_with_bkv(bkv_idx, _)` — [`L1834`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1834)
- `compute_with_bq(bq_idx, _)` — [`L1822`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1822)
- `copy_partial(new_kv_idx, new_kv_start, bkv_idx, bkv_start)` — [`L1067`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1067)
- `debug_print(msg, *args)` — [`L348`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L348)
- `epilogue()` — [`L2074`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L2074)
- `flash_attention_step1_qk_softmax(ql_nope, q_pe, kv_c, k_pe, *, kv_len, q_len, bq_idx, bkv_idx, head_l_ref, head_m_ref)` — [`L464`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L464)
- `flash_attention_step2_pv(p, v, exp_m_diff, bkv_idx, head_acc_ref)` — [`L544`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L544)
- `get_dtype_bitwidth(dtype)` — [`L66`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L66)
- `get_dtype_packing(dtype)` — [`L70`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L70)
- `get_kv_cache_shape(total_num_pages, page_size, kv_dim, kv_dtype)` — [`L75`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L75)
- `get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx)` — [`L1811`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1811)
- `get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx)` — [`L1803`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1803)
- `load_batch_bkv(load_bkv_fn, bkv_sem_idx)` — [`L1748`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1748)
- `load_batch_bq(bq_sem_idx, *, actual_bq_sz=bq_sz)` — [`L1686`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1686)
- `load_bkv(batch_item_idx, bkv_sem_idx)` — [`L1728`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1728)
- `load_bq(batch_item_idx, bq_sem_idx, *, actual_bq_sz=bq_sz)` — [`L1707`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1707)
- `load_transposed_bkv(batch_item_idx, bkv_sem_idx)` — [`L1743`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1743)
- `load_with_init(ref, init_val)` — [`L392`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L392)
- `load_with_init(ref, init_val)` — [`L493`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L493)
- `load_with_init(ref, init_val)` — [`L551`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L551)
- `loop_body(i, states)` — [`L1353`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1353)
- `loop_body(i, states)` — [`L1471`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1471)
- `merge_loop_body(i, vals)` — [`L1205`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1205)
- `mla_ragged_paged_attention(ql_nope: jax.Array, q_pe: jax.Array, new_kv_c: jax.Array, new_k_pe: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, *, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, num_kv_pages_per_block: tuple[int, int, int] | int | None = None, num_queries_per_block: tuple[int, int, int] | int | None = None, vmem_limit_bytes: int | None = None, decode_batch_size: int = 1, s_dtype: jnp.dtype = jnp.bfloat16, transpose_kv_cache: bool = False, two_step_flash_attention: bool = True, p_same_dtype_as_v: bool = True, debug_mode: bool = False)` — [`L2183`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L2183) — MLA Ragged paged attention that supports mixed prefill and decode. — documented in [tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel](../../../../../../concepts/tokamax-_src-ops-experimental-mla-pallas_mosaic_tpu_kernel.md)
- `prefetch_next_bkv()` — [`L1844`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1844)
- `prefetch_next_bq()` — [`L1830`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1830)
- `prepare_kv_inputs(kv: jax.Array)` — [`L2110`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L2110)
- `prepare_kv_inputs_for_transposed_kv_cache(kv: jax.Array)` — [`L2126`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L2126)
- `prepare_outputs(out, actual_num_q_heads: int, actual_head_dim: int)` — [`L2142`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L2142)
- `prepare_q_inputs(q: jax.Array)` — [`L2082`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L2082)
- `process()` — [`L1787`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1787)
- `prologue()` — [`L2041`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L2041)
- `run_mla_kernel(ql_nope: jax.Array, q_pe: jax.Array, new_kv_c: jax.Array, new_k_pe: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, start_seq_idx: jax.Array, end_seq_idx: jax.Array, static_q_len: int | None, num_kv_pages_per_block: int, num_queries_per_block: int, s_dtype: jnp.dtype, p_same_dtype_as_v: bool, batch_size: int = 1, case: MlaCase = MlaCase.MIXED)` — [`L2325`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L2325)
- `start_fetch_bkv(batch_start_seq_idx, bkv_idx, bkv_sem_idx)` — [`L1618`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1618)
- `start_fetch_bq(batch_start_seq_idx, bq_idx, bq_sem_idx)` — [`L1624`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1624)
- `start_send_bo(batch_start_seq_idx, bo_idx, bo_sem_idx)` — [`L1630`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1630)
- `start_update_kv_cache(start_seq_idx, bkv_sem_idx, offsets, update_szs)` — [`L1649`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1649)
- `static_validate_inputs(ql_nope: jax.Array, q_pe: jax.Array, new_kv_c: jax.Array, new_k_pe: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, *, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, decode_batch_size: int = 1, transpose_kv_cache: bool = False, debug_mode: bool = False)` — [`L114`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L114) — Validate inputs to the MLA RPA kernel statically.
- `unsigned_align_to(a, b)` — [`L52`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L52)
- `unsigned_cdiv(a, b)` — [`L36`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L36)
- `unsigned_floor_div(a, b)` — [`L44`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L44)
- `unsigned_mod(a, b)` — [`L28`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L28)
- `update_body(idx, state)` — [`L1108`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1108)
- `update_cur_bkv_to_cache()` — [`L1870`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1870)
- `wait_cur_bq()` — [`L1850`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1850)
- `wait_fetch_bkv(batch_start_seq_idx, bkv_idx, bkv_sem_idx)` — [`L1621`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1621)
- `wait_fetch_bq(batch_start_seq_idx, bq_idx, bq_sem_idx)` — [`L1627`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1627)
- `wait_send_bo(bo_sem_idx)` — [`L1635`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1635)
- `wait_update_kv_cache(bkv_sem_idx)` — [`L1668`](../../../../../../../../../raw/code/tokamax/tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py#L1668)

