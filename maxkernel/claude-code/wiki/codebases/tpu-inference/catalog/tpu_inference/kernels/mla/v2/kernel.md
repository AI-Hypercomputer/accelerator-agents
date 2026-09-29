---
title: 'Module: tpu_inference/kernels/mla/v2/kernel.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/mla/v2/kernel.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.mla.v2.kernel`/
symbols:
  _mla_ragged_paged_attention_kernel.debug_print: _mla_ragged_paged_attention_kernel().debug_print().
  _mla_ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv: _mla_ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().
  mla_ragged_paged_attention: mla_ragged_paged_attention().
  _mla_ragged_paged_attention_kernel._async_copy: _mla_ragged_paged_attention_kernel()._async_copy().
  _mla_ragged_paged_attention_kernel._update_transposed_kv_cache: _mla_ragged_paged_attention_kernel()._update_transposed_kv_cache().
  mla_ragged_paged_attention.run_mla_kernel: mla_ragged_paged_attention().run_mla_kernel().
  _mla_ragged_paged_attention_kernel: _mla_ragged_paged_attention_kernel().
  _mla_ragged_paged_attention_kernel._fetch_transposed_bkv: _mla_ragged_paged_attention_kernel()._fetch_transposed_bkv().
  _mla_ragged_paged_attention_kernel._pack_new_transposed_kv._update: _mla_ragged_paged_attention_kernel()._pack_new_transposed_kv()._update().
  _mla_ragged_paged_attention_kernel._update_kv_cache: _mla_ragged_paged_attention_kernel()._update_kv_cache().
  unsigned_floor_div: unsigned_floor_div().
  unsigned_align_to: unsigned_align_to().
  MlaCase.symbol: MlaCase#symbol().
  _mla_ragged_paged_attention_kernel._fetch_bkv: _mla_ragged_paged_attention_kernel()._fetch_bkv().
  _mla_ragged_paged_attention_kernel.process.compute_with_bq: _mla_ragged_paged_attention_kernel().process().compute_with_bq().
  get_dtype_packing: get_dtype_packing().
  unsigned_cdiv: unsigned_cdiv().
  align_to: align_to().
  prepare_q_nope_inputs: prepare_q_nope_inputs().
  prepare_outputs: prepare_outputs().
  MlaCase: MlaCase#
  unsigned_mod: unsigned_mod().
  static_validate_inputs: static_validate_inputs().
  _mla_ragged_paged_attention_kernel.batch_flash_attention: _mla_ragged_paged_attention_kernel().batch_flash_attention().
  _mla_ragged_paged_attention_kernel.flash_attention_step1_qk_softmax: _mla_ragged_paged_attention_kernel().flash_attention_step1_qk_softmax().
  _mla_ragged_paged_attention_kernel._pack_new_transposed_kv._update.update_body: _mla_ragged_paged_attention_kernel()._pack_new_transposed_kv()._update().update_body().
  _mla_ragged_paged_attention_kernel.process: _mla_ragged_paged_attention_kernel().process().
  _mla_ragged_paged_attention_kernel.wait_update_kv_cache: _mla_ragged_paged_attention_kernel().wait_update_kv_cache().
  _mla_ragged_paged_attention_kernel.broadcast_minor: _mla_ragged_paged_attention_kernel().broadcast_minor().
  _mla_ragged_paged_attention_kernel.flash_attention_step2_pv: _mla_ragged_paged_attention_kernel().flash_attention_step2_pv().
  get_kv_cache_shape: get_kv_cache_shape().
  _mla_ragged_paged_attention_kernel._update_transposed_kv_cache.loop_body: _mla_ragged_paged_attention_kernel()._update_transposed_kv_cache().loop_body().
  _mla_ragged_paged_attention_kernel.start_update_kv_cache: _mla_ragged_paged_attention_kernel().start_update_kv_cache().
  prepare_q_inputs: prepare_q_inputs().
  _mla_ragged_paged_attention_kernel.prologue: _mla_ragged_paged_attention_kernel().prologue().
  _mla_ragged_paged_attention_kernel.epilogue: _mla_ragged_paged_attention_kernel().epilogue().
  logger: logger.
  _mla_ragged_paged_attention_kernel._pack_new_transposed_kv._update.copy_partial: _mla_ragged_paged_attention_kernel()._pack_new_transposed_kv()._update().copy_partial().
  _mla_ragged_paged_attention_kernel._fetch_bq: _mla_ragged_paged_attention_kernel()._fetch_bq().
  _mla_ragged_paged_attention_kernel._send_bo: _mla_ragged_paged_attention_kernel()._send_bo().
  _mla_ragged_paged_attention_kernel.start_fetch_bq: _mla_ragged_paged_attention_kernel().start_fetch_bq().
  _mla_ragged_paged_attention_kernel.wait_send_bo: _mla_ragged_paged_attention_kernel().wait_send_bo().
  prepare_kv_inputs: prepare_kv_inputs().
  _mla_ragged_paged_attention_kernel._pack_new_transposed_kv: _mla_ragged_paged_attention_kernel()._pack_new_transposed_kv().
  _mla_ragged_paged_attention_kernel._pack_new_kv: _mla_ragged_paged_attention_kernel()._pack_new_kv().
  _mla_ragged_paged_attention_kernel._update_kv_cache.loop_body: _mla_ragged_paged_attention_kernel()._update_kv_cache().loop_body().
  _mla_ragged_paged_attention_kernel.wait_fetch_bq: _mla_ragged_paged_attention_kernel().wait_fetch_bq().
  _mla_ragged_paged_attention_kernel.start_send_bo: _mla_ragged_paged_attention_kernel().start_send_bo().
  MlaCase.MIXED: MlaCase#MIXED.
  _mla_ragged_paged_attention_kernel.batch_flash_attention.load_with_init: _mla_ragged_paged_attention_kernel().batch_flash_attention().load_with_init().
  _mla_ragged_paged_attention_kernel.start_fetch_bkv: _mla_ragged_paged_attention_kernel().start_fetch_bkv().
  _mla_ragged_paged_attention_kernel._pack_new_kv._update: _mla_ragged_paged_attention_kernel()._pack_new_kv()._update().
  _mla_ragged_paged_attention_kernel._fetch_bq._copy: _mla_ragged_paged_attention_kernel()._fetch_bq()._copy().
  _mla_ragged_paged_attention_kernel._send_bo._copy: _mla_ragged_paged_attention_kernel()._send_bo()._copy().
  _mla_ragged_paged_attention_kernel.process.compute_with_bq.prefetch_next_bq: _mla_ragged_paged_attention_kernel().process().compute_with_bq().prefetch_next_bq().
  _mla_ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv.prefetch_next_bkv: _mla_ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().prefetch_next_bkv().
  _mla_ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv.wait_cur_bq: _mla_ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().wait_cur_bq().
  _mla_ragged_paged_attention_kernel.process.compute_with_bq.compute_with_bkv.update_cur_bkv_to_cache: _mla_ragged_paged_attention_kernel().process().compute_with_bq().compute_with_bkv().update_cur_bkv_to_cache().
  MlaCase.DECODE: MlaCase#DECODE.
  MlaCase.PREFILL: MlaCase#PREFILL.
  MlaCase.BATCHED_DECODE: MlaCase#BATCHED_DECODE.
  _mla_ragged_paged_attention_kernel.flash_attention_step1_qk_softmax.load_with_init: _mla_ragged_paged_attention_kernel().flash_attention_step1_qk_softmax().load_with_init().
  prepare_kv_inputs_for_transposed_kv_cache: prepare_kv_inputs_for_transposed_kv_cache().
  _XPOSE_N_TILE_SIZE: _XPOSE_N_TILE_SIZE.
  get_dtype_bitwidth: get_dtype_bitwidth().
  _mla_ragged_paged_attention_kernel.flash_attention_step2_pv.load_with_init: _mla_ragged_paged_attention_kernel().flash_attention_step2_pv().load_with_init().
  _mla_ragged_paged_attention_kernel.wait_fetch_bkv: _mla_ragged_paged_attention_kernel().wait_fetch_bkv().
  _mla_ragged_paged_attention_kernel.load_batch_bq: _mla_ragged_paged_attention_kernel().load_batch_bq().
  _mla_ragged_paged_attention_kernel.load_bq: _mla_ragged_paged_attention_kernel().load_bq().
  _mla_ragged_paged_attention_kernel.load_bkv: _mla_ragged_paged_attention_kernel().load_bkv().
  _mla_ragged_paged_attention_kernel.load_transposed_bkv: _mla_ragged_paged_attention_kernel().load_transposed_bkv().
  _mla_ragged_paged_attention_kernel.load_batch_bkv: _mla_ragged_paged_attention_kernel().load_batch_bkv().
  _mla_ragged_paged_attention_kernel.process.get_next_bq_ids: _mla_ragged_paged_attention_kernel().process().get_next_bq_ids().
  _mla_ragged_paged_attention_kernel.process.get_next_bkv_ids: _mla_ragged_paged_attention_kernel().process().get_next_bkv_ids().
---
# Module: [`tpu_inference/kernels/mla/v2/kernel.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py)

## Classes
### `MlaCase`  ·  implements/extends Enum
- def: [`tpu_inference/kernels/mla/v2/kernel.py:109`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L109) — documented in [tpu_inference-kernels-mla-v2-kernel](../../../../../concepts/tpu_inference-kernels-mla-v2-kernel.md)
- doc: Represents the different cases for MLA.
- signature: `class MlaCase(Enum):`
- members:
  - `symbol(self)` — [`L123`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L123)
  - `BATCHED_DECODE` — [`L120`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L120) — documented in [tpu_inference-kernels-mla-v2-kernel](../../../../../concepts/tpu_inference-kernels-mla-v2-kernel.md)
  - `DECODE` — [`L117`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L117) — documented in [tpu_inference-kernels-mla-v2-kernel](../../../../../concepts/tpu_inference-kernels-mla-v2-kernel.md)
  - `MIXED` — [`L119`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L119) — documented in [tpu_inference-kernels-mla-v2-kernel](../../../../../concepts/tpu_inference-kernels-mla-v2-kernel.md)
  - `PREFILL` — [`L118`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L118) — documented in [tpu_inference-kernels-mla-v2-kernel](../../../../../concepts/tpu_inference-kernels-mla-v2-kernel.md)
- used by: [`mla_ragged_paged_attention`](kernel.md#mla_ragged_paged_attention), [`run_mla_kernel`](kernel.md#mla_ragged_paged_attention.run_mla_kernel)

## Functions
- `_async_copy(src, dst, sem, wait)` — [`L657`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L657) — documented in [tpu_inference-kernels-mla-v2-kernel](../../../../../concepts/tpu_inference-kernels-mla-v2-kernel.md)
- `_copy(q_len_start=q_len_start, sz=sz, bq_nope_vmem_ref=bq_nope_vmem_ref, bq_rope_vmem_ref=bq_rope_vmem_ref, sem=sem)` — [`L1497`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1497)
- `_copy(vmem_ref=vmem_ref, sz=sz, q_len_start=q_len_start, sem=sem)` — [`L1538`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1538)
- `_fetch_bkv(batch_start_seq_idx, bkv_idx, bkv_sem_idx, *, wait=False)` — [`L871`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L871)
- `_fetch_bq(batch_start_seq_idx, bq_idx, bq_sem_idx, *, wait=False)` — [`L1474`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1474)
- `_fetch_transposed_bkv(batch_start_seq_idx, bkv_idx, bkv_sem_idx, *, wait=False)` — [`L667`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L667)
- `_mla_ragged_paged_attention_kernel(kv_lens_ref, page_indices_ref, cu_q_lens_ref, start_end_seq_idx_ref, sem_ids_ref, bo_ids_ref, bkv_update_ids_ref, ql_nope_hbm_ref, q_pe_hbm_ref, new_kv_c_hbm_ref, new_k_pe_hbm_ref, cache_kv_hbm_ref, o_hbm_ref, updated_cache_kv_hbm_ref, bkvc_x2_ref, bkpe_x2_ref, bq_nope_x2_ref, bq_rope_x2_ref, bo_x2_ref, sems, l_ref, m_ref, acc_ref, *, static_q_len: int, sm_scale: float, mask_value: float, s_dtype: jnp.dtype, transpose_kv_cache: bool, two_step_flash_attention: bool, p_same_dtype_as_v: bool, sliding_window: int | None = None, soft_cap: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, bkv_p, bq_sz, batch_size: int = 1, debug_mode: bool = False)` — [`L286`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L286)
- `_pack_new_kv(bkv_sem_idx, offsets, update_szs)` — [`L1210`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1210) — Packs newly computed KVs into the correct sub-word alignment in VMEM.
- `_pack_new_transposed_kv(bkv_sem_idx, bkv_idx, offsets, update_szs)` — [`L1082`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1082) — Packs newly computed KVs into the correct sub-word alignment in VMEM.
- `_send_bo(batch_start_seq_idx, bo_idx, bo_sem_idx, *, wait=False)` — [`L1516`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1516)
- `_update(b=b, update_sz=update_sz)` — [`L1098`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1098)
- `_update(b=b, offset=offset, update_sz=update_sz)` — [`L1242`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1242)
- `_update_kv_cache(batch_start_seq_idx, b, bkv_sem_idx, offset, update_sz, *, wait=False)` — [`L1366`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1366)
- `_update_transposed_kv_cache(batch_start_seq_idx, b, bkv_sem_idx, offset, update_sz, *, wait=False)` — [`L1255`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1255)
- `align_to(a, b)` — [`L70`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L70)
- `batch_flash_attention(ql_nope, q_pe, kv_c, k_pe, *, bq_idx, bkv_idx)` — [`L394`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L394)
- `broadcast_minor(src, shape)` — [`L1692`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1692)
- `compute_with_bkv(bkv_idx, _)` — [`L1764`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1764) — documented in [tpu_inference-kernels-mla-v2-kernel](../../../../../concepts/tpu_inference-kernels-mla-v2-kernel.md)
- `compute_with_bq(bq_idx, _)` — [`L1753`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1753) — documented in [tpu_inference-kernels-mla-v2-kernel](../../../../../concepts/tpu_inference-kernels-mla-v2-kernel.md)
- `copy_partial(new_kv_idx, new_kv_start, bkv_idx, bkv_start)` — [`L1148`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1148)
- `debug_print(msg, *args)` — [`L380`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L380)
- `epilogue()` — [`L1994`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1994)
- `flash_attention_step1_qk_softmax(ql_nope, q_pe, kv_c, k_pe, *, kv_len, q_len, bq_idx, bkv_idx, head_l_ref, head_m_ref)` — [`L490`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L490) — documented in [tpu_inference-kernels-mla-v2-kernel](../../../../../concepts/tpu_inference-kernels-mla-v2-kernel.md)
- `flash_attention_step2_pv(p, v, exp_m_diff, bkv_idx, head_acc_ref)` — [`L628`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L628) — documented in [tpu_inference-kernels-mla-v2-kernel](../../../../../concepts/tpu_inference-kernels-mla-v2-kernel.md)
- `get_dtype_bitwidth(dtype)` — [`L74`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L74)
- `get_dtype_packing(dtype)` — [`L78`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L78)
- `get_kv_cache_shape(total_num_pages, page_size, kv_dim, kv_dtype, kv_packing: int | None = None, transpose_kv_cache: bool = False)` — [`L83`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L83)
- `get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx)` — [`L1741`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1741)
- `get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx)` — [`L1732`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1732)
- `load_batch_bkv(load_bkv_fn, bkv_sem_idx)` — [`L1683`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1683)
- `load_batch_bq(bq_sem_idx, *, actual_bq_sz=bq_sz)` — [`L1627`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1627)
- `load_bkv(batch_item_idx, bkv_sem_idx)` — [`L1661`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1661)
- `load_bq(batch_item_idx, bq_sem_idx, *, actual_bq_sz=bq_sz)` — [`L1644`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1644)
- `load_transposed_bkv(batch_item_idx, bkv_sem_idx)` — [`L1678`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1678)
- `load_with_init(ref, init_val)` — [`L423`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L423)
- `load_with_init(ref, init_val)` — [`L519`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L519)
- `load_with_init(ref, init_val)` — [`L636`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L636)
- `loop_body(i, states)` — [`L1302`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1302)
- `loop_body(i, states)` — [`L1405`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1405)
- `mla_ragged_paged_attention(ql_nope: jax.Array, q_pe: jax.Array, new_kv_c: jax.Array, new_k_pe: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, *, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, num_kv_pages_per_block: tuple[int, int, int] | int | None = None, num_queries_per_block: tuple[int, int, int] | int | None = None, vmem_limit_bytes: int | None = None, decode_batch_size: int = 1, s_dtype: jnp.dtype = jnp.bfloat16, transpose_kv_cache: bool = False, two_step_flash_attention: bool = True, p_same_dtype_as_v: bool = True, debug_mode: bool = False)` — [`L2138`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L2138) — MLA Ragged paged attention that supports mixed prefill and decode. — documented in [tpu_inference-kernels-mla-v2-kernel](../../../../../concepts/tpu_inference-kernels-mla-v2-kernel.md)
- `prefetch_next_bkv()` — [`L1773`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1773)
- `prefetch_next_bq()` — [`L1760`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1760)
- `prepare_kv_inputs(kv: jax.Array, kv_packing: int)` — [`L2060`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L2060) — documented in [tpu_inference-kernels-mla-v2-kernel](../../../../../concepts/tpu_inference-kernels-mla-v2-kernel.md)
- `prepare_kv_inputs_for_transposed_kv_cache(kv: jax.Array)` — [`L2074`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L2074) — documented in [tpu_inference-kernels-mla-v2-kernel](../../../../../concepts/tpu_inference-kernels-mla-v2-kernel.md)
- `prepare_outputs(out, actual_num_q_heads: int, actual_max_num_tokens: int, actual_head_dim: int)` — [`L2090`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L2090) — documented in [tpu_inference-kernels-mla-v2-kernel](../../../../../concepts/tpu_inference-kernels-mla-v2-kernel.md)
- `prepare_q_inputs(q: jax.Array)` — [`L2002`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L2002) — documented in [tpu_inference-kernels-mla-v2-kernel](../../../../../concepts/tpu_inference-kernels-mla-v2-kernel.md)
- `prepare_q_nope_inputs(q: jax.Array)` — [`L2020`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L2020) — Packs and physically transposes q_nope to the layout expected by the MLA kernel. — documented in [tpu_inference-kernels-mla-v2-kernel](../../../../../concepts/tpu_inference-kernels-mla-v2-kernel.md)
- `process()` — [`L1716`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1716) — documented in [tpu_inference-kernels-mla-v2-kernel](../../../../../concepts/tpu_inference-kernels-mla-v2-kernel.md)
- `prologue()` — [`L1961`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1961)
- `run_mla_kernel(ql_nope: jax.Array, q_pe: jax.Array, new_kv_c: jax.Array, new_k_pe: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, start_seq_idx: jax.Array, end_seq_idx: jax.Array, static_q_len: int | None, num_kv_pages_per_block: int, num_queries_per_block: int, s_dtype: jnp.dtype, p_same_dtype_as_v: bool, batch_size: int = 1, case: MlaCase = MlaCase.MIXED)` — [`L2297`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L2297) — documented in [tpu_inference-kernels-mla-v2-kernel](../../../../../concepts/tpu_inference-kernels-mla-v2-kernel.md)
- `start_fetch_bkv(batch_start_seq_idx, bkv_idx, bkv_sem_idx)` — [`L1555`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1555)
- `start_fetch_bq(batch_start_seq_idx, bq_idx, bq_sem_idx)` — [`L1564`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1564)
- `start_send_bo(batch_start_seq_idx, bo_idx, bo_sem_idx)` — [`L1570`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1570)
- `start_update_kv_cache(start_seq_idx, bkv_sem_idx, offsets, update_szs)` — [`L1589`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1589)
- `static_validate_inputs(ql_nope: jax.Array, q_pe: jax.Array, new_kv_c: jax.Array, new_k_pe: jax.Array, cache_kv: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, *, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, num_kv_pages_per_blocks: tuple[int, int, int] | None = None, num_queries_per_blocks: tuple[int, int, int] | None = None, vmem_limit_bytes: int | None = None, decode_batch_size: int = 1, transpose_kv_cache: bool = False, debug_mode: bool = False)` — [`L133`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L133) — Validate inputs to the MLA RPA kernel statically. — documented in [tpu_inference-kernels-mla-v2-kernel](../../../../../concepts/tpu_inference-kernels-mla-v2-kernel.md)
- `unsigned_align_to(a, b)` — [`L60`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L60)
- `unsigned_cdiv(a, b)` — [`L44`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L44)
- `unsigned_floor_div(a, b)` — [`L52`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L52)
- `unsigned_mod(a, b)` — [`L36`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L36)
- `update_body(idx, state)` — [`L1182`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1182)
- `update_cur_bkv_to_cache()` — [`L1800`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1800)
- `wait_cur_bq()` — [`L1780`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1780)
- `wait_fetch_bkv(batch_start_seq_idx, bkv_idx, bkv_sem_idx)` — [`L1558`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1558)
- `wait_fetch_bq(batch_start_seq_idx, bq_idx, bq_sem_idx)` — [`L1567`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1567)
- `wait_send_bo(bo_sem_idx)` — [`L1575`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1575)
- `wait_update_kv_cache(bkv_sem_idx)` — [`L1608`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L1608)

## Module values
- `_XPOSE_N_TILE_SIZE` — [`L33`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L33)
- `logger` — [`L31`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/kernel.py#L31)

