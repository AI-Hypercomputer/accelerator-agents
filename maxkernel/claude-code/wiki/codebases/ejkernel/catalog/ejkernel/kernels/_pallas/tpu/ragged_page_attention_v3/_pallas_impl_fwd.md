---
title: 'Module: ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.ragged_page_attention_v3._pallas_impl_fwd`/
symbols:
  ragged_paged_attention: ragged_paged_attention().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq: _ragged_paged_attention_kernel_loop().process().compute_with_bq().
  ref_ragged_paged_attention: ref_ragged_paged_attention().
  ragged_paged_attention.run_case: ragged_paged_attention().run_case().
  RpaCase: RpaCase#
  _ragged_paged_attention_kernel_loop: _ragged_paged_attention_kernel_loop().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq.compute_with_bkv: _ragged_paged_attention_kernel_loop().process().compute_with_bq().compute_with_bkv().
  get_vmem_estimate_bytes: get_vmem_estimate_bytes().
  RpaCase.symbol: RpaCase#symbol().
  RpaCase.get_range: RpaCase#get_range().
  ragged_paged_attention.get_case_block_sizes: ragged_paged_attention().get_case_block_sizes().
  _ragged_paged_attention_kernel_loop._fetch_bkv: _ragged_paged_attention_kernel_loop()._fetch_bkv().
  static_validate_inputs: static_validate_inputs().
  prepare_inputs: prepare_inputs().
  dynamic_validate_inputs: dynamic_validate_inputs().
  _ragged_paged_attention_kernel_loop.strided_load_bkv._convert_to_target_bitwidth: _ragged_paged_attention_kernel_loop().strided_load_bkv()._convert_to_target_bitwidth().
  _ragged_paged_attention_kernel_loop._update_kv_cache: _ragged_paged_attention_kernel_loop()._update_kv_cache().
  merge_kv: merge_kv().
  _ragged_paged_attention_kernel_loop.strided_load: _ragged_paged_attention_kernel_loop().strided_load().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  _clamp_block_sizes_to_vmem: _clamp_block_sizes_to_vmem().
  _ragged_paged_attention_kernel_loop.flash_attention: _ragged_paged_attention_kernel_loop().flash_attention().
  _ragged_paged_attention_kernel_loop.strided_load_bkv: _ragged_paged_attention_kernel_loop().strided_load_bkv().
  _ragged_paged_attention_kernel_loop.process: _ragged_paged_attention_kernel_loop().process().
  _ragged_paged_attention_kernel_loop.broadcast_minor: _ragged_paged_attention_kernel_loop().broadcast_minor().
  get_smem_estimate_bytes: get_smem_estimate_bytes().
  get_kv_cache_shape: get_kv_cache_shape().
  _ragged_paged_attention_kernel_loop._async_copy: _ragged_paged_attention_kernel_loop()._async_copy().
  _ragged_paged_attention_kernel_loop.prologue: _ragged_paged_attention_kernel_loop().prologue().
  _ragged_paged_attention_kernel_loop.epilogue: _ragged_paged_attention_kernel_loop().epilogue().
  _ragged_paged_attention_kernel_loop._fetch_bq: _ragged_paged_attention_kernel_loop()._fetch_bq().
  _ragged_paged_attention_kernel_loop._send_bo: _ragged_paged_attention_kernel_loop()._send_bo().
  _ragged_paged_attention_kernel_loop.start_fetch_bkv: _ragged_paged_attention_kernel_loop().start_fetch_bkv().
  _ragged_paged_attention_kernel_loop.start_fetch_bq: _ragged_paged_attention_kernel_loop().start_fetch_bq().
  _ragged_paged_attention_kernel_loop.wait_send_bo: _ragged_paged_attention_kernel_loop().wait_send_bo().
  _ragged_paged_attention_kernel_loop.wait_update_kv_cache: _ragged_paged_attention_kernel_loop().wait_update_kv_cache().
  RpaCase.DECODE: RpaCase#DECODE.
  RpaCase.PREFILL: RpaCase#PREFILL.
  RpaCase.MIXED: RpaCase#MIXED.
  _ragged_paged_attention_kernel: _ragged_paged_attention_kernel().
  _ragged_paged_attention_kernel_loop._fetch_bkv.loop_body: _ragged_paged_attention_kernel_loop()._fetch_bkv().loop_body().
  _ragged_paged_attention_kernel_loop._update_kv_cache.loop_body: _ragged_paged_attention_kernel_loop()._update_kv_cache().loop_body().
  _ragged_paged_attention_kernel_loop.wait_fetch_bkv: _ragged_paged_attention_kernel_loop().wait_fetch_bkv().
  _ragged_paged_attention_kernel_loop.wait_fetch_bq: _ragged_paged_attention_kernel_loop().wait_fetch_bq().
  _ragged_paged_attention_kernel_loop.start_send_bo: _ragged_paged_attention_kernel_loop().start_send_bo().
  _ragged_paged_attention_kernel_loop.start_update_kv_cache: _ragged_paged_attention_kernel_loop().start_update_kv_cache().
  _ragged_paged_attention_kernel_loop.load_bq: _ragged_paged_attention_kernel_loop().load_bq().
  _ragged_paged_attention_kernel_loop.process.get_next_bkv_ids: _ragged_paged_attention_kernel_loop().process().get_next_bkv_ids().
  _ragged_paged_attention_kernel_loop.flash_attention.load_with_init: _ragged_paged_attention_kernel_loop().flash_attention().load_with_init().
  _ragged_paged_attention_kernel_loop._fetch_bkv._fetch_bkv_from_new_kv: _ragged_paged_attention_kernel_loop()._fetch_bkv()._fetch_bkv_from_new_kv().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq.prefetch_next_bq: _ragged_paged_attention_kernel_loop().process().compute_with_bq().prefetch_next_bq().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq.compute_with_bkv.prefetch_next_bkv: _ragged_paged_attention_kernel_loop().process().compute_with_bq().compute_with_bkv().prefetch_next_bkv().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq.compute_with_bkv.wait_cur_bq: _ragged_paged_attention_kernel_loop().process().compute_with_bq().compute_with_bkv().wait_cur_bq().
  _ragged_paged_attention_kernel_loop.process.compute_with_bq.compute_with_bkv.update_cur_bkv_to_cache: _ragged_paged_attention_kernel_loop().process().compute_with_bq().compute_with_bkv().update_cur_bkv_to_cache().
  _ragged_paged_attention_kernel_loop.pipeline: _ragged_paged_attention_kernel_loop().pipeline().
  has_bank_conflicts: has_bank_conflicts().
  _ragged_paged_attention_kernel_loop.get_start_bkv_idx: _ragged_paged_attention_kernel_loop().get_start_bkv_idx().
  DEFAULT_SCOPED_VMEM_LIMIT_BYTES: DEFAULT_SCOPED_VMEM_LIMIT_BYTES.
  _ragged_paged_attention_kernel_loop.process.get_next_bq_ids: _ragged_paged_attention_kernel_loop().process().get_next_bq_ids().
  prepare_outputs: prepare_outputs().
  DEFAULT_VMEM_LIMIT_BYTES: DEFAULT_VMEM_LIMIT_BYTES.
---
# Module: [`ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py)

## Classes
### `RpaCase`  ·  implements/extends Enum
- def: [`ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py:97`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L97) — documented in [ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd.md)
- doc: Case split used to specialize decode, prefill, and mixed launches.
- signature: `class RpaCase(Enum):`
- members:
  - `get_range(self, distribution)` — [`L112`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L112) — documented in [ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd.md)
  - `symbol(self)` — [`L105`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L105) — documented in [ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd.md)
  - `DECODE` — [`L100`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L100) — documented in [ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd.md)
  - `MIXED` — [`L102`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L102) — documented in [ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd.md)
  - `PREFILL` — [`L101`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L101) — documented in [ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd.md)
- used by: [`ragged_paged_attention`](_pallas_impl_fwd.md#ragged_paged_attention), [`run_case`](_pallas_impl_fwd.md#ragged_paged_attention.run_case), [`_ragged_paged_attention_kernel_loop`](_pallas_impl_fwd.md#_ragged_paged_attention_kernel_loop), [`get_case_block_sizes`](_pallas_impl_fwd.md#ragged_paged_attention.get_case_block_sizes)

## Functions
- `_async_copy(src, dst, sem, wait)` — [`L695`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L695)
- `_clamp_block_sizes_to_vmem(*, actual_num_kv_heads, actual_num_q_heads_per_kv_head, actual_head_dim, page_size, q_dtype, kv_dtype, bkv_p, bq_sz, vmem_limit_bytes)` — [`L378`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L378) — Clamp TPU block sizes until scratch buffers fit within scoped VMEM. — documented in [ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd.md)
- `_convert_to_target_bitwidth(val, target_bitwidth: int)` — [`L912`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L912)
- `_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False)` — [`L702`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L702)
- `_fetch_bkv_from_new_kv()` — [`L743`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L743)
- `_fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False)` — [`L792`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L792)
- `_ragged_paged_attention_kernel(*args, **kwargs)` — [`L477`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L477) — documented in [ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd.md)
- `_ragged_paged_attention_kernel_loop(seq_idx, kv_lens_ref, page_indices_ref, cu_q_lens_ref, distribution_ref, sem_ids_ref, bo_ids_ref, bkv_update_ids_ref, q_hbm_ref, kv_hbm_ref, kv_cache_hbm_ref, attention_sink_ref, o_hbm_ref, updated_kv_cache_hbm_ref, bkv_x2_ref, bq_x2_ref, bo_x2_ref, sems, l_ref, m_ref, acc_ref, *, softmax_scale: float, sliding_window: int | None = None, logits_soft_cap: float | None = None, mask_value: float = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, bkv_p, bq_sz, static_q_len: int | None = None, case: RpaCase = RpaCase.MIXED)` — [`L490`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L490) — Pallas TPU kernel for ragged paged attention V3 with KV cache write. — documented in [ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd.md)
- `_send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False)` — [`L806`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L806)
- `_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, *, wait=False)` — [`L754`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L754)
- `broadcast_minor(src, shape)` — [`L945`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L945)
- `compute_with_bkv(bkv_idx, _)` — [`L1007`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L1007)
- `compute_with_bq(bq_idx, _)` — [`L989`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L989) — documented in [ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd.md)
- `dynamic_validate_inputs(queries: jax.Array, keys: jax.Array, values: jax.Array, kv_cache: jax.Array, kv_lens: jax.Array, block_tables: jax.Array, query_start_loc: jax.Array, distribution: jax.Array, softmax_aux: jax.Array | None = None, *, softmax_scale: float = 1, sliding_window: int | None = None, logits_soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, num_kv_pages_per_block: int | None = None, num_queries_per_block: int | None = None, vmem_limit_bytes: int | None = None)` — [`L1248`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L1248) — Validate inputs with both static shape checks and dynamic value checks.
- `epilogue()` — [`L1095`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L1095)
- `flash_attention(q, k, v, *, processed_q_len, processed_kv_len, bkv_idx, start_bkv_idx, kv_head_idx)` — [`L620`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L620)
- `get_case_block_sizes(case: RpaCase)` — [`L1630`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L1630)
- `get_kv_cache_shape(total_num_pages, page_size, actual_num_kv_heads, actual_head_dim, kv_dtype)` — [`L432`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L432) — Compute the TPU-optimized KV cache tensor shape.
- `get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx, *, num_bkv)` — [`L975`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L975)
- `get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx)` — [`L967`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L967)
- `get_smem_estimate_bytes(max_num_seqs, pages_per_seq)` — [`L304`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L304) — Estimate scalar memory (SMEM) usage for the kernel's prefetched data.
- `get_start_bkv_idx(processed_q_len)` — [`L954`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L954)
- `get_vmem_estimate_bytes(actual_num_kv_heads, actual_num_q_heads_per_kv_head, actual_head_dim, bq_sz, bkv_sz, q_dtype, kv_dtype)` — [`L330`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L330) — Estimate vector memory (VMEM) usage for the kernel's scratch buffers.
- `has_bank_conflicts(stride, distance=24, num_banks=32)` — [`L466`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L466) — Return whether a VMEM stride is likely to trigger bank conflicts.
- `load_bq(bq_sem_idx, kv_head_idx)` — [`L876`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L876)
- `load_with_init(ref, init_val)` — [`L640`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L640)
- `loop_body(i, offset)` — [`L724`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L724)
- `loop_body(i, states)` — [`L767`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L767)
- `merge_kv(k: jax.Array, v: jax.Array)` — [`L1101`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L1101) — Interleave key and value tensors into TPU-packed KV format.
- `pipeline()` — [`L1091`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L1091)
- `prefetch_next_bkv()` — [`L1024`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L1024)
- `prefetch_next_bq()` — [`L1003`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L1003)
- `prepare_inputs(q: jax.Array, k: jax.Array, v: jax.Array, softmax_aux: jax.Array | None = None)` — [`L1141`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L1141) — Transform query and KV tensors into TPU-optimized packed layouts.
- `prepare_outputs(out, actual_num_q_heads_per_kv_head: int, actual_head_dim: int)` — [`L1208`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L1208) — Transform kernel output from TPU-packed layout back to standard shape.
- `process()` — [`L959`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L959)
- `prologue()` — [`L1086`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L1086)
- `ragged_paged_attention(queries: jax.Array, keys: jax.Array, values: jax.Array, kv_cache: jax.Array, kv_lens: jax.Array, block_tables: jax.Array, query_start_loc: jax.Array, distribution: jax.Array, softmax_aux: jax.Array | None = None, *, softmax_scale: float = 1, sliding_window: int | None = None, logits_soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, num_kv_pages_per_block: int | None = None, num_queries_per_block: int | None = None, vmem_limit_bytes: int | None = None)` — [`L1521`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L1521) — Ragged paged attention that supports mixed prefill and decode. — documented in [ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd.md)
- `ref_ragged_paged_attention(queries: jax.Array, keys: jax.Array, values: jax.Array, kv_cache: jax.Array, kv_lens: jax.Array, block_tables: jax.Array, query_start_loc: jax.Array, distribution: jax.Array, softmax_aux: jax.Array | None = None, *, softmax_scale: float = 1, sliding_window: int | None = None, logits_soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None)` — [`L123`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L123) — Reference implementation of V3 ragged paged attention with KV cache write. — documented in [ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd.md)
- `run_case(q, kv_cache, *, static_q_len: int | None, case: RpaCase)` — [`L1662`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L1662) — documented in [ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd.md)
- `start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)` — [`L820`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L820)
- `start_fetch_bq(seq_idx, bq_idx, bq_sem_idx)` — [`L826`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L826)
- `start_send_bo(seq_idx, bo_idx, bo_sem_idx)` — [`L832`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L832)
- `start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)` — [`L845`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L845)
- `static_validate_inputs(queries: jax.Array, keys: jax.Array, values: jax.Array, kv_cache: jax.Array, kv_lens: jax.Array, block_tables: jax.Array, query_start_loc: jax.Array, distribution: jax.Array, softmax_aux: jax.Array | None = None, *, softmax_scale: float = 1, sliding_window: int | None = None, logits_soft_cap: float | None = None, mask_value: float | None = DEFAULT_MASK_VALUE, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, num_kv_pages_per_block: int | None = None, num_queries_per_block: int | None = None, vmem_limit_bytes: int | None = None)` — [`L1362`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L1362) — Validate static shape and dtype constraints for the ragged paged attention kernel.
- `strided_load(ref, start, sz, step, *, dtype=None)` — [`L861`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L861)
- `strided_load_bkv(bkv_sem_idx, start, step, *, bkv_mask)` — [`L890`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L890)
- `update_cur_bkv_to_cache()` — [`L1035`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L1035)
- `wait_cur_bq()` — [`L1029`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L1029)
- `wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)` — [`L823`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L823)
- `wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)` — [`L829`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L829)
- `wait_send_bo(bo_sem_idx)` — [`L837`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L837)
- `wait_update_kv_cache(bkv_sem_idx)` — [`L851`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L851)

## Module values
- `DEFAULT_MASK_VALUE` — [`L92`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L92) — documented in [ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-ragged_page_attention_v3-_pallas_impl_fwd.md)
- `DEFAULT_SCOPED_VMEM_LIMIT_BYTES` — [`L375`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L375)
- `DEFAULT_VMEM_LIMIT_BYTES` — [`L94`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_page_attention_v3/_pallas_impl_fwd.py#L94)

