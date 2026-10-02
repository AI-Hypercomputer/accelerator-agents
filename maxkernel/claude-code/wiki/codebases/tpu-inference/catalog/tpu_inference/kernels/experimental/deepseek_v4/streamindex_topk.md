---
title: 'Module: tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.experimental.deepseek_v4.streamindex_topk`/
symbols:
  streamindex_topk: streamindex_topk().
  _streamindex_topk_kernel.process.compute_with_bq.compute_with_bkv: _streamindex_topk_kernel().process().compute_with_bq().compute_with_bkv().
  streamindex_topk.run_topk_kernel: streamindex_topk().run_topk_kernel().
  MlaCase.symbol: MlaCase#symbol().
  _streamindex_topk_kernel.process.compute_with_bq: _streamindex_topk_kernel().process().compute_with_bq().
  MlaCase: MlaCase#
  _streamindex_topk_kernel._async_copy: _streamindex_topk_kernel()._async_copy().
  _streamindex_topk_kernel: _streamindex_topk_kernel().
  _streamindex_topk_kernel.process: _streamindex_topk_kernel().process().
  prepare_q_inputs: prepare_q_inputs().
  prepare_index_weights: prepare_index_weights().
  align_to: align_to().
  get_dtype_packing: get_dtype_packing().
  _streamindex_topk_kernel.prologue: _streamindex_topk_kernel().prologue().
  _streamindex_topk_kernel._fetch_bkv: _streamindex_topk_kernel()._fetch_bkv().
  _streamindex_topk_kernel._fetch_bq: _streamindex_topk_kernel()._fetch_bq().
  _streamindex_topk_kernel._send_bo: _streamindex_topk_kernel()._send_bo().
  _streamindex_topk_kernel.start_fetch_bkv: _streamindex_topk_kernel().start_fetch_bkv().
  _streamindex_topk_kernel.start_fetch_bq: _streamindex_topk_kernel().start_fetch_bq().
  _streamindex_topk_kernel.wait_send_bo: _streamindex_topk_kernel().wait_send_bo().
  cdiv: cdiv().
  _streamindex_topk_kernel.wait_fetch_bkv: _streamindex_topk_kernel().wait_fetch_bkv().
  _streamindex_topk_kernel.wait_fetch_bq: _streamindex_topk_kernel().wait_fetch_bq().
  _streamindex_topk_kernel.start_send_bo: _streamindex_topk_kernel().start_send_bo().
  MlaCase.MIXED: MlaCase#MIXED.
  _streamindex_topk_kernel.process.compute_with_bq.prefetch_next_bq: _streamindex_topk_kernel().process().compute_with_bq().prefetch_next_bq().
  _streamindex_topk_kernel.process.compute_with_bq.compute_with_bkv.prefetch_next_bkv: _streamindex_topk_kernel().process().compute_with_bq().compute_with_bkv().prefetch_next_bkv().
  _streamindex_topk_kernel.process.compute_with_bq.compute_with_bkv.wait_cur_bq: _streamindex_topk_kernel().process().compute_with_bq().compute_with_bkv().wait_cur_bq().
  _streamindex_topk_kernel.epilogue: _streamindex_topk_kernel().epilogue().
  MlaCase.DECODE: MlaCase#DECODE.
  prepare_outputs: prepare_outputs().
  Enum: Enum.
  DEFAULT_VMEM_LIMIT_BYTES: DEFAULT_VMEM_LIMIT_BYTES.
  get_dtype_bitwidth: get_dtype_bitwidth().
  MlaCase.PREFILL: MlaCase#PREFILL.
  _streamindex_topk_kernel.compute_topk: _streamindex_topk_kernel().compute_topk().
  _streamindex_topk_kernel.load_bq: _streamindex_topk_kernel().load_bq().
  _streamindex_topk_kernel.load_bq_weights: _streamindex_topk_kernel().load_bq_weights().
  _streamindex_topk_kernel.load_bkv: _streamindex_topk_kernel().load_bkv().
  _streamindex_topk_kernel.process.get_next_bq_ids: _streamindex_topk_kernel().process().get_next_bq_ids().
  _streamindex_topk_kernel.process.get_next_bkv_ids: _streamindex_topk_kernel().process().get_next_bkv_ids().
---
# Module: [`tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py)

## Classes
### `MlaCase`  ·  implements/extends Enum
- def: [`tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py:47`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L47)
- doc: Represents the different cases for MLA.
- signature: `class MlaCase(Enum):`
- members:
  - `symbol(self)` — [`L60`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L60)
  - `DECODE` — [`L55`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L55)
  - `MIXED` — [`L57`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L57)
  - `PREFILL` — [`L56`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L56)
- uses (calls/refs, reference-scoped): [`Enum`](streamindex_topk.md#Enum)
- used by: [`streamindex_topk`](streamindex_topk.md#streamindex_topk), [`run_topk_kernel`](streamindex_topk.md#streamindex_topk.run_topk_kernel)

## Functions
- `_async_copy(src, dst, sem, wait)` — [`L210`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L210)
- `_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False)` — [`L217`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L217)
- `_fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False)` — [`L255`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L255)
- `_send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False)` — [`L288`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L288)
- `_streamindex_topk_kernel(seq_lens_ref, page_indices_ref, cu_q_lens_ref, start_end_seq_idx_ref, sem_ids_ref, bo_ids_ref, q_hbm_ref, indexer_weights_hbm_ref, cache_kv_hbm_ref, topk_idxs_hbm_ref, bkv_x2_ref, bq_x2_ref, bq_weights_x2_ref, bo_idxs_x2_ref, sems, topk_vals_scratch, topk_idxs_scratch, *, k: int, compression_ratio: int, static_q_len: int, bkv_p: int, bq_sz: int, actual_num_q_heads: int, kv_packing: int)` — [`L68`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L68)
- `align_to(x, a)` — [`L34`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L34)
- `cdiv(a, b)` — [`L29`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L29)
- `compute_topk(q, kv, scale_val, *, bkv_idx, bq_pos_compressed, bq_weights)` — [`L129`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L129)
- `compute_with_bkv(bkv_idx, _)` — [`L413`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L413)
- `compute_with_bq(bq_idx, _)` — [`L390`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L390)
- `epilogue()` — [`L473`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L473)
- `get_dtype_bitwidth(dtype)` — [`L38`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L38)
- `get_dtype_packing(dtype)` — [`L42`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L42)
- `get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx)` — [`L379`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L379)
- `get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx)` — [`L371`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L371)
- `load_bkv(bkv_sem_idx)` — [`L345`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L345)
- `load_bq(bq_sem_idx)` — [`L336`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L336)
- `load_bq_weights(bq_sem_idx)` — [`L341`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L341)
- `prefetch_next_bkv()` — [`L420`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L420)
- `prefetch_next_bq()` — [`L397`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L397)
- `prepare_index_weights(index_weights: jax.Array, q_dtype)` — [`L499`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L499)
- `prepare_outputs(out)` — [`L517`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L517)
- `prepare_q_inputs(q: jax.Array)` — [`L480`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L480)
- `process()` — [`L364`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L364)
- `prologue()` — [`L466`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L466)
- `run_topk_kernel(q, prepared_indexer_weights, cache_kv, seq_lens, page_indices, cu_q_lens, start_seq_idx, end_seq_idx, static_q_len, num_kv_pages_per_block, num_queries_per_block, case=MlaCase.MIXED)` — [`L602`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L602)
- `start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)` — [`L311`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L311)
- `start_fetch_bq(seq_idx, bq_idx, bq_sem_idx)` — [`L317`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L317)
- `start_send_bo(seq_idx, bo_idx, bo_sem_idx)` — [`L323`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L323)
- `streamindex_topk(q: jax.Array, indexer_weights: jax.Array, cache_kv: jax.Array, seq_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, *, k: int, compression_ratio: int, num_kv_pages_per_block: tuple[int, int, int] | int | None = None, num_queries_per_block: tuple[int, int, int] | int | None = None, vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES)` — [`L533`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L533) — StreamIndex Top-K retrieval.
- `wait_cur_bq()` — [`L427`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L427)
- `wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)` — [`L314`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L314)
- `wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)` — [`L320`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L320)
- `wait_send_bo(bo_sem_idx)` — [`L328`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L328)

## Module values
- `DEFAULT_VMEM_LIMIT_BYTES` — [`L26`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L26)
- `Enum` — [`L25`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py#L25)

