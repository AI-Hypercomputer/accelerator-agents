---
title: 'Module: ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.flash_mla._pallas_impl_bwd`/
symbols:
  _flash_mla_bwd_dkv: _flash_mla_bwd_dkv().
  _flash_mla_bwd_dq: _flash_mla_bwd_dq().
  _flash_mla_bwd_impl: _flash_mla_bwd_impl().
  _flash_mla_dkv_kernel._run: _flash_mla_dkv_kernel()._run().
  _flash_mla_dq_kernel._run: _flash_mla_dq_kernel()._run().
  ROPE_NONE: ROPE_NONE.
  ROPE_DECOUPLED: ROPE_DECOUPLED.
  _flash_mla_bwd_dkv.do_index_map: _flash_mla_bwd_dkv().do_index_map().
  _flash_mla_bwd_dq.kv_index_map: _flash_mla_bwd_dq().kv_index_map().
  ROPE_FUSED: ROPE_FUSED.
  _flash_mla_bwd_dq.q_index_map: _flash_mla_bwd_dq().q_index_map().
  _flash_mla_dkv_kernel: _flash_mla_dkv_kernel().
  _flash_mla_bwd_dkv.q_index_map: _flash_mla_bwd_dkv().q_index_map().
  _flash_mla_bwd_dkv.bq_index_map: _flash_mla_bwd_dkv().bq_index_map().
  _flash_mla_bwd_dkv.bias_index_map: _flash_mla_bwd_dkv().bias_index_map().
  _flash_mla_dq_kernel: _flash_mla_dq_kernel().
  _flash_mla_bwd_dq.bk_index_map: _flash_mla_bwd_dq().bk_index_map().
  _flash_mla_bwd_dq.bias_index_map: _flash_mla_bwd_dq().bias_index_map().
  _flash_mla_bwd_dkv.dkv_index_map: _flash_mla_bwd_dkv().dkv_index_map().
  _flash_mla_bwd_dq.lm_index_map: _flash_mla_bwd_dq().lm_index_map().
  _flash_mla_dkv_kernel._init: _flash_mla_dkv_kernel()._init().
  _flash_mla_dkv_kernel._store: _flash_mla_dkv_kernel()._store().
  _flash_mla_dq_kernel._init: _flash_mla_dq_kernel()._init().
  _flash_mla_dq_kernel._store: _flash_mla_dq_kernel()._store().
  _flash_mla_bwd_dkv.kv_index_map: _flash_mla_bwd_dkv().kv_index_map().
  _flash_mla_bwd_dkv.lm_index_map: _flash_mla_bwd_dkv().lm_index_map().
  _flash_mla_bwd_dkv.bk_index_map: _flash_mla_bwd_dkv().bk_index_map().
  _flash_mla_bwd_dq.bq_index_map: _flash_mla_bwd_dq().bq_index_map().
  _flash_mla_bwd_dq.dbias_out_index_map: _flash_mla_bwd_dq().dbias_out_index_map().
  _flash_mla_dq_kernel._zero_dbias: _flash_mla_dq_kernel()._zero_dbias().
---
# Module: [`ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py)

## Functions
- `_flash_mla_bwd_dkv(q, k_nope, v, b_q, b_k, bias, l, m, do, di, *, rope_mode, d_nope, gqa_ratio, block_q, block_k, softmax_scale, sliding_window, logits_soft_cap, causal)` — [`L252`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L252) — Compute gradients w.r.t. the projected keys, values, and RoPE keys.
- `_flash_mla_bwd_dq(q, k_nope, v, b_q, b_k, bias, l, m, do, di, *, rope_mode, d_nope, gqa_ratio, block_q, block_k, softmax_scale, sliding_window, logits_soft_cap, causal)` — [`L668`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L668) — Compute gradient w.r.t. queries, optional bias, and optional b_q.
- `_flash_mla_bwd_impl(rope_mode, causal, softmax_scale, sliding_window, logits_soft_cap, block_q, block_k, residuals, do)` — [`L872`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L872) — Compute gradients for Flash MLA using Pallas kernels + JAX post-processing.
- `_flash_mla_dkv_kernel(q_tile_ref, k_nope_tile_ref, v_tile_ref, bq_tile_ref, bk_tile_ref, bias_tile_ref, l_tile_ref, m_tile_ref, do_tile_ref, di_tile_ref, dk_tile_ref, dv_tile_ref, db_k_tile_ref, dk_scratch_ref, dv_scratch_ref, db_k_scratch_ref, *, rope_mode, d_nope, causal, softmax_scale, sliding_window, logits_soft_cap, mask_value, q_seq_len, block_q, block_k)` — [`L54`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L54) — Pallas kernel for computing ``dk_nope``, ``dv``, and optionally ``db_k``.
- `_flash_mla_dq_kernel(q_tile_ref, k_nope_tile_ref, v_tile_ref, bq_tile_ref, bk_tile_ref, bias_tile_ref, l_tile_ref, m_tile_ref, do_tile_ref, di_tile_ref, dq_tile_ref, dbias_tile_ref, db_q_tile_ref, dq_scratch_ref, db_q_scratch_ref, *, rope_mode, d_nope, causal, softmax_scale, sliding_window, logits_soft_cap, mask_value, kv_seq_len, block_q, block_k)` — [`L465`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L465) — Pallas kernel for computing ``dq`` and optionally ``dbias`` / ``db_q``.
- `_init()` — [`L125`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L125)
- `_init()` — [`L533`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L533)
- `_run()` — [`L137`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L137)
- `_run()` — [`L546`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L546)
- `_store()` — [`L245`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L245)
- `_store()` — [`L662`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L662)
- `_zero_dbias()` — [`L657`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L657)
- `bias_index_map(batch_idx, head_idx, kv_seq_idx, q_seq_idx)` — [`L355`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L355)
- `bias_index_map(batch_idx, head_idx, q_seq_idx, kv_seq_idx)` — [`L771`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L771)
- `bk_index_map(batch_idx, head_idx, kv_seq_idx, _)` — [`L351`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L351)
- `bk_index_map(batch_idx, head_idx, q_seq_idx, kv_seq_idx)` — [`L759`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L759)
- `bq_index_map(batch_idx, head_idx, kv_seq_idx, q_seq_idx)` — [`L339`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L339)
- `bq_index_map(batch_idx, head_idx, q_seq_idx, _)` — [`L755`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L755)
- `dbias_out_index_map(batch_idx, head_idx, q_seq_idx, kv_seq_idx)` — [`L803`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L803)
- `dkv_index_map(batch_idx, head_idx, kv_seq_idx, _)` — [`L379`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L379)
- `do_index_map(batch_idx, head_idx, kv_seq_idx, q_seq_idx)` — [`L368`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L368)
- `kv_index_map(batch_idx, head_idx, kv_seq_idx, _)` — [`L336`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L336)
- `kv_index_map(batch_idx, head_idx, q_seq_idx, kv_seq_idx)` — [`L744`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L744)
- `lm_index_map(batch_idx, head_idx, _, q_seq_idx)` — [`L365`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L365)
- `lm_index_map(batch_idx, head_idx, q_seq_idx, _)` — [`L781`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L781)
- `q_index_map(batch_idx, head_idx, kv_seq_idx, q_seq_idx)` — [`L325`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L325)
- `q_index_map(batch_idx, head_idx, q_seq_idx, _)` — [`L741`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L741)

## Module values
- `ROPE_DECOUPLED` — [`L51`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L51)
- `ROPE_FUSED` — [`L50`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L50)
- `ROPE_NONE` — [`L49`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/flash_mla/_pallas_impl_bwd.py#L49)

