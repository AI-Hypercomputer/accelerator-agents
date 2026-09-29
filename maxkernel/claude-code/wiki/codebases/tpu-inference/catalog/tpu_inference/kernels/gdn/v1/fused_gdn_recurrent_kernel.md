---
title: 'Module: tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.gdn.v1.fused_gdn_recurrent_kernel`/
symbols:
  fused_recurrent_gdn: fused_recurrent_gdn().
  calculate_chunk_indices: calculate_chunk_indices().
  _MetadataIndexMaps.token_map: _MetadataIndexMaps#token_map().
  _recurrent_gdn_main: _recurrent_gdn_main().
  GDNChunkIndices.block_id_to_t_offset: GDNChunkIndices#block_id_to_t_offset.
  GDNChunkIndices: GDNChunkIndices#
  calculate_chunk_indices._kernel: calculate_chunk_indices()._kernel().
  calculate_chunk_indices._kernel.outer_seq_loop: calculate_chunk_indices()._kernel().outer_seq_loop().
  _recurrent_gdn_main._inner_kernel_body: _recurrent_gdn_main()._inner_kernel_body().
  GDNChunkIndices.num_blocks: GDNChunkIndices#num_blocks.
  GDNChunkIndices.block_id_to_seq_idx: GDNChunkIndices#block_id_to_seq_idx.
  _MetadataIndexMaps.__init__: _MetadataIndexMaps#__init__().
  _MetadataIndexMaps.meta: _MetadataIndexMaps#meta.
  get_default_block_sizes: get_default_block_sizes().
  calculate_chunk_indices._kernel.inner_block_loop: calculate_chunk_indices()._kernel().inner_block_loop().
  _MetadataIndexMaps: _MetadataIndexMaps#
  _recurrent_gdn_main._inner_kernel_body.step: _recurrent_gdn_main()._inner_kernel_body().step().
  _recurrent_gdn_main._inner_kernel_body._prefetch: _recurrent_gdn_main()._inner_kernel_body()._prefetch().
  _recurrent_gdn_main._inner_kernel_body._wait_h0: _recurrent_gdn_main()._inner_kernel_body()._wait_h0().
  _recurrent_gdn_main._inner_kernel_body._zero_h0: _recurrent_gdn_main()._inner_kernel_body()._zero_h0().
  _recurrent_gdn_main._inner_kernel_body._wait_prev_store: _recurrent_gdn_main()._inner_kernel_body()._wait_prev_store().
  _recurrent_gdn_main._inner_kernel_body._start_store: _recurrent_gdn_main()._inner_kernel_body()._start_store().
  _recurrent_gdn_main._drain_other: _recurrent_gdn_main()._drain_other().
---
# Module: [`tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py)

## Classes
### `GDNChunkIndices`
- def: [`tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py:77`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py#L77)
- signature: `class GDNChunkIndices:`
- members:
  - `block_id_to_seq_idx` — [`L79`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py#L79)
  - `block_id_to_t_offset` — [`L80`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py#L80)
  - `num_blocks` — [`L78`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py#L78)
- used by: [`fused_recurrent_gdn`](fused_gdn_recurrent_kernel.md#fused_recurrent_gdn), [`calculate_chunk_indices`](fused_gdn_recurrent_kernel.md#calculate_chunk_indices), [`token_map`](fused_gdn_recurrent_kernel.md#_MetadataIndexMaps.token_map), [`__init__`](fused_gdn_recurrent_kernel.md#_MetadataIndexMaps.__init__)

### `_MetadataIndexMaps`
- def: [`tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py:176`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py#L176)
- doc: Index maps driven by pre-computed metadata arrays.
- signature: `class _MetadataIndexMaps:`
- members:
  - `token_map(self, block_id)` — [`L182`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py#L182)
  - `meta` — [`L180`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py#L180)
- protocol/private: `__init__`[`L179`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py#L179)
- uses (calls/refs, reference-scoped): [`block_id_to_t_offset`](fused_gdn_recurrent_kernel.md#GDNChunkIndices.block_id_to_t_offset), [`GDNChunkIndices`](fused_gdn_recurrent_kernel.md#GDNChunkIndices)
- used by: [`_recurrent_gdn_main`](fused_gdn_recurrent_kernel.md#_recurrent_gdn_main)

## Functions
- `_drain_other()` — [`L468`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py#L468)
- `_inner_kernel_body(q_ref, k_ref, v_ref, g_ref, b_ref, a_log_ref, dt_bias_ref, o_ref, h_bufs_s, meta_s, state_indices_s, has_initial_state_s, h_load_sems_s, h_store_sems_s)` — [`L247`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py#L247)
- `_kernel(cu_seqlens_ref, distribution_ref, meta_out, *, bt: int)` — [`L94`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py#L94)
- `_prefetch()` — [`L299`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py#L299)
- `_recurrent_gdn_main(meta, q_hbm, k_hbm, v_hbm, g_hbm, b_hbm, state_indices_ref, a_log_hbm, dt_bias_hbm, _state_init_ref, has_initial_state_ref, o_hbm, state_hbm, h_bufs, h_load_sems, h_store_sems, *, H_qk: int, H_v: int, K: int, V: int, scale: float, use_qk_l2norm: bool, use_gate_in_kernel: bool, lower_bound: float | None, bt: int)` — [`L192`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py#L192)
- `_start_store()` — [`L417`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py#L417)
- `_wait_h0()` — [`L303`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py#L303)
- `_wait_prev_store()` — [`L412`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py#L412)
- `_zero_h0()` — [`L316`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py#L316)
- `calculate_chunk_indices(cu_seqlens, distribution, max_num_blocks, bt: int)` — [`L84`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py#L84) — Pre-compute per-block metadata as a standalone Pallas kernel.
- `fused_recurrent_gdn(q, k, v, cu_seqlens, g, initial_state, state_indices, b, has_initial_state, *, scale, use_qk_l2norm, use_gate_in_kernel=False, A_log=None, dt_bias=None, lower_bound=None, distribution)` — [`L475`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py#L475) — Run the pre-computed-metadata recurrent GDN pallas kernel.
- `get_default_block_sizes(H_qk: int, H_v: int, K: int, V: int, dtype, use_gate_in_kernel: bool, has_dt_bias: bool, vmem_bytes_limit: int, state_dtype=jnp.float32)` — [`L32`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py#L32) — Choose bt to maximize VMEM utilization.
- `inner_block_loop(blk_rel, carry, *, seq_idx, eos)` — [`L106`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py#L106)
- `outer_seq_loop(seq_rel, carry)` — [`L120`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py#L120)
- `step(local_t, h)` — [`L334`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v1/fused_gdn_recurrent_kernel.py#L334)

