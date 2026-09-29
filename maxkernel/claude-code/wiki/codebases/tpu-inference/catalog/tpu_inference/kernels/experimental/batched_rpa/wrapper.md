---
title: 'Module: tpu_inference/kernels/experimental/batched_rpa/wrapper.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/experimental/batched_rpa/wrapper.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.experimental.batched_rpa.wrapper`/
symbols:
  ragged_paged_attention: ragged_paged_attention().
  calculate_block_sizes: calculate_block_sizes().
  ragged_paged_attention.run_rpa_kernel: ragged_paged_attention().run_rpa_kernel().
  calculate_block_sizes.find_best_block_sizes: calculate_block_sizes().find_best_block_sizes().
  calculate_block_sizes.calculate_vmem_usage: calculate_block_sizes().calculate_vmem_usage().
  prepare_inputs: prepare_inputs().
  calculate_block_sizes.calculate_compute_buffer_time: calculate_block_sizes().calculate_compute_buffer_time().
  get_kv_cache_shape: get_kv_cache_shape().
  prepare_outputs: prepare_outputs().
---
# Module: [`tpu_inference/kernels/experimental/batched_rpa/wrapper.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/wrapper.py)

## Functions
- `calculate_block_sizes(model_cfgs: configs.ModelConfigs, serve_cfgs: configs.ServingConfigs, vmem_limit_bytes: int)` — [`L134`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/wrapper.py#L134) — Calculate optimal block size for decode and prefill.
- `calculate_compute_buffer_time(batch_size: int, bq_c_sz: int, bkv_sz: int)` — [`L212`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/wrapper.py#L212) — Calculate computational complexity of a single compute block.
- `calculate_vmem_usage(batch_size: int, n_buffer: int, bq_sz: int, bkv_sz: int)` — [`L161`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/wrapper.py#L161) — Given tile size, calculate VMEM usage of the kernel.
- `find_best_block_sizes(max_batch_size: int, max_n_buffer: int, fixed_bq_sz: int | None = None)` — [`L223`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/wrapper.py#L223) — Loop through different block sizes to find the most optimal one.
- `get_kv_cache_shape(total_num_pages, page_size, actual_num_kv_heads, actual_head_dim, kv_dtype)` — [`L116`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/wrapper.py#L116)
- `prepare_inputs(q: jax.Array, k: jax.Array, v: jax.Array, q_dtype: jnp.dtype, kv_dtype: jnp.dtype)` — [`L41`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/wrapper.py#L41)
- `prepare_outputs(out: jax.Array)` — [`L111`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/wrapper.py#L111)
- `ragged_paged_attention(queries: jax.Array, keys: jax.Array, values: jax.Array, kv_cache: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array, *, sm_scale: float = 1, sliding_window: int | None = None, soft_cap: float | None = None, mask_value: float | None = None, q_scale: float | None = None, k_scale: float | None = None, v_scale: float | None = None, chunk_prefill_size: int | None = None, decode_block_sizes: configs.BlockSizes | None = None, prefill_block_sizes: configs.BlockSizes | None = None, vmem_limit_bytes: int | None = None, debug_mode: bool = False, out_dtype: jnp.dtype | None = None, use_causal_mask: bool = True, update_kv_cache: bool = True)` — [`L340`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/wrapper.py#L340) — Perform batched ragged paged attention.
- `run_rpa_kernel(mode: configs.RpaCase, o_hbm_alias_q_hbm: jax.Array, kv_cache: jax.Array)` — [`L459`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/wrapper.py#L459)

