---
title: 'Module: src/maxtext/models/deepseek_batchsplit_fp8.py'
type: catalog
provenance: extracted
module: src/maxtext/models/deepseek_batchsplit_fp8.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.models.deepseek_batchsplit_fp8`/
symbols:
  kv_projection: kv_projection().
  route_compute_unroute: route_compute_unroute().
  scan_batch_split_layers: scan_batch_split_layers().
  dot: dot().
  batch_split_schedule: batch_split_schedule().
  mla_with_norms.fn: mla_with_norms().fn().
  mla: mla().
  query_projection: query_projection().
  route_compute_unroute.route_fn: route_compute_unroute().route_fn().
  scan_batch_split_layers.batch_split_scan_fn: scan_batch_split_layers().batch_split_scan_fn().
  mla_with_norms: mla_with_norms().
  moe: moe().
  expert_selection: expert_selection().
  compute: compute().
  staggered_call: staggered_call().
  rms_norm: rms_norm().
  compute.get_active_sharding_axes: compute().get_active_sharding_axes().
  quantized_psum_scatter: quantized_psum_scatter().
  gather_weights: gather_weights().
  get_key_value: get_key_value().
  route_compute_unroute.compute_fn: route_compute_unroute().compute_fn().
  route_compute_unroute.unroute_fn: route_compute_unroute().unroute_fn().
  process_activations: process_activations().
  split: split().
  with_data_parallel_constraint: with_data_parallel_constraint().
  _q_psum_scatter_fwd: _q_psum_scatter_fwd().
  yarn: yarn().
  _q_psum_scatter_bwd: _q_psum_scatter_bwd().
  fetch_weights: fetch_weights().
  merge: merge().
  gather_weights.fn: gather_weights().fn().
  expert_indices_and_weights: expert_indices_and_weights().
  route: route().
  unroute: unroute().
  compute.gmm: compute().gmm().
---
# Module: [`src/maxtext/models/deepseek_batchsplit_fp8.py`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py)

## Functions
- `_q_psum_scatter_bwd(axis_name: str, scatter_dimension: int, tiled: bool, res: Any, grads: jax.Array)` — [`L64`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L64) — Backward pass for quantized_psum_scatter.
- `_q_psum_scatter_fwd(x: jax.Array, axis_name: str, scatter_dimension: int, tiled: bool)` — [`L59`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L59)
- `batch_split_scan_fn(inputs, weights, dpos, dseg)` — [`L304`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L304)
- `batch_split_schedule(inputs, weights, positions, segment_ids, *, model_mode, mesh, quant, cfg)` — [`L357`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L357) — Applies the DeepSeek MoE layer with batch-split schedule.
- `compute(x, w0, w1, wo, group_sizes, weights, *, config, mesh)` — [`L941`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L941) — Processes routed tokens through the MLP.
- `compute_fn(inputs)` — [`L1106`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L1106)
- `dot(x, y, quant=None, axes=1)` — [`L444`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L444) — Computes the dot product of two arrays, optionally using quantization.
- `expert_indices_and_weights(gate_logits: jax.Array, pre_bias_logits: jax.Array, num_experts_per_tok: int, routed_scaling_factor: float)` — [`L855`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L855) — Computes expert indices for each token and their corresponding weights.
- `expert_selection(x, routing_kernel, routing_bias, *, num_experts, num_experts_per_tok, routed_scaling_factor, quant)` — [`L871`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L871) — Selects experts for each token and calculates group sizes for each expert.
- `fetch_weights(params, dtype)` — [`L106`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L106) — Fetches weights from params in the proper format for batch-split schedule.
- `fn(weights)` — [`L173`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L173)
- `fn(args)` — [`L489`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L489)
- `gather_weights(weights, mesh)` — [`L170`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L170) — all-gathers FSDP sharded weights.
- `get_active_sharding_axes(pspec_dim_axes, tensor_dim_index)` — [`L1009`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L1009)
- `get_key_value(low_rank_main, key_rope, wkv_b_weights, *, qk_nope_head_dim, num_query_heads, quant)` — [`L735`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L735) — Gets key and value from compressed KV latent vector and key rope.
- `gmm(inputs, kernel, tiling, group_sizes, preferred_element_type, weight_gather_axes)` — [`L944`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L944)
- `kv_projection(inputs, inputs_positions, wkv_a_weights, wkv_b_weights, kv_norm_scale_weights, *, kv_lora_rank, kv_norm_epsilon, qk_rope_head_dim, rope_theta, max_position_embeddings, original_max_position_embeddings, beta_fast, beta_slow, rope_factor, dtype, qk_nope_head_dim, num_query_heads, quant)` — [`L680`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L680) — Performs KV projection.
- `merge(x, split_factor=2)` — [`L162`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L162) — Merges the input microbatches back into a single tensor.
- `mla(inputs, positions, segment_ids, weights, *, model_mode, epsilon, kv_lora_rank, kv_norm_epsilon, qk_nope_head_dim, qk_rope_head_dim, num_query_heads, rope_theta, max_position_embeddings, original_max_position_embeddings, beta_fast, beta_slow, rope_factor, mscale, attention_op_fn, dtype, quant)` — [`L533`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L533) — Performs MLA.
- `mla_with_norms(inputs, weights, decoder_positions, decoder_segment_ids, *, mesh, model_mode, attn_op, normalization_layer_epsilon, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, rope_max_timescale, num_query_heads, max_position_embeddings, original_max_position_embeddings, beta_fast, beta_slow, rope_factor, mscale, dtype, quant)` — [`L462`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L462) — Performs MLA with pre- and post-normalization.
- `moe(inputs, weights, *, mesh, num_experts, num_experts_per_tok, routed_scaling_factor, expert_axis_name, use_gather_mosaic_kernel, config, quant)` — [`L822`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L822) — Performs dropless MoE with tensor/expert parallelism.
- `process_activations(xs, weights, *, mesh, num_experts, num_experts_per_tok, routed_scaling_factor, expert_axis_name, use_gather_mosaic_kernel, config, quant)` — [`L1136`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L1136) — Processes activations, which are fully sharded on the batch axis, with tensor/expert sharded weights.
- `quantized_psum_scatter(x: jax.Array, axis_name: str, scatter_dimension: int, tiled: bool)` — [`L42`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L42) — Forward: Standard BF16 Reduce-Scatter.
- `query_projection(inputs_q, inputs_positions, wq_a_weights, wq_b_weights, q_norm_scale_weights, *, epsilon, qk_nope_head_dim, qk_rope_head_dim, rope_theta, max_position_embeddings, original_max_position_embeddings, beta_fast, beta_slow, rope_factor, dtype, mscale, quant)` — [`L623`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L623) — Performs query projection.
- `rms_norm(x, scale, *, epsilon, dtype)` — [`L756`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L756) — RMS normalization.
- `route(x, selected_experts, weights, group_sizes, *, expert_axis_name, use_gather_mosaic_kernel)` — [`L895`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L895) — All-gather tokens and then perform local routing.
- `route_compute_unroute(xs, weights, *, num_experts, num_experts_per_tok, routed_scaling_factor, expert_axis_name, use_gather_mosaic_kernel, config, mesh, quant)` — [`L1059`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L1059) — Routes, processes, and unroutes activations.
- `route_fn(inputs)` — [`L1080`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L1080)
- `scan_batch_split_layers(inputs, params, positions, segment_ids, *, model_mode, mesh, quant, cfg, policy)` — [`L290`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L290) — Scans the layers with batch-split schedule.
- `split(x, split_factor=2)` — [`L150`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L150) — Splits the input into `split_factor` parts along the batch dimension.
- `staggered_call(fn, xs)` — [`L426`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L426)
- `unroute(x, selected_experts, *, expert_axis_name, use_gather_mosaic_kernel)` — [`L922`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L922) — Undo `route()`.
- `unroute_fn(inputs)` — [`L1120`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L1120)
- `with_data_parallel_constraint(x, mesh)` — [`L435`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L435)
- `yarn(inputs, positions, *, embedding_dims, rope_theta, max_position_embeddings, original_max_position_embeddings, beta_fast, beta_slow, rope_factor, fprop_dtype)` — [`L764`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit_fp8.py#L764) — Performs YaRN rotary embedding.

