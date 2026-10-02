---
title: 'Module: src/maxtext/models/deepseek_batchsplit.py'
type: catalog
provenance: extracted
module: src/maxtext/models/deepseek_batchsplit.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.models.deepseek_batchsplit`/
symbols:
  scan_batch_split_layers.process_all_layers_bwd: scan_batch_split_layers().process_all_layers_bwd().
  batch_split_layer: batch_split_layer().
  scan_batch_split_layers: scan_batch_split_layers().
  scan_batch_split_layers.process_all_layers_bwd.process_layer_bwd_scannable: scan_batch_split_layers().process_all_layers_bwd().process_layer_bwd_scannable().
  route_compute_unroute_bwd: route_compute_unroute_bwd().
  scan_batch_split_layers.process_all_layers_fwd: scan_batch_split_layers().process_all_layers_fwd().
  scan_batch_split_layers.process_all_layers_fwd.process_layer_scannable: scan_batch_split_layers().process_all_layers_fwd().process_layer_scannable().
  batch_split_schedule_bwd: batch_split_schedule_bwd().
  kv_projection: kv_projection().
  shared_expert_and_route: shared_expert_and_route().
  batch_split_layer.process_layer_bwd: batch_split_layer().process_layer_bwd().
  mla: mla().
  mla_remat: mla_remat().
  route_compute_unroute: route_compute_unroute().
  route_compute_unroute_bwd.compute_fn_remat: route_compute_unroute_bwd().compute_fn_remat().
  dot: dot().
  gather_weights: gather_weights().
  batch_split_layer.process_layer_fwd: batch_split_layer().process_layer_fwd().
  query_projection: query_projection().
  batch_split_schedule: batch_split_schedule().
  scheduling_group: scheduling_group().
  reduce_scatter_ws_grad: reduce_scatter_ws_grad().
  all_reduce_ws_grad_dcn: all_reduce_ws_grad_dcn().
  unroute_ubatch_remat_and_bwd_shard_mapped: unroute_ubatch_remat_and_bwd_shard_mapped().
  gmm: gmm().
  unroute_ubatch_shard_mapped: unroute_ubatch_shard_mapped().
  extract_layer_weights: extract_layer_weights().
  mla_with_norms: mla_with_norms().
  mla_with_norms.fn: mla_with_norms().fn().
  mla_with_norms_remat.remat_fn: mla_with_norms_remat().remat_fn().
  mla_with_norms_bwd: mla_with_norms_bwd().
  expert_selection: expert_selection().
  route_compute_unroute.compute_fn: route_compute_unroute().compute_fn().
  unroute_ubatch_remat_and_bwd_shard_mapped.unroute_ubatch_remat_and_bwd_fn: unroute_ubatch_remat_and_bwd_shard_mapped().unroute_ubatch_remat_and_bwd_fn().
  compute_gating: compute_gating().
  rms_norm: rms_norm().
  batch_split_layer.process_layer: batch_split_layer().process_layer().
  scan_batch_split_layers.process_all_layers: scan_batch_split_layers().process_all_layers().
  route_impl: route_impl().
  unroute_impl: unroute_impl().
  ragged_dot_with_mask: ragged_dot_with_mask().
  megablox_gmm_with_mask: megablox_gmm_with_mask().
  compute_linear: compute_linear().
  unroute_ubatch_fn: unroute_ubatch_fn().
  unroute_ubatch_fn_remat: unroute_ubatch_fn_remat().
  split: split().
  insert_layer_ws_grad: insert_layer_ws_grad().
  route_impl_fwd: route_impl_fwd().
  tpu_flash_attention: tpu_flash_attention().
  tpu_flash_attention_bwd: tpu_flash_attention_bwd().
  mla_with_norms_remat: mla_with_norms_remat().
  mla_with_norms_bwd.bwd_fn: mla_with_norms_bwd().bwd_fn().
  get_key_value: get_key_value().
  shared_expert: shared_expert().
  expert_indices_and_weights: expert_indices_and_weights().
  route: route().
  unroute: unroute().
  route_impl_bwd: route_impl_bwd().
  unroute_impl_bwd: unroute_impl_bwd().
  route_compute_unroute.route_fn: route_compute_unroute().route_fn().
  route_compute_unroute.compute_gating_fn: route_compute_unroute().compute_gating_fn().
  route_compute_unroute.compute_linear_fn: route_compute_unroute().compute_linear_fn().
  route_compute_unroute.unroute_fn: route_compute_unroute().unroute_fn().
  route_compute_unroute_bwd.route_fn_remat: route_compute_unroute_bwd().route_fn_remat().
  route_compute_unroute_bwd.compute_fn_remat.remat: route_compute_unroute_bwd().compute_fn_remat().remat().
  route_compute_unroute_bwd.compute_gating_fn_remat: route_compute_unroute_bwd().compute_gating_fn_remat().
  route_compute_unroute_bwd.compute_linear_fn_remat: route_compute_unroute_bwd().compute_linear_fn_remat().
  route_compute_unroute_bwd.unroute_fn_remat: route_compute_unroute_bwd().unroute_fn_remat().
  route_compute_unroute_bwd.unroute_fn_bwd: route_compute_unroute_bwd().unroute_fn_bwd().
  moe: moe().
  moe_bwd: moe_bwd().
  staggered_call: staggered_call().
  unroute_impl_fwd: unroute_impl_fwd().
  sum_grads: sum_grads().
  fetch_weights: fetch_weights().
  merge: merge().
  init_splash_kernel: init_splash_kernel().
  initialize_yarn_mask: initialize_yarn_mask().
  initialize_yarn_freqs: initialize_yarn_freqs().
  yarn: yarn().
  ragged_dot_with_mask_fwd: ragged_dot_with_mask_fwd().
  megablox_gmm_with_mask_fwd: megablox_gmm_with_mask_fwd().
  unroute_ubatch_fn_bwd: unroute_ubatch_fn_bwd().
  gather_weights.fn: gather_weights().fn().
  reduce_scatter_ws_grad.fn: reduce_scatter_ws_grad().fn().
  all_reduce_ws_grad_dcn.fn: all_reduce_ws_grad_dcn().fn().
  tpu_flash_attention.wrap_flash_attention_manual: tpu_flash_attention().wrap_flash_attention_manual().
  tpu_flash_attention_bwd.wrap_flash_attention_manual_bwd: tpu_flash_attention_bwd().wrap_flash_attention_manual_bwd().
  mla_bwd: mla_bwd().
  expert_group_mask: expert_group_mask().
  ragged_dot_with_mask_bwd: ragged_dot_with_mask_bwd().
  megablox_gmm_with_mask_bwd: megablox_gmm_with_mask_bwd().
  route_compute_unroute_bwd.route_fn_bwd: route_compute_unroute_bwd().route_fn_bwd().
  route_compute_unroute_bwd.compute_fn_remat.use_res: route_compute_unroute_bwd().compute_fn_remat().use_res().
  route_compute_unroute_bwd.compute_fn_bwd: route_compute_unroute_bwd().compute_fn_bwd().
---
# Module: [`src/maxtext/models/deepseek_batchsplit.py`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py)

## Functions
- `all_reduce_ws_grad_dcn(ws_grad, mesh)` — [`L386`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L386) — all-reduces weight gradients across DCN axes. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `batch_split_layer(inputs, params, positions, *, mesh, cfg)` — [`L648`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L648) — Processes a single layer with batch-split schedule. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `batch_split_schedule(inputs, weights, positions, *, mesh, cfg, splash_kernel, activation_pspec, pairwise_swap_and_negate_mask)` — [`L1098`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1098) — Applies the DeepSeek MoE layer with batch-split schedule. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `batch_split_schedule_bwd(residuals, outputs_grad, weights, positions, *, mesh, cfg, splash_kernel, activation_pspec, pairwise_swap_and_negate_mask)` — [`L1156`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1156) — Performs the backward pass for a single layer. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `bwd_fn(args)` — [`L1370`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1370)
- `compute_fn(inputs)` — [`L2237`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2237)
- `compute_fn_bwd(inputs)` — [`L2531`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2531) — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `compute_fn_remat(inputs)` — [`L2511`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2511) — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `compute_gating(x, w0, w1, group_sizes, *, dtype, config)` — [`L2150`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2150) — Computes the gating GMMs.
- `compute_gating_fn(inputs)` — [`L2244`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2244)
- `compute_gating_fn_remat(inputs)` — [`L2541`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2541)
- `compute_linear(layer_w0, layer_w1, wo, group_sizes, weights, *, dtype, config)` — [`L2169`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2169) — Combines the outputs of the gating GMMs and computes the final GMM.
- `compute_linear_fn(inputs)` — [`L2256`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2256)
- `compute_linear_fn_remat(inputs)` — [`L2556`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2556)
- `dot(x, y, axes=1)` — [`L1227`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1227)
- `expert_group_mask(gate_logits, *, n_routing_groups, topk_routing_group, top_k_in_group)` — [`L1841`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1841) — Computes expert group mask for node-limited routing.
- `expert_indices_and_weights(gate_logits: jax.Array, pre_bias_logits: jax.Array, num_experts_per_tok: int, routed_scaling_factor: float, n_routing_groups: int, topk_routing_group: int, top_k_in_group: int)` — [`L1870`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1870) — Computes expert indices for each token and their corresponding weights.
- `expert_selection(x, routing_kernel, routing_bias, *, num_experts: int, num_experts_per_tok: int, routed_scaling_factor: float, n_routing_groups: int, topk_routing_group: int, top_k_in_group: int)` — [`L1897`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1897) — Selects experts for each token and calculates group sizes for each expert.
- `extract_layer_weights(all_weights, layer_idx, layer_axis)` — [`L108`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L108) — Extracts the weights for given layer. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `fetch_weights(params, dtype)` — [`L46`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L46) — Fetches weights from params in the proper format for batch-split schedule. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `fn(weights)` — [`L128`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L128)
- `fn(ws_grad)` — [`L269`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L269)
- `fn(ws_grad)` — [`L392`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L392)
- `fn(args)` — [`L1255`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1255)
- `gather_weights(weights, mesh)` — [`L125`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L125) — all-gathers FSDP sharded weights. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `get_key_value(low_rank_main, key_rope, wkv_b_weights, *, qk_nope_head_dim, num_query_heads)` — [`L1684`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1684) — Gets key and value from compressed KV latent vector and key rope.
- `gmm(inputs, kernel, group_sizes, preferred_element_type, config)` — [`L2109`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2109) — Performs a Grouped Matrix Multiplication (GMM).
- `init_splash_kernel(config)` — [`L499`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L499) — Initializes the Splash kernel. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `initialize_yarn_freqs(positions, embedding_dims, rope_theta, max_position_embeddings, original_max_position_embeddings, beta_fast, beta_slow, rope_factor, mesh, activation_pspec)` — [`L1723`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1723) — Initializes YaRN frequencies. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `initialize_yarn_mask(embedding_dims)` — [`L1713`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1713) — Initializes YaRN mask. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `insert_layer_ws_grad(all_ws_grad, ws_grad, layer_idx, layer_axis)` — [`L116`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L116) — Inserts the weight gradients for given layer. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `kv_projection(inputs, yarn_freqs, wkv_a_weights, wkv_b_weights, kv_norm_scale_weights, *, kv_lora_rank, kv_norm_epsilon, pairwise_swap_and_negate_mask, dtype, qk_nope_head_dim, num_query_heads, config, mesh, activation_pspec)` — [`L1640`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1640) — Performs KV projection.
- `megablox_gmm_with_mask(x, ws, group_sizes, dtype, qwix_rule)` — [`L2074`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2074)
- `megablox_gmm_with_mask_bwd(dtype, qwix_rule, res, grad)` — [`L2098`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2098) — Dual of Megablox GMM and masks gradients of unprocessed tokens.
- `megablox_gmm_with_mask_fwd(x, ws, group_sizes, dtype, qwix_rule)` — [`L2078`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2078) — Performs Megablox GMM and masks unprocessed tokens.
- `merge(x, split_factor=2)` — [`L100`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L100) — Merges the input microbatches back into a single tensor. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `mla(inputs, yarn_freqs, weights, *, epsilon, kv_lora_rank, kv_norm_epsilon, qk_nope_head_dim, qk_rope_head_dim, num_query_heads, max_position_embeddings, original_max_position_embeddings, rope_factor, mscale, config, splash_kernel, pairwise_swap_and_negate_mask, dtype, mesh, activation_pspec)` — [`L1385`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1385) — Performs MLA.
- `mla_bwd(out_grad, bwds)` — [`L1564`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1564) — Performs the backward pass for the mla function.
- `mla_remat(residuals, yarn_freqs, weights, *, epsilon, kv_lora_rank, kv_norm_epsilon, qk_nope_head_dim, qk_rope_head_dim, num_query_heads, max_position_embeddings, original_max_position_embeddings, rope_factor, mscale, config, splash_kernel, pairwise_swap_and_negate_mask, dtype, mesh, activation_pspec)` — [`L1464`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1464) — Performs remat for the mla function. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `mla_with_norms(inputs, weights, yarn_freqs, *, mesh, config, splash_kernel, normalization_layer_epsilon, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, num_query_heads, max_position_embeddings, original_max_position_embeddings, rope_factor, mscale, pairwise_swap_and_negate_mask, dtype, activation_pspec)` — [`L1231`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1231) — Performs MLA with pre-normalization. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `mla_with_norms_bwd(outputs_grad, bwds)` — [`L1364`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1364) — Performs the backward pass for the mla_with_norms function. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `mla_with_norms_remat(residuals, weights, yarn_freqs, *, mesh, config, splash_kernel, normalization_layer_epsilon, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, num_query_heads, max_position_embeddings, original_max_position_embeddings, rope_factor, mscale, pairwise_swap_and_negate_mask, dtype, activation_pspec)` — [`L1294`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1294) — Performs remat for the mla_with_norms function. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `moe(xs, weights, *, mesh, num_experts, num_experts_per_tok, routed_scaling_factor, n_routing_groups, topk_routing_group, top_k_in_group, expert_axis_name, num_expert_shards, capacity_factor, use_gather_mosaic_kernel, config, normalization_layer_epsilon, dtype, activation_pspec)` — [`L2614`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2614) — Performs dropless MoE with tensor/expert parallelism. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `moe_bwd(residuals, outputs_grad, weights, *, mesh, num_experts, num_experts_per_tok, routed_scaling_factor, n_routing_groups, topk_routing_group, top_k_in_group, expert_axis_name, num_expert_shards, use_gather_mosaic_kernel, config, normalization_layer_epsilon, dtype, activation_pspec)` — [`L2692`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2692) — Performs the backward pass for the moe function. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `process_all_layers(inputs, all_weights, yarn_freqs)` — [`L805`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L805) — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `process_all_layers_bwd(res, g)` — [`L926`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L926) — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `process_all_layers_fwd(inputs, all_weights, yarn_freqs)` — [`L808`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L808) — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `process_layer(inputs, all_weights, yarn_freqs)` — [`L683`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L683) — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `process_layer_bwd(res, g)` — [`L717`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L717) — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `process_layer_bwd_scannable(carry, res_and_layer_idx, group_id)` — [`L944`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L944) — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `process_layer_fwd(inputs, weights, yarn_freqs)` — [`L686`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L686)
- `process_layer_scannable(carry, layer_idx, group_id)` — [`L809`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L809) — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `query_projection(inputs_q, yarn_freqs, wq_a_weights, wq_b_weights, q_norm_scale_weights, *, epsilon, qk_nope_head_dim, qk_rope_head_dim, max_position_embeddings, original_max_position_embeddings, rope_factor, pairwise_swap_and_negate_mask, dtype, mscale, config, mesh, activation_pspec)` — [`L1589`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1589) — Performs query projection.
- `ragged_dot_with_mask(x, ws, group_sizes, dtype)` — [`L2038`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2038)
- `ragged_dot_with_mask_bwd(dtype, res, grad)` — [`L2059`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2059) — Dual of ragged dot and masks gradients of unprocessed tokens.
- `ragged_dot_with_mask_fwd(x, ws, group_sizes, dtype)` — [`L2042`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2042) — Performs ragged dot and masks unprocessed tokens.
- `reduce_scatter_ws_grad(ws_grad, mesh)` — [`L263`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L263) — reduce-scatters weight gradients to FSDP sharding. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `remat()` — [`L2517`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2517)
- `remat_fn(args)` — [`L1319`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1319)
- `rms_norm(x, scale, *, epsilon, dtype, out_sharding=None)` — [`L1705`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1705) — RMS normalization. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `route(x, selected_experts, weights, group_sizes, *, expert_axis_name, num_experts, num_expert_shards, use_gather_mosaic_kernel)` — [`L1926`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1926) — All-gather tokens and then perform local routing.
- `route_compute_unroute(xs, weights, *, num_experts, num_experts_per_tok, routed_scaling_factor, n_routing_groups, topk_routing_group, top_k_in_group, expert_axis_name, num_expert_shards, capacity_factor, use_gather_mosaic_kernel, normalization_layer_epsilon, dtype, config)` — [`L2186`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2186) — Routes, processes, and unroutes activations. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `route_compute_unroute_bwd(residuals, outputs_grad, weights, *, num_experts, num_experts_per_tok, num_expert_shards, routed_scaling_factor, n_routing_groups, topk_routing_group, top_k_in_group, expert_axis_name, use_gather_mosaic_kernel, normalization_layer_epsilon, dtype, config)` — [`L2430`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2430) — Performs the backward pass for route_compute_unroute. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `route_fn(inputs)` — [`L2213`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2213)
- `route_fn_bwd(inputs)` — [`L2484`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2484)
- `route_fn_remat(inputs)` — [`L2457`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2457) — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `route_impl(x, selected_experts, expert_axis_name, use_gather_mosaic_kernel)` — [`L1969`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1969)
- `route_impl_bwd(expert_axis_name, use_gather_mosaic_kernel, res, grad)` — [`L1988`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1988)
- `route_impl_fwd(x, selected_experts, expert_axis_name, use_gather_mosaic_kernel)` — [`L1975`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1975) — Routes the activations and all-gathers across the expert axis.
- `scan_batch_split_layers(inputs, params, positions, *, mesh, cfg, num_layers)` — [`L767`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L767) — Scans the layers with batch-split schedule. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `scheduling_group(group_id)` — [`L42`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L42) — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `shared_expert(inputs, shared_w0, shared_w1, shared_wo)` — [`L1836`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1836)
- `shared_expert_and_route(inputs, post_attn_scale, shared_w0, shared_w1, shared_wo, gate_kernel, gate_bias, *, num_experts, num_experts_per_tok, num_expert_shards, routed_scaling_factor, n_routing_groups, topk_routing_group, top_k_in_group, expert_axis_name, use_gather_mosaic_kernel, config, normalization_layer_epsilon, dtype)` — [`L1780`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1780) — Computes the shared expert and routes the activations.
- `split(x, split_factor=2)` — [`L88`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L88) — Splits the input into `split_factor` parts along the batch dimension. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `staggered_call(fn, xs)` — [`L1213`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1213) — Calls a function in a staggered manner while accumulating residuals. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `sum_grads(grads)` — [`L2426`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2426) — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `tpu_flash_attention(query, key, value, mesh, splash_kernel, activation_pspec)` — [`L528`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L528) — TPU Flash Attention.
- `tpu_flash_attention_bwd(attention_out_grad, query, key, value, attention_output, logsumexp, mesh, splash_kernel, activation_pspec)` — [`L578`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L578) — TPU Flash Attention backward.
- `unroute(x, selected_experts, *, expert_axis_name, use_gather_mosaic_kernel)` — [`L1955`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1955) — Undo `route()`.
- `unroute_fn(inputs)` — [`L2269`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2269)
- `unroute_fn_bwd(inputs)` — [`L2580`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2580)
- `unroute_fn_remat(inputs)` — [`L2572`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2572) — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `unroute_impl(x, selected_experts, expert_axis_name, use_gather_mosaic_kernel)` — [`L1999`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1999)
- `unroute_impl_bwd(expert_axis_name, use_gather_mosaic_kernel, res, grad)` — [`L2020`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2020)
- `unroute_impl_fwd(x, selected_experts, expert_axis_name, use_gather_mosaic_kernel)` — [`L2005`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2005) — Unroutes the activations and reduce-scatters across the expert axis.
- `unroute_ubatch_fn(moe_inputs, routed_expert_out, shared_expert_out, selected_experts, *, expert_axis_name, use_gather_mosaic_kernel, target_length)` — [`L2330`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2330) — Performs the unroute operation for a single microbatch.
- `unroute_ubatch_fn_bwd(outputs_grad, unroute_bwd)` — [`L2415`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2415)
- `unroute_ubatch_fn_remat(selected_experts, *, expert_axis_name, use_gather_mosaic_kernel)` — [`L2400`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2400)
- `unroute_ubatch_remat_and_bwd_fn(selected_experts, outputs_grad, *, expert_axis_name, use_gather_mosaic_kernel)` — [`L2362`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2362)
- `unroute_ubatch_remat_and_bwd_shard_mapped(selected_experts, outputs_grad, *, expert_axis_name, use_gather_mosaic_kernel, mesh, activation_pspec)` — [`L2351`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2351) — Performs remat and backward pass for unroute_ubatch in a shard map. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `unroute_ubatch_shard_mapped(moe_inputs, routed_expert_out, shared_expert_out, selected_experts, *, expert_axis_name, use_gather_mosaic_kernel, target_length, mesh, activation_pspec)` — [`L2288`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2288) — Performs the unroute operation for a single microbatch in a shard map. — documented in [maxtext-models-deepseek_batchsplit](../../../../concepts/maxtext-models-deepseek_batchsplit.md)
- `use_res()` — [`L2520`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L2520)
- `wrap_flash_attention_manual(query, key, value)` — [`L560`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L560)
- `wrap_flash_attention_manual_bwd(res, grad)` — [`L624`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L624)
- `yarn(inputs, freqs, *, pairwise_swap_and_negate_mask, fprop_dtype)` — [`L1767`](../../../../../../../raw/code/maxtext/src/maxtext/models/deepseek_batchsplit.py#L1767) — Performs YaRN rotary embedding.

