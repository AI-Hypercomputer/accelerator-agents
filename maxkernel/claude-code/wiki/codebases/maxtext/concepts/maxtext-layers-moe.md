---
title: "MaxText RoutedMoE — routing, dispatch/combine, and the megablox vs dense paths"
type: concept
provenance: mixed
concept: maxtext-layers-moe
updated: 2026-07-03
status: fresh
---
# MaxText RoutedMoE — routing, dispatch/combine, and the megablox vs dense paths

## Overview
`RoutedMoE` is MaxText's mixture-of-experts feed-forward block: for every token it scores all
experts with a gate, keeps the top-k, and runs each token only through its selected experts' MLPs.
The single most important thing to understand is that the *same* mathematical block is realized by
**three physically different compute paths**, chosen at call time by config, and they have wildly
different TPU cost profiles: a **dense/capacity path** ([`dense_matmul`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.dense_matmul)) that runs every expert over a padded, token-dropping capacity buffer as ordinary einsums; a **sparse/grouped-matmul path** ([`sparse_matmul`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.sparse_matmul)) that sorts tokens by expert and calls a ragged grouped GEMM (`gmm`, megablox/tokamax/`ragged_dot`); and a **fused inference path** ([`fused_moe_matmul`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.fused_moe_matmul)). Under expert parallelism the sparse path is where the expensive collectives live: a `ragged_all_to_all` dispatch of tokens to the shard that owns their expert, and its dual on the way back. This page is about where the FLOPs and the communication actually go, because that is what an optimizer tunes.

## Diagram
```mermaid
flowchart TD
  call["__call__"] --> gate["gate → gate_logits"]
  call -->|"cfg.attention == vllm_rpa"| fused["fused_moe_matmul"]
  call -->|"cfg.sparse_matmul"| sparse["sparse_matmul"]
  call -->|"else"| dense["dense_matmul"]

  sparse --> shardmap["sparse_matmul_route_and_compute (jax.shard_map)"]
  shardmap --> route["route"]
  route --> topk["get_topk (top-k / hash / deepseek)"]
  route --> permute["permute (argsort → group by expert)"]
  route -->|"EP>1"| ra2a["ragged_all_to_all + local_permute"]
  shardmap --> gmmup["gmm_up → gmm (ragged grouped GEMM)"]
  gmmup --> act["apply_ffn_activation"]
  shardmap --> gmmdown["gmm (wo)"]
  gmmdown -->|"EP>1"| unsort["unsort_output_and_ra2a"]
  unsort --> unpermute["unpermute (scatter + weight combine)"]
  gmmdown --> unpermute

  dense --> topk2["get_topk"]
  dense --> masks["generate_masks (capacity, token dropping)"]
  masks --> einsum["get_einsum: dispatch → wi_0/wi_1 → wo → combine"]
```

## Design rationale (why it's built this way)
The dense path and the sparse path exist because "route each token to k experts" has no single good
TPU lowering. The dense path expressed in [`generate_masks`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.generate_masks) turns routing into a fixed-shape `expert_capacity` buffer and a one-hot `dispatch_mask`/`combine_mask` pair, so the whole block becomes dense einsums (`BSM,BSEC->EBCM`, etc.) that XLA compiles beautifully — at the cost of computing padding for under-full experts and *dropping* tokens for over-full ones once `capacity_factor` is exceeded. The sparse path instead physically **sorts tokens by expert** in [`permute`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.permute) so that a single ragged grouped-matmul ([`gmm`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.gmm)) processes contiguous per-expert token blocks with no padding and no dropping — but now the sort, the scatter-back, and (under EP) an all-to-all become explicit costs.

> [!inferred]
> The dense path's capacity/dropping design is the classic Switch-Transformer trade: fixed shapes
> and clean fusion in exchange for dropped tokens. The sparse path is the "dropless" MoE trade: no
> waste, but you pay for irregular gather/scatter and ragged collectives. Which one wins on a given
> TPU config is exactly the kind of question this wiki's experiments settle empirically.

A second, subtler decision: the *entire* route-and-compute body of the sparse path runs inside one [`sparse_matmul_route_and_compute`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.sparse_matmul_route_and_compute) closure wrapped in a single `jax.shard_map`. Doing routing, the grouped GEMMs, and the all-to-alls under one manual-collective region lets the author place `ragged_all_to_all`/`psum_scatter` exactly where they belong relative to the GEMMs, rather than hoping the compiler infers the communication from logical shardings.

## Entry points
- [`__call__`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.__call__) — the block entry. It runs the [`gate`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.gate) to produce `gate_logits`/`pre_bias_logits`, materializes the expert kernels from [`wi`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.wi)/[`wo`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.wo) (optionally splitting the fused `wi` into w0/w1, applying [`per_expert_scale`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.per_expert_scale) and the [`wi_0_sparsity_module`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.wi_0_sparsity_module)/[`wi_1_sparsity_module`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.wi_1_sparsity_module)/[`wo_sparsity_module`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.wo_sparsity_module)), then dispatches to exactly one of the three compute paths. Control reaches it once per MoE decoder layer per forward.
- [`sparse_matmul`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.sparse_matmul) — the grouped-matmul path, selected when `cfg.sparse_matmul` is set. It first computes all `PartitionSpec`s via [`get_routed_moe_shardings`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.get_routed_moe_shardings), shards the inputs/kernels with [`_maybe_shard_with_pspec`](../catalog/src/maxtext/layers/moe.md#RoutedMoE._maybe_shard_with_pspec), then enters the shard-mapped [`sparse_matmul_route_and_compute`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.sparse_matmul_route_and_compute).
- [`dense_matmul`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.dense_matmul) — the capacity/einsum path, the `else` branch. It calls [`get_topk`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.get_topk), the load-balance auxiliary loss via [`load_balance_loss`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.load_balance_loss), and either the capacity path through [`generate_masks`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.generate_masks)/[`generate_masks_subgroup`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.generate_masks_subgroup) or the no-capacity dense einsum.
- [`fused_moe_matmul`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.fused_moe_matmul) — the inference-only path (`cfg.attention == "vllm_rpa"`, non-hash), delegating to `tpu_inference`'s fused MoE kernel. Bypassed for [`is_hash_routing`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.is_hash_routing) layers.

## Mechanism (step-by-step)

1. **Gate and weight prep.** [`__call__`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.__call__) casts inputs to [`dtype`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.dtype), runs the [`gate`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.gate) (optionally in float32 for stable logits), and prepares the three expert kernels. When `prefuse_moe_weights` is set the gate+up projections are stored fused in [`wi`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.wi) and split into `w0_kernel`/`w1_kernel` here; otherwise separate `wi_0`/`wi_1` params are read. This is also where quantization sparsity masks are applied. The routing cost begins here: the gate is a dense `[B,S,E]` matmul over *all* [`num_experts`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.num_experts).

2. **Top-k selection.** [`get_topk`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.get_topk) chooses [`num_experts_per_tok`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.num_experts_per_tok) experts per token. It has several routing dialects: plain `jax.lax.top_k`+softmax; DeepSeek grouped routing (`deepseek_routing` selects experts only from the top expert-groups, and `deepseek_scale_weights` applies the sigmoid renorm + `routed_scaling_factor`); Gemma4/Llama4 variants; and hash routing, where [`is_hash_routing`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.is_hash_routing) uses the static [`tid2eid`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.tid2eid) token-id→expert-id table indexed by `input_ids` instead of a learned gate. Hash layers deterministically route and emit no auxiliary loss.

3. **Sparse path — sort tokens by expert.** Inside the shard-mapped body, [`route`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.route) calls [`permute`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.permute), which flattens `[B,S]` to a token list, replicates each token `num_experts_per_tok` times, `argsort`s by selected expert, and computes `group_sizes = bincount(selected_experts)` — the contiguous per-expert block lengths the grouped GEMM needs. It also computes the load-balance loss and, when `should_update_load_balance()` returns true ([`should_update_load_balance`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.should_update_load_balance)), the loss-free routed-bias updates. With `use_ragged_sort` + ring-of-experts it instead uses the fused [`ring_ragged_sort`](../catalog/src/maxtext/kernels/ragged/ragged_sort.md#ring_ragged_sort) Pallas kernel, sizing a `ragged_buffer` that can intentionally drop tokens when `ragged_buffer_factor > 0`.

4. **Sparse path — dispatch across expert-parallel shards.** When [`get_expert_parallelism_size`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.get_expert_parallelism_size) > 1, [`route`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.route) must move each token to the shard that owns its expert. It all-gathers per-shard group sizes, computes the `ragged_all_to_all` offset/size vectors with [`get_all_to_all_params`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.get_all_to_all_params) (whose [`transform_array`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.transform_array)/[`TransformStrategy`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.get_all_to_all_params.TransformStrategy) build input/send/output/recv offsets), issues `jax.lax.ragged_all_to_all` over [`_expert_parallelism_name`](../catalog/src/maxtext/layers/moe.md#RoutedMoE._expert_parallelism_name), then re-sorts the received tokens locally with [`local_permute`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.local_permute) (which uses [`a2a_ragged_sort`](../catalog/src/maxtext/kernels/ragged/ragged_sort.md#a2a_ragged_sort)). **This all-to-all is the dominant EP communication cost of the block.**

5. **Sparse path — grouped matmul (gmm).** [`get_gmm_for_local_experts`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.get_gmm_for_local_experts) binds the local `group_sizes`/`group_offset` into a partial [`gmm`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.gmm). [`gmm_up`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.gmm_up) runs the two up-projections (gate + up), fusing them into one grouped GEMM when `prefuse_moe_weights` is set, then applies the SwiGLU via [`apply_ffn_activation`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.apply_ffn_activation); the down-projection is a second `gmm`. `gmm` itself has three backends selected by config: tokamax `ragged_dot` (`use_tokamax_gmm`), the forked megablox `mblx.gmm` (`config.megablox`, the historical fast path), or stock `jax.lax.ragged_dot`. It also carries the per-dimension `wi_tile_*`/`wo_tile_*` tiling knobs and `weight_gather_axes` for gathering FSDP-sharded weights inside the kernel.

6. **Sparse path — combine and return tokens.** After the down-projection, under tensor parallelism a `psum_scatter` reduces the hidden dim; under EP [`unsort_output_and_ra2a`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.unsort_output_and_ra2a) does the *reverse* ragged-all-to-all (via [`a2a_ragged_unsort`](../catalog/src/maxtext/kernels/ragged/ragged_sort.md#a2a_ragged_unsort)) to return each token's result to its original shard. Finally [`unpermute`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.unpermute) inverts the argsort and does the weighted combine (`einsum("BKE,BK->BE")`) that sums each token's k expert outputs back into one vector.

7. **Dense path — capacity, masks, token dropping.** When `capacity_factor > 0`, [`dense_matmul`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.dense_matmul) calls [`generate_masks`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.generate_masks) (training) or [`generate_masks_subgroup`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.generate_masks_subgroup) (inference). `generate_masks` computes `expert_capacity = ceil(tokens_per_batch/num_experts)*capacity_factor`, cumsums the one-hot assignment to find each token's position within its expert, and truncates anything past capacity — literally dropping tokens — producing a boolean `dispatch_mask` and a float `combine_mask`. The MLP is then pure einsums (`dispatch`, `wi_0`/`wi_1`, `wo`, `combine`) sharded across the expert axis, with [`_maybe_shard_moe_dispatch`](../catalog/src/maxtext/layers/moe.md#RoutedMoE._maybe_shard_moe_dispatch) peeling the `expert` axis off the batch dim so the GEMM stays expert-parallel (all-to-all) rather than falling back to FSDP-style double-mapping.

8. **Dense path — no-capacity dense fallback.** When `capacity_factor <= 0` the block computes *every* expert over *every* token (`BSM,EMH->BSEH`) and then masks the combine with the routing [`reshape_and_update_weights`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.reshape_and_update_weights) scattered `[B,S,E]` weights. This is the simplest lowering and never drops a token, but its FLOPs scale with the *full* expert count, not k — the most expensive path for large `num_experts`.

## Key data structures
- **`gate_logits` / `pre_bias_logits`** — `[batch, seq, num_experts]`. `pre_bias_logits` exists only for deepseek3/4, where routing *selects* on biased logits but *weights* with the pre-bias logits.
- **`group_sizes`** — `[num_experts]` (or local experts under EP); the contiguous per-expert row counts that make the grouped GEMM ragged. The single most load-bearing array in the sparse path.
- **`dispatch_mask` / `combine_mask`** — `[batch, seq, num_experts, expert_capacity]` one-hot tensors in the dense path; encode both *which* expert-slot each token occupies and whether it was dropped.
- **`RouteOutput` / `RouteMetadata`** — the tuple [`route`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.route) returns: sorted inputs, `selected_experts`, `sorted_selected_experts`, `weights`, `lb_loss`, `bias_updates`, plus EP bookkeeping (shard id, `local_sorted_indices`, `all_shards_group_sizes`) needed to reverse the all-to-all.
- **Kernel params** — [`wi`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.wi)/[`wo`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.wo) are `[experts, in, hidden]`/`[experts, hidden, out]`; their logical partitioning is [`wi_kernel_axes`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.wi_kernel_axes) `(embed_moe, None, mlp_moe)` and [`wo_kernel_axes`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.wo_kernel_axes) `(embed_moe, mlp_moe, None)`.

## Dynamics (design intent)
The sharding of the whole sparse path is orchestrated through a small helper stack: [`_logical_to_mesh_axes`](../catalog/src/maxtext/layers/moe.md#RoutedMoE._logical_to_mesh_axes) resolves logical names to mesh `PartitionSpec`s, [`_maybe_shard_with_logical`](../catalog/src/maxtext/layers/moe.md#RoutedMoE._maybe_shard_with_logical) / [`_maybe_shard_with_pspec`](../catalog/src/maxtext/layers/moe.md#RoutedMoE._maybe_shard_with_pspec) apply them, and [`get_routed_moe_shardings`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.get_routed_moe_shardings) assembles the full in/out spec set for the `shard_map`. The default weight specs deliberately use "funky rules to AG over FSDP" (per the source comment) so the grouped-GEMM weights are all-gathered off the FSDP axis, optionally via a two-stage all-gather guarded by `optimization_barrier` to stop XLA from re-fusing the two gathers. [`maybe_all_gather_kernel_weight_in_expert_parallelism`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.maybe_all_gather_kernel_weight_in_expert_parallelism) handles the dense-path analogue. The tensor-parallel reductions ([`get_tensor_parallelism_size`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.get_tensor_parallelism_size), [`_tensor_parallelism_name`](../catalog/src/maxtext/layers/moe.md#RoutedMoE._tensor_parallelism_name)) and the `checkpoint_name` rematerialization markers (`moe_mlpwi_0`, `moe_mlpwo`) are the other perf-relevant hooks, letting the optimizer choose what to keep vs recompute.

> [!inferred]
> No tests in the packet exercise this subgraph, so all ordering/collective claims above are read
> from source, not observed. The relative cost of `ragged_all_to_all` vs the grouped GEMM, and of
> ragged-sort vs argsort, is a profiling question for the experiment loop.

## Edge cases
- **Token dropping** happens two ways: dense-path capacity truncation in [`generate_masks`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.generate_masks), and sparse-path `ragged_buffer_factor > 0` sizing in [`permute`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.permute). Both change model semantics — a "supported" speedup that silently increased dropping would be invalid under this wiki's rules.
- **Hash routing** ([`is_hash_routing`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.is_hash_routing)) requires `input_ids`, produces no load-balance loss, and forces the block off the fused inference kernel back onto [`sparse_matmul`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.sparse_matmul).
- **Inference vs training** diverge sharply: training uses [`generate_masks`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.generate_masks) and the expert-peeling dispatch; inference uses [`generate_masks_subgroup`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.generate_masks_subgroup) with an extra context-parallel `N` axis, and skips the auxiliary loss.
- **Llama4** replaces the softmax combine weights with 1.0 and multiplies a sigmoid router score into the inputs instead — an easy correctness trap when porting.
- **ring-of-experts** changes the whole EP strategy: inputs are all-gathered to every shard and routed within-shard, so the reverse path is a `psum_scatter` rather than a second `ragged_all_to_all`.

## Open questions
- The megablox forked kernel (`mblx.gmm`) and tokamax `ragged_dot` are referenced from [`gmm`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.gmm) but live outside this subgraph; their tiling/autotuning internals need the megablox kernel packet.
- [`ragged_gather`](../catalog/src/maxtext/kernels/ragged/ragged_gather.md#ragged_gather) and [`ragged_gather_reduce`](../catalog/src/maxtext/kernels/ragged/ragged_gather_reduce.md#ragged_gather_reduce) are in the subgraph as the primitives under the ragged-sort kernels, but the fallback-vs-Pallas selection logic (`enforce_fallback`, cost overrides) is only visible at their own def sites.
- `calculate_load_balance_updates` (the loss-free bias update) and `RouteOutput`/`RouteMetadata` are used by [`route`](../catalog/src/maxtext/layers/moe.md#RoutedMoE.route) but are not in-subgraph, so their exact update rule is unconfirmed here.

## See also
- [DeepSeek model (MLA + MoE)](maxtext-models-deepseek.md) — the primary consumer, wiring `RoutedAndSharedMoE` into each sparse decoder layer.
