---
title: llama_ref/model_with_scan — GSPMD-sharded, scan-over-layers Llama3
type: concept
provenance: mixed
concept: llama_ref-model_with_scan
updated: 2026-07-03
status: fresh
---
# llama_ref/model_with_scan — GSPMD-sharded, scan-over-layers Llama3
The [`model.py`](llama_ref-model.md) Llama3 port with two additions: every layer's transformer block is compiled once and executed via `jax.lax.scan` over a stacked-weights tensor (instead of one Python object per layer), and explicit `with_sharding_constraint` calls steer GSPMD's auto-partitioner at the attention-projection boundaries. It is the `model_impl='scan'` option in `llama_ref.run.main` — sharding is expressed as *hints* to XLA's partitioner, in contrast to [model_with_collectives](llama_ref-model_with_collectives.md), which issues the same collectives by hand.

## Entry points
- [`Attention.forward`](../catalog/llama_ref/model_with_scan.md#Attention.forward) — identical structure to [`model.py`'s version](llama_ref-model.md), but inserts [`with_sharding_constraint`](../catalog/llama_ref/model_with_scan.md#with_sharding_constraint) calls on `xq`/`xk`/`xv` (spec `P('fsdp', None, 'tp', None)`) right after the head-dim reshape and on the `wo` output (spec `P('fsdp', None, None)`) before returning — reached once per layer body invocation inside the enclosing scan.

## Mechanism (step-by-step)
1. [`Attention.forward`](../catalog/llama_ref/model_with_scan.md#Attention.forward) projects through [`wq`](../catalog/llama_ref/model_with_scan.md#Attention.wq)/[`wk`](../catalog/llama_ref/model_with_scan.md#Attention.wk)/[`wv`](../catalog/llama_ref/model_with_scan.md#Attention.wv), reshapes using [`head_dim`](../catalog/llama_ref/model_with_scan.md#Attention.head_dim), then immediately applies [`with_sharding_constraint`](../catalog/llama_ref/model_with_scan.md#with_sharding_constraint) to pin the batch axis to `fsdp` and the head axis to `tp` — this is a *hint*, not a collective: GSPMD's auto-partitioner is still free to insert whatever communication it decides is needed to honor the constraint, unlike an explicit `all_gather`/`psum` call.
2. K/V head-repeat ([`repeat_kv`](../catalog/llama_ref/model_with_scan.md#repeat_kv), expanding by [`n_rep`](../catalog/llama_ref/model_with_scan.md#Attention.n_rep) from [`n_local_kv_heads`](../catalog/llama_ref/model_with_scan.md#Attention.n_local_kv_heads) to [`n_local_heads`](../catalog/llama_ref/model_with_scan.md#Attention.n_local_heads)) and the SDPA call proceed exactly as in `model.py`; the projection back through [`wo`](../catalog/llama_ref/model_with_scan.md#Attention.wo) is followed by a second `with_sharding_constraint` re-pinning the result to `('fsdp', None, None)` before the block's residual add.
3. [`ModelArgs.__init__`](../catalog/llama_ref/model_with_scan.md#ModelArgs.__init__) performs the same `n_kv_heads`/`n_heads`/`dim` validation as `model.py`'s, confirming the two files share config semantics even though their execution shape (per-layer object vs. stacked-and-scanned) differs.
4. [`Transformer.norm`](../catalog/llama_ref/model_with_scan.md#Transformer.norm), [`Transformer.output`](../catalog/llama_ref/model_with_scan.md#Transformer.output), and [`Transformer.tok_embeddings`](../catalog/llama_ref/model_with_scan.md#Transformer.tok_embeddings) remain plain `RMSNorm`/`nn.Linear`/`nn.Embedding` — the sharding-constraint treatment is applied only inside `Attention.forward` in this packet's cited subgraph, not at the embedding/output boundary.

## Key data structures
> [!inferred] Not part of this packet's citable subgraph, but visible in the surrounding source: `ScanLayer` (defined in this same file) stacks every `TransformerBlock`'s state-dict tensors along a new leading layer axis (`torch.stack([v for _ in range(num_layers)])`), registers the stack as a single `nn.ParameterDict`, and in `forward` calls `jax.lax.scan` over a `jax.checkpoint`-wrapped `eval_one_layer` closure that uses `torch.func.functional_call` to run one shared `TransformerBlock` module against a different weight slice per scan step — the compiler only ever sees *one* layer's HLO, replayed `num_layers` times, instead of `num_layers` separately-compiled blocks.

## Dynamics (design intent)
> [!inferred] The `jax.checkpoint` policy passed to the scanned layer (`save_and_offload_only_these_names`, naming `decoder_layer_input`/`query_proj`/`key_proj`/`value_proj`/`out_proj` as offloadable) is visible in the surrounding source but not in this packet's cited subgraph — it offloads exactly the activations tagged by `checkpoint_name` calls inside `Attention.forward`/`TransformerBlock.forward` to host memory rather than recomputing them, trading host-device transfer bandwidth for HBM capacity across the scan's `num_layers` steps.
- The two `with_sharding_constraint` calls this file *does* cite are the mechanism by which GSPMD is told "this per-layer scan body should look the same, shard-wise, on every iteration" — without them, XLA's auto-partitioner would have to infer a per-layer sharding independently (or fail to find a consistent one across scan steps).

## Edge cases
- `with_sharding_constraint` is a hint, not a guarantee: if XLA's partitioner cannot satisfy the requested `PartitionSpec` given the surrounding computation's actual sharding, the constraint either forces extra resharding communication or (in stricter modes) can fail to compile — this file's correctness therefore depends on the constraint being *satisfiable*, not merely stated.

## Open questions
> [!inferred] Whether the `with_sharding_constraint` placement chosen here (immediately after the QKV reshape, and again after `wo`) was empirically tuned against alternatives, or is simply the first placement that worked, is not resolvable from the cited subgraph.

## See also
- [llama_ref/model](llama_ref-model.md) — the un-sharded, un-scanned ancestor.
- [llama_ref/model_with_collectives](llama_ref-model_with_collectives.md) — the sibling variant that replaces `with_sharding_constraint` hints with explicit `all_gather`/`psum` collectives (including per-layer FSDP weight-gathering inside the scan body itself).
- [llama_ref/model_original](llama_ref-model_original.md) — the fairscale-parallel root both descend from.
