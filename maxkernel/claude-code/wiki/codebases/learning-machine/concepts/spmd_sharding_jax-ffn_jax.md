---
title: spmd_sharding_jax/ffn_jax — Megatron-style FFN sharding benchmark (Flax)
type: concept
provenance: mixed
concept: spmd_sharding_jax-ffn_jax
updated: 2026-07-03
status: fresh
---
# spmd_sharding_jax/ffn_jax — Megatron-style FFN sharding benchmark (Flax)

<!-- connect:up:begin -->
> **Cross-repo concept:** part of [sharding](../../../concepts/sharding.md) across this wiki's repos.
<!-- connect:up:end -->
A standalone Flax/Optax training-step benchmark for a deep stack of feed-forward blocks under a 2-D `(batch, model)` device mesh, using the classic Megatron column-parallel→row-parallel weight split (so only one resharding communication is needed per block) plus explicit `with_sharding_constraint` calls to keep GSPMD's auto-partitioner aligned with that intent, both on activations during the forward pass and on gradients before the optimizer step.

## Entry points
- [`main`](../catalog/spmd_sharding_jax/ffn_jax.md#main) — builds the `(batch, model)` mesh, constructs [`StackedFFN`](../catalog/spmd_sharding_jax/ffn_jax.md#StackedFFN), assigns explicit per-parameter shardings, and runs the timed training loop under `jax.profiler.trace`.
- [`StackedFFN.__call__`](../catalog/spmd_sharding_jax/ffn_jax.md#StackedFFN.__call__) — the forward pass reached once per training step; chains [`num_layers`](../catalog/spmd_sharding_jax/ffn_jax.md#StackedFFN.num_layers) instances of [`FeedForwardNetwork`](../catalog/spmd_sharding_jax/ffn_jax.md#FeedForwardNetwork) then [`out_proj`](../catalog/spmd_sharding_jax/ffn_jax.md#StackedFFN.out_proj).
- [`train_step`](../catalog/spmd_sharding_jax/ffn_jax.md#main.train_step) — the `jax.jit`-compiled, `donate_argnums=(0,3)`-optimized step that computes gradients via [`loss`](../catalog/spmd_sharding_jax/ffn_jax.md#main.loss), re-applies sharding constraints to every gradient tensor, then runs the AdamW update.

## Mechanism (step-by-step)
1. [`main`](../catalog/spmd_sharding_jax/ffn_jax.md#main) builds a 2-D mesh via `jax.make_mesh((jax.device_count() // model_axis, model_axis), ('batch', 'model'))` — a fixed `model_axis=4` tensor-parallel width times however many devices remain for data parallelism.
2. Each [`FeedForwardNetwork`](../catalog/spmd_sharding_jax/ffn_jax.md#FeedForwardNetwork) holds `dense1`/`dense2` (`flax.linen.Dense`, sized by [`hidden_dim`](../catalog/spmd_sharding_jax/ffn_jax.md#FeedForwardNetwork.hidden_dim)/[`output_dim`](../catalog/spmd_sharding_jax/ffn_jax.md#FeedForwardNetwork.output_dim)); after `dense1`, `jax.lax.with_sharding_constraint` pins the activation to `P('batch', None, 'model')` (hidden dimension sharded on the model axis), and after `dense2`, to `P('batch', None, None)` (back to fully batch-sharded) — the classic column-then-row Megatron pattern, expressed as sharding hints on activations rather than as separate parallel-layer classes.
3. [`main`](../catalog/spmd_sharding_jax/ffn_jax.md#main) assigns each layer's `dense1.kernel` to `NamedSharding(mesh, P('batch', 'model'))` and `dense2.kernel` to `P('model', 'batch')` (note the axis order flip between the two) — this is what makes `dense1`'s output land sharded on `model` and `dense2`'s output need only one row-reduce to become batch-sharded again, matching the `with_sharding_constraint` hints inside each [`FeedForwardNetwork`](../catalog/spmd_sharding_jax/ffn_jax.md#FeedForwardNetwork)'s call (visible in source, described in step 2 above).
4. [`StackedFFN.__call__`](../catalog/spmd_sharding_jax/ffn_jax.md#StackedFFN.__call__) loops every [`FeedForwardNetwork`](../catalog/spmd_sharding_jax/ffn_jax.md#FeedForwardNetwork) in [`layers`](../catalog/spmd_sharding_jax/ffn_jax.md#StackedFFN.layers), applies [`out_proj`](../catalog/spmd_sharding_jax/ffn_jax.md#StackedFFN.out_proj), then constrains the final output to `P('batch', None, None)` — one more explicit re-shard at the very end of the whole stack.
5. [`loss`](../catalog/spmd_sharding_jax/ffn_jax.md#main.loss) — decorated `@jax.jit` and `@partial(jax.profiler.annotate_function, name='loss')` — runs `model.apply(params, x)`, constrains the prediction's sharding, then computes plain MSE against `y`; the `annotate_function` decorator is what makes this show up as a named region in the xprof trace produced by the surrounding `jax.profiler.trace` context.
6. [`train_step`](../catalog/spmd_sharding_jax/ffn_jax.md#main.train_step) calls `gradf = jax.jit(jax.grad(loss))`, then — critically — re-applies `with_sharding_constraint` to *every gradient tensor* individually (`dense1.kernel` grad to `P('batch','model')`, `dense2.kernel` grad to `P('model','batch')`, `out_proj.kernel` grad to `P('batch','model')`) before calling `solver.update`/`optax.apply_updates` — this guards against GSPMD's autodiff-through-sharded-forward-pass producing a gradient sharding that doesn't match the corresponding parameter's sharding, which would otherwise force an implicit reshard inside the optimizer step.
7. The training loop in [`main`](../catalog/spmd_sharding_jax/ffn_jax.md#main) reuses the same `(dummy_x, dummy_y)` batch for every one of `num_steps` iterations (the commented-out `dataset_iter` line shows real data cycling was tried and disabled) and calls `jax.block_until_ready(params)` after each `train_step` so the printed per-step wall-clock time reflects actual device completion, not async dispatch.

## Key data structures
- [`RandomTensorDataset`](../catalog/spmd_sharding_jax/ffn_jax.md#RandomTensorDataset) — pre-generates `element_count` batches of random `(bs, num_tokens, input_dim)` inputs and integer labels on CPU at construction time (via [`data`](../catalog/spmd_sharding_jax/ffn_jax.md#RandomTensorDataset.data)/[`labels`](../catalog/spmd_sharding_jax/ffn_jax.md#RandomTensorDataset.labels)), then `jax.device_put`s each batch onto the configured [`sharding`](../catalog/spmd_sharding_jax/ffn_jax.md#RandomTensorDataset.sharding) lazily inside [`__iter__`](../catalog/spmd_sharding_jax/ffn_jax.md#RandomTensorDataset.__iter__) — synthetic-data generation is decoupled from device placement.
- [`StackedFFN`](../catalog/spmd_sharding_jax/ffn_jax.md#StackedFFN) — a `flax.linen.Module` holding a Python list of [`FeedForwardNetwork`](../catalog/spmd_sharding_jax/ffn_jax.md#FeedForwardNetwork) submodules (not a scanned/stacked representation — each of `num_layers` layers is a distinct traced submodule), plus one final [`out_proj`](../catalog/spmd_sharding_jax/ffn_jax.md#StackedFFN.out_proj) `Dense` layer to a smaller `out_channels` output width.

## Dynamics (design intent)
> [!inferred] The alternating `P('batch','model')`/`P('model','batch')` kernel sharding, paired with matching `with_sharding_constraint`s on the activations between the two dense layers, is the standard tensor-parallel FFN pattern (column-parallel first matmul, row-parallel second matmul) — it needs exactly one communication op (an all-reduce or reduce-scatter, chosen by GSPMD) per `FeedForwardNetwork` block rather than one per matmul, which is the whole point of pairing the two projections this way instead of sharding each independently.

## Edge cases
- [`main`](../catalog/spmd_sharding_jax/ffn_jax.md#main) hardcodes `model_axis = 4` on the line immediately after the parameter `model_axis=4` default — the function's `model_axis`/`num_layers` parameters are shadowed by literal reassignment inside the body, so calling `main(model_axis=8, ...)` would silently have no effect on the mesh's tensor-parallel width (though `num_layers` is reassigned to itself, so that parameter does take effect).
- `train_step`'s `donate_argnums=(0,3)` donates the `params` and `opt_state` buffers — callers must not reuse the pre-call `params`/`opt_state` references afterward, since XLA is permitted to alias/overwrite their device memory for the updated values.

## Open questions
> [!inferred] Whether the per-gradient `with_sharding_constraint` calls in `train_step` are load-bearing (i.e. GSPMD actually produces a mismatched gradient sharding without them) or defensive/no-op in practice is not verifiable from source alone — verifying it would require comparing compiled HLO with and without those constraints.

## See also
- [basic_ops](basic_ops.md) — a smaller, JAX-only sibling benchmark comparing `shard_map` vs GSPMD-auto sharding for a similar gated-FFN shape, without the two-axis tensor+data-parallel mesh this file uses.
- [llama_ref/model_with_scan](llama_ref-model_with_scan.md) — uses the same `with_sharding_constraint`-as-GSPMD-hint pattern, but on a PyTorch/torch_xla2 Llama model rather than a native Flax one.
