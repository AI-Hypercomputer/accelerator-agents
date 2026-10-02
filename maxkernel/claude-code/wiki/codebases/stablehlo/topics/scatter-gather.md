---
title: "scatter / gather — dynamic indexing semantics"
type: concept
tags: [stablehlo, gather, scatter, indexing, embedding, moe, reference]
created: 2026-07-04
updated: 2026-07-04
---

`gather` and `scatter` are StableHLO's **data-dependent indexing** ops — reads and writes at runtime-computed offsets. They are the primitives behind **embedding lookups**, **MoE token dispatch/combine**, and any advanced-indexing pattern. In a dump these are perf-sensitive because their access pattern is irregular (gather/scatter don't tile like a matmul); an unexpectedly slow step often traces to a large `gather`/`scatter` that didn't fold to a `slice`.

## gather — read slices at offsets

`result[result_index] = operand[operand_index]`, where the operand index is assembled from `start_indices` plus per-axis offsets ([gather](src:docs/spec.md#gather)). The dimension-number attributes:

- `offset_dims` / `collapsed_slice_dims` — which result dims come from the sliced window vs. are collapsed (size-1) away.
- `start_index_map` — maps `start_indices` columns to operand dims.
- `index_vector_dim` — which axis of `start_indices` holds the index vector.
- `slice_sizes` — the window pulled per index (constraint C20: one per operand dim).
- `operand_batching_dims` / `start_indices_batching_dims` — batched gather (per-slice indices).
- `indices_are_sorted` — if `true`, the impl may assume sorted indices (else undefined); an optimization hint.

> [!inferred]
> An embedding lookup is a `gather` with `slice_sizes = [1, embed_dim]` collapsing the vocab axis — i.e. pull one row per token id. `indices_are_sorted=true` is a real speed knob the compiler can exploit; unsorted token ids force the general path.

## scatter — write slices at offsets

`results` equal `inputs` except that slices at `scatter_indices` are combined via `update_computation` ([scatter](src:docs/spec.md#scatter)). Mirror attributes: `update_window_dims`, `inserted_window_dims`, `scatter_dims_to_operand_dims`, `index_vector_dim`, plus:

- `indices_are_sorted` and `unique_indices` — both are *assumptions* the impl may exploit; if `unique_indices=true` but indices collide, behavior is **undefined** ([scatter](src:docs/spec.md#scatter)).
- `update_computation` — the combiner (add for scatter-add / gradient accumulation, or overwrite).

> [!inferred]
> Scatter is the backward of gather: embedding-gradient accumulation is a `scatter` with an `add` update_computation. `unique_indices=false` (the safe default) serializes colliding writes and is markedly slower — a scatter-add over duplicate token ids can't assume uniqueness. This is a frequent MoE-combine and embedding-grad hotspot in a trace.

## Relation to dynamic_slice and collectives

- **[dynamic_slice](src:docs/spec.md#dynamic_slice)** / `dynamic_gather` are the simpler single-window cousins; the simplification passes rewrite `gather(X, cst_start_indices)` → `slice` and `dynamic_gather(x, cst) → gather` when indices are constant ([generated-stablehlo_optimization_passes](../sources/generated-stablehlo_optimization_passes.md)). A gather that *should* be a static slice but isn't folding is worth checking.
- MoE **dispatch/combine across devices** uses [all_to_all](collectives.md), a *collective*, distinct from the local `gather`/`scatter` here — don't conflate them in a dump.

## See also

- [collectives](collectives.md), [dot_general](dot_general.md), [optimization-passes](optimization-passes.md) (gather→slice folding)
- Source: [spec.md](../sources/spec.md)

## Sources

- `raw/code/stablehlo/docs/spec.md`
