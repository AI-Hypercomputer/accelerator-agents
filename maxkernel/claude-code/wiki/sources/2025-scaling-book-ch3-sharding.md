---
title: "How to Scale Your Model — Ch 3: Sharded Matrices and How to Multiply Them"
type: source
tags: [docs, book, scaling-book, sharding, all-gather, all-reduce, reduce-scatter, fsdp, tensor-parallelism, megatron]
author: DeepMind / JAX ML team
book: "How to Scale Your Model"
book_date: "2025-02-04"
chapter: 3
upstream: https://jax-ml.github.io/scaling-book/sharding
created: 2026-04-23
updated: 2026-04-23
---

Chapter 3 of the scaling-book. **Four-case taxonomy for sharded matmul** (neither / one / both contracting dims sharded / conflicting non-contracting shardings) with the exact collective each case requires. The algebra that underlies FSDP, tensor parallelism, and every other partitioned op in this wiki.

## Key claims

1. Sharding notation: `A[I_X, J_Y]` means axis `I` partitioned across mesh axis `X`, axis `J` across `Y`; unnamed axes are replicated.
2. **Case 1** — **neither** contracting dim sharded (`A[I, J]` × `B[J, K]` both replicated): no communication; output follows whichever sharding is specified.
3. **Case 2** — **one** contracting dim sharded (`A[I, J_X]` × `B[J, K]`): **AllGather** the sharded input along `X` before matmul; cost ≈ `2 × sharded_size / (BW × X)`.
4. **Case 3** — **both** contracting dims sharded matchingly (`A[I, J_X]` × `B[J_X, K]`): multiply local blocks, then **AllReduce** (or **ReduceScatter** if output is to be sharded too); total cost ≈ `2 × output_size / BW`.
5. **Case 4** — **conflicting non-contracting shardings** (`A[I_X, J]` × `B[J, K_X]`): invalid — must AllGather one matrix first.
6. **Collective identity**: `AllReduce = AllGather + ReduceScatter` — same total cost, but as two async phases the decomposition enables overlap.
7. **FSDP** replaces full parameter-replication's AllReduce with `AllGather(params, fwd) + ReduceScatter(grads, bwd)` — same total cost as DP AllReduce but lower memory footprint.

## Key data points

### Collective costs (N-way sharded, total byte count `B`, bandwidth `W` per direction)

| Collective | Time (1D ring, bidi) | Time (2D mesh) |
|---|---|---|
| AllGather | `2 B / (W · N)` (per device volume unchanged) | `2 B / (W · N_axis)` |
| ReduceScatter | `2 B / (W · N)` | same |
| AllReduce | `2 B / W` (ring) | `2 B / (W · M_axis)` |

### Practical thresholds

- 1D ring AllReduce on v5p ICI: ~1 ms for 2 GB (`2 × 2e9 / 9e10 ≈ 22 ms` single-direction; halved with bidi — so ~1 ms on short ring).
- Per-token KV scatter at inference: 10–100 µs.

## Techniques referenced

- Block matrix multiplication (the algebraic basis).
- Ring algorithms (AllReduce / AllGather / ReduceScatter on toroidal ICI).
- Bidirectional ICI rings (both directions active).
- Optical-switch superpod reconfiguration (v5p wrap).
- **Megatron-style TP** — Case 2 + Case 3 chained: column-sharded first matmul (AllGather-then-matmul), row-sharded second matmul (matmul-then-ReduceScatter).
- **FSDP / ZeRO-3** — AllGather-forward + ReduceScatter-backward pattern.

## Gaps & caveats

- Assumes uniform ICI with no contention or hotspots; real superpods have optical-switch congestion.
- Cost model ignores latency for small messages and pipelining gains from overlap.
- Doesn't cover **3D tensor parallelism**, **pipeline parallelism**, or **expert parallelism** (MoE / AllToAll) — deferred to Ch 5.
- Assumes single-precision transfers; mixed-precision modifies intensity.
- Book is 2025-02-04 — **v6e and v7 not covered**; collective costs shift with generation.

## Connections

- [concepts/sharding](../concepts/sharding.md)
- [concepts/all-gather](../concepts/all-gather.md) / [all-reduce](../concepts/all-reduce.md) / [reduce-scatter](../concepts/reduce-scatter.md)
- [concepts/fsdp](../concepts/fsdp.md)
- [concepts/tensor-parallelism](../concepts/tensor-parallelism.md)
- [codebases/jax-huggingface](../codebases/jax-huggingface.md) — the Llama-2 TP sharding recipe is a worked Case 3 pairing.

## See also

- [Ch 2 — TPUs](2025-scaling-book-ch2-tpus.md)
- [Ch 5 — Parallelize for Training](2025-scaling-book-ch5-training.md)
- [Ch 10 — JAX Programming](2025-scaling-book-ch10-jax.md)

## Sources

- `raw/code/scaling-book/sharding.md`
- Upstream: <https://jax-ml.github.io/scaling-book/sharding>
