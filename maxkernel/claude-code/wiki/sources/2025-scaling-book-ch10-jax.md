---
title: "How to Scale Your Model — Ch 10: Programming TPUs in JAX"
type: source
tags: [docs, book, scaling-book, jax, shard-map, pjit, collective-matmul, ppermute, auto-mode, explicit-mode, manual-mode]
author: DeepMind / JAX ML team
book: "How to Scale Your Model"
book_date: "2025-02-04"
chapter: 10
upstream: https://jax-ml.github.io/scaling-book/jax-stuff
created: 2026-04-23
updated: 2026-04-23
---

Chapter 10 of the scaling-book. **Three parallelism programming models in JAX** — Auto (Shardy-decides), Explicit (type-system-propagated), and Manual `shard_map` (device-local, write-collectives-by-hand). Worked **collective-matmul** example: 1.27× speedup over naive AllGather+matmul via `ppermute` in a `fori_loop`.

## Key claims

1. **Auto (Shardy)**: XLA decides sharding from input shardings + program structure; adds AllGather / AllReduce / ReduceScatter where needed. Simplest but opaque — XLA can make mistakes, requiring `with_sharding_constraint` to "tickle" the compiler.
2. **Explicit**: sharding is part of JAX's type system (via `typeof`); JAX propagates automatically for unambiguous ops, errors on ambiguity. User supplies `out_sharding=` to resolve.
3. **Manual `shard_map`**: function sees a **device-local** view (1/N of global shape); collectives (`psum`, `all_gather`, `ppermute`, `all_to_all`) written explicitly.
4. **Collective matmul** worked example (2×4 mesh, `1024 × 2048 × 8192`):
   - Unsharded baseline: 224 µs.
   - Naive AllGather + matmul: 311 µs (37% overhead).
   - **Overlapped collective (`ppermute` inside `fori_loop`)**: 244 µs (**9% overhead** — 1.27× speedup vs naive).
5. **Mesh construction**: `jax.make_mesh(axis_shapes, axis_names)`; axis `AxisType.Explicit` vs `AxisType.Auto` controls compiler behavior.
6. **PartitionSpec `P`**: `P('X', 'Y')` logical sharding; `P(None, 'Y')` replicated on X, sharded on Y.
7. **MoE** pattern: route tokens to k of E experts; local impl avoids materializing `[S, D, F]`; `all_to_all` + `while` loop for ragged processing.

## Key data points

### Collective-matmul microbench (2×4 mesh, `1024 × 2048` × `2048 × 8192`)

| Scheme | Time | Overhead vs unsharded |
|---|---:|---:|
| Unsharded (baseline) | 224 µs | — |
| Naive AllGather + matmul | 311 µs | +37% |
| Overlapped (`ppermute` in loop) | 244 µs | **+9%** |

Overlap saves `(311 - 244) / 311 ≈ 22%` vs naive.

## Techniques referenced

- `jax.jit`, `jax.shard_map` (see [codebases/jax](../codebases/jax.md)).
- `jax.lax.all_gather` / `psum` / `ppermute` / `all_to_all`.
- `jax.lax.with_sharding_constraint` — compiler hint for Auto mode.
- Ring algorithms (ppermute-based).
- MoE token routing / dynamic slicing for expert assignment.

## Gaps & caveats

- Examples are small-array / CPU-emulated; real TPU cluster required for perf validation.
- `shard_map` syntax is complex (device-local semantics, `axes`, `pvary`, `axis_index`).
- Custom-gradient rules for sharded functions not covered.
- Mixed-dtype sharding (quantized weights + full-precision activations) not deeply explored.
- **Auto mode "tickling"** via `with_sharding_constraint` is empirical — book doesn't formalize when it's needed.
- Shardy internals opaque.

## Connections

- [codebases/jax](../codebases/jax.md) — `jax.experimental.shard_map`, `jax.experimental.custom_partitioning`, `jax.experimental.layout`.
- [codebases/jax-huggingface](../codebases/jax-huggingface.md) — Explicit-mode TP example.
- [codebases/tpu-inference](../codebases/tpu-inference.md) — production `all_gather_matmul` applying the overlap pattern.
- [codebases/ejkernel](../codebases/ejkernel.md) — community `all_gather_matmul` / `reduce_scatter_matmul` Pallas kernels.
- [concepts/sharding](../concepts/sharding.md), [all-gather](../concepts/all-gather.md), [async-collectives](../concepts/async-collectives.md).

## See also

- [Ch 3 — Sharded matrices](2025-scaling-book-ch3-sharding.md)
- [Ch 9 — Profiling](2025-scaling-book-ch9-profiling.md)

## Sources

- `raw/code/scaling-book/jax-stuff.md`
- Upstream: <https://jax-ml.github.io/scaling-book/jax-stuff>
