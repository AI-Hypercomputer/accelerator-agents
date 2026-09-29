---
title: basic_ops — shard_map vs GSPMD-auto microbenchmark harness
type: concept
provenance: mixed
concept: basic_ops
updated: 2026-07-03
status: fresh
---
# basic_ops — shard_map vs GSPMD-auto microbenchmark harness

<!-- connect:up:begin -->
> **Cross-repo concept:** part of [autotuning](../../../concepts/autotuning.md), [sharding](../../../concepts/sharding.md) across this wiki's repos.
<!-- connect:up:end -->
A standalone script that times a fixed list of sharded JAX ops (replicated/sharded matmul,
`all_gather`, `all_reduce`, and a 3-layer Llama FFN) under two different sharding
strategies — explicit `shard_map` vs GSPMD auto-partitioning via
`with_sharding_constraint` — to compare their wall-clock cost on a real device mesh.

## Entry points
- [`main`](../catalog/basic_ops.md#main) — the script's `if __name__ == "__main__"` target; prints the device count and hands the whole [`allcases`](../catalog/basic_ops.md#allcases) list to the (uncited) benchmark-running loop.
- [`allcases`](../catalog/basic_ops.md#allcases) — the literal list of [`BenchmarkCase`](../catalog/basic_ops.md#BenchmarkCase) entries that defines what gets measured: three matmul sharding variants, [`_all_gather`](../catalog/basic_ops.md#_all_gather), [`_all_reduce`](../catalog/basic_ops.md#_all_reduce), and the shard_map vs GSPMD FFN pair ([`_llama_ffn_shmap`](../catalog/basic_ops.md#_llama_ffn_shmap) vs [`_llama_ffn_spmd`](../catalog/basic_ops.md#_llama_ffn_spmd)). This list is the whole point of the file — everything else exists to define or execute its entries.

## Mechanism (step-by-step)
1. A 1-D device [`mesh`](../catalog/basic_ops.md#mesh) named `"x"` is built once at import time from `jax.devices()`, and [`P`](../catalog/basic_ops.md#P) is bound to `PartitionSpec` as shorthand — every sharding spec in the file (`P()`, `P("x")`, `P(None, "x")`) is expressed against this single mesh axis.
2. [`_llama_ffn`](../catalog/basic_ops.md#_llama_ffn) is the shared numerical core (`silu(x@w1) * (x@w3)) @ w2`, i.e. a gated FFN block matching Llama's MLP shape) called identically by both sharding strategies — the benchmark isolates the *sharding mechanism*, not the math, as the variable under test.
3. [`_llama_ffn_shmap`](../catalog/basic_ops.md#_llama_ffn_shmap) wraps three stacked `_llama_ffn` calls in `shard_map.shard_map` with explicit `in_specs`/`out_specs` (`w1`,`w3` column-sharded on `"x"`, `w2` row-sharded), and manually inserts `jax.lax.psum(x, "x")` after each layer to resolve the partial sums — the programmer, not the compiler, decides exactly where communication happens.
4. [`_llama_ffn_spmd`](../catalog/basic_ops.md#_llama_ffn_spmd) runs the same three-layer loop under plain `jax.jit` with no `shard_map`, instead calling `jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P()))` after each layer — this hands the actual collective-placement decision to GSPMD's auto-partitioner, which only receives a *hint* (replicate `x`) rather than an explicit `psum`.
5. [`_all_gather`](../catalog/basic_ops.md#_all_gather) and [`_all_reduce`](../catalog/basic_ops.md#_all_reduce) are minimal one-op `shard_map` kernels (`jax.lax.all_gather`/`jax.lax.psum` over axis `"x"`) included as a baseline pair — their timings calibrate how much of the FFN cases' cost is pure collective overhead versus the matmul itself.
6. The [`BenchmarkCase`](../catalog/basic_ops.md#BenchmarkCase) dataclass (`name`, `function`, [`args_shape`](../catalog/basic_ops.md#BenchmarkCase.args_shape), [`args_sharding`](../catalog/basic_ops.md#BenchmarkCase.args_sharding), optional `profiler_output`) is the uniform record type every entry in `allcases` conforms to, letting the (uncited) run loop generate matching random inputs and place them on-device generically regardless of which op is being timed.

## Key data structures
- [`BenchmarkCase`](../catalog/basic_ops.md#BenchmarkCase) — a `@dataclasses.dataclass` pairing a callable with the shapes/shardings needed to synthesize its inputs; [`name`](../catalog/basic_ops.md#BenchmarkCase.name) is what gets printed in the results line.
- The module-level FFN size constants — [`dim`](../catalog/basic_ops.md#dim) `= 4096`, [`hidden_dim`](../catalog/basic_ops.md#hidden_dim) `= 11008`, [`BATCH`](../catalog/basic_ops.md#BATCH) `= 1024` — match Llama-family FFN dimensions, and [`L`](../catalog/basic_ops.md#L) `= 1 << 15` (32768) sizes the pure matmul cases to be large enough that sharding overhead is visible against compute.

## Dynamics (design intent)
> [!inferred] The uncited `_run_case`/`_new_args`/`_run_call_cases` functions (visible in the surrounding source but not part of this packet's citable subgraph) run each `BenchmarkCase` under both `bfloat16` and `int8` dtypes, print the lowered-HLO text on warmup iterations, and optionally wrap the final timed iteration in `jax.profiler.start_trace`/`stop_trace` when `profiler_output` is set — i.e. this file doubles as a profiling harness, not just a stopwatch benchmark.

## Edge cases
- [`_all_gather`](../catalog/basic_ops.md#_all_gather)'s `shard_map` call passes `check_rep=False`, disabling shard_map's output-replication verification — necessary because `all_gather`'s result is intentionally fully-replicated from a sharded input, a pattern shard_map's default checker would otherwise flag.

## Open questions
> [!inferred] Whether the shard_map (explicit collective) or GSPMD-auto (`with_sharding_constraint` hint) FFN variant actually wins on real hardware is exactly what this script measures — the source alone doesn't record an outcome; that would live in a run's printed output or an attached profile, neither of which is part of this packet.

## See also
- [custom_kernel_spmd](custom_kernel_spmd.md) — a related benchmark harness in the same repo comparing sharding strategies with a custom Pallas kernel in the mix.
