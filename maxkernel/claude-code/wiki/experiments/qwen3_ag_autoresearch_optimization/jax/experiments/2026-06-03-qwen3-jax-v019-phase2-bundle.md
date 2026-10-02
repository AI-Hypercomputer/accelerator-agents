# Experiment: v019 Phase 2 Known-Good Bundle

**Status:** Completed (Refuted/Crashed)
**Lane:** jax
**Date:** 2026-06-03

## Hypothesis under test
- **Hypothesis**: Our step time is currently dominated by TP=2 collectives (`all-gather` and `all-reduce-scatter fusion` take up ~23% of the step). The `model-optimization-blueprint.md` documents a "Phase 2 known-good high-yield flag bundle" for the JAX stack, which includes `HOST_OFFLOAD_FLAGS`, `DATA_PARALLEL_OVERLAP`, `LAYOUT_FOR_ALL_REDUCE_SCATTER`, `CF_FOR_ALL_GATHER`, `CF_FOR_ALL_REDUCE`, `DISABLE_BUNDLE_AWARE_COST_MODEL`, and disabling collective matmuls. By applying this full suite of compiler flags, we will optimize collective scheduling, overlap data parallel collectives with compute, and disable inefficient fusions, yielding a major MFU boost.
- **Mechanism**: Use the `v015` image (Splash + TP=2 + Tokamax CE + Remat + Fused QK-norm/RoPE). Update `LIBTPU_INIT_ARGS` to include the full suite of flags derived from `xla_flags_library.py`.

## Results
- **Status**: Completed (Exit Code 1)
- **Throughput**: N/A
- **Approx MFU**: N/A
- **Note**: The workload failed during compilation with an HBM OOM: `Ran out of memory in memory space hbm. Used 34.00G of 31.25G hbm. Exceeded hbm capacity by 2.75G.`

## Verdict
**Refuted/Crashed**. The Phase 2 flags attempt to overlap data-parallel collectives and host transfers, and they alter the layout and continuation fusion heuristics. While these flags optimize step time, they dramatically increase the compilation-time memory footprint (requiring an additional 2.75G of HBM). Because we are already at 99.94% HBM utilization at `bs=8` (with only 14MB free), we cannot afford the memory overhead of these scheduling optimizations.

We must either drop the batch size back to `bs=4` to use these flags, or find another way to reduce memory pressure (like sequence-parallelism / fully-sharded data parallelism) before applying them.
