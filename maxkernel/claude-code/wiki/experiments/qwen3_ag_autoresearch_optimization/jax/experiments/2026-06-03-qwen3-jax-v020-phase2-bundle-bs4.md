# Experiment: v020 Phase 2 Known-Good Bundle at bs=4

**Status:** Completed (Refuted)
**Lane:** jax
**Date:** 2026-06-03

## Hypothesis under test
- **Hypothesis**: The `v019` experiment applying the Phase 2 known-good flag bundle crashed with an HBM OOM at `bs=8` (exceeded capacity by 2.75GB). The Phase 2 flags overlap data-parallel collectives with compute and alter layout/fusion heuristics, which drastically increases the compilation-time memory footprint. Because we are currently pinned at 99.94% HBM utilization at `bs=8`, we cannot afford this memory overhead. By stepping the batch size back to `bs=4`, we free up enough HBM for the compiler to apply these scheduling optimizations. While we lose some batch amortization, the dramatic reduction in collective wait times from the Phase 2 bundle might outweigh the loss, giving us a higher net MFU.
- **Mechanism**: Use the `v016` codebase (or equivalent TP=2 bs=4 configuration like `v013` but with the fused-rope optimizations). `bs=4` provides ~15GB of HBM headroom. Apply the full Phase 2 flag bundle via `LIBTPU_INIT_ARGS`.

## Results
- **Status**: Completed (Exit Code 0)
- **Throughput**: 43664 tok/s (5458/chip)
- **Approx MFU**: 28.1%

## Verdict
**Refuted**.
The Phase 2 bundle compiled successfully at `bs=4`, but the MFU (28.1%) is lower than our `bs=4` baseline `v008` (28.6% MFU). This indicates two things:
1. The batch amortization loss (going from `bs=8` to `bs=4`) is not overcome by the collective optimizations.
2. Even at `bs=4`, the Phase 2 flags either do not successfully overlap the TP=2 collectives, or the overlapping creates scheduling inefficiencies that regress the step time.

The `model-optimization-blueprint.md` explicitly lists `HOST_OFFLOAD_FLAGS` and `DATA_PARALLEL_OVERLAP` as the main Phase 2 levers. However, these target *Data Parallelism* (FSDP / DCN transfers), whereas our 28% overhead is purely from *Tensor Parallelism* (TP=2) `all-gather` and `all-reduce-scatter` operations.

We must find a way to eliminate TP entirely without OOMing the Tokamax CE kernel.
