# Experiment: v018 TP=2 Collective Matmul Mode None

**Status:** Completed (Refuted/Invalid)
**Lane:** jax
**Date:** 2026-06-03

## Hypothesis under test
- **Hypothesis**: Our step time is currently dominated by TP=2 collectives (`all-gather` and `all-reduce-scatter fusion` take up ~23% of the step). The `model-optimization-blueprint.md` documents that setting `xla_tpu_reduce_scatter_collective_matmul_mode=none` and `xla_tpu_all_gather_collective_matmul_mode=none` (the Phase 2 known-good bundle) beats the default collective-matmul codegen on JAX. By passing these in `LIBTPU_INIT_ARGS`, we will disable the inefficient collective-matmul fusion and significantly reduce the time spent in collectives.
- **Mechanism**: Use the `v015` image (Splash + TP=2 + Tokamax CE + Remat + Fused QK-norm/RoPE). Update `LIBTPU_INIT_ARGS` to include `--xla_tpu_reduce_scatter_collective_matmul_mode=none --xla_tpu_all_gather_collective_matmul_mode=none`.

## Results
- **Status**: Completed (Exit Code 0)
- **Throughput**: 46526 tok/s
- **Approx MFU**: 29.9%
- **Note**: The subagent accidentally launched this workload with `bs=6` instead of `bs=8` (global batch 24 instead of 32). Regardless, the MFU dropped from 30.7% to 29.9%.

## Verdict
**Refuted/Invalid**. The collective matmul mode flags alone did not yield a massive speedup, and the batch size mistake invalidates a direct comparison. However, the blueprint specifies a FULL Phase 2 bundle (including `HOST_OFFLOAD_FLAGS`, `DATA_PARALLEL_OVERLAP`, etc.) that must be applied together. We will proceed to `v019` with the full bundle and the correct batch size 8.
