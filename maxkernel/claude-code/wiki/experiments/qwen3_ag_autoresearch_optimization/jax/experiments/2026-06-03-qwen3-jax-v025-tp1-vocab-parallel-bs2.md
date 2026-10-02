# Experiment: v025 TP=1 + Vocab Parallel (bs=2)

**Status:** Completed
**Lane:** jax
**Date:** 2026-06-03

## Hypothesis under test
- **Hypothesis**: By dropping the per-chip batch size to 2, we can successfully compile the TP=1 + Vocab Parallelism graph without hitting the HLO compilation OOMs that plagued v021 and v024 (bs=4).
- **Mechanism**: Use the `v021` codebase (which correctly shards the vocabulary dimension across FSDP shards to avoid Tokamax CE OOMs), but reduce `--batch_size=2`.
- **Predicted signal**: The model compiles successfully and runs. `all-gather` communication bottlenecks from TP=2 are eliminated. MFU will be lower than the 30.8% of `v017` due to the small batch size, but this will prove the TP=1 architecture works.

## Configuration
- **Model**: Qwen/Qwen3-8B
- **Framework**: JAX (Flax NNX)
- **Batch Size**: 2
- **Seq Length**: 2048
- **Remat Policy**: `nnx.scan(remat=True)` (inherited from v017)
- **Attention**: Splash Attention
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260603-v021-tp1-vocab-parallel-bs4`
- **Flags**: `--tp_parallelism=1 --batch_size=2 --use_splash=True`

## Results
- **Status**: Completed (Exit Code 0)
- **Throughput**: 37811 tok/s (4726/chip)
- **Approx MFU**: 24.3%

## Profile Analysis
- `convolution fusion` (main matmuls) accounts for 33.9% of the step.
- `collective-permute-start` and `collective-permute-done` accounts for ~25.6% of the step.
- `loop fusion` (elementwise / activations) accounts for 17.9%.
- `all-gather` time has plummeted to near zero (< 1 ms).
- HBM Peak: 27.23 GB / 31.25 GB (87.1%).

## Verdict
**Confirmed**.
The workload successfully compiled and ran at `bs=2` with `TP=1`. As predicted, the `all-gather` communication bottleneck from `v017` (which took multiple seconds) is completely gone. However, the smaller batch size (2 vs 8) drops the MXU utilization, leading to a lower MFU of 24.3%.

## Next Steps
Now that we have a working TP=1 baseline, we need to push the batch size up (to 4 or 8) to reach the >35% MFU goal. However, `v021` (bs=4) OOM'd during compilation. We need to analyze the compiler OOM to see what is blowing up the HBM during compilation so we can mitigate it and scale up the batch size.
