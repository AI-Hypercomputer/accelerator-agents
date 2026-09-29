# Experiment: v012 Splash + TP=4 (bs=8)

**Status:** Completed
**Lane:** jax
**Date:** 2026-06-02

## Hypothesis under test
- **Hypothesis**: The OOM at batch size 8 under TP=2 requested exactly 35.21GB of HBM, indicating we are ~4GB short. Changing rematerialization policies (v011) yielded no gains. To fundamentally fit `bs=8` into 31.25GB of memory, we need to increase structural parallelism. Scaling to Tensor Parallelism 4 (`TP=4`) and FSDP 2 (`FSDP=2`) across our 8 chips will divide the massive sequence dimension and KV cache footprint by 4 instead of 2. This will provide the necessary memory relief to compile and run `bs=8`.
- **Mechanism**: Use the `v008` Splash Attention image and pass `--tp_parallelism=4` and `--batch_size=8` via the launch flags.
- **Predicted signal**: The workload will compile and run successfully without OOM.
- **Falsification criterion**: The workload still OOMs, or the TP=4 communication overhead (which will be larger than TP=2) degrades throughput so severely that MFU drops below the 28.6% achieved by `bs=4, TP=2`.

## Configuration
- **Model**: Qwen/Qwen3-8B
- **Framework**: JAX (Flax NNX)
- **Batch Size**: 8
- **Seq Length**: 2048
- **Remat Policy**: `checkpoint_dots_with_no_batch_dims`
- **Attention**: Splash Attention
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v008-splash-tp2-bs4`
- **Flags**: `--tp_parallelism=4 --batch_size=8 --use_splash=True`

## Setup
Launched directly via `gke-cluster-runner`.

## Results
- **Status**: Completed (Exit Code 0)
- **Throughput**: 39438 tok/s (4930/chip)
- **Approx MFU**: 25.4%

## Verdict
**Refuted** (on MFU grounds, though memory hypothesis was Supported).
Increasing to `TP=4` successfully slashed the memory footprint enough to compile and run `bs=8` without OOM. However, the added communication overhead across the 4-chip tensor parallel group completely negated the compute efficiency gains of the larger batch size. MFU regressed to 25.4%, significantly worse than the 28.6% achieved by `v008` (TP=2, bs=4).

## Next Steps
We should profile this run to confirm the exact breakdown of the TP=4 overhead.
Since increasing TP to 4 degrades MFU, and `bs=8` cannot fit in TP=2, we need a new approach to reach >35% MFU.
Ideas:
1. Try Tokamax CE + TP=2 on bs=4 (to see if it improves on v008's 28.6%).
2. Try fully overlapping communications.
3. Try Fused QK-norm + RoPE (as mentioned in program.md) to reduce elementwise fusion overhead.
