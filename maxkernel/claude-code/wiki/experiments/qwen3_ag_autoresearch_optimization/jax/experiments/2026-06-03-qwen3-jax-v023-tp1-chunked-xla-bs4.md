# Experiment: v023 TP=1 Vocab-Parallel CE using injected chunked_xla (bs=4)

**Status:** Pending
**Lane:** jax
**Date:** 2026-06-03

## Hypothesis under test
- **Hypothesis**: The previous `v022` crashed before compilation because `chunked_xla` wasn't present in the `tokamax` library packaged within the docker image. By manually copying the newer `tokamax` code from the workspace into the Docker image, we can use standard JAX primitives instead of the Pallas kernel. This bypasses the opaque bounds constraints of the `mosaic_tpu` custom-call that was triggering full tensor materializations and massive (16.87 GB) memory fragmentation during GSPMD vocab-parallel auto-partitioning.
- **Mechanism**: Copied `tokamax` source directly to `/opt/venv/lib/python3.12/site-packages/tokamax` in the Dockerfile. Used `implementation="chunked_xla"` in `train.py`.
- **Predicted signal**: The experiment should bypass the OOM limits, achieve `bs=4`, and yield a significant boost in MFU (>35%) over `v017` by completely removing Tensor Parallelism (TP) overhead.
- **Falsification criterion**: The model continues to OOM or performance remains < 32% MFU.

## Configuration
- **Model**: Qwen/Qwen3-8B
- **Framework**: JAX (Flax NNX)
- **Batch Size**: 4
- **Seq Length**: 2048
- **Remat Policy**: `nnx.scan(remat=True)`
- **Attention**: Splash Attention
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260603-v023-tp1-chunked-xla-bs4`
- **Flags**: `--tp_parallelism=1 --batch_size=4 --use_splash=True`

## Setup
Derived from `v022`. Updated the Docker image to include the newest `tokamax` library from the workspace to support `implementation="chunked_xla"`.

## Results
- **Status**: Pending
- **Throughput**: TBA
- **Approx MFU**: TBA

## Profile Analysis
TBA

## Verdict
TBA

## Next Steps
TBA
