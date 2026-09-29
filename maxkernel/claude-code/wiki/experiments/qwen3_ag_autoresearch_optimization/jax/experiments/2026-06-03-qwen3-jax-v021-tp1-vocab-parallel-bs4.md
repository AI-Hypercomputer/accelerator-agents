# Experiment: v021 FSDP Vocab-Parallel CE (bs=4, TP=1)

**Status:** Pending
**Lane:** jax
**Date:** 2026-06-03

## Hypothesis under test
- **Hypothesis**: The Tokamax Cross Entropy (CE) loss requires Tensor Parallelism by default, which introduces significant communication overheads (around 28% of the step time in v017). Pure FSDP (`TP=1`) crashed previously. By modifying the sharding plan for `lm_head.weight` to `("fsdp", "tp")`, under pure FSDP (`tp=1`), this resolves to `("fsdp", None)`. This shards the vocabulary dimension over the FSDP axis and prevents the large weight gather that caused the OOM. This will allow the chunked XLA CE loss to utilize FSDP as a vocab-parallel axis, eliminating TP overhead.
- **Mechanism**: Use the `v017` codebase (which includes Tokamax CE, Splash, remat, and scan). In `model/sharding.py`, change `SHARDING_PLAN["lm_head.weight"]` from `("tp", "fsdp")` to `("fsdp", "tp")`. We keep `train.py` exactly as it was, as Tokamax CE auto-falls back to `chunked_xla` under GSPMD and supports vocab-parallel sharding.
- **Predicted signal**: The experiment should run successfully without crashing from OOM under `TP=1`, and the MFU should dramatically improve over 35% because the TP communication overhead is eliminated.
- **Falsification criterion**: The experiment crashes, hits an OOM, or MFU does not significantly improve over 30.7%.

## Configuration
- **Model**: Qwen/Qwen3-8B
- **Framework**: JAX (Flax NNX)
- **Batch Size**: 4
- **Seq Length**: 2048
- **Remat Policy**: `nnx.scan(remat=True)`
- **Attention**: Splash Attention
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260603-v021-tp1-vocab-parallel-bs4`
- **Flags**: `--tp_parallelism=1 --batch_size=4 --use_splash=True`

## Setup
Derived from `v017`. Modified `model/sharding.py` to change `lm_head.weight` sharding to `("fsdp", "tp")`. Built new image and launched via `gke-cluster-runner`.

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
