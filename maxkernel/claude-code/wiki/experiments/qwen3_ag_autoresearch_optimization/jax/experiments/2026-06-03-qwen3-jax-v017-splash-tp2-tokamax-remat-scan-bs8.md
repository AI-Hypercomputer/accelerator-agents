# Experiment: v017 Splash + TP=2 + Tokamax CE + Scan Remat (bs=8)

**Status:** Completed
**Lane:** jax
**Date:** 2026-06-03

## Hypothesis under test
- **Hypothesis**: We are stuck at 30.7% MFU (`bs=8`, `TP=2`). Pure FSDP (`TP=1`) crashed, confirming Tokamax CE demands Tensor Parallelism. To bridge the remaining gap to >35% MFU, we need to eliminate python unrolling overheads and improve XLA's scheduling over the 36 decoder blocks.
Currently, the `Qwen3Model.__call__` uses a Python `for` loop to apply the 36 decoder layers, forcing XLA to compile a massive flattened graph. By replacing the python loop with `nnx.scan` (which natively supports `nnx.remat` as an argument), we fold the 36 layers into a single XLA `while` loop construct. This drastically reduces the size of the compiled graph, improves HLO instruction cache locality, and allows XLA to schedule the compute and TP=2 collective communications far more efficiently.
- **Mechanism**: Use the `v015` codebase. Edit `modeling_qwen3.py` to use `nnx.vmap` in `Qwen3Model.__init__` to stack the layers, and use `nnx.scan` in `Qwen3Model.__call__` to execute them, passing `remat=True` to the scan.
- **Predicted signal**: The workload compiles significantly faster and MFU jumps over the 35% goal due to improved XLA scheduling and instruction cache locality.
- **Falsification criterion**: MFU regresses or stays at 30.7%, indicating graph size was not the bottleneck.

## Configuration
- **Model**: Qwen/Qwen3-8B
- **Framework**: JAX (Flax NNX)
- **Batch Size**: 8
- **Seq Length**: 2048
- **Remat Policy**: `nnx.scan(remat=True)`
- **Attention**: Splash Attention
- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260603-v017-splash-tp2-tokamax-remat-scan-bs8`
- **Flags**: `--tp_parallelism=2 --batch_size=8 --use_splash=True`

## Setup
Derived from `v015`. Modifed `Qwen3Model.__init__` and `__call__` to use `nnx.vmap` and `nnx.scan`. Launched via `gke-cluster-runner`.

## Results
- **Status**: Completed (Exit Code 0)
- **Throughput**: 47864 tok/s (5983/chip)
- **Approx MFU**: 30.8%

## Profile Analysis
- `convolution fusion` (main matmuls) accounts for 35.9% of the step.
- `all-reduce-scatter fusion`, `all-gather`, and `all-to-all` accounts for a combined ~28.1% of the step.
- `loop fusion` (elementwise / activations) accounts for 12.9%.
- HBM Peak: 31.23 GB / 31.25 GB (99.94%). Free HBM: 14 MB.

## Verdict
**Refuted**.
The throughput increased marginally from 47781 to 47864 tok/s, putting us at 30.8% MFU. While the XLA compilation was significantly faster, the runtime step time barely budged. This proves that instruction cache trashing was not our bottleneck, and XLA is already perfectly scheduling the unrolled graph. Furthermore, the profile confirms that HBM utilization remained identical at 99.94%, meaning `nnx.scan` did not magically free up any buffer space for `bs=16`.

## Next Steps
We are at an impasse. We have stacked every optimization known to us (Splash, TP=2, Tokamax, remat, scan, fused kernels), but we cannot scale to `bs=16` due to a strict 14MB memory headroom, and we cannot eliminate the 28.1% TP=2 communication overhead without Tokamax crashing. We need to formulate a new hypothesis or write a retrospective.
