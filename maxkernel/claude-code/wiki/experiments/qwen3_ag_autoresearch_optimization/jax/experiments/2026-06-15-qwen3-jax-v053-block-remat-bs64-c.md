---
title: "qwen3-jax-v053-block-remat-bs64-c"
type: experiment
model: qwen3-cc-jax
variant: 8B/v6e-8
hypothesis: "Block-wise activation checkpointing (K=4) saves layer boundary HBM, allowing batch size 64."
status: filed
created: 2026-06-15
origin: 2026-06-13-qwen3-jax-v053-block-remat-bs64
---

# qwen3-jax-v053-block-remat-bs64-c

## Hypothesis under test

**Hypothesis**: Grouping layers into blocks of K=4 and applying `nnx.remat` only at the block boundary (instead of per-layer) will reduce the layer boundary memory footprint by 75% (~11 GB), allowing `batch_size=8` (global 64) to fit within the 31.25 GB HBM limit.

**Mechanism**: We reshaped the `self.layers` parameters from `(36, ...)` to `(9, 4, ...)` in `modeling_qwen3.py`. We use a nested `nnx.scan`: an outer scan over the 9 blocks with `nnx.remat` applied to the block, and an inner scan over the 4 layers without `remat`. This means during the backward pass, we recompute 4 layers at a time from a single saved boundary, rather than saving 36 boundaries.

**Predicted signal**: Expect peak HBM usage to drop from >33 GB to ~22-26 GB, allowing compilation and execution to succeed at `bs=64` per chip.

**Falsification criterion**: HBM exceeds 31.25 GB or step time degrades excessively compared to baseline.

## Setup

Rerun of v053-b with the correct `launch_cmd` that omits the `cd wiki/.../jax` instruction since the container `WORKDIR` is already `/app/trainer`.

## Results
The workload crashed on launch with `bash: line 5: conda: command not found`. The container image uses a standard `/opt/venv/bin/python` environment, not conda. The launch command was malformed.

## Profile
missing

## HLO Dump
missing

## Verdict
**Refuted (Crash)**. The launch command was malformed for the container environment.
## Results
**Verdict**: inconclusive
Workload terminated/crashed or loop stopped before profile completion.
