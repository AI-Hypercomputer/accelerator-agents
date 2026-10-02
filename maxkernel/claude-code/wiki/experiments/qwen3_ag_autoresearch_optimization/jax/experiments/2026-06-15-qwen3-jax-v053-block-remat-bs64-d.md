---
title: "qwen3-jax-v053-block-remat-bs64-d"
type: experiment
model: qwen3-cc-jax
variant: 8B/v6e-8
hypothesis: "Block-wise activation checkpointing (K=4) saves layer boundary HBM, allowing batch size 64."
status: invalid
created: 2026-06-15
origin: 2026-06-13-qwen3-jax-v053-block-remat-bs64
---

# qwen3-jax-v053-block-remat-bs64-d

## Hypothesis under test

**Hypothesis**: Grouping layers into blocks of K=4 and applying `nnx.remat` only at the block boundary (instead of per-layer) will reduce the layer boundary memory footprint by 75% (~11 GB), allowing `batch_size=8` (global 64) to fit within the 31.25 GB HBM limit.

**Mechanism**: We reshaped the `self.layers` parameters from `(36, ...)` to `(9, 4, ...)` in `modeling_qwen3.py`. We use a nested `nnx.scan`: an outer scan over the 9 blocks with `nnx.remat` applied to the block, and an inner scan over the 4 layers without `remat`. This means during the backward pass, we recompute 4 layers at a time from a single saved boundary, rather than saving 36 boundaries.

**Predicted signal**: Expect peak HBM usage to drop from >33 GB to ~22-26 GB, allowing compilation and execution to succeed at `bs=64` per chip.

**Falsification criterion**: HBM exceeds 31.25 GB or step time degrades excessively compared to baseline.

## Setup

Rerun of v053-c with the correct `launch_cmd` that omits the `conda run -n py312` instruction since the container uses a standard `/opt/venv/bin/python` environment.

## Results

The experiment crashed with a `NotImplementedError` regarding Mosaic kernels lacking a `shard_map`.
```python
  File "/opt/venv/lib/python3.12/site-packages/jax/_src/tpu_custom_call.py", line 713, in lower_module_to_custom_call
    return _tpu_custom_call_lowering(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/jax/_src/tpu_custom_call.py", line 349, in _tpu_custom_call_lowering
    raise NotImplementedError(
  NotImplementedError: Mosaic kernels cannot be automatically partitioned. Please wrap the call in a shard_map.
```

## Profile

- xprof unavailable: The trace directory does not exist at `gs://<your-project>-alekseyv-asia-northeast1/autoresearch/qwen3-jax-v053-block-remat-bs64-d/plugins/profile` (matched no objects). The run was not found, likely due to the experiment crashing before profiling could complete.

## HLO Dump

- Phase 2 + Phase 3 skipped — Phase 1 unavailable

## Verdict

`invalid` — Crash due to unpartitioned Mosaic kernel. The block-wise remat requires `shard_map` encapsulation or similar to ensure the underlying custom calls (like Pallas attention) partition correctly over the mesh when embedded inside nested block scans.
