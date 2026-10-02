---
variant: 8B/v6e-8
hypothesis: Block-wise activation checkpointing (K=4) saves layer boundary HBM, allowing batch size 64.
status: filed
created: 2026-06-13
---

## Hypothesis under test

**Hypothesis**: Grouping layers into blocks of K=4 and applying `nnx.remat` only at the block boundary (instead of per-layer) will reduce the layer boundary memory footprint by 75% (~11 GB), allowing `batch_size=8` (global 64) to fit within the 31.25 GB HBM limit.

**Mechanism**: We will reshape the `self.layers` parameters from `(36, ...)` to `(9, 4, ...)` in `modeling_qwen3.py`. We will use a nested `nnx.scan`: an outer scan over the 9 blocks with `nnx.remat` applied to the block, and an inner scan over the 4 layers without `remat`. This means during the backward pass, we recompute 4 layers at a time from a single saved boundary, rather than saving 36 boundaries.

**Predicted signal**: Expect peak HBM usage to drop from >33 GB to ~22-26 GB, allowing compilation and execution to succeed at `bs=64` per chip.

**Falsification criterion**: HBM exceeds 31.25 GB or step time degrades excessively compared to baseline.

## Setup
(planned setup)
## Results
The workload crashed on launch with an `ImportError: cannot import name 'set_splash_mesh' from 'model'`.
This occurred because `__all__` in `__init__.py` was modified and `set_splash_mesh` was removed or not exposed properly during the code refactoring for block-wise remat.

## Profile
missing

## HLO Dump
missing

## Verdict
**Refuted (Crash)**. The code refactoring inadvertently caused a syntax/import error. We will mark this run as crashed.
## Results
**Verdict**: inconclusive
Workload terminated/crashed or loop stopped before profile completion.
