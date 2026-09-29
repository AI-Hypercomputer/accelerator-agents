---
title: "qwen3-jax-v053-block-remat-bs64-e"
type: experiment
model: qwen3-cc-jax
variant: 8B/v6e-8
hypothesis: "Encapsulate the block-wise rematerialization scan in a `shard_map` to prevent GSPMD's `NotImplementedError` on auto-partitioning Mosaic kernels."
status: filed
created: 2026-06-15
origin: 2026-06-15-qwen3-jax-v053-block-remat-bs64-d
---

# qwen3-jax-v053-block-remat-bs64-e

## Hypothesis under test

**Hypothesis**: Grouping layers into blocks of K=4 and applying `nnx.remat` only at the block boundary (instead of per-layer) will reduce the layer boundary memory footprint by 75% (~11 GB), allowing `batch_size=8` (global 64) to fit within the 31.25 GB HBM limit.

**Mechanism**: We reshaped the `self.layers` parameters from `(36, ...)` to `(9, 4, ...)` in `modeling_qwen3.py`. We use a nested `nnx.scan`: an outer scan over the 9 blocks with `nnx.remat` applied to the block, and an inner scan over the 4 layers without `remat`. This means during the backward pass, we recompute 4 layers at a time from a single saved boundary, rather than saving 36 boundaries.

**Predicted signal**: Expect peak HBM usage to drop from >33 GB to ~22-26 GB, allowing compilation and execution to succeed at `bs=64` per chip.

**Falsification criterion**: HBM exceeds 31.25 GB or step time degrades excessively compared to baseline.

## Setup

Fork from v053-d. Add explicit `shard_map` encapsulation around the `nnx.scan` block to prevent GSPMD from auto-partitioning Mosaic kernels.

## Results

The experiment crashed with an Out-of-Memory (OOM) error during the execution of an `allgather` within the nested remat scan configuration:
```python
    File "/opt/venv/lib/python3.12/site-packages/jax/experimental/multihost_utils.py", line 134, in _handle_array_process_allgather
      out = jax.jit(_identity_fn, out_shardings=P())(global_arr)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Error loading program 'jit__identity_fn': Attempting to reserve 13.50G at the bottom of memory. That was not possible. There are 7.48G free, 0B reserved, and 7.48G reservable.
```

## Profile

- **Phase 1 unavailable**: xprof trace not found at `gs://<your-project>-alekseyv-asia-northeast1/autoresearch/qwen3-jax-v053-block-remat-bs64-e/plugins/profile`.

## HLO Dump

**Source**: `gs://<your-project>-alekseyv-asia-northeast1/autoresearch/qwen3-jax-v053-block-remat-bs64-e/hlo`
**Modules**: 200+ total, top by size: `jit__normal`, `jit__uniform`, `jit_threefry_fold_in`

**Inspected modules** (top by size, fallback due to missing xprof):
- `jit__normal` (~29 KB): Eager dispatch of Python-level op.
- `jit__uniform` (~26 KB): Eager dispatch of Python-level op.

**Fusion verification**:
- N/A — eager execution detected (100+ generic `module_XXXX` ops). No `train_step` to verify.

**Hypothesis-firing audit** (Phase 3 — silent-noop check vs the hypothesis's predicted mechanism):
- SKIPPED — no hypothesis on file (setup error: experiment page missing).
- Result: **SKIPPED — no hypothesis on file**

**Notable patterns**:
- **Eager execution detected**: The HLO module count fingerprint indicates eager execution (100+ small, generic-named modules like `jit_broadcast_in_dim`, `jit_convert_element_type`). No fused `train_step` module is present.

## Verdict

`invalid` — OOM crash during eager compilation step due to excessive allgather buffering for block-wise parameters. Wrapping the block-wise scan in a `shard_map` solved the custom call partitioning but resulted in an insurmountable memory penalty on V6e nodes (out of HBM budget when preparing intermediate tensors across multiple outer blocks).
## Results
**Verdict**: inconclusive
Workload terminated/crashed or loop stopped before profile completion.
