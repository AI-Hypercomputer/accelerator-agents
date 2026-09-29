---
variant: 8B/v6e-8
hypothesis: Integrating Native Sparse Attention (NSA) will reduce HBM usage during compilation, allowing batch size 24 to compile and run, increasing MFU.
commit: 2026-06-05-v045-nsa-attention
status: running
verdict: open
---

# v045: Native Sparse Attention (NSA) Integration

## Setup
- **Base Config**: `v044` (which failed with `RESOURCE_EXHAUSTED` at bs=24, seq=2048).
- **Changes**: Modified `Qwen3Attention` in `jax/trainer/model/modeling_qwen3.py` to use `ejkernel.modules.operations.native_sparse_attention`. Added `g_cmp` and `g_slc` linear projections.

## Hypothesis under test
- **Mechanism**: Replacing standard SDPA with Native Sparse Attention (NSA), which computes attention over mean-pooled blocks (global context) and selectively attends to fine-grained blocks dynamically using `g_cmp`. Set `block_size=64` and `block_counts=16`, which reduces sequence length complexity.
- **Predicted signal**: The reduction in attention memory overhead should allow the XLA compiler to succeed under the 31.25GB limit for batch size 24, and the resulting run should yield improved MFU compared to batch size 16.

## Results
- **Outcome**: `[Waiting for gke-cluster-runner]`
- **MFU**: `[Waiting]`

## Verdict
**OPEN**
