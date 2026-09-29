---
title: "Qwen3 jax — Selective SparseCore offload"
type: hypothesis
model: qwen3-cc-jax
variants: ["8B/v6e-8"]
status: refuted
expected_gain: "1-3% throughput / MFU"
confidence: medium
effort: S
origin: 2026-06-02-qwen3-jax-v036-sc-bs3
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3 jax — Selective SparseCore offload

*Hypothesis*: Offloading only a subset of FSDP collectives (e.g., only `reduce-scatter` or `all-gather` instead of all three) will minimize SparseCore-to-TensorCore transfer overhead and avoid PCIe link bottlenecks, while still reclaiming some TensorCore cycles for matmul computation.

*Rationale*: Our [v036 experiment](../../experiments/qwen3_ag_autoresearch_optimization/jax/experiments/2026-06-02-qwen3-jax-v036-sc-bs3.md) showed that offloading all three collectives (all-reduce, reduce-scatter, all-gather) led to a minor throughput regression (-0.6%), likely due to transfer overhead or PCIe saturation. Offloading selectively might strike a better balance.

*Proposed experiment*: Run with `--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true` but leave all-reduce and all-gather on TensorCore, measuring step time.

*Falsification criterion*: Step time does not decrease compared to the v035 baseline (4077.5 ms) or further degrades performance → refuted.
