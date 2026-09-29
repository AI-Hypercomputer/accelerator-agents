---
name: qwen3-ag-maxtext
title: Qwen3 8B MaxText Baseline
date: 2026-06-05
tags: [model, maxtext, qwen3]
status: active
---

# Qwen3 8B on MaxText

This model page tracks the baseline and progression of the `qwen3` model running via MaxText (native JAX) on TPUs.

## Run Configuration
- **Hardware**: v6e-8 (Trillium)
- **Framework**: MaxText (JAX)
- **Model**: Qwen3 8B
- **Sequence Length**: 8192
- **Parallelism**: TP=1, FSDP=1
- **Rematerialization**: Custom (proj, decoder_layer_input offload)
- **Attention**: Flash (Splash sizes = 1024)

## Metrics Progression

| Size | Hardware | Status | Baseline (step / TPS / MFU) | Current best (step / TPS / MFU) | Open hyps | Frontier exp |
|------|----------|--------|-----------------------------|----------------------------------|-----------|--------------|
| 8B | v6e-8 | live | 3.06s / 5,355 TPS / 34.9% MFU @ seq8192 bs2 | 3.06s / 5,355 TPS / 34.9% MFU @ seq8192 bs2 | 0 | [2026-06-05 v006](../experiments/qwen3_ag_autoresearch_optimization/maxtext/experiments/2026-06-05-qwen3-maxtext-v006-baseline-seq8192-bs2.md) |

## Implementation Notes

- MaxText requires specific `sa_block` size tuning to fit within the `vmem` scoped limit on v6e hardware. `sa_block_*=1024` fits successfully, while `2048` causes a `CompileTimeScopedVmemOom` failure.
- `per_device_batch_size=2` fits in HBM. Higher batch sizes like `bs=4` cause a `CompileTimeHbmOom`.
