---
title: "MaxText Qwen3-8B v6e-8 bs3 rerun profile"
date: 2026-06-12
status: supported
model: qwen3-8b
lane: maxtext
hardware: v6e-8
workload: alekseyv-qwen3-maxtext-v007-bs3r
---

# MaxText Qwen3-8B v6e-8 bs3 rerun profile

## Hypothesis

The previously found MaxText bs3 recipe should reproduce the ~45% MFU result
and provide a fresh XProf/HLO reference for native-JAX comparison.

## Configuration

- `model_name=qwen3-8b`
- `per_device_batch_size=3`
- `max_target_length=8192`
- `ici_fsdp_parallelism=-1`
- `remat_policy=custom`
- offload: `decoder_layer_input`, `query_proj`, `key_proj`, `value_proj`,
  `out_proj`
- Splash attention: all 2048 tiles, fused backward enabled
- XLA flags: Trillium dense VMEM, reduce-scatter layout assignment,
  data-parallel overlap, async all-gather collective fusion, SparseCore
  all-reduce offload, host-offload scheduler flags, collective matmul disabled.

## Result

Supported. Workload completed with `EXIT_CODE=0`.

- Output root: `gs://<your-bucket>/autoresearch/qwen3_maxtext/2026-06-12-qwen3-8b-v6e8-recipe-v007-bs3-rerun/alekseyv-qwen3-maxtext-v007-bs3r/`
- Profile: `tensorboard/plugins/profile/2026_06_12_06_07_07/`
- HLO: `xla_dump/gke-tpu-01f93eb9-43vk/module_0103.jit_train_step.cl_759148519.after_optimizations.txt`

Steady post-profile steps 15-19:

| Step | seconds | TFLOP/s/device | tok/s/device |
|------|---------|----------------|--------------|
| 15 | 3.590 | 410.049 | 6845.097 |
| 16 | 3.594 | 409.591 | 6837.443 |
| 17 | 3.595 | 409.488 | 6835.732 |
| 18 | 3.597 | 409.334 | 6833.154 |
| 19 | 3.598 | 409.196 | 6830.847 |

XProf summary: **3611.7 ms**, **62.2% MXU**, **0.3% idle**. Op profile:
train step **99.7%** of compute time, convolution fusion **49.6%**, Splash
custom-calls **32.6%**.

Optimized HLO:

- SHA256: `598fe616e4031e66c7a0709c0e0dfd04bec2566041590378c0a4fb1ea931cf84`
- Size: **1,498,186 bytes / 8,013 lines**
- Counts: `dot_general=387`, `fusion=1190`, `copy=667`, `all-gather=185`,
  `all-reduce=100`, `reduce-scatter=20`, `convolution=122`,
  `custom-call=68`, `splash=53`, `_one_hot=8`.

## Interpretation

This reproduces the prior MaxText v006 compiler artifact exactly. The profile
does not reveal a native-JAX throughput gap; instead the cross-lane MFU gap was
mostly accounting. The MaxText recipe image counts attention as the full
non-causal QK/AV matmul term, which gives **1472.20 TFLOPs/device-step** at
bs3 seq8192. The native Qwen3 trainers previously used triangular causal
attention, giving **1294.08 TFLOPs/device-step**. Native reporting has been
updated to match the MaxText-style convention.
