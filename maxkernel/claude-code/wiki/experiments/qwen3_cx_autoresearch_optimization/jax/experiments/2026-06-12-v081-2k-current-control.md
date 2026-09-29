---
title: "Qwen3 JAX current v081 - 2k current-stack control"
type: experiment
hypothesis: "At 2k context length, the carried native-JAX explicit-SiLU stack should compile with much larger memory headroom, but the 8k-tuned batch and scheduling choices may not be optimal for short sequence attention."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: supported
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, sparsecore, collectives, scheduler, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 JAX current v081 - 2k current-stack control

## Hypothesis under test

**Hypothesis**: Moving the carried explicit-SiLU shmem90 native-JAX stack from
8k to **2k** context length should reduce attention and activation memory
pressure enough to compile cleanly with substantial HBM headroom. The resulting
profile will establish whether the current 8k frontier remains compute-bound at
short context or whether 2k needs a different batch/scheduler/CE strategy.

**Mechanism**: Reuse the current durable v067 explicit-SiLU image and runtime
stack, changing the training shape to `--seqlen=2048` while keeping
`--batch_size=4`, no-scan remat, activation sharding, Tokamax Splash, MaxText CE,
scoped VMEM **100352 KiB**, latency-hiding rerun **3**, SparseCore
reduce-scatter/all-reduce offload, SparseCore collective aggregator, and shared
memory limit **90**.

**Predicted signal**: Support requires clean finite loss, a valid XProf trace,
and an HLO dump on `v6e-demo-hjajoo`. The run is a control: throughput/MFU and
profile composition will pick the next 2k-specific optimization axis. Low MXU,
high host idle, or a cross-entropy-heavy profile would motivate batch-size,
compilation, or logits/CE experiments rather than more 8k scheduler sweeps.

## Setup

- Workload: `alekseyv-qwen3-jax-v081-2kctl`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v067-explicit-silu`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v081-2k-bs4-current-control`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v081_2k_bs4ctl`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`
- Mesh: `fsdp=8,tp=1`; per-chip batch 4 / global batch 32; sequence length
  **2048**; profile window steps 12-14.

## Status

Completed cleanly on `v6e-demo-hjajoo` after verifying capacity: Kueue
`multislice-queue` had zero pending and zero admitted workloads, and the two
2x4 `tpu-v6e-slice` nodes in `v6e-demo-hjajoo-np-0` were Ready.

## Results

Worker summaries:

- `gke-tpu-77a6681c-wwzz`: **65,448 tok/s**, **8,181 tok/s/chip**,
  **42.1%** script MFU, `EXIT_CODE=0`.
- `gke-tpu-77a6681c-6221`: **65,505 tok/s**, **8,188 tok/s/chip**,
  **42.1%** script MFU, `EXIT_CODE=0`.

Warm steady-state steps after compile were tightly clustered near **1000 ms**.
Worker1's post-profile summary was the best worker average at **65,505 tok/s**.
Loss was finite and decreased from **12.0987** at step 0 to **12.0718** at step
19.

XProf runs:

- `2026-06-12-qwen3-jax-v081-2k-bs4-current-control/2026_06_12_22_27_47`
- `2026-06-12-qwen3-jax-v081-2k-bs4-current-control/2026_06_12_22_27_48`

Host `gke-tpu-77a6681c-wwzz` reported **1008.1 ms** average step time,
**0.1 ms** stddev, **56.5% MXU**, and about **1.0%** device idle in the
op-profile view. Peak HBM was **18.95 GiB / 31.25 GiB**, leaving
**12.296 GiB** free.

Top XProf operation buckets for `jit_train_step`:

- `convolution fusion`: **7525.467 ms / 62.0%**, **5477.328 TFLOP**,
  **2832.561 GiB** accessed.
- `custom-call`: **1503.348 ms / 12.4%**.
- `loop fusion`: **1486.835 ms / 12.3%**.
- `custom fusion`: **804.665 ms / 6.6%**.
- `data formatting`: **449.972 ms / 3.7%**.

Train-step HLO:

- Modules: `module_0264.jit_train_step` and `module_0267.jit_train_step`.
- SHA256: `c6b001141eb5b66b941054abd357bc8683e8e95bece3fb3b6d74b494348ff786`
  for both hosts.
- Size: **19,968,324 bytes / 121,278 lines**.
- Compile memory report: **16.00 GiB** total, with **10.25 GiB**
  preallocated temp.
- Counts: `all-gather=4595`, `all-reduce=154`, `reduce-scatter=1780`,
  `async-start=802`, `async-done=802`, `convolution=2100`,
  `custom-call=2514`, `copy=10614`, `fusion=22515`,
  `dot_general=7035`, `splash=1778`, `tokamax=2`,
  `jvp_jit_silu__=0`, `exponential=183`, `dynamic-slice=150`,
  `dynamic-update-slice=216`.

## Verdict

Supported as the new 2k current-stack control. It is substantially better than
the older 2k native-JAX baseline (**31,955 tok/s**) and the earlier TP/scan
2k branch (**47,846 tok/s** best recorded in the model table), but it is not
yet an optimized 2k frontier.

The profile points to batch scaling as the next axis: this 2k shape has
large HBM headroom, low idle, and lower MXU than the 8k frontier. Run
`--batch_size=6` / global batch **48** next before changing kernels.
