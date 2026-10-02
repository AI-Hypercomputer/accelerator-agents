---
title: "Qwen3 JAX current v105 - 2k batch16 Splash BKV2048 all scheduler features"
type: experiment
hypothesis: "Re-enabling the broad experimental scheduler feature bundle on the confirmed BKV2048 frontier may recover schedule quality that the simplified false setting misses."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v067-explicit-silu
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, maxtext-ce, splash, tokamax-splash, splash-blocks, sparsecore, collectives, scheduler, all-experimental-scheduler, scoped-vmem, shmem100, batch16, compile-cache, short-seq, context-2k, v6e-8]
created: 2026-06-13
updated: 2026-06-13
---

# Qwen3 JAX current v105 - 2k batch16 Splash BKV2048 all scheduler features

## Hypothesis under test

**Hypothesis**: The current BKV2048 frontier uses
`--xla_tpu_enable_all_experimental_scheduler_features=false`. Earlier qseq
experiments found that disabling the broad scheduler bundle could be an
in-band/slight regression, while later simplified stacks accepted `false`.
Because BKV2048 changes the attention/custom-call schedule and leaves HBM
tight, re-enabling the broad bundle on the confirmed v095/v096 stack may find a
better latency-hiding schedule without model-code changes.

**Mechanism**: Reuse v095/v096 exactly and change only:

```bash
--xla_tpu_enable_all_experimental_scheduler_features=true
```

Keep `--batch_size=16`, `--seqlen=2048`, scoped VMEM **98304 KiB**,
`--xla_tpu_scheduler_percent_shared_memory_limit=100`, no-scan remat,
activation sharding, Tokamax Splash, MaxText CE, SparseCore reduce-scatter and
all-reduce offload, SparseCore collective aggregator, and:

```bash
export SPLASH_BKV=2048
export SPLASH_BKV_COMPUTE=2048
```

**Predicted signal**: Support requires beating the confirmed v095/v096
**70,959-70,976 tok/s** band or matching it with lower XProf step time. A
byte-identical HLO and in-band/slower runtime refutes this scheduler axis.

## Setup

- Workload: `alekseyv-qwen3-jax-v105-2ksp2048allsched`
- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v067-explicit-silu`
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-13-qwen3-jax-v105-2k-bs16-vmem98304-splash2048-allsched`
- Compile cache:
  `gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache_v105_2k_bs16_vmem98304_splash2048_allsched`
- Placement: cluster `v6e-demo-hjajoo`, nodepool `v6e-demo-hjajoo-np-0`
- Mesh: `fsdp=8,tp=1`; per-chip batch 16 / global batch 128; sequence length
  **2048**; profile window steps 12-14.

## Status

Completed on 2026-06-13.

## Results

- Workload: `alekseyv-qwen3-jax-v105-2ksp2048allsched`
- Worker throughput:
  - worker0: **70,932 tok/s**, **45.6%** script MFU.
  - worker1: **70,943 tok/s**, **45.6%** script MFU.
- XProf:
  `2026-06-13-qwen3-jax-v105-2k-bs16-vmem98304-splash2048-allsched/2026_06_13_05_24_05`
  - Step time: **3725.5 ms** average.
  - MXU utilization: **67.8%**.
  - Idle: **0.2%**.
- Op profile:
  - `convolution fusion`: **60,171.262 ms / 67.4%**.
  - `custom-call`: **11,695.170 ms / 13.1%**.
  - `loop fusion`: **10,931.714 ms / 12.2%**.
  - Forward residuals: **5,905.176 ms**.
  - DKV total: **5,788.314 ms**.
- Memory profile: **31.14 GiB / 31.25 GiB** peak, **0.1018 GiB** free,
  **23.2933 GiB** stack, **7.8510 GiB** heap.
- HLO: `/tmp/qwen3-v105-hlo/`
  - SHA256:
    `a8b0c4b3833fb22bb72597d9e68cdf28fd70e18ba0a695e247b2ebf56ed69bd2`.
  - Size: **19,375,297 bytes / 115,424 lines**.
  - Compiled memory: **27.88 GiB** total, **22.12 GiB** preallocated temp.
  - Counts: `all-gather=7072`, `all-reduce=165`, `reduce-scatter=2774`,
    `async-start=391`, `async-done=391`, `convolution=3071`,
    `custom-call=3846`, `copy=17231`, `fusion=30274`,
    `dot_general=7301`, `tokamax=1`.

## Verdict

Refuted for speed. Re-enabling the broad experimental scheduler feature bundle
does not beat the confirmed v095/v096 BKV2048 frontier
(**70,959-70,976 tok/s**) and leaves the same tight-HBM runtime profile. The
optimized train-step HLO is byte-identical to v095/v096/v103/v104, so this axis
is runtime noise/scheduler no-gain rather than a new compiler shape. Keep
`--xla_tpu_enable_all_experimental_scheduler_features=false` on the current
frontier.
