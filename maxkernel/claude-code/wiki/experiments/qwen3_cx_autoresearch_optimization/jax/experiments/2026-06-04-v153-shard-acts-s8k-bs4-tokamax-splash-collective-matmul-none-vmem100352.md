---
title: "Qwen3 JAX v153 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 100352"
type: experiment
hypothesis: "If v152's 102400 KiB setting overshoots the useful scoped-VMEM band, the midpoint 100352 KiB may preserve most of v150/v151's gain or identify the high-side edge more precisely."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, scoped-vmem, latency-hiding, scheduler, selective-resources, async-collective-fusion, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v153 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 100352

## Hypothesis under test

**Hypothesis**: v152 showed 102400 KiB is a valid but slower upper overshoot.
The midpoint 100352 KiB may recover the v150/v151 98304 performance band or
show that the high-side edge is already below this midpoint.

**Mechanism**: Keep the v151 frontier stack fixed and change only:

```text
--xla_tpu_scoped_vmem_limit_kib=100352
```

from the carried 98304 setting.

**Predicted signal**: Support requires clean loss plus best-worker average
above the confirmed 98304 frontier, or at least an XProf step time at or below
v150's 1959.9 ms with a clean profile. A result between v151 and v152 would
sharpen the high-side bracket but not move the frontier.

**Falsification criterion**: Compile/runtime failure, invalid loss, HBM/OOM,
or clean completion below the v150/v151 98304 band.

## Setup

GKE workload `alekseyv-qwen3-v153-bs4-vmem100352`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v153-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem100352`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v151 frontier stack plus `--xla_tpu_scoped_vmem_limit_kib=100352`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers with normal loss ending at 12.0461.

- worker0: 57,757 tok/s average, 7,220 tok/s/chip, 41.4% MFU.
- worker1: 57,734 tok/s average, 7,217 tok/s/chip, 41.4% MFU.
- Best observed steady steps: worker0 57,886 tok/s; worker1 57,888 tok/s.
- This beats the confirmed v150/v151 98304 KiB frontier at 57,578 tok/s
  best-worker average and 57,707 best observed steady step.

## Profile

XProf run:
`2026-06-04-qwen3-jax-v153-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem100352/2026_06_04_17_22_28`.

- Step average: 1954.6 ms.
- Total profile time: 109,331.284 ms, with 301.238 ms idle (0.3%).
- MXU utilization: 65.6%.
- Train-step program: `jit_train_step(10728347668315821976)`.
- Program work: 57,371.383 TFLOPs, 40,648.976 GiB accessed.
- Top buckets include convolution fusion 56,946.666 ms / 52.1%, data
  formatting 2,895.687 ms / 2.6%, and custom fusion 1,827.759 ms / 1.7%.
- Device event split for Tokamax custom calls: DKV 7,748.967 ms per host event
  family / 14.2% combined across devices; forward residuals approximately
  6,982 ms / 12.8% combined.
- Memory profile: 26.92 GiB peak HBM, 21.12 GiB stack reservation,
  5.80 GiB heap allocation, and 4.33 GiB free.

## HLO Dump

Fresh optimized train-step HLO was emitted under
`module_0267.jit_train_step.cl_854318611.after_optimizations.txt`.

- Optimized HLO size: 783,949 bytes.
- Content hash:
  `3734fae066fc35c4b9c6c4cce9b0015413ab4513c34a09ba33a0e1862bff3a3d`.
- Compiled memory report: 28,841,540,536 bytes / 26.86 GiB.
- The train-step program changed from v150/v151's
  `6628478088121256461` and v152's `12059987095245408810`, so 100352 KiB
  selected a real third schedule in the high-VMEM bracket.

## Verdict

Provisional supported. Scoped VMEM 100352 is valid, selects a fresh schedule,
improves XProf step average to 1954.6 ms, and raises best-worker average to
57,757 tok/s / 41.4% MFU. Rerun exactly before carrying 100352 as the durable
frontier.
