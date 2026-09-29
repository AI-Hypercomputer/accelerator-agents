---
title: "Qwen3 JAX v146 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 90112"
type: experiment
hypothesis: "If the scoped-VMEM 86016 lift is from extra custom-call scheduling slack, a further increase to 90112 may improve or may reveal the upper over-shoot point."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, scoped-vmem, latency-hiding, scheduler, selective-resources, async-collective-fusion, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v146 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 90112

## Hypothesis under test

**Hypothesis**: v144/v145 establish scoped VMEM 86016 as a repeatable
improvement over 81920 in the current batch-4 collective-matmul-none family. A
further increase to 90112 KiB may give additional scoped-memory slack and
improve scheduling, or it may over-shoot and reproduce the old v070 regression.

**Mechanism**: Keep the new v145 frontier stack fixed and change only:

```text
--xla_tpu_scoped_vmem_limit_kib=90112
```

from 86016.

**Predicted signal**: Support requires clean loss plus best-worker average
above v144/v145 or a lower XProf step time. If it falls back to the old 56.45k
band or regresses below v145, the VMEM upper bracket is closed and 86016 remains
the carried setting.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed HLO
with no throughput gain, or clean completion below the v144/v145 confirmed
frontier.

## Setup

GKE workload `alekseyv-qwen3-v146-bs4-vmem90112`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v146-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem90112`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v145 frontier stack plus `--xla_tpu_scoped_vmem_limit_kib=90112`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers with normal loss ending near 12.0460.

- worker0: 56,635 tok/s average, 7,079 tok/s/chip, 40.6% MFU.
- worker1: 56,624 tok/s average, 7,078 tok/s/chip, 40.6% MFU.
- Best observed steady steps: worker0 56,763 tok/s; worker1 56,764 tok/s.
- This is above the confirmed v144/v145 scoped-VMEM 86016 frontier by about
  26-32 tok/s on best-worker average and by about 36 tok/s on best observed
  step.

## Profile

XProf run:
`2026-06-04-qwen3-jax-v146-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem90112/2026_06_04_16_19_46`.

- Step average: 1990.9 ms.
- Total profile time: 111,433.546 ms, with 236.744 ms idle (0.2%).
- MXU utilization: 64.9%.
- Train-step program: `jit_train_step(13912291835542571918)`.
- Program work: 57,371.383 TFLOPs, 40,569.465 GiB accessed.
- Top buckets: convolution fusion 59,634.359 ms / 53.5%; custom-call
  29,354.917 ms / 26.3%; loop fusion 11,497.4 ms / 10.3%;
  all-reduce-scatter fusion 3,755.133 ms / 3.4%.
- Custom-call split: DKV 15,484.605 ms / 13.9%; forward residuals combined
  13,870.311 ms / 12.4%.

## HLO Dump

Fresh optimized train-step HLO was emitted under
`module_0267.jit_train_step.cl_854318611.after_optimizations.txt`.

- Optimized HLO size: 787,246 bytes.
- Content hash:
  `eec5d587c0d511d4a7c6eb30dbdacbe4563121cc1321870f957d686be63b0ea6`.
- Compiled memory report: 28,853,864,892 bytes / 26.87 GiB.
- Flagfile confirms `--xla_tpu_scoped_vmem_limit_kib=90112`,
  `--xla_tpu_scheduler_percent_shared_memory_limit=100`, scheduler rerun3,
  host send/recv 100, host-transfer overlap 24, and both collective-matmul
  modes set to `NONE`.

## Verdict

Provisional supported. Scoped VMEM 90112 selects a fresh train-step program,
slightly lowers compiled HLO size/memory, improves XProf step time versus
v144/v145, and reaches the best observed throughput so far. Rerun exactly before
carrying 90112 as the durable frontier because the gain is small enough to need
noise confirmation.
