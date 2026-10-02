---
title: "Qwen3 JAX v150 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 98304"
type: experiment
hypothesis: "If the 90112 -> 94208 scoped-VMEM gain reflects useful scheduling slack, increasing to 98304 may continue the trend or reveal the upper over-shoot point."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, scoped-vmem, latency-hiding, scheduler, selective-resources, async-collective-fusion, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v150 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 98304

## Hypothesis under test

**Hypothesis**: v148/v149 confirmed scoped VMEM 94208 as better than 90112 in
the current batch-4 collective-matmul-none family. A further increase to 98304
may provide more scheduling slack and improve step time, or it may over-shoot
and close the upper VMEM bracket.

**Mechanism**: Keep the v149 frontier stack fixed and change only:

```text
--xla_tpu_scoped_vmem_limit_kib=98304
```

from 94208.

**Predicted signal**: Support requires clean loss plus best-worker average
above v149's 56,768 tok/s or a lower XProf step time. If it falls back to the
v147/v149 band or regresses, 94208 remains the carried setting.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed HLO
with no throughput gain, or clean completion below the v149 confirmed frontier.

## Setup

GKE workload `alekseyv-qwen3-v150-bs4-vmem98304`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v150-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem98304`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v149 frontier stack plus `--xla_tpu_scoped_vmem_limit_kib=98304`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers with normal loss ending at 12.0460.

- worker0: 57,578 tok/s average, 7,197 tok/s/chip, 41.3% MFU.
- worker1: 57,561 tok/s average, 7,195 tok/s/chip, 41.3% MFU.
- Best observed steady steps: worker0 57,682 tok/s; worker1 57,685 tok/s.
- This is a large provisional improvement over the confirmed v149 94208
  frontier at 56,768 tok/s average and 56,856 best observed.

## Profile

XProf run:
`2026-06-04-qwen3-jax-v150-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem98304/2026_06_04_16_52_56`.

- Step average: 1959.9 ms.
- Total profile time: 109,644.798 ms, with 286.513 ms idle (0.3%).
- MXU utilization: 65.3%.
- Train-step program: `jit_train_step(6628478088121256461)`.
- Program work: 57,371.383 TFLOPs, 40,582.440 GiB accessed.
- Top buckets: convolution fusion 57,298.241 ms / 52.3%; custom-call
  29,381.061 ms / 26.8%; loop fusion 11,475.258 ms / 10.5%;
  all-reduce-scatter fusion 3,758.216 ms / 3.4%; data formatting
  2,898.903 ms / 2.6%; custom fusion 1,651.842 ms / 1.5%.
- Custom-call split: DKV 15,510.604 ms / 14.1%; forward residuals combined
  13,870.456 ms / 12.7%.

## HLO Dump

Fresh optimized train-step HLO was emitted under
`module_0267.jit_train_step.cl_854318611.after_optimizations.txt`.

- Optimized HLO size: 787,060 bytes.
- Content hash:
  `40d497c2bad650f5da214fd82b9636af17633b718094f8561a4b5a9ba7c2fdd9`.
- Compiled memory report: 28,845,509,048 bytes / 26.86 GiB.
- Flagfile confirms `--xla_tpu_scoped_vmem_limit_kib=98304`,
  `--xla_tpu_scheduler_percent_shared_memory_limit=100`, scheduler rerun3,
  host send/recv 100, host-transfer overlap 24, and both collective-matmul
  modes set to `NONE`.

## Verdict

Provisional supported. Scoped VMEM 98304 selects a fresh train-step program,
lowers compiled memory slightly, drops XProf step time to 1959.9 ms, and raises
observed throughput to 57,578 tok/s average / 57,685 best. Rerun exactly before
carrying 98304 as the durable frontier.
