---
title: "Qwen3 JAX v144 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 86016"
type: experiment
hypothesis: "A slightly higher scoped VMEM limit may give the batch-4 collective-matmul-none schedule more room than the 81920 frontier and improve overlap, even though the old pre-collective-matmul-none VMEM upper bracket regressed."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, scoped-vmem, latency-hiding, scheduler, selective-resources, async-collective-fusion, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v144 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 86016

## Hypothesis under test

**Hypothesis**: v143 showed the lower VMEM cap 73728 is harmful in the current
batch-4 collective-matmul-none family. A slightly higher cap, 86016 KiB, may
give custom calls more scoped-memory slack and improve overlap, even though old
v111 regressed before collective-matmul modes were disabled.

**Mechanism**: Restore the v127 simplified batch-4/global-32 frontier stack,
including scheduler shared-memory limit 100, scheduler rerun 3, host
send/recv 100, host-transfer overlap 24, and collective-matmul modes `NONE`;
change only:

```text
--xla_tpu_scoped_vmem_limit_kib=86016
```

from the current baseline value 81920.

**Predicted signal**: Valid train-step compile with better step time or
best-worker average than the v123/v127/v136-v142 band, without loss drift.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed HLO
with no throughput gain, or a regression like v111/v143. If this regresses,
close the post-collective-matmul-none VMEM scalar sweep and keep 81920.

## Setup

GKE workload `alekseyv-qwen3-v144-bs4-vmem86016`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v144-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem86016`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v127 simplified stack plus `--xla_tpu_scoped_vmem_limit_kib=86016`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v144-bs4-vmem86016` completed cleanly on both
workers with normal loss ending at 12.0458.

- Worker 0: 56,609 tok/s average, 7,076 tok/s/chip, approx 40.6% MFU.
- Worker 1: 56,591 tok/s average, 7,074 tok/s/chip, approx 40.6% MFU.
- Best observed steady steps: worker0 56,728 tok/s, worker1 56,727 tok/s.

This is the best result so far by both best-worker average and best observed
step, beating v123's 56,599 best observed step and the v127/v136-v142 average
band.

## Profile

XProf run
`2026-06-04-qwen3-jax-v144-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem86016/2026_06_04_16_04_23`:

- Step time average: 1992.6 ms.
- Total profile time: 111,492.2 ms; idle 264.6 ms / 0.2%.
- MXU utilization: 64.2%.
- Train-step program: `jit_train_step(11563614913035853403)`.
- Top buckets: convolution fusion 59,830.2 ms / 53.7%, custom-call
  29,355.0 ms / 26.3%, loop fusion 11,701.7 ms / 10.5%,
  all-reduce-scatter fusion 3,761.2 ms / 3.4%.
- Splash custom calls: DKV 15,485.0 ms / 13.9%; forward residuals combined
  13,870.0 ms / 12.4%.

Compared with v141/v142, total profile time and convolution-fusion time improve
while custom-call time is effectively unchanged.

## HLO Dump

Fresh train-step HLO dump:

- Module: `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`.
- Optimized HLO size: 787,753 bytes.
- SHA256:
  `aa02d3286a36812ae812c94b82e0bd8fb733cc06e99abf6db17d42aa896cf594`.
- Memory report: 28,858,091,964 bytes / 26.88 GiB.
- Flagfile confirms `--xla_tpu_scoped_vmem_limit_kib=86016`,
  `--xla_tpu_scheduler_percent_shared_memory_limit=100`, scheduler rerun 3,
  host send/recv 100, host-transfer overlap 24, and both collective-matmul
  modes `NONE`.

## Verdict

Provisional supported. Raising scoped VMEM from 81920 to 86016 in the current
batch-4 collective-matmul-none family changes the train-step HLO, improves
observed throughput and XProf step time, and preserves normal loss. Because the
gain is meaningful but still close enough to require noise control, rerun this
exact configuration before carrying 86016 as the durable frontier.
