---
title: "Qwen3 JAX v141 - shard acts s8k bs4 tokamax splash collective matmul none shared memory 95"
type: experiment
hypothesis: "A mild scheduler shared-memory cap may alter the batch-4 collective-matmul-none schedule enough to reduce contention while preserving the valid frontier HLO family."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, scheduler-shmem, latency-hiding, scheduler, selective-resources, async-collective-fusion, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v141 - shard acts s8k bs4 tokamax splash collective matmul none shared memory 95

## Hypothesis under test

**Hypothesis**: On the batch-4 collective-matmul-none frontier, reducing the
scheduler shared-memory limit from 100 to 95 may nudge scheduling enough to
improve steady throughput or reduce memory pressure without the wall-time
regression seen in stronger/older shared-memory probes.

**Mechanism**: Restore the v127 simplified batch-4/global-32 frontier stack,
including scheduler rerun 3, and change only:

```text
--xla_tpu_scheduler_percent_shared_memory_limit=95
```

from the current baseline value 100.

**Predicted signal**: Same or valid changed train-step HLO with lower step time,
higher best-worker average than v127/v136-v140, and ideally a best observed step
above v123's 56,599 tok/s.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed HLO
with no throughput gain, or clean completion that remains in the 56.43k-56.46k
tok/s frontier band.

## Setup

GKE workload `alekseyv-qwen3-v141-bs4-shmem95`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v141-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-shmem95`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v127 simplified stack plus `--xla_tpu_scheduler_percent_shared_memory_limit=95`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v141-bs4-shmem95` completed cleanly on both
workers with normal loss ending at 12.0461.

- Worker 0: 56,455 tok/s average, 7,057 tok/s/chip, approx 40.5% MFU.
- Worker 1: 56,429 tok/s average, 7,054 tok/s/chip, approx 40.5% MFU.
- Best observed steady steps: worker0 56,567 tok/s, worker1 56,560 tok/s.

This is inside the same noise band as v127/v136/v137 and remains below v123's
56,599 tok/s best observed step.

## Profile

XProf run
`2026-06-04-qwen3-jax-v141-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-shmem95/2026_06_04_15_23_31`:

- Step time average: 1999.4 ms.
- Total profile time: 111,830.8 ms; idle 306.6 ms / 0.3%.
- MXU utilization: 64.1%.
- Train-step program: `jit_train_step(16296087512277873197)`.
- Top buckets: convolution fusion 60,209.5 ms / 53.8%, custom-call
  29,348.6 ms / 26.2%, loop fusion 11,663.6 ms / 10.4%,
  all-reduce-scatter fusion 3,761.3 ms / 3.4%.
- Splash custom calls: DKV 15,477.7 ms / 13.8%; forward residuals combined
  13,871.0 ms / 12.4%.

The XProf memory-profile endpoint returned an internal/connection error for
this run; use the HLO memory report below for compiled HBM evidence.

## HLO Dump

Fresh train-step HLO dump:

- Module: `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`.
- Optimized HLO size: 787,978 bytes.
- SHA256:
  `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`.
- Memory report: 28,862,302,652 bytes / 26.88 GiB.
- Flagfile confirms `--xla_tpu_scheduler_percent_shared_memory_limit=95`
  with rerun 3, host send/recv 100, host-transfer overlap 24, and both
  collective-matmul modes `NONE`.

This is byte-identical to the v123/v127/v135-v140 frontier HLO.

## Verdict

Refuted/tie. The mild shared-memory cap does not change the final train-step
program or profile shape, does not reduce compiled HBM, and does not improve
throughput beyond the current frontier noise band. Do not carry
`--xla_tpu_scheduler_percent_shared_memory_limit=95`; keep the simplified v127
baseline with shared-memory limit 100 for normal runs. The next scalar probe is
the stronger shared-memory cap 90 to close the post-collective-matmul-none
shared-memory sweep.
