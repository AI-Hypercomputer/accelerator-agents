---
title: "Qwen3 JAX v142 - shard acts s8k bs4 tokamax splash collective matmul none shared memory 90"
type: experiment
hypothesis: "A stronger scheduler shared-memory cap may force a distinct lower-memory batch-4 collective-matmul-none schedule that improves contention enough to offset any extra scheduling cost."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, scheduler-shmem, latency-hiding, scheduler, selective-resources, async-collective-fusion, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v142 - shard acts s8k bs4 tokamax splash collective matmul none shared memory 90

## Hypothesis under test

**Hypothesis**: v141 showed that a 95% scheduler shared-memory cap is a no-op
on the batch-4 collective-matmul-none frontier. A stronger 90% cap may force a
real schedule change, potentially lowering HBM pressure or improving collective
overlap enough to beat the current 56.45k tok/s frontier band.

**Mechanism**: Restore the v127 simplified batch-4/global-32 frontier stack,
including scheduler rerun 3, host send/recv 100, host-transfer overlap 24, and
collective-matmul modes `NONE`; change only:

```text
--xla_tpu_scheduler_percent_shared_memory_limit=90
```

from the current baseline value 100.

**Predicted signal**: Valid train-step compile with either a changed HLO and
lower compiled/profile HBM, or a same-HLO no-op. Support requires a best-worker
average above the v123/v127/v136-v141 band and ideally a best observed step
above v123's 56,599 tok/s.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed HLO
with no throughput gain, or clean completion in the same 56.43k-56.46k tok/s
frontier band. A same-HLO no-op also refutes carrying the cap.

## Setup

GKE workload `alekseyv-qwen3-v142-bs4-shmem90`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v142-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-shmem90`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v127 simplified stack plus `--xla_tpu_scheduler_percent_shared_memory_limit=90`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v142-bs4-shmem90` completed cleanly on both
workers with normal loss ending at 12.0461.

- Worker 0: 56,452 tok/s average, 7,057 tok/s/chip, approx 40.5% MFU.
- Worker 1: 56,426 tok/s average, 7,053 tok/s/chip, approx 40.5% MFU.
- Best observed steady steps: worker0 56,550 tok/s, worker1 56,556 tok/s.

This remains inside the same 56.43k-56.46k tok/s frontier band and below
v123's 56,599 tok/s best observed step.

## Profile

XProf run
`2026-06-04-qwen3-jax-v142-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-shmem90/2026_06_04_15_36_10`:

- Step time average: 2000.0 ms.
- Total profile time: 111,840.6 ms; idle 316.4 ms / 0.3%.
- MXU utilization: 64.1%.
- Train-step program: `jit_train_step(8671917837109355706)`.
- Top buckets: convolution fusion 60,200.9 ms / 53.8%, custom-call
  29,348.5 ms / 26.2%, loop fusion 11,671.6 ms / 10.4%,
  all-reduce-scatter fusion 3,990.8 ms / 3.6%.
- Splash custom calls: DKV 15,477.6 ms / 13.8%; forward residuals combined
  13,871.0 ms / 12.4%.

## HLO Dump

Fresh train-step HLO dump:

- Module: `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`.
- Optimized HLO size: 787,978 bytes.
- SHA256:
  `149d09d651b30397ae9474d92bfada657c6d99896fac2686ff580b16b328df5a`.
- Memory report: 28,862,302,652 bytes / 26.88 GiB.
- Flagfile confirms `--xla_tpu_scheduler_percent_shared_memory_limit=90`
  with rerun 3, host send/recv 100, host-transfer overlap 24, and both
  collective-matmul modes `NONE`.

Compared with v141/v127, the HLO is the same size and same memory but not
byte-identical. A text diff shows scheduling reorder around copy/all-reduce
placement and async slice/custom-call staging, not a model math change.

## Verdict

Refuted/tie. The stronger shared-memory cap creates a real train-step program
hash/content-hash change, but does not reduce compiled HBM and does not improve
the profile or throughput. Do not carry
`--xla_tpu_scheduler_percent_shared_memory_limit=90`; keep the simplified v127
baseline with shared-memory limit 100. The post-collective-matmul-none
shared-memory scalar sweep is closed.
