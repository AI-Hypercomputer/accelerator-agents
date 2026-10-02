---
title: "Qwen3 JAX v138 - shard acts s8k bs4 tokamax splash collective matmul none host sendrecv128"
type: experiment
hypothesis: "After collective-matmul-none changed the communication profile, a higher host send/recv concurrency limit may improve the batch-4 frontier even though it tied on the older batch-3 layeragg stack."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, host-sendrecv, latency-hiding, scheduler, selective-resources, async-collective-fusion, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v138 - shard acts s8k bs4 tokamax splash collective matmul none host sendrecv128

## Hypothesis under test

**Hypothesis**: The collective-matmul-none stack shifted the communication and
host-transfer balance enough that `--xla_max_concurrent_host_send_recv=128` may
perform differently from the older batch-3 layeragg v109 tie.

**Mechanism**: Restore the v127 simplified batch-4/global-32 frontier stack,
including host-transfer overlap limit 24, and change only:

```text
--xla_max_concurrent_host_send_recv=128
```

from the current baseline value 100.

**Predicted signal**: Same train-step HLO/profile family with lower step time,
higher best-worker average than v127/v136/v137, and ideally a best observed step
above v123's 56,599 tok/s.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed HLO
with no throughput gain, or clean completion that remains in the 56.43k-56.46k
tok/s frontier band.

## Setup

GKE workload `alekseyv-qwen3-v138-bs4-sendrecv128`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v138-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-host-sendrecv128`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v127 simplified stack plus `--xla_max_concurrent_host_send_recv=128`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers with normal loss ending at 12.0461.

- Worker0: 56,447 tok/s avg, 7,056 tok/s/chip, 40.5% MFU; best observed steady step 56,550 tok/s.
- Worker1: 56,428 tok/s avg, 7,053 tok/s/chip, 40.5% MFU; best observed steady step 56,550 tok/s.

This stays in the normal v127/v135-v137 frontier band but does not beat v127's
best-worker average or v123's best observed step.

## Profile

XProf run:
`2026-06-04-qwen3-jax-v138-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-host-sendrecv128/2026_06_04_14_52_26`

- Step average: 1999.6 ms, stddev 2489.2 ms.
- Total time: 111,860.2 ms; compute time 111,569.4 ms; idle 290.9 ms / 0.3%.
- MXU utilization: 64.1%.
- Peak HBM: 26.91 GiB (21.118 GiB stack, 5.789 GiB heap, 4.339 GiB free).
- Train-step program: `jit_train_step(16296087512277873197)`.
- Top buckets: convolution fusion 60,206.4 ms / 53.8%; custom-call 29,348.8 ms / 26.2%; loop fusion 11,675.1 ms / 10.4%; all-reduce-scatter fusion 3,762.3 ms / 3.4%.
- Splash attention custom calls: DKV 15,477.8 ms / 13.8%; forward residuals 13,871.0 ms / 12.4% combined.

Profile shape is indistinguishable from the v123/v127 frontier family. The
runtime average is below v127/v136 despite the profile total landing in-band.

## HLO Dump

Fresh HLO dumps were emitted. The train-step optimized HLO is byte-identical to
the v123/v127/v135-v136 frontier program:

- Train-step module: `module_0267.jit_train_step.cl_854318611`
- Optimized HLO size: 787,978 bytes
- SHA256: `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`
- Compiler memory report: 28,862,302,652 bytes / 26.88 GiB
- Flagfile confirms `--xla_max_concurrent_host_send_recv=128`,
  `--xla_tpu_host_transfer_overlap_limit=24`, and collective-matmul modes
  `NONE`.

## Verdict

Refuted/tie. Raising max concurrent host send/recv from 100 to 128 keeps the
exact frontier HLO/profile but does not improve throughput. Do not carry it.

Return to `--xla_max_concurrent_host_send_recv=100`; next test the symmetric
lower value, 64, on the same batch-4 collective-matmul-none baseline.
