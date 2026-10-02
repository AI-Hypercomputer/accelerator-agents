---
title: "Qwen3 JAX v140 - shard acts s8k bs4 tokamax splash collective matmul none scheduler rerun4"
type: experiment
hypothesis: "The batch-4 collective-matmul-none frontier may benefit from one additional latency-hiding scheduler rerun even though rerun4 tied on the older batch-3 layeragg stack."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, latency-hiding, scheduler-rerun, scheduler, selective-resources, async-collective-fusion, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v140 - shard acts s8k bs4 tokamax splash collective matmul none scheduler rerun4

## Hypothesis under test

**Hypothesis**: The collective-matmul-none HLO has a different scheduling
surface than the older batch-3 layeragg frontier. Raising
`--xla_latency_hiding_scheduler_rerun` from 3 to 4 may find a better schedule
without changing model semantics.

**Mechanism**: Restore the v127 simplified batch-4/global-32 frontier stack,
including host-transfer overlap 24 and host send/recv 100, and change only:

```text
--xla_latency_hiding_scheduler_rerun=4
```

from the current baseline value 3.

**Predicted signal**: Same train-step HLO/profile family or a valid changed
schedule with lower step time, higher best-worker average than v127/v136-v139,
and ideally a best observed step above v123's 56,599 tok/s.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed HLO
with no throughput gain, or clean completion that remains in the 56.43k-56.46k
tok/s frontier band.

## Setup

GKE workload `alekseyv-qwen3-v140-bs4-rerun4`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v140-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-rerun4`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v127 simplified stack plus `--xla_latency_hiding_scheduler_rerun=4`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers with normal loss ending at 12.0461.

- Worker0: 56,445 tok/s avg, 7,056 tok/s/chip, 40.5% MFU; best observed steady step 56,549 tok/s.
- Worker1: 56,424 tok/s avg, 7,053 tok/s/chip, 40.5% MFU; best observed steady step 56,549 tok/s.

The result stays in the v127/v135-v139 frontier band but does not improve the
average or best observed step.

## Profile

XProf run:
`2026-06-04-qwen3-jax-v140-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-rerun4/2026_06_04_15_13_34`

- Step average: 1999.0 ms, stddev 2488.4 ms.
- Total time: 111,824.4 ms; compute time 111,531.3 ms; idle 293.2 ms / 0.3%.
- MXU utilization: 64.1%.
- Peak HBM: 26.91 GiB (21.118 GiB stack, 5.789 GiB heap, 4.339 GiB free).
- Train-step program: `jit_train_step(16296087512277873197)`.
- Top buckets: convolution fusion 60,213.4 ms / 53.8%; custom-call 29,348.5 ms / 26.2%; loop fusion 11,664.5 ms / 10.4%; all-reduce-scatter fusion 3,762.2 ms / 3.4%.
- Splash attention custom calls: DKV 15,477.7 ms / 13.8%; forward residuals 13,870.8 ms / 12.4% combined.

The profile total is in-band, but runtime throughput remains below v127 and the
best observed steps stay below v123.

## HLO Dump

Fresh HLO dumps were emitted. The train-step optimized HLO is byte-identical to
the v123/v127 frontier program:

- Train-step module: `module_0267.jit_train_step.cl_854318611`
- Optimized HLO size: 787,978 bytes
- SHA256: `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`
- Compiler memory report: 28,862,302,652 bytes / 26.88 GiB
- Flagfile confirms `--xla_latency_hiding_scheduler_rerun=4`,
  `--xla_max_concurrent_host_send_recv=100`, host-transfer overlap 24, and
  collective-matmul modes `NONE`.

## Verdict

Refuted/tie. Raising latency-hiding scheduler reruns from 3 to 4 keeps the exact
frontier HLO/profile but does not improve throughput. Do not carry it.

Return to `--xla_latency_hiding_scheduler_rerun=3`; next test the mild
shared-memory cap `--xla_tpu_scheduler_percent_shared_memory_limit=95` on the
same batch-4 collective-matmul-none baseline.
