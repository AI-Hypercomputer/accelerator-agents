---
title: "Qwen3 JAX v139 - shard acts s8k bs4 tokamax splash collective matmul none host sendrecv64"
type: experiment
hypothesis: "If the batch-4 collective-matmul-none frontier is mildly host-transfer/concurrency sensitive, lowering max host send/recv concurrency to 64 may reduce contention and improve steady throughput."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, host-sendrecv, latency-hiding, scheduler, selective-resources, async-collective-fusion, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v139 - shard acts s8k bs4 tokamax splash collective matmul none host sendrecv64

## Hypothesis under test

**Hypothesis**: v138 showed that higher host send/recv concurrency 128 preserves
the frontier HLO but does not improve throughput. The symmetric lower setting,
64, may reduce host/dispatch contention for the batch-4 collective-matmul-none
stack.

**Mechanism**: Restore the v127 simplified batch-4/global-32 frontier stack and
change only:

```text
--xla_max_concurrent_host_send_recv=64
```

from the current baseline value 100.

**Predicted signal**: Same train-step HLO/profile family with higher
best-worker average than v127/v136-v138 and ideally a best observed step above
v123's 56,599 tok/s.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed HLO
with no throughput gain, or clean completion that remains in the 56.43k-56.46k
tok/s frontier band.

## Setup

GKE workload `alekseyv-qwen3-v139-bs4-sendrecv64`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v139-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-host-sendrecv64`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v127 simplified stack plus `--xla_max_concurrent_host_send_recv=64`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers with normal loss ending at 12.0461.

- Worker0: 56,439 tok/s avg, 7,055 tok/s/chip, 40.5% MFU; best observed steady step 56,553 tok/s.
- Worker1: 56,416 tok/s avg, 7,052 tok/s/chip, 40.4% MFU; best observed steady step 56,557 tok/s.

This regresses relative to v127 and remains below v123's 56,599 tok/s best
observed step.

## Profile

XProf run:
`2026-06-04-qwen3-jax-v139-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-host-sendrecv64/2026_06_04_15_03_14`

- Step average: 2000.3 ms, stddev 2489.9 ms.
- Total time: 111,872.8 ms; compute time 111,544.2 ms; idle 328.6 ms / 0.3%.
- MXU utilization: 64.1%.
- Peak HBM: 26.91 GiB (21.118 GiB stack, 5.789 GiB heap, 4.339 GiB free).
- Train-step program: `jit_train_step(16296087512277873197)`.
- Top buckets: convolution fusion 60,210.3 ms / 53.8%; custom-call 29,348.7 ms / 26.2%; loop fusion 11,668.9 ms / 10.4%; all-reduce-scatter fusion 3,762.5 ms / 3.4%.
- Splash attention custom calls: DKV 15,477.7 ms / 13.8%; forward residuals 13,871.0 ms / 12.4% combined.

The profile is in the exact v123/v127 frontier family and shows no useful
redistribution from lower host send/recv concurrency.

## HLO Dump

Fresh HLO dumps were emitted. The train-step optimized HLO is byte-identical to
the v123/v127/v138 frontier program:

- Train-step module: `module_0267.jit_train_step.cl_854318611`
- Optimized HLO size: 787,978 bytes
- SHA256: `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`
- Compiler memory report: 28,862,302,652 bytes / 26.88 GiB
- Flagfile confirms `--xla_max_concurrent_host_send_recv=64`,
  `--xla_tpu_host_transfer_overlap_limit=24`, and collective-matmul modes
  `NONE`.

## Verdict

Refuted/tie. Lowering max concurrent host send/recv from 100 to 64 keeps the
exact frontier HLO/profile but slightly regresses throughput. Do not carry it.

The host send/recv scalar sweep is closed for the current batch-4
collective-matmul-none frontier; keep the baseline value 100. Next test
`--xla_latency_hiding_scheduler_rerun=4` on the v127 simplified stack.
