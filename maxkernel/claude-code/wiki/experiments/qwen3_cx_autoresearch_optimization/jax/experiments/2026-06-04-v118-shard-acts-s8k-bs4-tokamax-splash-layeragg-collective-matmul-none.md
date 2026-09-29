---
title: "Qwen3 JAX v118 - shard acts s8k bs4 tokamax splash layeragg collective matmul none"
type: experiment
hypothesis: "The confirmed collective-matmul-none frontier frees enough HBM to scale per-chip batch from 3 to 4 at seq8192."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, all-gather, reduce-scatter, latency-hiding, scheduler, layer-scheduler, sparse-core-collective-aggregator, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v118 - shard acts s8k bs4 tokamax splash layeragg collective matmul none

## Hypothesis under test

**Hypothesis**: v116/v117's collective-matmul-none compiled program lowers peak
HBM enough to reopen per-chip batch 4 / global batch 32 at seq8192. Older batch
4 attempts were blocked or unattractive under different CE/HLO schedules; the
confirmed frontier now has peak HBM 22.21 GiB with about 9.03 GiB free.

**Mechanism**: Keep the confirmed v116/v117 stack fixed and change only:

```text
--batch_size=3 -> --batch_size=4
```

This increases global batch from 24 to 32 while preserving `fsdp=8,tp=1`,
scan, remat, tokamax Splash, MaxText CE, activation sharding, layer scheduler,
sparse-core collective aggregation, and both collective-matmul modes set to
`none`.

**Predicted signal**: Clean compile/run with finite loss and higher total
tokens/sec/MFU than v116/v117. Because per-step work rises by 33%, the key
metric is total throughput, not step time. A good outcome should clear 56k
tok/s and ideally improve MFU materially above 40.2%.

**Falsification criterion**: OOM, compile/runtime failure, invalid loss, or a
clean run whose total throughput does not beat the confirmed v116/v117 56k
tok/s frontier.

## Setup

GKE workload `alekseyv-qwen3-v118-bs4-cmnone`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v118-shard-acts-s8k-bs4-tokamax-splash-layeragg-collective-matmul-none`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: confirmed v116/v117 collective-matmul-none stack.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers (`EXIT_CODE=0`, no pod restarts).

| Worker | Avg throughput | Tok/s/chip | Approx MFU | Best observed step | Final loss |
|--------|----------------|------------|------------|--------------------|------------|
| 0 | 56,445 tok/s | 7,056 | 40.5% | 56,557 tok/s | 12.0461 |
| 1 | 56,418 tok/s | 7,052 | 40.5% | 56,553 tok/s | 12.0461 |

Batch 4 fits and is a small throughput frontier versus the confirmed
v116/v117 batch-3 band (56,017 repeated avg, 56,172 best observed). The gain is
real but sublinear: global batch rises 33%, while total throughput rises only
about 0.7-0.8%.

## Profile

XProf run:

```text
2026-06-04-qwen3-jax-v118-shard-acts-s8k-bs4-tokamax-splash-layeragg-collective-matmul-none/2026_06_04_11_11_24
```

Summary:

- XProf step-time average: 2000.8 ms (same reporting caveat as prior runs).
- Total device time: 111,891.2 ms.
- Idle: 331.7 ms / 0.3%.
- Train-step program: `jit_train_step(16296087512277873197)`.
- MXU utilization: 64.1%.
- Peak HBM: 26.91 GiB (`21.118` GiB stack, `5.7894` GiB heap, `4.3388` GiB free).

Top train-step buckets:

| Bucket | Time | Share | Notes |
|--------|------|-------|-------|
| convolution fusion | 60,203.4 ms | 53.8% | scales from v117's 44,278.5 ms |
| custom-call | 29,348.5 ms | 26.2% | Splash attention total |
| splash DKV | 15,477.4 ms | 13.8% | within custom-call |
| splash forward residuals | 13,871.1 ms | 12.4% | two residual custom-calls |
| loop fusion | 11,666.9 ms | 10.4% | slightly higher share than v117 |
| all-reduce-scatter fusion | 3,761.5 ms | 3.4% | nearly flat absolute time, lower share |

## HLO Dump

Train-step module:

```text
module_0112.jit_train_step.cl_854318611.after_optimizations.txt
```

- Optimized train-step text size: 787,978 bytes.
- HLO content hash: `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`.
- HLO prefix object count: 49.
- Train-step program hash: `16296087512277873197`.
- Flagfile confirms both collective-matmul modes are `NONE`, layer scheduler
  and sparse-core collective aggregator are enabled, async collective fusion
  multiple-steps is enabled, and scoped VMEM remains 81920 KiB.

## Verdict

**Supported, small gain.** Batch 4/global 32 is the new measured throughput
frontier, but it is not a qualitative scaling breakthrough. The useful learning
is that collective-matmul-none creates enough HBM margin for batch 4, while
batch scaling is mostly compute/custom-call proportional and leaves only
4.34 GiB free.

Carry batch 4 provisionally. Next fit probe: batch 5/global 40. Linear HBM
scaling from v117 to v118 predicts about 31.6 GiB peak, slightly above nominal
31.25 GiB capacity, but the estimate is close enough that a direct compile/run
test is worthwhile.
