---
title: "Qwen3 JAX v145 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 86016 rerun"
type: experiment
hypothesis: "The v144 scoped-VMEM 86016 improvement is repeatable and should be carried as the new batch-4 collective-matmul-none frontier if the rerun stays above the v123/v127 band."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, scoped-vmem, latency-hiding, scheduler, selective-resources, async-collective-fusion, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8, rerun]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v145 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 86016 rerun

## Hypothesis under test

**Hypothesis**: v144's scoped-VMEM 86016 run is a real improvement, not a
single-run noise event. If the rerun preserves the same HLO family and remains
above the v123/v127/v136-v142 throughput band, 86016 should become the new
frontier setting.

**Mechanism**: Exact rerun of v144:

```text
--xla_tpu_scoped_vmem_limit_kib=86016
```

with the v127 simplified batch-4/global-32 collective-matmul-none stack.

**Predicted signal**: Clean completion, normal loss, train-step HLO matching
v144 or an equivalent 86016 schedule, best-worker average clearly above the
56.45k band, and best observed steps near or above v144's 56,728.

**Falsification criterion**: Compile/runtime failure, invalid loss, or rerun
average falling back into the 56.43k-56.46k frontier band.

## Setup

GKE workload `alekseyv-qwen3-v145-bs4-vmem86016r`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v145-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem86016-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v144 exact stack with `--xla_tpu_scoped_vmem_limit_kib=86016`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v145-bs4-vmem86016r` completed cleanly on both
workers with normal loss ending at 12.0458.

- Worker 0: 56,603 tok/s average, 7,075 tok/s/chip, approx 40.6% MFU.
- Worker 1: 56,574 tok/s average, 7,072 tok/s/chip, approx 40.6% MFU.
- Best observed steady steps: worker0 56,708 tok/s, worker1 56,697 tok/s.

This confirms the v144 throughput lift remains above the old v123/v127 frontier
band, although v144 still holds the highest single-run average and best step.

## Profile

XProf run
`2026-06-04-qwen3-jax-v145-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem86016-rerun/2026_06_04_16_10_05`:

- Step time average: 1994.4 ms.
- Total profile time: 111,550.1 ms; idle 303.1 ms / 0.3%.
- MXU utilization: 64.0%.
- Train-step program: `jit_train_step(11563614913035853403)`, matching v144.
- Top buckets: convolution fusion 59,855.9 ms / 53.7%, custom-call
  29,354.8 ms / 26.3%, loop fusion 11,697.6 ms / 10.5%,
  all-reduce-scatter fusion 3,762.1 ms / 3.4%.
- Splash custom calls: DKV 15,485.0 ms / 13.9%; forward residuals combined
  13,869.8 ms / 12.4%.

## HLO Dump

No fresh HLO dump was emitted because this rerun hit the persistent compilation
cache. Use v144's compiled-program evidence:

- Train-step program: `11563614913035853403`.
- Optimized HLO size: 787,753 bytes.
- SHA256:
  `aa02d3286a36812ae812c94b82e0bd8fb733cc06e99abf6db17d42aa896cf594`.
- Memory report: 28,858,091,964 bytes / 26.88 GiB.

## Verdict

Supported. The scoped-VMEM 86016 setting repeats above the old frontier and
preserves the same train-step program/profile shape as v144. Carry
`--xla_tpu_scoped_vmem_limit_kib=86016` as the new batch-4
collective-matmul-none frontier setting. Next test the upper bracket 90112 KiB
once from this new frontier to see whether the improvement continues or
over-shoots.
