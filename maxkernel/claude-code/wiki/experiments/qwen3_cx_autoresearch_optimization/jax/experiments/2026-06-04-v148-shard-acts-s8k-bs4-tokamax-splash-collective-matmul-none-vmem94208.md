---
title: "Qwen3 JAX v148 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 94208"
type: experiment
hypothesis: "If the 86016 -> 90112 scoped-VMEM gain reflects useful scheduling slack, increasing to 94208 may continue the trend or reveal the upper over-shoot point."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, scoped-vmem, latency-hiding, scheduler, selective-resources, async-collective-fusion, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v148 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 94208

## Hypothesis under test

**Hypothesis**: v146/v147 confirmed scoped VMEM 90112 as better than 86016 in
the current batch-4 collective-matmul-none family. A further 4096 KiB increase
to 94208 may provide more scheduling slack and improve step time, or it may
over-shoot and close the upper VMEM bracket.

**Mechanism**: Keep the v147 frontier stack fixed and change only:

```text
--xla_tpu_scoped_vmem_limit_kib=94208
```

from 90112.

**Predicted signal**: Support requires clean loss plus best-worker average
above v147's 56,644 tok/s or a lower XProf step time. If it falls back to the
v144/v145/v146 band or regresses, 90112 remains the carried setting.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed HLO
with no throughput gain, or clean completion below the v147 confirmed frontier.

## Setup

GKE workload `alekseyv-qwen3-v148-bs4-vmem94208`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v148-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem94208`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v147 frontier stack plus `--xla_tpu_scoped_vmem_limit_kib=94208`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers with normal loss ending at 12.0460.

- worker0: 56,773 tok/s average, 7,097 tok/s/chip, 40.7% MFU.
- worker1: 56,750 tok/s average, 7,094 tok/s/chip, 40.7% MFU.
- Best observed steady steps: worker0 56,904 tok/s; worker1 56,902 tok/s.
- This is a clear provisional improvement over the confirmed v147 90112
  frontier at 56,644 tok/s average and 56,785 best observed.

## Profile

XProf run:
`2026-06-04-qwen3-jax-v148-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem94208/2026_06_04_16_37_52`.

- Step average: 1988.6 ms.
- Total profile time: 111,227.166 ms, with 304.996 ms idle (0.3%).
- MXU utilization: 65.3%.
- Train-step program: `jit_train_step(15797032912653574046)`.
- Program work: 57,371.383 TFLOPs, 40,577.401 GiB accessed.
- Top buckets: convolution fusion 59,137.824 ms / 53.2%; custom-call
  29,340.198 ms / 26.4%; loop fusion 11,487.871 ms / 10.3%;
  all-reduce-scatter fusion 3,756.532 ms / 3.4%; data formatting
  2,881.279 ms / 2.6%; custom fusion 1,410.315 ms / 1.3%.
- Custom-call split: DKV 15,468.661 ms / 13.9%; forward residuals combined
  13,871.536 ms / 12.5%.

## HLO Dump

Fresh optimized train-step HLO was emitted under
`module_0267.jit_train_step.cl_854318611.after_optimizations.txt`.

- Optimized HLO size: 779,591 bytes.
- Content hash:
  `e78989b8aa0a3c726e47566c51f7fd42cd5d218e129b2bf0bff2fc5b4f51fc93`.
- Compiled memory report: 28,849,097,120 bytes / 26.87 GiB.
- Flagfile confirms `--xla_tpu_scoped_vmem_limit_kib=94208`,
  `--xla_tpu_scheduler_percent_shared_memory_limit=100`, scheduler rerun3,
  host send/recv 100, host-transfer overlap 24, and both collective-matmul
  modes set to `NONE`.

## Verdict

Provisional supported. Scoped VMEM 94208 selects a fresh train-step program,
shrinks optimized HLO versus v146/v147, lowers XProf step time to 1988.6 ms, and
raises observed throughput to 56,773 tok/s average / 56,904 best. Rerun exactly
before carrying 94208 as the durable frontier.
