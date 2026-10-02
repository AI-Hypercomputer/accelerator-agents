---
title: "Qwen3 JAX v152 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 102400"
type: experiment
hypothesis: "If the 98304 scoped-VMEM frontier is not yet the upper scheduling limit, 102400 KiB may continue the step-time improvement; otherwise it should reveal the overshoot point."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, scoped-vmem, latency-hiding, scheduler, selective-resources, async-collective-fusion, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v152 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 102400

## Hypothesis under test

**Hypothesis**: v151 confirmed scoped VMEM 98304 as a real frontier shift. A
further increase to 102400 KiB may provide more scheduler slack and lower step
time, or it may overshoot the useful VMEM band.

**Mechanism**: Keep the v151 frontier stack fixed and change only:

```text
--xla_tpu_scoped_vmem_limit_kib=102400
```

from 98304.

**Predicted signal**: Support requires clean loss plus best-worker average
above the confirmed 98304 frontier, ideally above v150's 57,578 tok/s average
or v151's 57,707 best observed step. Profile should show lower step time or a
clearly improved train-step schedule without increasing idle or collective
exposure.

**Falsification criterion**: Compile/runtime failure, invalid loss, HBM/OOM,
or clean completion that falls back to or below the v150/v151 98304 band.

## Setup

GKE workload `alekseyv-qwen3-v152-bs4-vmem102400`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v152-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem102400`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v151 frontier stack plus `--xla_tpu_scoped_vmem_limit_kib=102400`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers with normal loss ending at 12.0461.

- worker0: 57,410 tok/s average, 7,176 tok/s/chip, 41.2% MFU.
- worker1: 57,385 tok/s average, 7,173 tok/s/chip, 41.1% MFU.
- Best observed steady steps: worker0 57,516 tok/s; worker1 57,513 tok/s.
- This is below the confirmed v150/v151 98304 KiB frontier: 57,578 tok/s
  best-worker average and 57,707 best observed steady step.

## Profile

XProf run:
`2026-06-04-qwen3-jax-v152-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem102400/2026_06_04_17_09_36`.

- Step average: 1966.8 ms.
- Total profile time: 110,002.110 ms, with 323.753 ms idle (0.3%).
- MXU utilization: 65.2%.
- Train-step program: `jit_train_step(12059987095245408810)`.
- Program work: 57,371.383 TFLOPs, 39,970.496 GiB accessed.
- Top buckets include convolution fusion 57,506.802 ms / 52.3%, custom-call
  approximately 29,369 ms / 26.7%, data formatting 2,897.602 ms / 2.6%, and
  custom fusion 1,789.027 ms / 1.6%.
- Custom-call split from device events: DKV approximately 15,495 ms / 14.1%;
  forward residuals approximately 13,874 ms / 12.6% combined.
- Memory profile: 26.92 GiB peak HBM, 21.12 GiB stack reservation,
  5.80 GiB heap allocation, and 4.32 GiB free.

## HLO Dump

Fresh optimized train-step HLO was emitted under
`module_0267.jit_train_step.cl_854318611.after_optimizations.txt`.

- Optimized HLO size: 782,971 bytes.
- Content hash:
  `ed58479a073ffd6c56449ddaf4da1cd5799d781d25011f602b379bf57fcb897f`.
- Compiled memory report: 28,841,150,904 bytes / 26.86 GiB.
- The train-step program changed from v150/v151's
  `6628478088121256461` to `12059987095245408810`, so the higher VMEM setting
  selected a real schedule rather than hitting the same compile artifact.

## Verdict

Refuted. Scoped VMEM 102400 is valid and selects a fresh train-step schedule,
but it regresses both live throughput and XProf step time relative to the
confirmed 98304 frontier. Keep scoped VMEM 98304 as the carried setting and
probe a narrower interpolation below 102400 rather than increasing further.
