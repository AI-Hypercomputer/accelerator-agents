---
title: "Qwen3 JAX v160 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 100864"
type: experiment
hypothesis: "If the high-side regression begins only near 101376 KiB, then the midpoint 100864 KiB may retain or slightly improve the confirmed 100352 KiB schedule."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, scoped-vmem, scheduler, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v160 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 100864

## Hypothesis under test

**Hypothesis**: v153/v154 confirmed 100352 KiB, v155 refuted 101376 KiB with a
byte-identical HLO but lower measured throughput, and v159 refuted the
lower-side 99328 KiB point with a changed slower HLO. The high-side midpoint
100864 KiB checks whether the true local optimum is a narrow plateau above
100352 or whether 100352 remains the peak.

**Mechanism**: Keep v154 fixed except change only:

```text
--xla_tpu_scoped_vmem_limit_kib=100864
```

All other command shape, image, mesh, batch, profile window, Splash settings,
and scheduler flags stay identical to v154.

**Predicted signal**: Support requires clean loss and a best-worker average
above the confirmed v153/v154 band (57,754-57,757 tok/s), ideally with stable
or improved HLO/profile evidence. A byte-identical HLO without throughput gain
or a regression confirms 100352 as the local VMEM peak.

**Falsification criterion**: Compile/runtime failure, invalid loss, unchanged
HLO with no throughput gain, or throughput below the confirmed frontier.

## Setup

GKE workload `alekseyv-qwen3-v160-bs4-vmem100864`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v160-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem100864`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: exact v154 stack except `--xla_tpu_scoped_vmem_limit_kib=100864`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v160-bs4-vmem100864` completed cleanly on both workers
with normal loss ending at 12.0461.

- Worker 0: average 57,762 tok/s, 7,220 tok/s/chip, approx 41.4% MFU.
- Worker 1: average 57,736 tok/s, 7,217 tok/s/chip, approx 41.4% MFU.
- Best observed steady steps: worker0 57,854 tok/s, worker1 57,854 tok/s.
- Final loss: 12.0461 on both workers.

The worker0 average is only 5 tok/s above v153 and 8 tok/s above v154, while
the best observed step is below v153's 57,888. This is inside measurement noise.

## Profile

Profile files were written under:

```text
gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v160-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem100864/plugins/profile/2026_06_04_18_26_45/
```

Both TPU hosts produced `.xplane.pb` and trace JSON files. Local XProf indexing
was still unreliable in this session, so the verdict relies on worker summaries
plus the optimized HLO identity.

## HLO Dump

Fresh train-step HLO:

```text
gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v160-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem100864/hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt
```

- Optimized HLO size: 783,949 bytes.
- SHA256: `3734fae066fc35c4b9c6c4cce9b0015413ab4513c34a09ba33a0e1862bff3a3d`.
- Compiled memory: 26.86 GiB.

This is byte-identical to the v153/v158 optimized train-step HLO.

## Verdict

Refuted/tie. The flag is accepted and the run completes, but the optimized
train-step HLO is byte-identical to the confirmed v153/v154 schedule family.
The tiny worker0 average increase is noise, and the best step does not beat
v153. Do not carry scoped VMEM 100864; keep 100352 KiB as the confirmed local
VMEM peak and pivot to a structural hypothesis.
