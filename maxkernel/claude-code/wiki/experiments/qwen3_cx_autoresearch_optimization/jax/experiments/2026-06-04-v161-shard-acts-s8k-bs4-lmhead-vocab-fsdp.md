---
title: "Qwen3 JAX v161 - shard acts s8k bs4 lm head vocab fsdp"
type: experiment
hypothesis: "If the residual CE/lm-head boundary is constrained by hidden-dimension FSDP sharding of the untied lm_head, then sharding lm_head.weight on the vocab axis may reduce collective/logit pressure or unlock a better HLO schedule."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v161-lmhead-vocab-fsdp
status: provisional-supported
tags: [qwen3-cc, jax, flax-nnx, lm-head, sharding, ce, maxtext-ce, splash, tokamax-splash, collective-matmul, scoped-vmem, remat, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v161 - shard acts s8k bs4 lm head vocab fsdp

## Hypothesis under test

**Hypothesis**: The retrospective identifies CE/lm-head materialization and
collective layout as the next structural target after the local VMEM bracket
closed. The current frontier shards `lm_head.weight` as `("tp", "fsdp")`; with
`tp=1`, that leaves vocab replicated and shards the hidden dimension across
FSDP. Sharding `lm_head.weight` as `("fsdp", None)` instead partitions the
vocab axis across FSDP and may change the logits/CE collective boundary.

**Mechanism**: Build a layered image from
`qwen3-8b-jax:v041-shard-acts` and change only this sharding-plan entry:

```python
"lm_head.weight": ("fsdp", None)
```

Run the confirmed v154/v160 stack otherwise unchanged:
seq8192, global batch 32, MaxText CE, Splash/tokamax attention, collective
matmul modes disabled, scoped VMEM 100352 KiB.

**Predicted signal**: Support requires clean loss and either a best-worker
average above the confirmed v153/v154 band (57,754-57,757 tok/s), a materially
different train-step HLO/profile that reduces CE/logit/collective pressure, or
a compiled-memory reduction that plausibly reopens batch 5.

**Falsification criterion**: Compile/runtime failure, invalid loss, unchanged
HLO/profile with no throughput gain, or a clean regression versus v154.

## Setup

GKE workload `alekseyv-qwen3-v161-bs4-lmheadvocab`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v161-lmhead-vocab-fsdp`
- **Base image**: `qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v161-shard-acts-s8k-bs4-lmhead-vocab-fsdp`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: confirmed v154 stack with `--xla_tpu_scoped_vmem_limit_kib=100352`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v161-bs4-lmheadvocab` completed cleanly on both
workers with normal loss ending near 12.046.

- Worker 0: average 57,765 tok/s, 7,221 tok/s/chip, approx 41.4% MFU.
- Worker 1: average 57,744 tok/s, 7,218 tok/s/chip, approx 41.4% MFU.
- Best observed steady steps: worker0 57,908 tok/s, worker1 57,904 tok/s.
- Final loss: worker0 12.0458, worker1 12.0458.

This is a small average gain over v153/v154, but unlike v160 it comes with a
distinct HLO and lower compiled memory. Treat as provisional until an exact
rerun confirms it.

## Profile

Profile files were written under:

```text
gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v161-shard-acts-s8k-bs4-lmhead-vocab-fsdp/plugins/profile/2026_06_04_18_35_40/
```

Both TPU hosts produced `.xplane.pb` and trace JSON files. Local XProf indexing
was unreliable in this session, so the first-pass verdict uses worker summaries,
HLO identity, and compiled-memory reports.

## HLO Dump

Fresh train-step HLO:

```text
gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v161-shard-acts-s8k-bs4-lmhead-vocab-fsdp/hlo/module_0105.jit_train_step.cl_854318611.after_optimizations.txt
```

- Optimized HLO size: 785,947 bytes.
- SHA256: `3648b17f7b1fdb754e9b9330400997ed2e06c3f25543cb549bb90559a8b7a348`.
- Compiled memory: 26.76 GiB.

This is a distinct train-step HLO from v153/v154/v160 and reduces compiled
memory by about 0.10 GiB versus the 26.86 GiB frontier family.

## Verdict

Provisional supported. Vocab-axis FSDP sharding for `lm_head.weight` changes the
HLO, reduces compiled memory, preserves loss, and slightly improves the observed
best-worker average/best step. Exact rerun required before carrying.
