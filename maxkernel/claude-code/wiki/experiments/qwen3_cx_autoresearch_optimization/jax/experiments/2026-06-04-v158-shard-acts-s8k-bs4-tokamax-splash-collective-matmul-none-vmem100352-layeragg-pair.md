---
title: "Qwen3 JAX v158 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 100352 layer scheduler aggregator pair"
type: experiment
hypothesis: "If the layer-scheduler plus sparse-core collective-aggregator pair interacts with the v153/v154 scoped-VMEM 100352 schedule, explicitly enabling the pair may improve or at least change the current frontier HLO."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: refuted
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, scoped-vmem, scheduler, sparse-core-collective-aggregator, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v158 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 100352 layer scheduler aggregator pair

## Hypothesis under test

**Hypothesis**: The current v154 frontier carries
`--xla_tpu_enable_latency_hiding_layer_scheduler=false` and
`--xla_tpu_enable_sparse_core_collective_aggregator=false`. Earlier in the
batch-3 lineage, the explicit true/true pair was required for a scheduler
frontier, while later batch-4 collective-matmul-none runs often compiled to the
same final graph with or without the explicit pair. On the newer vmem100352
schedule family, explicitly enabling the pair may select a changed or faster
schedule.

**Mechanism**: Keep v154 fixed except replace only the layer-scheduler pair:

```text
--xla_tpu_enable_latency_hiding_layer_scheduler=true
--xla_tpu_enable_sparse_core_collective_aggregator=true
```

All other command shape, image, mesh, batch, profile window, Splash settings,
and scoped VMEM stay identical to v154.

**Predicted signal**: Support requires clean loss and a best-worker average
above the confirmed v153/v154 band (57,754-57,757 tok/s), ideally with a changed
train-step HLO/profile. A byte-identical or slower run refutes the interaction.

**Falsification criterion**: Compile/runtime failure, invalid loss, unchanged
HLO with no throughput gain, or throughput below the confirmed frontier.

## Setup

GKE workload `alekseyv-qwen3-v158-bs4-layeragg`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v158-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem100352-layeragg-pair`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: exact v154 stack except the layer-scheduler pair is true/true.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers with normal loss and v153/v154-band
throughput:

- Worker 0: 57,730 tok/s average, 7,216 tok/s/chip, 41.4% MFU; final loss
  12.0461.
- Worker 1: 57,755 tok/s average, 7,219 tok/s/chip, 41.4% MFU; final loss
  12.0461.
- Best steady steps were about 57,870 tok/s early after warmup.

This is a valid tie with the confirmed v153/v154 band, not a clear improvement.

## Profile

Profile artifacts were written under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v158-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem100352-layeragg-pair/plugins/profile/2026_06_04_18_08_02/`

Both host xplane files are present. XProf HTTP indexing remained unreliable, so
the decision uses worker logs plus direct GCS HLO/profile artifacts.

## HLO Dump

Fresh train-step HLO was emitted as
`module_0267.jit_train_step.cl_854318611.after_optimizations.txt`:

- Size: 783,949 bytes
- SHA256:
  `3734fae066fc35c4b9c6c4cce9b0015413ab4513c34a09ba33a0e1862bff3a3d`
- Compiled memory: 26.86 GiB

This is byte-identical to v153's optimized train-step HLO, despite the flagfile
confirming:

```text
--xla_tpu_enable_latency_hiding_layer_scheduler='true'
--xla_tpu_enable_sparse_core_collective_aggregator='true'
--xla_tpu_scoped_vmem_limit_kib=100352
```

## Verdict

**Refuted/no-gain tie.** The explicit layer-scheduler plus sparse-core
collective-aggregator pair is accepted but compiles to the same optimized HLO as
v153 and does not improve throughput beyond the confirmed frontier. Keep the
simpler v154 false/false form unless a later mechanism needs this pair.
