---
title: "Qwen3 JAX v123 - shard acts s8k bs4 tokamax splash layeragg collective matmul none rerun"
type: experiment
hypothesis: "Exact rerun of v118 to confirm the batch-4 throughput frontier before banking it."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, all-gather, reduce-scatter, latency-hiding, scheduler, layer-scheduler, sparse-core-collective-aggregator, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, rerun, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v123 - shard acts s8k bs4 tokamax splash layeragg collective matmul none rerun

## Hypothesis under test

**Hypothesis**: v118's batch-4/global-32 gain over the confirmed batch-3
collective-matmul-none frontier is durable, not noise. An exact rerun should
repeat the 56.4k tok/s / 40.5% MFU band with finite loss and the same train-step
program shape.

**Mechanism**: Exact rerun of v118:

```text
--batch_size=4
--xla_tpu_scoped_vmem_limit_kib=81920
--xla_tpu_scheduler_percent_shared_memory_limit=100
--xla_tpu_all_gather_collective_matmul_mode=none
--xla_tpu_reduce_scatter_collective_matmul_mode=none
```

**Predicted signal**: Clean completion with throughput near v118's
56,445/56,418 tok/s worker band and finite loss.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion that falls back to the v116/v117 batch-3 throughput band without
repeating the v118 gain.

## Setup

GKE workload `alekseyv-qwen3-v123-bs4-rerun`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v123-shard-acts-s8k-bs4-tokamax-splash-layeragg-collective-matmul-none-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: exact v118 stack.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers (`EXIT_CODE=0`, no pod restarts).

| Worker | Avg throughput | Tok/s/chip | Approx MFU | Best observed step | Final loss |
|--------|----------------|------------|------------|--------------------|------------|
| 0 | 56,448 tok/s | 7,056 | 40.5% | 56,599 tok/s | 12.0461 |
| 1 | 56,432 tok/s | 7,054 | 40.5% | 56,596 tok/s | 12.0461 |

This confirms v118's batch-4/global-32 throughput band. It is a small but
repeatable gain over the confirmed batch-3 collective-matmul-none band.

## Profile

XProf run:

```text
2026-06-04-qwen3-jax-v123-shard-acts-s8k-bs4-tokamax-splash-layeragg-collective-matmul-none-rerun/2026_06_04_11_52_08
```

Summary:

- XProf step-time average: 1999.2 ms.
- Total device time: 111,868.7 ms.
- Idle: 260.1 ms / 0.2%.
- Train-step program: `jit_train_step(16296087512277873197)`, same as v118.
- MXU utilization: 64.0%.
- Peak HBM: 26.91 GiB (`21.118` GiB stack, `5.7894` GiB heap, `4.3388` GiB free).

Top train-step buckets:

| Bucket | Time | Share |
|--------|------|-------|
| convolution fusion | 60,233.4 ms | 53.8% |
| custom-call | 29,348.3 ms | 26.2% |
| splash DKV | 15,477.4 ms | 13.8% |
| splash forward residuals | 13,870.9 ms | 12.4% |
| loop fusion | 11,676.7 ms | 10.4% |
| all-reduce-scatter fusion | 3,762.5 ms | 3.4% |

## HLO Dump

No new HLO objects were emitted under the v123 prefix, consistent with a
persistent-cache hit. XProf reports the same train-step program hash as v118
(`16296087512277873197`), so use v118's HLO evidence for this exact compiled
program:

- Optimized train-step text size: 787,978 bytes.
- HLO content hash: `3efb113cd72f8c9d08804e6cec46cf9373d83c06fae34ae9e3c27468e6b83fdf`.
- HLO prefix object count: 49.
- Flagfile confirms both collective-matmul modes are `NONE`, scoped VMEM is
  81920 KiB, and scheduler shared-memory limit is 100.

## Verdict

**Supported.** v123 confirms v118's batch-4/global-32 frontier. Adopt batch 4
with collective-matmul-none as the durable current best:

```text
global_batch=32
avg throughput ~=56.44k tok/s
approx MFU=40.5%
```

Batch 5 scalar fit attempts v119-v122 missed HBM even after shared-memory and
scoped-VMEM tweaks, so continue from this batch-4 frontier.
