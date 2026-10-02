---
title: "Qwen3 JAX v117 - shard acts s8k bs3 tokamax splash layeragg collective matmul none rerun"
type: experiment
hypothesis: "Exact rerun of v116 to confirm that the collective-matmul-none frontier is durable."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, all-gather, reduce-scatter, latency-hiding, scheduler, layer-scheduler, sparse-core-collective-aggregator, selective-resources, custom-call, remat, maxtext-ce, activation-sharding, long-seq, rerun, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v117 - shard acts s8k bs3 tokamax splash layeragg collective matmul none rerun

## Hypothesis under test

**Hypothesis**: v116's jump to 56,009 tok/s avg / 40.2% MFU is a durable
compiler frontier, not a one-off noisy run. An exact rerun should reproduce the
56k tok/s band, finite loss, smaller train-step HLO, lower peak HBM, and the
same `DISABLE_COLLECTIVE_MATMUL` flagfile state.

**Mechanism**: Exact rerun of v116:

```text
--xla_tpu_all_gather_collective_matmul_mode=none
--xla_tpu_reduce_scatter_collective_matmul_mode=none
```

on top of the v090 qseq1 layeragg frontier.

**Predicted signal**: Loss stays finite and throughput repeats near 56,000
tok/s avg / 40% MFU; XProf repeats the lower train-step device time and lower
HBM profile.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion that falls back into the old 51k tok/s / 36.6% MFU frontier band.

## Setup

GKE workload `alekseyv-qwen3-v117-cmnone-rerun`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v117-shard-acts-s8k-bs3-tokamax-splash-layeragg-collective-matmul-none-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: exact v116 stack.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers (`EXIT_CODE=0`, no pod restarts).

| Worker | Avg throughput | Tok/s/chip | Approx MFU | Best observed step | Final loss |
|--------|----------------|------------|------------|--------------------|------------|
| 0 | 56,041 tok/s | 7,005 | 40.2% | 56,151 tok/s | 12.0512 |
| 1 | 55,993 tok/s | 6,999 | 40.1% | 56,151 tok/s | 12.0512 |

This reproduces v116's 56k tok/s band and confirms the large jump over the
previous v090 51,033 tok/s / 36.6% MFU frontier. The final losses match v116
within normal run-to-run noise.

## Profile

XProf run:

```text
2026-06-04-qwen3-jax-v117-shard-acts-s8k-bs3-tokamax-splash-layeragg-collective-matmul-none-rerun/2026_06_04_10_59_36
```

Summary:

- XProf step-time average: 1515.3 ms (same reporting caveat as v116).
- Total device time: 84,645.3 ms.
- Idle: 366.1 ms / 0.4%.
- Train-step program: `jit_train_step(16402808057163658277)`, same as v116.
- MXU utilization: 62.9%.
- Peak HBM: 22.21 GiB (`16.4208` GiB stack, `5.7908` GiB heap, `9.0345` GiB free).

Top train-step buckets:

| Bucket | Time | Share | Notes |
|--------|------|-------|-------|
| convolution fusion | 44,278.5 ms | 52.3% | matmul-heavy bucket, same shape as v116 |
| custom-call | 22,060.9 ms | 26.1% | Splash attention total |
| splash DKV | 11,650.0 ms | 13.8% | within custom-call |
| splash forward residuals | 10,411.0 ms | 12.3% | two residual custom-calls |
| loop fusion | 8,517.5 ms | 10.1% | remains far below the old 18% frontier |
| all-reduce-scatter fusion | 3,757.9 ms | 4.4% | same as v116 |

## HLO Dump

No HLO dump objects were emitted under the v117 run prefix:

```text
gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v117-shard-acts-s8k-bs3-tokamax-splash-layeragg-collective-matmul-none-rerun/
```

The profile reports the same train-step program hash as v116
(`16402808057163658277`), so this is consistent with a persistent-cache hit.
Use v116's HLO as the compiled-program evidence for this exact rerun:

- Optimized train-step text size: 795,784 bytes.
- HLO content hash: `ff53351f8c3f828d212665c2505f4f865079a5f42232a66c05bf345a35044a8b`.
- HLO prefix object count: 1,289.
- Approximate instruction-line count: 3,848.
- Flagfile confirms both collective-matmul modes are `NONE`.

## Verdict

**Supported.** v117 confirms that the v116 `DISABLE_COLLECTIVE_MATMUL` pair is
a durable new frontier, not a one-off noisy run. Carry both flags forward:

```text
--xla_tpu_all_gather_collective_matmul_mode=none
--xla_tpu_reduce_scatter_collective_matmul_mode=none
```

The important secondary signal is the persistent HBM drop from 28.32 GiB to
22.21 GiB. That reopens batch-scaling probes that were previously blocked or
unattractive, especially per-chip batch 4 / global batch 32 with the now-smaller
collective-matmul-none compiled program.
