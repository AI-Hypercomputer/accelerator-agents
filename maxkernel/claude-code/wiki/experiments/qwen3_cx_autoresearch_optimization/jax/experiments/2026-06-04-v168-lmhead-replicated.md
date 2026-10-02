---
title: "Qwen3 JAX v168 - fully replicated lm_head"
type: experiment
hypothesis: "If the final lm-head hidden-axis FSDP sharding creates a costly collective/logits boundary, fully replicating only lm_head.weight may improve the final projection and MaxText CE schedule despite higher memory use."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v168-lmhead-repl
status: refuted
tags: [qwen3-cc, jax, flax-nnx, lm-head, sharding, ce, maxtext-ce, splash, tokamax-splash, collective-matmul, scoped-vmem, remat, activation-sharding, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v168 - fully replicated lm_head

## Hypothesis under test

**Hypothesis**: The confirmed v153/v154 frontier still leaves the untied
`lm_head.weight` sharded on the hidden dimension across FSDP:
`("tp", "fsdp")`, which collapses to `(None, "fsdp")` at `tp=1`. The input
activation to `lm_head` is batch-sharded on the same FSDP axis. If this mismatch
forces a costly final projection / full-logits / CE collective boundary, fully
replicating only `lm_head.weight` may select a cleaner schedule.

**Mechanism**: Derive a thin image from `v041-shard-acts` and change only the
lm-head parameter sharding plan entry:

```python
"lm_head.weight": ()
```

All decoder, embedding, CE, attention, optimizer, batch/sequence shape, and XLA
runtime flags remain fixed.

**Predicted signal**: Support requires clean loss, no OOM, a distinct HLO with
reduced final lm-head/CE collective pressure or improved profile attribution,
and throughput above the confirmed v153/v154 frontier band (57,754-57,757 tok/s
average, 57,888 best observed).

**Falsification criterion**: Compile/runtime OOM, invalid loss, HLO no-op, or a
clean run at/below the confirmed frontier without a clear profile win.

## Setup

GKE workload `alekseyv-qwen3-v168-lmrepb`.

Note: `alekseyv-qwen3-v168-lmrepl` was an inert first JobSet created from a
completed template that retained queue-owned metadata and never reached a real
run. It was deleted and replaced with this clean retry before logs/compile.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v168-lmhead-repl`
- **Image digest**: `sha256:fb51d0b9122d675575d1d63db1fc00fb80cb3ad267c237ce3dfbda9a5ac91506`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v168-lmhead-replicated`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: confirmed v154 stack with `--xla_tpu_scoped_vmem_limit_kib=100352`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers with normal loss and `EXIT_CODE=0`.

- Worker 0: 56,464 tok/s average, 7,058 tok/s/chip, 40.5% MFU.
- Worker 1: 56,445 tok/s average, 7,056 tok/s/chip, 40.5% MFU.
- Best observed post-warmup step: 56,875 tok/s.
- Final loss: 12.0461 on both workers.

This is a large regression from the confirmed v153/v154 frontier
(57,754-57,757 tok/s average, 57,888 best observed), despite clean loss.

## Profile

Profile captured successfully:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v168-lmhead-replicated/plugins/profile/2026_06_04_19_28_50/`

Artifacts:

- `gke-tpu-32cb1c36-gs53.xplane.pb`
- `gke-tpu-32cb1c36-jj8f.xplane.pb`
- matching `trace.json.gz` files for both hosts

## HLO Dump

Fresh optimized train-step HLO was emitted:

- `module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 780,820 bytes
- SHA256: `49953ae38096108b72b65ed59d492d32348ebada51efa9070bd396e100b67c68`

The change fired structurally. Quick textual opcode counts versus v153:

| Opcode text | v153 | v168 |
|---|---:|---:|
| `all-gather` | 211 | 136 |
| `all-reduce` | 45 | 36 |
| `collective-permute` | 6 | 3 |
| `convolution` | 30 | 66 |
| `custom-call` | 36 | 41 |
| `copy` | 14 | 410 |
| `fusion` | 225 | 752 |

The reduced collective text count did not translate into wall time. The
replicated lm-head schedule greatly increases copy/fusion complexity and is
slower end to end.

## Verdict

Refuted. Fully replicating `lm_head.weight` preserves math and produces a
distinct HLO, but it regresses throughput by roughly 1.3k tok/s from the
confirmed frontier:

- v168: 56,464 tok/s best-worker average, 40.5% MFU.
- v153/v154 frontier: 57,754-57,757 tok/s average, 41.4% MFU.

Do not carry full `lm_head.weight` replication. The CE/lm-head sharding branch
has now refuted hidden-axis FSDP replacement with vocab-axis FSDP, full logits
layout constraints, final-hidden constraints, and full lm-head replication.
