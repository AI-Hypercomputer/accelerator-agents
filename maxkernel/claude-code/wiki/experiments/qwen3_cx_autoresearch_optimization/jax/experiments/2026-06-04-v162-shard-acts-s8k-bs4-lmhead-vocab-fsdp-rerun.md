---
title: "Qwen3 JAX v162 - shard acts s8k bs4 lm head vocab fsdp rerun"
type: experiment
hypothesis: "If v161's lm_head vocab-axis FSDP gain is real, an exact rerun should preserve the distinct HLO family and remain above the confirmed v153/v154 frontier band."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v161-lmhead-vocab-fsdp
status: refuted
tags: [qwen3-cc, jax, flax-nnx, lm-head, sharding, ce, maxtext-ce, splash, tokamax-splash, collective-matmul, scoped-vmem, remat, activation-sharding, batch-scaling, long-seq, v6e-8, rerun]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v162 - shard acts s8k bs4 lm head vocab fsdp rerun

## Hypothesis under test

**Hypothesis**: v161 changed only `lm_head.weight` sharding to vocab-axis FSDP,
emitted a distinct train-step HLO, reduced compiled memory to 26.76 GiB, and
slightly improved best-worker throughput. If this is real rather than run noise,
an exact rerun should preserve the same HLO family and remain above the
confirmed v153/v154 band.

**Mechanism**: Exact rerun of v161:

```python
"lm_head.weight": ("fsdp", None)
```

All runtime flags, batch/sequence shape, image, and profile window are held
fixed.

**Predicted signal**: Support requires clean loss, HLO matching v161 or an
equivalent lm-head-vocab-FSDP schedule, compiled memory near 26.76 GiB, and a
best-worker average above the confirmed v153/v154 band (57,754-57,757 tok/s) or
best observed step near v161's 57,908.

**Falsification criterion**: Compile/runtime failure, invalid loss, HLO falling
back to the v153 family, or average throughput falling back into/below the
v153/v154 noise band without a memory/profile win.

## Setup

GKE workload `alekseyv-qwen3-v162-bs4-lmheadvocabr`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v161-lmhead-vocab-fsdp`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v162-shard-acts-s8k-bs4-lmhead-vocab-fsdp-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: confirmed v154 stack with `--xla_tpu_scoped_vmem_limit_kib=100352`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers with normal loss and `EXIT_CODE=0`, but did
not reproduce v161's throughput.

- Worker 0: 56,494 tok/s average, 7,062 tok/s/chip, 40.5% MFU.
- Worker 1: 56,482 tok/s average, 7,060 tok/s/chip, 40.5% MFU.
- Best observed post-warmup step: 56,914 tok/s at step 2 on both workers.
- Final loss: 12.0458 on both workers, matching the valid baseline loss path.

This is below the confirmed v153/v154 repeated frontier band
(57,754-57,757 tok/s average, 57,888 best observed) and below v161's
provisional 57,765 tok/s average / 57,908 best observed.

## Profile

Profile captured successfully:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v162-shard-acts-s8k-bs4-lmhead-vocab-fsdp-rerun/plugins/profile/2026_06_04_18_42_10/`

Artifacts:

- `gke-tpu-32cb1c36-gs53.xplane.pb`
- `gke-tpu-32cb1c36-jj8f.xplane.pb`
- matching `trace.json.gz` files for both hosts

## HLO Dump

No new HLO dump objects were emitted under the v162 run prefix. Since the image,
shape, runtime flags, and persistent compilation cache are identical to v161,
this is consistent with a persistent-cache replay of v161's train-step program.
The v161 optimized train-step HLO remains:

- `module_0105.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 785,947 bytes
- SHA256: `3648b17f7b1fdb754e9b9330400997ed2e06c3f25543cb549bb90559a8b7a348`

## Verdict

Refuted. The lm-head vocab-axis FSDP sharding is valid and appears to reduce
compiled memory in the v161 compile, but the exact rerun fell to 56,494 tok/s
best-worker average / 40.5% MFU, well below the confirmed v153/v154 frontier.
Do not carry this sharding change. Treat v161 as run noise unless a future
profile-level explanation shows a controllable runtime difference.
