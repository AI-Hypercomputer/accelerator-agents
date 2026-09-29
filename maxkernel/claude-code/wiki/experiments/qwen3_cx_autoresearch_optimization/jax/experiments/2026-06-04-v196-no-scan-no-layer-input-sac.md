---
title: "Qwen3 JAX v196 - no scan, no decoder layer input sharding constraint"
type: experiment
hypothesis: "The no-scan unrolled graph may benefit from removing only the decoder layer input residual-stream `_sac`, even though the same boundary was slower in the scan family."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v175-no-layer-input-sac-runtime-no-scan
status: refuted
tags: [qwen3-cc, jax, flax-nnx, scan, no-scan, remat, activation-sharding, residual, sharding-constraint, splash, tokamax-splash, collective-matmul, maxtext-ce, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v196 - no scan, no decoder layer input sharding constraint

## Hypothesis under test

**Hypothesis**: v192-v195 show that removing all activation constraints or
individual local attention/MLP norm-output boundaries under no-scan is valid
but below frontier. The remaining existing single-boundary image, v175, removes
only the decoder layer input `_sac(hidden_states)` residual-stream boundary.
Under no-scan, this could reduce cross-layer residual relayout pressure without
relaxing the local attention/MLP boundaries.

**Mechanism**: Reuse the numerically-equivalent v175 image and change only the
runtime shape from scan to no-scan: `--use_scan=False`, keeping
`--shard_acts=True`.

Compared with v183, the model-code difference is removing only:

```python
hidden_states = _sac(hidden_states)
```

at decoder layer entry, while keeping the v169 MLP-intermediate `_sac` removal.

**Predicted signal**: Support requires clean loss and steady throughput above
the confirmed no-scan band: v182/v183 at 58,288-58,330 tok/s confirmed average
and 58,443 best observed.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the no-scan frontier without a profile/memory win.

## Setup

GKE workload `alekseyv-qwen3-v196-noscan-nolayersac`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v175-no-layer-input-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v196-no-scan-no-layer-input-sac`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v183 stack.
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

Containerized CPU equivalence was rerun for the v175 image before launch. It
passed: logits max delta `2.682e-07`, loss delta `0`, all 25 gradients OK, max
gradient delta `7.823e-08`.

## Results

The run completed cleanly with normal loss and `EXIT_CODE=0`.

- Worker0: 58,099 tok/s average / 7,262 tok/s/chip / 41.7% MFU.
- Worker1: 58,083 tok/s average / 7,260 tok/s/chip / 41.6% MFU.
- Best observed post-warmup step: 58,251 tok/s.
- Final loss: 12.0459.

This is a near-tie but still below the confirmed no-scan frontier:
v182/v183 reached 58,288-58,330 tok/s confirmed average, with 58,443 tok/s best
observed.

## Profile

Profile files:
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v196-no-scan-no-layer-input-sac/plugins/profile/2026_06_04_23_33_01/`

Device-side `jit_train_step(403931025898375493)` spans:

- `gke-tpu-964065d9-9vh8`: 4.524858 s average, 57,934 tok/s, 12 spans.
- `gke-tpu-964065d9-c180`: 4.517163 s average, 58,033 tok/s, 12 spans.

The profile confirms this belongs to the slower near-tie boundary-relaxation
class, not the v183 4.497 s / 58.29k tok/s class.

## HLO Dump

Fresh optimized train-step HLO:

- File: `module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 19,043,220 bytes
- SHA256:
  `f94aebd8de00f0a1311ee3960d32851f1dfc262c9f8f4b4a056cfc3cc02e0dfc`
- Memory report: 29,695,264,712 bytes / 27.66 GiB.
- Text counts: all-gather 7,039, all-reduce 1,148, collective-permute 6,
  convolution 3,071, custom-call 2,806, copy 21,271, fusion 31,686.

This is the same high-count HLO family observed in the no-scan
boundary-relaxation variants, and it remains below the confirmed frontier.

## Verdict

Refuted/near-tie. Removing the decoder layer input `_sac` under no-scan is
valid and numerically equivalent, but it does not beat the confirmed no-scan
frontier. The single-boundary activation-constraint sweep under no-scan is now
closed: keep the original layer/input and norm/attention boundaries, and carry
only the v169 MLP-intermediate `_sac` removal.
