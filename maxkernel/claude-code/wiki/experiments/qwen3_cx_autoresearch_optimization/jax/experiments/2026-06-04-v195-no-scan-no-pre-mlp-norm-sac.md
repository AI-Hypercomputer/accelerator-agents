---
title: "Qwen3 JAX v195 - no scan, no pre-MLP norm-output sharding constraint"
type: experiment
hypothesis: "The no-scan unrolled graph may benefit from removing only the pre-MLP norm-output activation constraint, even though the same boundary was slower in the scan family."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v174-no-pre-mlp-norm-sac-runtime-no-scan
status: refuted
tags: [qwen3-cc, jax, flax-nnx, scan, no-scan, remat, activation-sharding, mlp, rmsnorm, sharding-constraint, splash, tokamax-splash, collective-matmul, maxtext-ce, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v195 - no scan, no pre-MLP norm-output sharding constraint

## Hypothesis under test

**Hypothesis**: v192-v194 show that removing all activation constraints,
attention-output `_sac`, or pre-attention norm-output `_sac` under no-scan is
valid but below frontier. The remaining existing single-boundary image, v174,
removes only the `_sac` around `post_attention_layernorm(hidden_states)` before
the MLP input. In the no-scan HLO, this may reduce MLP-side relayout/copy
pressure while keeping the residual and attention boundaries constrained.

**Mechanism**: Reuse the numerically-equivalent v174 image and change only the
runtime shape from scan to no-scan:
`--use_scan=False`, keeping `--shard_acts=True`.

Compared with v183, the only model-code difference is:

```python
x = self.mlp(self.post_attention_layernorm(hidden_states))
```

instead of constraining the pre-MLP norm output before `self.mlp`.

**Predicted signal**: Support requires clean loss and steady throughput above
the confirmed no-scan band: v182/v183 at 58,288-58,330 tok/s confirmed average
and 58,443 best observed. A fresh HLO/hash with lower copy/collective pressure
would justify this boundary removal only in the no-scan schedule.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the no-scan frontier without a profile/memory win.

## Setup

GKE workload `alekseyv-qwen3-v195-noscan-nomlpnormsac`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v174-no-pre-mlp-norm-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v195-no-scan-no-pre-mlp-norm-sac`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v183 stack.
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

Containerized CPU equivalence was rerun for the v174 image before classifying
v195. It passed: logits max delta `2.682e-07`, loss delta `0`, all 25 gradients
OK, max gradient delta `7.823e-08`. This v195 run changes only
`--use_scan=False` at runtime.

## Results

The run completed cleanly with normal loss and `EXIT_CODE=0`.

- Worker0: 56,654 tok/s average / 7,082 tok/s/chip / 40.6% MFU.
- Worker1: 56,628 tok/s average / 7,079 tok/s/chip / 40.6% MFU.
- Best observed post-warmup step: 57,007 tok/s.
- Final loss: 12.0459.

This is a clear regression versus the confirmed no-scan frontier:
v182/v183 reached 58,288-58,330 tok/s confirmed average, with 58,443 tok/s best
observed.

## Profile

Profile files:
`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v195-no-scan-no-pre-mlp-norm-sac/plugins/profile/2026_06_04_23_22_20/`

Device-side `jit_train_step(14016405483169477827)` spans:

- `gke-tpu-32cb1c36-gs53`: 4.663221 s average, 56,215 tok/s, 12 spans.
- `gke-tpu-32cb1c36-jj8f`: 4.650469 s average, 56,369 tok/s, 12 spans.

The profile confirms the stdout regression and places this variant well below
the v183 profile-derived 58,288-58,291 tok/s span class.

## HLO Dump

Fresh optimized train-step HLO:

- File: `module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 19,043,256 bytes
- SHA256:
  `aa8c66042d4f0c579643e99c7a4e99a9b6291d73bf6f50ff9e095930e493a773`
- Memory report: 29,695,264,712 bytes / 27.66 GiB.
- Text counts: all-gather 7,039, all-reduce 1,148, collective-permute 6,
  convolution 3,071, custom-call 2,806, copy 21,271, fusion 31,686.

This is the same high-count boundary-relaxation family seen in v189 and
v191-v194, not the faster confirmed frontier family.

## Verdict

Refuted. Removing only the pre-MLP norm-output `_sac` under no-scan is valid
and numerically equivalent, but it loses roughly 1.8k tok/s versus the confirmed
frontier and lands in the repeatedly slower high-count HLO family. Do not carry
this boundary removal.
