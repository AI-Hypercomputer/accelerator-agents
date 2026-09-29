---
title: "Qwen3 JAX v194 - no scan, no pre-attention norm-output sharding constraint"
type: experiment
hypothesis: "The no-scan unrolled graph may benefit from removing only the pre-attention norm-output activation constraint, even though the same single-boundary change was slower in the scan family."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v173-no-pre-attn-norm-sac-runtime-no-scan
status: refuted
tags: [qwen3-cc, jax, flax-nnx, scan, no-scan, remat, activation-sharding, attention, rmsnorm, sharding-constraint, splash, tokamax-splash, collective-matmul, maxtext-ce, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v194 - no scan, no pre-attention norm-output sharding constraint

## Hypothesis under test

**Hypothesis**: v192 showed that removing all activation constraints under
no-scan is valid but below the frontier. v193 showed the attention-output
boundary alone also remains below frontier. The next existing single-boundary
image, v173, removes only the `_sac` around `input_layernorm(hidden_states)`
before attention. In the no-scan HLO, relaxing this QKV/attention input layout
may reduce relayout/copy pressure while preserving the residual-stream
attention-output boundary.

**Mechanism**: Reuse the numerically-equivalent v173 image and change only the
runtime shape from scan to no-scan:
`--use_scan=False`, keeping `--shard_acts=True`.

Compared with v183, the only model-code difference is:

```python
x = _sac(self.self_attn(self.input_layernorm(hidden_states), position_embeddings, attention_mask))
```

instead of constraining the pre-attention norm output before `self_attn`.

**Predicted signal**: Support requires clean loss and steady throughput above
the confirmed no-scan band: v182/v183 at 58,288-58,330 tok/s confirmed average
and 58,443 best observed. A fresh HLO/hash with lower copy/collective pressure
would justify this boundary removal only in the no-scan schedule.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the no-scan frontier without a profile/memory win.

## Setup

GKE workload `alekseyv-qwen3-v194-noscan-noinnormsac`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v173-no-pre-attn-norm-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v194-no-scan-no-pre-attn-norm-sac`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v183 stack.
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

The v173 image already passed CPU equivalence: logits max delta `2.682e-07`,
loss delta `0`, all 25 gradients OK, max gradient delta `7.823e-08`. This v194
run changes only `--use_scan=False` at runtime.

## Results

Refuted/near-tie. Workload `alekseyv-qwen3-v194-noscan-noinnormsac` completed
cleanly with valid loss ending at 12.0459 and `EXIT_CODE=0`, but stayed below
the confirmed no-scan frontier:

- Worker0 stdout: 58,086 tok/s average / 7,261 tok/s/chip / 41.6% MFU.
- Worker1 stdout: 58,064 tok/s average / 7,258 tok/s/chip / 41.6% MFU.
- Best observed worker0 steady step: 58,244 tok/s at step 2.
- Profile-derived host averages: `gke-tpu-964065d9-9vh8` 4.528267 s / 57,891
  tok/s, `gke-tpu-964065d9-c180` 4.517641 s / 58,027 tok/s.
- Final loss: 12.0459.

This matches the v192/v193 near-tie band but remains below v182/v183's
58,288-58,330 confirmed average and v182's 58,443 best observed step.

## Profile

Profile files:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v194-no-scan-no-pre-attn-norm-sac/plugins/profile/2026_06_04_23_14_45/`

The profile confirms the near-tie/regression. Device-side
`jit_train_step(403931025898375493)` spans are about 4.518-4.528 s across the
two hosts, slower than the v183 no-scan 4.497 s class.

## HLO Dump

Fresh train-step HLO:

- Optimized HLO:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v194-no-scan-no-pre-attn-norm-sac/hlo/module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 19,043,220 bytes.
- SHA256: `ad96ef695c73656ad5560cde57b941bfc1e0db58797e71143dc7cd4591360aba`.
- Memory report: 27.66 GiB total bytes used.
- Text counts: all-gather 7,039, all-reduce 1,148, collective-permute 6,
  convolution 3,071, custom-call 2,806, copy 21,271, fusion 31,686.

The no-scan pre-attention norm boundary relaxation lands in the same high-count
HLO family as v192/v193 and remains slower than the confirmed frontier.

## Verdict

Refuted/near-tie. Do not carry the pre-attention norm-output `_sac` removal in
the no-scan frontier. The v192-v194 boundary relaxations all converge to the
same slower high-count no-scan family.
