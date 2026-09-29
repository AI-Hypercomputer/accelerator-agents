---
title: "Qwen3 JAX v193 - no scan, no attention output sharding constraint"
type: experiment
hypothesis: "The no-scan unrolled graph may benefit from removing only the attention-output activation constraint, even though removing all activation constraints or removing this boundary in the scan family was slower."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v172-no-attn-output-sac-runtime-no-scan
status: refuted
tags: [qwen3-cc, jax, flax-nnx, scan, no-scan, remat, activation-sharding, attention, sharding-constraint, splash, tokamax-splash, collective-matmul, maxtext-ce, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v193 - no scan, no attention output sharding constraint

## Hypothesis under test

**Hypothesis**: v192 showed that removing all activation constraints under
no-scan is valid and near the frontier, but selects a slower high-count HLO
family. The existing v172 image removes only the attention-output `_sac`, which
was slower in the scan family. Under no-scan, this single-boundary relaxation
may reduce residual/attention relayout pressure without losing the useful
residual-stream and MLP-side constraints.

**Mechanism**: Reuse the numerically-equivalent v172 image and change only the
runtime shape from scan to no-scan:
`--use_scan=False`, keeping `--shard_acts=True`.

Compared with v183, the only model-code difference is:

```python
x = self.self_attn(_sac(self.input_layernorm(hidden_states)), position_embeddings, attention_mask)
```

instead of wrapping the attention output in `_sac(...)`.

**Predicted signal**: Support requires clean loss and steady throughput above
the confirmed no-scan band: v182/v183 at 58,288-58,330 tok/s confirmed average
and 58,443 best observed. A fresh HLO/hash with lower copy/collective pressure
would justify this boundary removal only in the no-scan schedule.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion at or below the no-scan frontier without a profile/memory win.

## Setup

GKE workload `alekseyv-qwen3-v193-noscan-noattnsac`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v172-no-attn-output-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v193-no-scan-no-attn-output-sac`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v183 stack.
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

The v172 image already passed CPU equivalence: logits max delta `2.682e-07`,
loss delta `0`, all 25 gradients OK, max gradient delta `7.823e-08`. This v193
run changes only `--use_scan=False` at runtime.

## Results

Refuted/near-tie. Workload `alekseyv-qwen3-v193-noscan-noattnsac` completed
cleanly with valid loss ending at 12.0459 and `EXIT_CODE=0`, but stayed below
the confirmed no-scan frontier:

- Worker0 stdout: 58,087 tok/s average / 7,261 tok/s/chip / 41.6% MFU.
- Worker1 stdout: 58,075 tok/s average / 7,259 tok/s/chip / 41.6% MFU.
- Best observed worker0 steady step: 58,236 tok/s at step 2.
- Profile-derived host averages: `gke-tpu-964065d9-9vh8` 4.525476 s / 57,926
  tok/s, `gke-tpu-964065d9-c180` 4.517539 s / 58,028 tok/s.
- Final loss: 12.0459.

This matches the v192 near-tie band but remains below v182/v183's 58,288-58,330
confirmed average and v182's 58,443 best observed step.

## Profile

Profile files:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v193-no-scan-no-attn-output-sac/plugins/profile/2026_06_04_23_06_48/`

The profile confirms the near-tie/regression. Device-side
`jit_train_step(403931025898375493)` spans are about 4.518-4.525 s across the
two hosts, slower than the v183 no-scan 4.497 s class.

## HLO Dump

Fresh train-step HLO:

- Optimized HLO:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v193-no-scan-no-attn-output-sac/hlo/module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 19,043,256 bytes.
- SHA256: `7d7516976eb9198a5898be22fb06e0364d73a6a35fb16e1d1fc2fc8dec2410fe`.
- Memory report: 27.66 GiB total bytes used.
- Text counts: all-gather 7,039, all-reduce 1,148, collective-permute 6,
  convolution 3,071, custom-call 2,806, copy 21,271, fusion 31,686.

Despite the v172 code change, the no-scan profile reports the same train-step
program id as v182/v183 (`403931025898375493`) while the dumped HLO text/hash
falls into the high-count changed family. In wall time it behaves like v192,
not like the confirmed frontier.

## Verdict

Refuted/near-tie. Do not carry the attention-output `_sac` removal in the
no-scan frontier. This individual boundary is no better than removing all
activation constraints under no-scan and remains below the confirmed v182/v183
schedule.
