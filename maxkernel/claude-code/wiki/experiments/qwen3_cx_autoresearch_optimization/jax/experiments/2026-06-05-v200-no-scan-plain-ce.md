---
title: "Qwen3 JAX v200 - no-scan plain integer CE"
type: experiment
hypothesis: "On the confirmed no-scan frontier, replacing MaxText/T5X one-hot custom-VJP CE with the trainer's built-in integer-label log_softmax + gather CE may reduce one-hot graph pressure or select a better CE/logits lowering without changing loss semantics."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-plain-ce
status: invalid
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, ce, plain-ce, maxtext-ce, splash, tokamax-splash, collective-matmul, scoped-vmem, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v200 - no-scan plain integer CE

## Hypothesis under test

**Hypothesis**: v182/v183 proves the no-scan frontier fits at batch 4/global
32 even with the full-logit MaxText CE path. The optimized HLO still contains
large one-hot/equality CE structures. The prior v043/v044 integer-label custom
VJP branch is invalid because custom VJP over full logits promotes/materializes
a 13.91 GiB f32 lm-head dot temp, but the trainer's original plain CE is a
different lowering: `jax.nn.log_softmax(logits.astype(f32))` plus integer
`take_along_axis`, with no custom VJP and no explicit one-hot target.

**Mechanism**: Reuse the confirmed v182/v183 runtime stack and change only
`--use_maxtext_ce=True` to `--use_maxtext_ce=False`, leaving
`--use_tokamax_ce=False` so the trainer uses its built-in `_ce` path.

Keep the v169 no-MLP-intermediate-`_sac` image, no-scan, remat, tokamax Splash,
activation sharding, batch/sequence shape, collective-matmul modes disabled,
and scoped VMEM 100352 fixed.

**Predicted signal**: Support requires clean compile/loss and throughput above
the v182/v183 confirmed 58,288-58,330 tok/s average / 41.8% MFU band, or a
material HBM/profile improvement that opens a follow-up batch-5 attempt.

**Falsification criterion**: Compile/runtime OOM, invalid loss, or clean
completion at or below the v182/v183 frontier without a useful memory/profile
improvement.

## Setup

GKE workload `alekseyv-qwen3-v200-noscan-plaince`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v200-no-scan-plain-ce`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v183 stack with `--xla_tpu_scoped_vmem_limit_kib=100352`,
  selective resources, custom-call VMEM adjustment, scheduler rerun 3, and
  both collective-matmul modes set to `none`.
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=False --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14 if the run reaches training.

## Equivalence

No model-code edit is made. This is a CE lowering/runtime selection only:
both CE paths compute mean cross entropy over the same logits and integer
labels, with no z-loss.

## Results

Failed before step 0 on both pods with `EXIT_CODE=1`. XLA reported a permanent
TPU compile HBM OOM:

- Used 48.39 GiB of 31.25 GiB HBM, exceeding capacity by 17.14 GiB.
- Total HBM usage >= 48.64 GiB.
- Program HBM requirement 48.39 GiB.
- HLO temp requirement 48.37 GiB, with 47.12 GiB unpadded and 1.25 GiB
  fragmentation.

The top allocations identify the mechanism:

- 18.55 GiB `f32[4,8192,151936]` from
  `jit(train_step)/forward_backward/jvp()/dot_general`.
- 18.55 GiB `f32[32768,151936]` from
  `jit(train_step)/forward_backward/jvp(jit(log_softmax))/sub`.

Plain integer CE removes the explicit MaxText one-hot path, but it causes the
lm-head dot and log-softmax subtract to materialize full f32 vocabulary tensors.
That is far worse than the carried MaxText CE frontier, whose v182/v183 program
fits at 27.66 GiB.

## Profile

No runtime profile: compile failed before training steps.

## HLO Dump

Fresh failed train-step HLO was emitted:

- Optimized HLO:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v200-no-scan-plain-ce/hlo/module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- OOM intermediate:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v200-no-scan-plain-ce/hlo/module_0109.jit_train_step.cl_854318611.oom_intermediate_module.txt`
- OOM top-memory report:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v200-no-scan-plain-ce/hlo/module_0109.jit_train_step.cl_854318611.oom_top_memory_consuming_instructions`
- Optimized HLO size: 19,057,269 bytes
- SHA256: `a5fdbd91bf3adfd6f99006e784753b2ffd6f15c79fa544c3d00c2d0ce40a8869`
- Text counts: all-gather 4,655, all-reduce 960, collective-permute 3,
  convolution 2,103, custom-call 1,472, copy 9,451, fusion 23,313,
  `log_softmax` 41, `_one_hot` 0.

## Verdict

Invalid / refuted. Do not replace MaxText CE with the trainer's plain integer
CE on the current no-scan frontier. MaxText CE remains required for batch
4/seq8192 because it avoids the full f32 vocab materialization selected by
plain `log_softmax + gather`.
