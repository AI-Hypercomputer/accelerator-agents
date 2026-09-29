---
title: "Qwen3 JAX v024 - splash TP2 tokamax remat fused s8k bs2"
type: experiment
hypothesis: "Seq8192 batch 2 may still fit because v023 batch 1 left 4.67 GiB free HBM, and it would recover target-sequence MFU through higher token count."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v6e8-qwen3-8b-jax-20260602-v015-splash-tp2-tokamax-remat-fused-rope-bs8
status: invalid
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax, remat, fused-rope, tp2, long-seq, batch-scaling, invalid, oom, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v024 - splash TP2 tokamax remat fused s8k bs2

## Hypothesis under test

**Hypothesis**: Seq8192 batch 2 may still fit because v023 batch 1 left 4.67 GiB
HBM free, and it would recover target-sequence MFU through higher token count.

**Mechanism**: Reuse the v015 image, successful XLA flag stack, TP2 sharding,
Splash attention, tokamax CE, remat, and fused RoPE. Keep `--seqlen=8192` and
increase per-chip batch from v023's 1 to 2, giving global batch 8.

**Predicted signal**: If it compiles, MFU should improve over v023's 26.6% at
the same target sequence length. If it OOMs, v023 is the current simple
batch-scaling limit at seq8192.

**Falsification criterion**: Compile/runtime OOM, or clean completion with no
MFU gain over v023.

## Setup

Planned GKE workload `alekseyv-qwen3-v024`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v015-splash-tp2-tokamax-remat-fused-rope-bs8`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v024-splash-tp2-tokamax-remat-fused-s8k-bs2`
- **Mesh**: `fsdp=4,tp=2`
- **Command shape**: `--batch_size=2 --seqlen=8192 --tp_parallelism=2 --use_splash=True`
- **Global batch**: 8
- **Profile window**: steps 12-14

## Results

Invalid. Workload `alekseyv-qwen3-v024` failed during train-step compilation on
both workers with `RESOURCE_EXHAUSTED`.

- **Exit**: `EXIT_CODE=1` at `Wed Jun 3 01:59:46 UTC 2026`
- **OOM**: used 33.99 GiB of 31.25 GiB HBM; exceeded capacity by 2.75 GiB
- **Memory accounting**: reserved 260.00 MiB, program 33.99 GiB,
  arguments 0B, HLO temp 33.98 GiB, fragmentation 117.36 MiB
- **Largest allocation**: 2.32 GiB tokamax backward dot
  `fusion.4291.remat5`, with labels shape `i32[65536]`
- **Other large allocations**: 593.50 MiB all-gather
  `all-gather.7921.remat2` plus repeated 192.00 MiB async collective fusion
  buffers of shape `bf16[2,8192,6144]`

## Profile

No runtime profile; the workload failed before executing training steps.

## HLO Dump

OOM artifacts were dumped under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v024-splash-tp2-tokamax-remat-fused-s8k-bs2/hlo/`

The prefix contains 41 objects totaling 186.41 MiB, including
`module_0092.jit_train_step.cl_854318611.oom_intermediate_module.txt` and
`module_0092.jit_train_step.cl_854318611.oom_top_memory_consuming_instructions`.

## Verdict

Refuted. The TP2 simple batch frontier at target sequence length is v023:
seq8192, per-chip batch 1, global batch 4, at 26.6% MFU. Moving to global
batch 8 reintroduces the compile-time HBM wall. Further target-sequence gains
need a memory mechanism or a different sharding/layout tradeoff, not plain
batch scaling under TP2.
