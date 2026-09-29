---
title: "Qwen3 JAX v179 - no MLP intermediate sac without remat"
type: experiment
hypothesis: "If v169's no-MLP-intermediate schedule freed enough memory headroom, disabling decoder-layer remat may trade higher HBM for less recompute and improve wall time."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-no-remat
status: invalid
tags: [qwen3-cc, jax, flax-nnx, remat, activation-sharding, mlp, sharding-constraint, splash, tokamax-splash, collective-matmul, scoped-vmem, maxtext-ce, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v179 - no MLP intermediate sac without remat

## Hypothesis under test

**Hypothesis**: The current v169/v170 frontier still uses per-layer
`jax.checkpoint` with `nothing_saveable`. v169 removed the large MLP intermediate
activation constraint and selected a slightly better schedule. If that schedule
also freed enough HBM, disabling remat may avoid recomputation and improve
throughput.

**Mechanism**: Reuse `v169-no-mlp-sac` and change only the runtime flag from
`--use_remat=True` to `--use_remat=False`. Keep scan, tokamax Splash,
MaxText CE, activation sharding, batch/sequence shape, sharding plan, and XLA
flags fixed.

**Predicted signal**: Support requires clean compile/loss and throughput above
the confirmed v169/v170 frontier band (57,935-57,947 tok/s average, 58,051 best
observed). A compile-time HBM OOM is informative and refutes the available
memory-headroom premise.

**Falsification criterion**: Compile/runtime OOM, invalid loss, or clean
completion at/below the confirmed frontier.

## Setup

GKE workload `alekseyv-qwen3-v179-noremat-mlpsac`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v179-no-mlp-sac-no-remat`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v170 stack with `--xla_tpu_scoped_vmem_limit_kib=100352`.
- **Command shape**: `--use_scan=True --use_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

Not required: this is a runtime rematerialization flag change only. It does not
change model math, parameter sharding, CE implementation, or weights.

## Results

Compile failed before the first training step on both workers with
`RESOURCE_EXHAUSTED`.

- Exit code: `1`
- Error: XLA TPU compile permanent error, HBM OOM.
- Reported usage: 340.87 GiB of 31.25 GiB HBM, exceeded by 309.62 GiB.
- Total memory report: 346.62 GiB total bytes used; 340.87 GiB preallocated temp.

The largest allocations are repeated scan-stacked activation tensors, led by
six 27.00 GiB `bf16[36,4,8192,12288]` temps and multiple 18.00 GiB
`f32[36,4,8192,4096]` / `f32[36,4,8192,32,128]` temps. Disabling remat causes
the 36-layer scan body to retain layer activations rather than recompute them.

## Profile

No runtime profile. The program failed during train-step compilation.

## HLO Dump

OOM HLO artifacts were emitted under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v179-no-mlp-sac-no-remat/hlo/`

Fresh optimized train-step HLO:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v179-no-mlp-sac-no-remat/hlo/module_0109.jit_train_step.cl_854318611.after_optimizations.txt`

- Size: 701,833 bytes
- SHA256: `f4dd4b1813a5c635e443c93d58e1ee927ceaa819615757aa52e253be51d73950`
- Text counts: all-gather 59, all-reduce 36, collective-permute 3, convolution 61, custom-call 41, copy 272, fusion 436.
- Memory report: `module_0109.jit_train_step.cl_854318611.after_optimizations-memory-usage-report.txt`
- OOM top instructions: `module_0109.jit_train_step.cl_854318611.oom_top_memory_consuming_instructions`

## Verdict

Invalid / refuted by compile-time HBM OOM. The v169 no-MLP-intermediate schedule
does not create enough headroom to disable remat. Keep `--use_remat=True`;
future no-remat attempts require a fundamentally different activation-memory
mechanism, not another local flag tweak.
