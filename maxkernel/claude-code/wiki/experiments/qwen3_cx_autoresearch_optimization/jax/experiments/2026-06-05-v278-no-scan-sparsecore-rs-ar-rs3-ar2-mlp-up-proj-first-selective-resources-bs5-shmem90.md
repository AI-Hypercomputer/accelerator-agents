---
title: "Qwen3 JAX v278 - Confirmed RS3 selective-resources frontier batch 5 with shared-memory limit 90"
type: experiment
hypothesis: "The shmem90 mode on the confirmed RS3+AR2 selective-resources family frees enough HBM to make the faster MaxText-CE batch-5/global-40 shape viable; if it fits, higher density may offset the same-shape shmem90 slowdown."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v258-mlp-up-first-runtime-rs3-ar2-selective-resources-bs5-shmem90
status: invalid/refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, mlp, swiglu, sparsecore, collectives, scheduler, selective-resources, latency, rs3, ar2, shared-memory, batch-scaling, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v278 - Confirmed RS3 selective-resources frontier batch 5 with shared-memory limit 90

## Hypothesis under test

**Hypothesis**: Direct MaxText-CE batch-5 attempts previously OOMed, while
Tokamax-CE batch 5 fit but was too slow. v277 showed that, on the newer v273
RS3+AR2 selective-resources HLO family, `shmem90` preserves the optimized
train-step text while reducing peak HBM from **30.93 GiB** to **29.03 GiB**.
That extra memory may let the faster MaxText-CE path fit at per-chip batch 5
/ global batch 40. If it fits, higher token density may recover enough
throughput/MFU to beat the v273 batch-4 frontier despite shmem90 being slower
at the same batch size.

**Mechanism**: Reuse v277's runtime stack and change only trainer batch size
from `4` to `5`. Keep scoped VMEM `100352`, scheduler shared-memory limit
`90`, RS latency multiplier `3`, AR latency multiplier `2`, selective
resources forced on, SparseCore reduce-scatter/all-reduce offload, SparseCore
collective aggregator, Tokamax Splash, MaxText CE, no-scan, remat, MLP
up-proj-first image, and activation sharding.

**Predicted signal**: Support requires clean finite loss and a valid XProf.
The primary success threshold is throughput/MFU above v273's **60,346 tok/s /
43.3% MFU**; a near-tie with materially better HBM headroom would require a
rerun before carry.

**Falsification criterion**: Compile/runtime OOM, invalid loss, missing
Tokamax markers, or valid runtime below the v273 batch-4 frontier.

## Setup

GKE workload: `alekseyv-q3-v278-bs5-shmem90`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v258-mlp-up-first`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v278-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-bs5-shmem90`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=5 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 40
- **Profile window**: steps 12-14

## Equivalence

Shape/density experiment only. Model code, params, math, fake-data setup,
image, mesh, scoped VMEM, scheduler shared-memory limit, latency multipliers,
selective resources, SparseCore offload stack, and all trainer flags match
v277 except per-chip batch is raised from `4` to `5`.

## Results

Invalid: both pods failed during XLA compilation before any training step or
profile window. The failure is a compile-time HBM OOM in `jit_train_step`.

- Worker pods:
  - `alekseyv-q3-v278-bs5-shmem90-slice-job-0-0-78q4l`: `Error`,
    `EXIT_CODE=1`
  - `alekseyv-q3-v278-bs5-shmem90-slice-job-0-1-49lkh`: `Error`,
    `EXIT_CODE=1`
- Error: `RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of
  memory in memory space hbm. Used 33.10G of 31.25G hbm. Exceeded hbm capacity
  by 1.86G.`
- Total HBM usage reported by XLA: at least **33.36 GiB**
  - reserved: **260.00 MiB**
  - program: **27.38 GiB**
  - arguments: **5.72 GiB**
- Program HBM requirement: **27.38 GiB**
  - reserved: **65.45 MiB**
  - global: **19.98 MiB**
  - scoped: **6.20 MiB**
  - HLO temp: **27.29 GiB**
- Output size: **5.72 GiB**, shared with arguments.

The largest allocation is the full-vocabulary logits temp:
`bf16[5,8192,151936]`, **11.59 GiB**, labelled `fusion.4976.remat` from
`jit(train_step)/forward_backward/jvp()/dot_general`. This is the same
structural class of memory pressure as the previous direct batch-5
MaxText-CE failures. The `shmem90` memory savings from v277 are not enough to
make global batch 40 viable.

## Profile

No XProf was produced; the workload failed during compile.

## HLO Dump

Train-step HLO artifacts were emitted:

- GCS prefix:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v278-no-scan-sparsecore-rs-ar-rs3-ar2-mlp-up-proj-first-selective-resources-bs5-shmem90/hlo/xla-dump/`
- Module: `module_0267.jit_train_step.cl_854318611`
- Optimized HLO SHA256:
  `ab88cab233d7fabd595e00652e22e66f18f26c37a849847de761e41db7e85d9a`
- Size: **17,671,203 bytes / 106,145 lines**
- Key text counts: `all-gather` 3748, `all-reduce` 158, `reduce-scatter` 1766,
  `async-start` 419, `async-done` 419, `convolution` 2142, `custom-call` 1926,
  `fusion` 17563, `copy` 9431, `tokamax` 2, `jvp_jit_silu__` 576.
- OOM reports:
  - `module_0267.jit_train_step.cl_854318611.after_optimizations-memory-usage-report.txt`
  - `module_0267.jit_train_step.cl_854318611.oom_top_memory_consuming_instructions`

## Verdict

Refuted/invalid. Do not carry batch 5 or `shmem90`.

This closes the direct batch-scaling route for the current MaxText-CE runtime
family. The next useful hypothesis should be structural: avoid materializing
the full `bf16[batch, seq, vocab]` logits tensor by implementing or porting a
streaming/fused lm-head+cross-entropy path, then retest batch 5 only after the
11.59 GiB logits temp is eliminated or tiled.
