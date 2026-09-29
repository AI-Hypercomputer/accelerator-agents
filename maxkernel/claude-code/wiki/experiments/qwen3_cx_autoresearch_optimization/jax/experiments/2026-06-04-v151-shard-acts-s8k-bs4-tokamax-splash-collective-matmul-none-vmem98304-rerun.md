---
title: "Qwen3 JAX v151 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 98304 rerun"
type: experiment
hypothesis: "If v150's scoped-VMEM 98304 gain is real, an exact rerun should remain above the confirmed 94208 frontier."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, scoped-vmem, latency-hiding, scheduler, selective-resources, async-collective-fusion, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v151 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 98304 rerun

## Hypothesis under test

**Hypothesis**: v150's scoped-VMEM 98304 point is a genuine improvement over
the confirmed v149 94208 frontier, not runtime noise.

**Mechanism**: Exact rerun of v150. Keep the v150 command, image, XLA flag
stack, mesh, batch, and profile window fixed; change only workload/run naming.

**Predicted signal**: Support requires clean loss and best-worker average above
the confirmed 94208 frontier, ideally near v150's 57,578 tok/s. If it falls back
to the v149 band or lower, keep 94208 as the carried frontier and treat 98304 as
provisional/noisy.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed
command shape, or clean completion that does not repeat the v150 improvement.

## Setup

GKE workload `alekseyv-qwen3-v151-bs4-vmem98304r`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v151-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem98304-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: exact v150 stack, including `--xla_tpu_scoped_vmem_limit_kib=98304`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers with normal loss ending at 12.0460.

- worker0: 57,572 tok/s average, 7,196 tok/s/chip, 41.3% MFU.
- worker1: 57,546 tok/s average, 7,193 tok/s/chip, 41.3% MFU.
- Best observed steady steps: worker0 57,707 tok/s; worker1 57,704 tok/s.
- This confirms v150's 98304 KiB scoped-VMEM improvement over the confirmed
  v149 94208 frontier.

## Profile

XProf run:
`2026-06-04-qwen3-jax-v151-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem98304-rerun/2026_06_04_16_58_05`.

- Step average: 1961.2 ms.
- Total profile time: 109,695.374 ms, with 311.157 ms idle (0.3%).
- MXU utilization: 65.3%.
- Train-step program: `jit_train_step(6628478088121256461)`.
- Program work: 57,371.383 TFLOPs, 40,582.440 GiB accessed.
- Top buckets: convolution fusion 57,307.840 ms / 52.2%; custom-call
  29,380.915 ms / 26.8%; loop fusion 11,477.084 ms / 10.5%;
  all-reduce-scatter fusion 3,757.967 ms / 3.4%; data formatting
  2,899.024 ms / 2.6%; custom fusion 1,668.492 ms / 1.5%;
  reduce 1,003.133 ms / 0.9%; all-gather 967.195 ms / 0.9%.
- Custom-call split: DKV 15,510.563 ms / 14.1%; forward residuals combined
  13,870.352 ms / 12.6%.

## HLO Dump

The rerun hit the persistent compilation cache and did not emit a fresh
train-step HLO under the v151 run prefix. XProf reports the same train-step
program as v150, `jit_train_step(6628478088121256461)`, so use v150's HLO
evidence for the compiled program:

- Optimized HLO file:
  `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`.
- Optimized HLO size: 787,060 bytes.
- Content hash:
  `40d497c2bad650f5da214fd82b9636af17633b718094f8561a4b5a9ba7c2fdd9`.
- Compiled memory report: 28,845,509,048 bytes / 26.86 GiB.

## Verdict

Supported. The exact rerun repeats v150's train-step program and stays in the
same 41.3% MFU band, with worker0 averaging 57,572 tok/s and a new best
observed steady step of 57,707 tok/s. Carry scoped VMEM 98304 as the current
frontier. Next probe the upper bracket at 102400 KiB.
