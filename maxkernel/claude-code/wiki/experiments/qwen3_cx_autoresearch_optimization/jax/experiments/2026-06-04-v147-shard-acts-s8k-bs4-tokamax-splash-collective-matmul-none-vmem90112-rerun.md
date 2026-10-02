---
title: "Qwen3 JAX v147 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 90112 rerun"
type: experiment
hypothesis: "If v146's scoped-VMEM 90112 gain is real, an exact rerun should remain above the confirmed 86016 frontier."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, scoped-vmem, latency-hiding, scheduler, selective-resources, async-collective-fusion, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v147 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 90112 rerun

## Hypothesis under test

**Hypothesis**: v146's 90112 KiB scoped-VMEM point is a genuine improvement over
the confirmed v144/v145 86016 KiB frontier, not runtime noise.

**Mechanism**: Exact rerun of v146. Keep the v146 command, image, XLA flag
stack, mesh, batch, and profile window fixed; change only workload/run naming.

**Predicted signal**: Support requires clean loss and best-worker average at or
above the confirmed 86016 frontier band, ideally near v146's 56,635 tok/s. If it
falls back to the v144/v145 band or lower, keep 86016 as the carried frontier and
treat 90112 as provisional/noisy.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed
command shape, or clean completion that does not repeat the v146 improvement.

## Setup

GKE workload `alekseyv-qwen3-v147-bs4-vmem90112r`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v147-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem90112-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: exact v146 stack, including `--xla_tpu_scoped_vmem_limit_kib=90112`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers with normal loss ending at 12.0460.

- worker0: 56,644 tok/s average, 7,081 tok/s/chip, 40.6% MFU.
- worker1: 56,618 tok/s average, 7,077 tok/s/chip, 40.6% MFU.
- Best observed steady steps: worker0 56,785 tok/s; worker1 56,784 tok/s.
- This confirms v146's 90112 KiB scoped-VMEM gain and raises the carried
  best-worker average from 56,609/56,603 tok/s at 86016 to 56,644 tok/s.

## Profile

XProf run:
`2026-06-04-qwen3-jax-v147-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem90112-rerun/2026_06_04_16_28_01`.

- Step average: 1993.9 ms.
- Total profile time: 111,507.492 ms, with 322.726 ms idle (0.3%).
- MXU utilization: 64.8%.
- Train-step program: `jit_train_step(13912291835542571918)`.
- Program work: 57,371.383 TFLOPs, 40,569.465 GiB accessed.
- Top buckets: convolution fusion 59,638.220 ms / 53.5%; custom-call
  29,354.048 ms / 26.3%; loop fusion 11,497.457 ms / 10.3%;
  all-reduce-scatter fusion 3,754.597 ms / 3.4%; data formatting
  2,898.169 ms / 2.6%; custom fusion 1,201.123 ms / 1.1%.
- Custom-call split: DKV 15,484.528 ms / 13.9%; forward residuals combined
  13,869.519 ms / 12.4%.

## HLO Dump

No fresh HLO directory was emitted under the v147 run prefix, consistent with a
persistent compile-cache hit. XProf reports the same train-step program as v146,
`13912291835542571918`, so use v146's compiled-program evidence:

- Optimized HLO size: 787,246 bytes.
- Content hash:
  `eec5d587c0d511d4a7c6eb30dbdacbe4563121cc1321870f957d686be63b0ea6`.
- Compiled memory report: 28,853,864,892 bytes / 26.87 GiB.
- Flagfile confirms `--xla_tpu_scoped_vmem_limit_kib=90112`,
  `--xla_tpu_scheduler_percent_shared_memory_limit=100`, scheduler rerun3,
  host send/recv 100, host-transfer overlap 24, and both collective-matmul
  modes set to `NONE`.

## Verdict

Supported. The exact rerun repeats and slightly improves v146, so carry scoped
VMEM 90112 as the current durable frontier for the batch-4/global-32
collective-matmul-none family. Next probe one higher scoped-VMEM point to find
the upper side of the new bracket.
