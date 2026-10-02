---
title: "Qwen3 JAX v149 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 94208 rerun"
type: experiment
hypothesis: "If v148's scoped-VMEM 94208 gain is real, an exact rerun should remain above the confirmed 90112 frontier."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: completed
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, scoped-vmem, latency-hiding, scheduler, selective-resources, async-collective-fusion, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v149 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 94208 rerun

## Hypothesis under test

**Hypothesis**: v148's scoped-VMEM 94208 point is a genuine improvement over
the confirmed v147 90112 frontier, not runtime noise.

**Mechanism**: Exact rerun of v148. Keep the v148 command, image, XLA flag
stack, mesh, batch, and profile window fixed; change only workload/run naming.

**Predicted signal**: Support requires clean loss and best-worker average above
the confirmed 90112 frontier, ideally near v148's 56,773 tok/s. If it falls back
to the v147 band or lower, keep 90112 as the carried frontier and treat 94208 as
provisional/noisy.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed
command shape, or clean completion that does not repeat the v148 improvement.

## Setup

GKE workload `alekseyv-qwen3-v149-bs4-vmem94208r`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v149-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem94208-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: exact v148 stack, including `--xla_tpu_scoped_vmem_limit_kib=94208`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers with normal loss ending at 12.0460.

- worker0: 56,768 tok/s average, 7,096 tok/s/chip, 40.7% MFU.
- worker1: 56,743 tok/s average, 7,093 tok/s/chip, 40.7% MFU.
- Best observed steady steps: worker0 56,856 tok/s; worker1 56,856 tok/s.
- This confirms v148's 94208 KiB scoped-VMEM improvement over the v147 90112
  frontier.

## Profile

XProf run:
`2026-06-04-qwen3-jax-v149-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem94208-rerun/2026_06_04_16_43_17`.

- Step average: 1988.9 ms.
- Total profile time: 111,236.251 ms, with 320.385 ms idle (0.3%).
- MXU utilization: 65.1%.
- Train-step program: `jit_train_step(15797032912653574046)`.
- Program work: 57,371.383 TFLOPs, 40,577.401 GiB accessed.
- Top buckets: convolution fusion 59,175.606 ms / 53.2%; custom-call
  29,339.466 ms / 26.4%; loop fusion 11,481.280 ms / 10.3%;
  all-reduce-scatter fusion 3,755.724 ms / 3.4%; data formatting
  2,880.820 ms / 2.6%; custom fusion 1,417.790 ms / 1.3%.
- Custom-call split: DKV 15,468.522 ms / 13.9%; forward residuals combined
  13,870.943 ms / 12.5%.

## HLO Dump

No fresh HLO directory was emitted under the v149 run prefix, consistent with a
persistent compile-cache hit. XProf reports the same train-step program as v148,
`15797032912653574046`, so use v148's compiled-program evidence:

- Optimized HLO size: 779,591 bytes.
- Content hash:
  `e78989b8aa0a3c726e47566c51f7fd42cd5d218e129b2bf0bff2fc5b4f51fc93`.
- Compiled memory report: 28,849,097,120 bytes / 26.87 GiB.
- Flagfile confirms `--xla_tpu_scoped_vmem_limit_kib=94208`,
  `--xla_tpu_scheduler_percent_shared_memory_limit=100`, scheduler rerun3,
  host send/recv 100, host-transfer overlap 24, and both collective-matmul
  modes set to `NONE`.

## Verdict

Supported. The exact rerun repeats v148's runtime profile and remains well above
the confirmed 90112 frontier, so carry scoped VMEM 94208 as the current durable
frontier for the batch-4/global-32 collective-matmul-none family. Next probe the
next upper scoped-VMEM point to find where the trend stops.
