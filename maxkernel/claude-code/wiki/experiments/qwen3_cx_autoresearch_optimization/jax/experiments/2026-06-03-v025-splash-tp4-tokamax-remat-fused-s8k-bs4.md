---
title: "Qwen3 JAX v025 - splash TP4 tokamax remat fused s8k bs4"
type: experiment
hypothesis: "TP4 may trade extra collectives for enough per-shard memory relief to fit seq8192 global batch 8 at target sequence length."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v6e8-qwen3-8b-jax-20260602-v015-splash-tp2-tokamax-remat-fused-rope-bs8
status: invalid
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax, remat, fused-rope, tp4, long-seq, batch-scaling, invalid, oom, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v025 - splash TP4 tokamax remat fused s8k bs4

## Hypothesis under test

**Hypothesis**: TP4 may trade extra collectives for enough per-shard memory
relief to fit seq8192 global batch 8 at target sequence length.

**Mechanism**: Reuse the v015 image, Splash attention, tokamax CE, remat, fused
RoPE, and the successful XLA flag stack. Keep `--seqlen=8192`, move from
`fsdp=4,tp=2` to `fsdp=2,tp=4`, and use per-chip batch 4 for global batch 8.

**Predicted signal**: If the extra tensor parallelism sharding covers the v024
2.75 GiB HBM excess, the run should compile and may improve target-sequence
throughput versus v023 by using global batch 8 instead of global batch 4. The
main risk is that added TP collectives erase the token-count gain.

**Falsification criterion**: Compile/runtime OOM, or clean completion with MFU
at or below v023's 26.6% despite the larger global batch.

## Setup

Planned GKE workload `alekseyv-qwen3-v025`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v015-splash-tp2-tokamax-remat-fused-rope-bs8`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v025-splash-tp4-tokamax-remat-fused-s8k-bs4`
- **Mesh**: `fsdp=2,tp=4`
- **Command shape**: `--batch_size=4 --seqlen=8192 --tp_parallelism=4 --use_splash=True`
- **Global batch**: 8
- **Profile window**: steps 12-14

## Results

Invalid. Workload `alekseyv-qwen3-v025` failed during train-step compilation
with `RESOURCE_EXHAUSTED`.

- **Exit**: worker 0 exited with compile-time HBM OOM; JobSet failed by policy
  at `2026-06-03T02:08:41Z`
- **OOM**: used 35.46 GiB of 31.25 GiB HBM; exceeded capacity by 4.21 GiB
- **Memory accounting**: total HBM usage at least 35.71 GiB, reserved 260.00
  MiB, program 35.46 GiB, arguments 0B, HLO temp 35.44 GiB
- **Largest allocation**: 2.32 GiB tokamax backward dot
  `fusion.4229.remat5`, with labels shape `i32[65536]`
- **Other large allocations**: 296.75 MiB and 256.00 MiB tokamax all-gathers,
  plus repeated 192.00 MiB activation/collective buffers of shape
  `bf16[4,8192,3072]`

## Profile

No runtime profile; the workload failed before executing training steps.

## HLO Dump

OOM artifacts were dumped under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v025-splash-tp4-tokamax-remat-fused-s8k-bs4/hlo/`

The prefix contains 272 objects totaling 187.16 MiB.

## Verdict

Refuted. TP4 made the seq8192 global-batch-8 shape worse than v024 under this
tokamax/fused-RoPE stack: HBM usage rose from 33.99 GiB to 35.46 GiB. More TP
is not the right memory lever here; target-sequence progress needs activation
sharding / CE changes rather than more tensor parallelism.
