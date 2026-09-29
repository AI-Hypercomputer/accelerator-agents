---
title: "Qwen3 JAX v097 - shard acts s8k bs3 tokamax splash qseq2 dense mask"
type: experiment
hypothesis: "The fast q_seq_shards=2 Tokamax Splash path NaNs because the lazy causal mask path miscomputes q-sharded partial masking; using a dense causal NumpyMask may preserve correctness."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v097-qseq2-dense-mask
status: invalid
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, qseq-shards, dense-mask, custom-call, remat, maxtext-ce, activation-sharding, long-seq, invalid, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v097 - shard acts s8k bs3 tokamax splash qseq2 dense mask

## Hypothesis under test

**Hypothesis**: v050/v051/v064 showed that `TOKAMAX_Q_SEQ_SHARDS=2` has a
large nominal throughput upside but produces NaN loss at step 1. Inspection of
the bundled Tokamax code on 2026-06-04 showed that `CausalMask(...,
shard_count=1)` and `CausalMask(..., shard_count=2)` produce the same processed
`MaskInfo`; `shard_count` only validates divisibility and does not change the
kernel mask metadata. The dense `NumpyMask(make_causal_mask(...))` path instead
materializes static partial-mask blocks and avoids the lazy `q_sequence` /
`mask_function` path.

**Mechanism**: Derive `/app/trainer/splash_attn.py` from the v064 image-layered
file and change only the Tokamax mask construction to:

`tokamax_splash_mask.NumpyMask(tokamax_splash_mask.make_causal_mask(shape=(T, T)))`

Runtime keeps `TOKAMAX_Q_SEQ_SHARDS=2` and the v090 layer-scheduler plus
sparse-core-aggregator scheduler frontier.

**Predicted signal**: Loss stays finite through the profiling window and
throughput remains near the invalid qseq2 band, ideally above the v090 valid
frontier.

**Falsification criterion**: Compile/runtime failure, NaN/invalid loss, or a
clean run whose throughput falls back to the v090 band without a profile
improvement.

## Setup

GKE workload `alekseyv-qwen3-v097-qseqdense`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v097-qseq2-dense-mask`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v097-shard-acts-s8k-bs3-tokamax-splash-qseq2-dense-mask`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_Q_SEQ_SHARDS=2 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v090 scheduler frontier with scoped VMEM 81920, custom-call
  VMEM adjustment, selective resources, scheduler rerun 3, async collective
  fusion flags, layer scheduler, and sparse-core collective aggregator.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=3 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 24
- **Profile window**: steps 12-14

## Results

Workload `alekseyv-qwen3-v097-qseqdense` completed on 2026-06-04 with exit code
0, but both workers produced NaN loss starting at step 1. This is invalid and
does not count as a frontier move.

- **Worker 0**: step 0 loss 12.0987, step 1 loss NaN, summary 54,832 tok/s /
  39.3% nominal MFU.
- **Worker 1**: step 1 loss NaN, summary 54,805 tok/s / 39.3% nominal MFU.
- **Best valid comparison**: v090 remains 51,033 tok/s best observed, 51,026+
  repeated / 36.6% MFU.

## Profile

XProf run:

- `2026-06-04-qwen3-jax-v097-shard-acts-s8k-bs3-tokamax-splash-qseq2-dense-mask/2026_06_04_07_26_25`

The profile is useful only as an invalid-fast diagnostic. It matches the qseq2
speedup signature: total device time 86,305.3 ms, idle 0.3%, train-step time
86,035.0 ms, convolution fusion 52.2%, loop fusion 19.3%, custom-call 18.6%,
data formatting 4.4%, and collective-permute-start 1.3%. Within custom-call,
`splash_mha_dkv_no_residuals.1` is 11,898.9 ms / 13.8%, and the two forward
residual calls total 4,126.9 ms / 4.8%.

Memory profile reports peak HBM 28.33 GiB of 31.25 GiB, with 22.51 GiB stack
reservation, 5.82 GiB heap allocation, and 90.66% utilization.

## HLO Dump

HLO was regenerated under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v097-shard-acts-s8k-bs3-tokamax-splash-qseq2-dense-mask/hlo/`

- 32 HLO objects, 102.14 MiB total.
- Train-step module:
  `module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Train-step optimized text size: 1,377,637 bytes.
- Flagfile confirms scoped VMEM 81920, custom-call scoped VMEM adjustments,
  selective resources, scheduler rerun 3, layer scheduler, and sparse-core
  collective aggregator.
- Dense static masking changed the Tokamax custom-call signatures and inserted
  static mask operands, including `s8[1,2048,2048]` on DKV and block-mask arrays
  on the forward custom calls.

## Verdict

**Invalid/refuted.** Replacing Tokamax's lazy causal mask with a dense static
`NumpyMask` preserves the invalid qseq2 speed band but does not preserve
correctness. The NaN is not caused solely by the lazy `CausalMask` q-sequence /
mask-function path, and mask-only qseq2 retries should stop.
