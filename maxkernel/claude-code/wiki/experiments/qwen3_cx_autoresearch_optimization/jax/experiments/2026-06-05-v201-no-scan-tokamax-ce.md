---
title: "Qwen3 JAX v201 - no-scan Tokamax CE"
type: experiment
hypothesis: "On the confirmed no-scan frontier, Tokamax streamed CE may recover enough throughput from the v182 schedule change while preserving its lower CE memory footprint; if not, it closes Tokamax CE under no-scan before any batch-5 attempt."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-tokamax-ce
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, ce, tokamax-ce, splash, tokamax-splash, collective-matmul, scoped-vmem, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v201 - no-scan Tokamax CE

## Hypothesis under test

**Hypothesis**: v181 replaced MaxText CE with Tokamax streamed CE on the scan
stack and reduced compiled memory to 23.15 GiB, but regressed throughput to
53,280 tok/s / 38.2% MFU. v182/v183 later showed that disabling scan changes
the schedule family and improves the carried MaxText CE frontier to 41.8% MFU.
Tokamax CE may benefit from the same no-scan scheduling freedom, or at minimum
it may provide enough memory relief to justify a separate batch-5 probe.

**Mechanism**: Reuse the v183 no-scan runtime stack and change only CE
implementation from `--use_maxtext_ce=True` to
`--use_tokamax_ce=True --tokamax_ce_impl=mosaic_tpu --use_maxtext_ce=False`.

Keep the v169 no-MLP-intermediate-`_sac` image, no-scan, remat, tokamax Splash,
activation sharding, batch/sequence shape, collective-matmul modes disabled,
and scoped VMEM 100352 fixed.

**Predicted signal**: Support requires clean compile/loss and throughput above
the v182/v183 confirmed 58,288-58,330 tok/s average / 41.8% MFU band. A clean
but slower run can still be useful only if HBM falls enough to make a batch-5
follow-up plausible.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion below the frontier without a compelling memory-opening signal.

## Setup

GKE workload `alekseyv-qwen3-v201-noscan-tokamaxce`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v201-no-scan-tokamax-ce`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: v183 stack with `--xla_tpu_scoped_vmem_limit_kib=100352`,
  selective resources, custom-call VMEM adjustment, scheduler rerun 3, and
  both collective-matmul modes set to `none`.
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False --use_splash=True --use_tokamax_ce=True --tokamax_ce_impl=mosaic_tpu --use_maxtext_ce=False --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14 if the run reaches training.

## Equivalence

No model-code edit is made. Tokamax CE is an alternate implementation of the
same mean cross entropy over the final hidden states, lm-head weight, and
integer labels; prior v181 loss stayed normal.

## Results

The job compiled and reached training with normal loss. Kubernetes reaped the
JobSet/pods before final stdout could be fetched, so the final `EXIT_CODE` line
is unavailable, but the captured logs cover compile, steps 1-14, and the full
profile window.

Observed stdout before pod cleanup:

- Step 1 compile/warmup: ~132.6 s, 1,976 tok/s.
- Worker 0 steady steps 7-14: 54,368, 54,367, 54,357, 54,349, 54,332,
  54,294, 54,309, 54,312 tok/s.
- Worker 1 steady/profiler steps: step 7-11 at 54,370, 54,370, 54,360,
  54,347, 54,331 tok/s; step 12 had profiler overhead at 53,904 tok/s; steps
  13-14 were 54,303 and 54,304 tok/s.
- Loss stayed normal through the captured window, ending at 12.0623 on step 14.

This is far below the v182/v183 no-scan MaxText CE frontier band
of 58,288-58,330 tok/s average / 41.8% MFU.

## Profile

Profile files were written under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v201-no-scan-tokamax-ce/plugins/profile/2026_06_05_00_28_12/`

- `gke-tpu-964065d9-9vh8.xplane.pb`
- `gke-tpu-964065d9-c180.xplane.pb`
- matching `.trace.json.gz` files for both hosts.

Trace JSON spans for `jit_train_step(2566476259130850218)` confirm the stdout
throughput:

- `gke-tpu-964065d9-9vh8`: average profiled span 4.840374 s / 54,158 tok/s.
- `gke-tpu-964065d9-c180`: average profiled span 4.826154 s / 54,317 tok/s.

## HLO Dump

Fresh train-step HLO was emitted. The two worker modules are byte-identical:

- Optimized HLO:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v201-no-scan-tokamax-ce/hlo/module_0109.jit_train_step.cl_854318611.after_optimizations.txt`
- Duplicate worker module:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v201-no-scan-tokamax-ce/hlo/module_0311.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 19,005,717 bytes
- SHA256: `3453e887ee7ccf1ccab52e2503ad14362a8e6d50b2c77c944f1aa10b1edfd5ab`
- Memory report: 22.50 GiB total bytes used, with a 16.75 GiB preallocated
  temp. This is much lower than v182/v183's 27.66 GiB.
- Text counts: all-gather 4,655, all-reduce 960, collective-permute 3,
  convolution 2,088, custom-call 1,474, copy 8,946, fusion 23,235, `tokamax`
  20, `linear_softmax_cross_entropy_loss` 16, `log_softmax` 0, `_one_hot` 0.

## Verdict

Refuted for throughput. No-scan does not rescue Tokamax CE; it remains about
4k tok/s slower than the current MaxText CE frontier. The memory reduction is
real, but a batch-5 follow-up should be treated as a separate high-risk memory
envelope test, not a carried frontier change.
