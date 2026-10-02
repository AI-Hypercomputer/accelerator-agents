---
title: "Qwen3 JAX v245 - RS2+AR2 Tokamax CE batch 5"
type: experiment
hypothesis: "Tokamax CE on the post-SparseCore frontier is slower at batch 4 but opens enough HBM headroom that batch 5/global batch 40 may recover aggregate throughput."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v169-no-mlp-sac-runtime-noscan-sc-rs2-ar2-tokamax-ce-bs5
status: completed
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, tokamax-ce, ce, batch-scaling, splash, tokamax-splash, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v245 - RS2+AR2 Tokamax CE batch 5

## Hypothesis under test

**Hypothesis**: v244 showed that post-SparseCore Tokamax CE reduces peak HBM to
**27.02 GiB** but remains too slow at batch 4. The freed memory may allow
per-chip batch 5/global batch 40 to fit and recover enough total throughput to
make the lower-memory CE path useful.

**Mechanism**: Keep v244 fixed except change `--batch_size=4` to
`--batch_size=5`.

**Predicted signal**: Support requires clean compile/loss and throughput above
the v230/v236 batch-4 MaxText CE frontier (**59,749-59,750 tok/s**), or a
profile showing enough headroom and scaling efficiency to justify further
Tokamax CE tuning.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion below the current frontier.

## Setup

GKE workload `alekseyv-q3-v245-tokamaxce-bs5`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v169-no-mlp-sac`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v245-no-scan-sparsecore-rs-ar-rs2-ar2-tokamax-ce-bs5`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False
  --use_splash=True --use_tokamax_ce=True --tokamax_ce_impl=mosaic_tpu
  --use_maxtext_ce=False --shard_acts=True --batch_size=5 --seqlen=8192
  --tp_parallelism=1`
- **Global batch**: 40
- **Profile window**: steps 12-14

## Equivalence

No model-code edit is made. This changes only runtime batch size relative to
v244.

## Result

Workload `alekseyv-q3-v245-tokamaxce-bs5` completed cleanly with
`EXIT_CODE=0` on both workers and normal loss ending at 12.0496.

- Worker0: **55,885 tok/s** average over steps 2-19, **56,097 tok/s** best
  step, **55,988 tok/s** average over steps 15-19, **55,897 tok/s** excluding
  profiled steps.
- Worker1: **55,866 tok/s** average over steps 2-19, **56,098 tok/s** best
  step, **55,987 tok/s** average over steps 15-19, **55,897 tok/s** excluding
  profiled steps.

Batch 5 improves over v244 batch 4 Tokamax CE (**55,493 tok/s**) but still
lands far below the v230/v236 MaxText CE frontier (**59,749-59,750 tok/s**).
Nominal MFU is about **40.0%**.

## Profile

XProf split the two hosts into adjacent one-host runs:

- `2026-06-05-qwen3-jax-v245-no-scan-sparsecore-rs-ar-rs2-ar2-tokamax-ce-bs5/2026_06_05_10_08_39`:
  **5876.4 ms** step time, **64.2% MXU**.
- `2026-06-05-qwen3-jax-v245-no-scan-sparsecore-rs-ar-rs2-ar2-tokamax-ce-bs5/2026_06_05_10_08_40`:
  **5888.2 ms** step time, **63.4% MXU**.

Representative memory profile:

- Peak HBM: **29.16 GiB** / 31.25 GiB
- Stack reservation: **21.3589 GiB**
- Heap allocation: **7.7977 GiB**
- Free memory at peak: **2.0895 GiB**

Representative one-host op profile:

- convolution fusion: **32,501.6 ms** / **46.1%**
- custom-call: **26,331.2 ms** / **37.3%**
- loop fusion: **6,687.3 ms** / **9.5%**

The Tokamax CE backward custom-call remains visible:
`linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu.1` takes
**6,306.5 ms** / **8.9%** of the profiled one-host train-step time.

## HLO Dump

Optimized train-step HLO:

- Local dump:
  `/tmp/qwen3-v245-hlo/module_0474.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256:
  `0ff06d31846d093b49a0690ea11c0fc37003bc030a39a51ec976e72796f26bb3`
- Size: **19,930,103 bytes**
- Lines: **119,317**

## Verdict

**Refuted.** The v244 HBM headroom is sufficient for per-chip batch 5/global
batch 40, and throughput rises by about **392 tok/s** over batch 4 Tokamax CE,
but the larger batch still cannot recover the CE custom-call wall-time cost.
Peak HBM remains below capacity at **29.16 GiB**, yet profiled step time is
about **5.88 s** and MXU remains around **64%**. Do not continue Tokamax CE
batch scaling on this frontier without first reducing the CE backward kernel
cost.
