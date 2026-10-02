---
title: "Qwen3 JAX v154 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 100352 rerun"
type: experiment
hypothesis: "If v153's scoped-VMEM 100352 gain is real, an exact rerun should remain above the confirmed 98304 frontier."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v041-shard-acts
status: supported
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax-splash, xla-flags, collective-matmul, scoped-vmem, latency-hiding, scheduler, selective-resources, async-collective-fusion, custom-call, remat, maxtext-ce, activation-sharding, batch-scaling, long-seq, v6e-8]
created: 2026-06-04
updated: 2026-06-04
---

# Qwen3 JAX v154 - shard acts s8k bs4 tokamax splash collective matmul none scoped vmem 100352 rerun

## Hypothesis under test

**Hypothesis**: v153's scoped-VMEM 100352 point is a genuine improvement over
the confirmed 98304 frontier, not runtime noise.

**Mechanism**: Exact rerun of v153. Keep the v153 command, image, XLA flag
stack, mesh, batch, and profile window fixed; change only workload/run naming.

**Predicted signal**: Support requires clean loss and best-worker average above
the confirmed 98304 frontier, ideally near v153's 57,757 tok/s. If it falls
back to the v150/v151 band or lower, keep 98304 as the carried frontier and
treat 100352 as provisional/noisy.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed
command shape, or clean completion that does not repeat the v153 improvement.

## Setup

GKE workload `alekseyv-qwen3-v154-bs4-vmem100352r`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v041-shard-acts`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v154-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem100352-rerun`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **XLA flags**: exact v153 stack, including `--xla_tpu_scoped_vmem_limit_kib=100352`.
- **Command shape**: `--use_scan=True --use_remat=True --use_splash=True --use_maxtext_ce=True --shard_acts=True --batch_size=4 --seqlen=8192 --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Results

Completed cleanly on both workers.

- Worker 0: average 57,754 tok/s (7,219/chip), approx 41.4% MFU; best steady step 57,862 tok/s; final loss 12.0461.
- Worker 1: average 57,724 tok/s (7,215/chip), approx 41.4% MFU; best steady step 57,857 tok/s; final loss 12.0461.
- JobSet: `Completed`, `EXIT_CODE=0` on both pods.

This confirms the v153 scoped-VMEM 100352 point: the best-worker average is
within 3 tok/s of v153's 57,757 tok/s and remains above the confirmed 98304
frontier (v151: 57,572 tok/s).

## Profile

Profile files were written under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-04-qwen3-jax-v154-shard-acts-s8k-bs4-tokamax-splash-collective-matmul-none-vmem100352-rerun/plugins/profile/2026_06_04_17_29_23/`

Files present:

- `gke-tpu-964065d9-c180.xplane.pb`
- `gke-tpu-964065d9-c180.trace.json.gz`
- `gke-tpu-964065d9-9vh8.xplane.pb`
- `gke-tpu-964065d9-9vh8.trace.json.gz`

Immediate XProf HTTP/MCP queries returned "No xplane file found" / 500 while
the GCS files were already present, so profile aggregate extraction is delayed.
Use v153's XProf for the same compiled program family until the server index
refreshes: step average 1954.6 ms, MXU 65.6%, peak HBM 26.92 GiB.

## HLO Dump

No fresh HLO dump appeared for v154; the rerun used the persistent compile
cache. This is expected for an exact rerun. The carried HLO evidence is v153:

- Train-step program: `10728347668315821976`
- HLO: `module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Size: 783,949 bytes
- SHA256: `3734fae066fc35c4b9c6c4cce9b0015413ab4513c34a09ba33a0e1862bff3a3d`
- Compiled memory: 28,841,540,536 bytes / 26.86 GiB

## Verdict

**Supported.** Carry `--xla_tpu_scoped_vmem_limit_kib=100352` as the confirmed
frontier. The next upper-bracket probe is `101376` KiB, between confirmed
100352 and refuted 102400.
