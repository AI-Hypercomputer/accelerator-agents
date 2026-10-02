---
title: "Qwen3 JAX v250 - Levanter CE h512 v768"
type: experiment
hypothesis: "The Levanter Pallas CE fallback/default blocks recovered much of v248's regression, and the source autotune candidate set includes v=768 for float32 compute; h512/v768 may reduce CE-adjacent fusion cost versus h512/v1024."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v248-levanter-ce
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, levanter-ce, pallas, ce, splash, tokamax-splash, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v250 - Levanter CE h512 v768

## Hypothesis under test

**Hypothesis**: v249 showed that Levanter CE is sensitive to block size:
`h512/v1024` recovered from v248's `h256/v512` but remained below the MaxText CE
frontier. Levanter's source autotune candidate set for TPU Pallas with float32
compute includes `v_block_size=768`. A smaller vocab tile at the same hidden
tile (`h=512`) may reduce CE-adjacent fusion memory traffic while retaining most
of the default tile's MXU recovery.

**Mechanism**: Reuse the v248 image and change only the Levanter CE block sizes:

- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v248-levanter-ce`
- CE path: `--use_levanter_ce=True --use_maxtext_ce=False`
- Levanter block sizes: `b=1024,h=512,v=768`
- Keep the v230 RS2+AR2 runtime stack, VMEM 100352, no-scan, remat,
  Tokamax Splash attention, activation sharding, and batch/sequence shape fixed.
- Use a concrete GCS path in `XLA_FLAGS` so the HLO dump lands in the run
  directory.

**Predicted signal**: Support requires improvement over v249's
**58,427 tok/s / 41.9% MFU** band, ideally approaching v230/v236
(**59,749-59,750 tok/s / 42.9% MFU**) or showing lower CE-adjacent fusion cost
and memory.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion below v249. If refuted, close Levanter CE block-size tuning and move
to a different structural CE/lm-head path.

## Setup

GKE workload `alekseyv-q3-v250-levce-v768`.

- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v250-no-scan-sparsecore-rs-ar-rs2-ar2-levanter-ce-h512-v768`
- **Mesh**: `fsdp=8,tp=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Result

Completed cleanly on both workers with `EXIT_CODE=0` and finite synthetic loss
ending at **12.0461**.

Worker0:

- Average steps 2-19: **57,792 tok/s**, **7,224 tok/s/chip**, about **41.4%
  nominal MFU**
- Late/no-profile steps 15-19: 57,838, 57,864, 57,862, 57,766, 57,775 tok/s

Worker1:

- Average steps 2-19: **57,777 tok/s**, **7,222 tok/s/chip**, about **41.4%
  nominal MFU**
- Same finite-loss curve as worker0.

This is below [v249](2026-06-05-v249-no-scan-sparsecore-rs-ar-rs2-ar2-levanter-ce-default-blocks.md)
(58,427 tok/s / 41.9% MFU) and far below the v230/v236 frontier
(59,749-59,750 tok/s / 42.9% MFU).

## Profile

XProf run:
`2026-06-05-qwen3-jax-v250-no-scan-sparsecore-rs-ar-rs2-ar2-levanter-ce-h512-v768/2026_06_05_11_28_55`.

Summary:

- Step time: **4553.0 ms**
- MXU utilization: **66.2%**
- Peak HBM: **29.38 GiB** of 31.25 GiB
- Memory profile: 21.51 GiB stack reservation, 7.87 GiB heap allocation,
  1.87 GiB free

Top op buckets:

- `convolution fusion`: **29,978.6 ms / 54.9%**
- `custom-call`: **15,296.1 ms / 28.0%**
- `loop fusion`: **5,316.1 ms / 9.7%**
- `data formatting`: **1,484.1 ms / 2.7%**
- `custom fusion`: **1,019.7 ms / 1.9%**
- `all-gather`: **492.7 ms / 0.9%**

The Levanter CE forward custom-call is **582.7 ms / 1.1%**, lower than v249's
775.3 ms / 1.4%, but total train-step time is worse. The loss comes from the
generated fusion surface: `convolution fusion` rises to **54.9%**, bytes
accessed rise to **14,104.6 GiB**, and `select_add_fusion.2` appears at
**2,357.1 ms / 4.3%**.

## HLO Dump

Train-step HLO:

- GCS:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v250-no-scan-sparsecore-rs-ar-rs2-ar2-levanter-ce-h512-v768/hlo/xla-dump/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Local analysis copy:
  `/tmp/qwen3-v250-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256:
  `430ec9f326f5ee6956871f7428c591fcbbf5b0cba2acf1334a52a828dd0273fa`
- Size: **19,562,599 bytes**
- Lines: **116,385**

The dump uses the intended Levanter source frames
`/app/levanter_src/levanter/kernels/pallas/fused_cross_entropy_loss/api.py` and
`pallas_tpu.py`, with
`linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu` metadata. The memory
usage report again shows **26.09 GiB** total allocation dominated by a
**20.34 GiB** preallocated temp.

## Verdict

**Refuted.** `h512/v768` lowers the forward Pallas CE custom-call time but
worsens the whole train step by increasing CE-adjacent fusion work and bytes.
Do not carry this block shape.

Close Levanter Pallas CE block-size tuning for this frontier: the best tested
Levanter setting is v249 (`h512/v1024`), and it still trails the MaxText-CE
frontier by about 1.3k tok/s. The next CE/lm-head attempt should use a different
implementation strategy, not more Pallas block-size probes.
