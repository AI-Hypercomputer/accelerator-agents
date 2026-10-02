---
title: "Qwen3 JAX v249 - Levanter CE default block sizes"
type: experiment
hypothesis: "v248 may have underperformed because it forced smaller Levanter CE tiles than Levanter's fallback/default TPU shape for Qwen's local CE dimensions."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v248-levanter-ce
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, levanter-ce, pallas, ce, splash, tokamax-splash, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v249 - Levanter CE default block sizes

## Hypothesis under test

**Hypothesis**: v248 compiled and ran the Levanter Pallas TPU fused CE path, but
it forced `b=1024,h=256,v=512`. For this Qwen local CE shape
`B=32768,H=4096,V=151936`, Levanter's tuned bucket table does not match because
the vocab is larger than the table's 131k-vocab buckets. Levanter would
therefore fall back to its default TPU block sizes `b=1024,h=512,v=1024`.
Testing those defaults isolates whether v248 was a bad block-size choice or a
bad kernel-family choice.

**Mechanism**: Reuse the v248 image and change only the Levanter CE block sizes:

- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v248-levanter-ce`
- CE path: `--use_levanter_ce=True --use_maxtext_ce=False`
- Levanter block sizes: `b=1024,h=512,v=1024`
- Keep the v230 RS2+AR2 runtime stack, VMEM 100352, no-scan, remat,
  Tokamax Splash attention, activation sharding, and batch/sequence shape fixed.

**Predicted signal**: Support requires a material recovery from v248's
**56,088 tok/s / 40.2% MFU** band, ideally toward the current v230/v236
frontier (**59,749-59,750 tok/s / 42.9% MFU**) or a profile showing reduced
CE-adjacent fusion cost/memory.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion in the same ~56k tok/s band. If refuted, close Levanter CE block-size
retuning as insufficient and move to a different structural CE/lm-head strategy.

## Setup

GKE workload `alekseyv-q3-v249-levce-def`.

- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v249-no-scan-sparsecore-rs-ar-rs2-ar2-levanter-ce-default-blocks`
- **Mesh**: `fsdp=8,tp=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Result

Completed cleanly on both workers with `EXIT_CODE=0` and finite synthetic loss
ending at **12.0461**.

Worker0:

- Average steps 2-19: **58,427 tok/s**, **7,303 tok/s/chip**, about **41.9%
  nominal MFU**
- Late/no-profile steps 15-19: 58,553, 58,479, 58,469, 58,393, 58,411 tok/s

Worker1:

- Average steps 2-19: **58,401 tok/s**, **7,300 tok/s/chip**, about **41.9%
  nominal MFU**
- Same loss curve and same operating band as worker0.

This recovers about **2.34k tok/s** versus
[v248](2026-06-05-v248-no-scan-sparsecore-rs-ar-rs2-ar2-levanter-ce.md), so
the smaller `h=256,v=512` tile was a bad Levanter choice. It still remains
about **1.32k tok/s** below the current v230/v236 frontier
(**59,749-59,750 tok/s / 42.9% MFU**).

## Profile

XProf run:
`2026-06-05-qwen3-jax-v249-no-scan-sparsecore-rs-ar-rs2-ar2-levanter-ce-default-blocks/2026_06_05_11_15_15`.

Summary:

- Step time: **4499.4 ms**
- MXU utilization: **67.2%**
- Peak HBM: **29.38 GiB** of 31.25 GiB
- Memory profile: 21.51 GiB stack reservation, 7.87 GiB heap allocation,
  1.87 GiB free

Top op buckets:

- `convolution fusion`: **29,193.2 ms / 54.1%**
- `custom-call`: **15,489.6 ms / 28.7%**
- `loop fusion`: **5,316.8 ms / 9.9%**
- `data formatting`: **1,485.7 ms / 2.8%**
- `custom fusion`: **1,027.8 ms / 1.9%**
- `all-gather`: **421.5 ms / 0.8%**

The Levanter CE forward custom-call
`linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu.1` is
**775.3 ms / 1.4%** on the representative host. The default blocks reduce the
CE-adjacent fusion drag seen in v248 and restore MXU to the frontier band, but
the overall train step is still about 99 ms slower than v230's **4400.7 ms**.

## HLO Dump

The intended XLA text dump did **not** land in the GCS run directory. This v249
launch set `XLA_FLAGS=--xla_dump_to=${RUN_DIR}/hlo/xla-dump ...` as a literal
Kubernetes environment value; profile output used `$RUN_DIR` from the shell
command correctly, but Kubernetes does not expand `${RUN_DIR}` inside another
environment value. XProf listed the train-step module
`jit_train_step(15480732180728451777)`, but the graph-viewer content endpoint
returned 500 / no HLO graph data for the text body.

Future launches should pass a concrete GCS path in `XLA_FLAGS` rather than
`${RUN_DIR}` when HLO text is required.

## Verdict

**Refuted, but partially supportive of block retuning.** Levanter CE default
blocks are much better than v248's smaller block shape, but still do not beat
the MaxText-CE frontier and do not provide enough memory headroom to justify
carrying Levanter CE as-is.

One final Levanter block candidate from the source autotune set is worth a
bounded probe: keep `h=512` and test `v=768`, with a corrected concrete HLO dump
path. If that remains below frontier, close Levanter CE block-size tuning and
move to a different structural CE/lm-head path.
