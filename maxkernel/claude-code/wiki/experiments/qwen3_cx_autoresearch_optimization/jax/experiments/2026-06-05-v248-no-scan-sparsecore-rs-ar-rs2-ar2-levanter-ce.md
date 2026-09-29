---
title: "Qwen3 JAX v248 - RS2+AR2 with Levanter Pallas fused CE"
type: experiment
hypothesis: "A Levanter Pallas TPU fused lm_head+CE path can keep fp32 CE math while avoiding full-logit materialization, reducing HBM without the Tokamax CE backward custom-call cost that refuted v244/v245."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v248-levanter-ce
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, levanter-ce, pallas, ce, splash, tokamax-splash, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v248 - RS2+AR2 with Levanter Pallas fused CE

## Hypothesis under test

**Hypothesis**: v244/v245 showed that streaming CE is the right memory direction
but the Tokamax `linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu`
custom-call cost is too high. Levanter's Pallas TPU fused CE is a different
kernel implementation. With fp32 CE compute, it may preserve the current MaxText
CE math while avoiding full `[B,L,V]` materialization and avoiding Tokamax's
observed backward custom-call cost.

**Mechanism**: Keep the v230 RS2+AR2 runtime stack fixed and change only the CE
path:

- Base image: `qwen3-8b-jax:v169-no-mlp-sac`
- New image: `qwen3-8b-jax:v248-levanter-ce`
- Image digest:
  `sha256:06bf336431f2f38b392b77913ecd1a33282657994f9ef317a8025ca0082d1748`
- Add `--use_levanter_ce=True --use_maxtext_ce=False`
- Keep `--use_scan=False --use_remat=True --offload_remat=False
  --use_splash=True --shard_acts=True --batch_size=4 --seqlen=8192
  --tp_parallelism=1`
- Levanter block sizes: `b=1024,h=256,v=512`
- Runtime flags keep RS2+AR2 SparseCore offload and VMEM 100352.

The thin image was derived from the bundled v169 trainer, not current trunk. The
only trainer change adds an explicit Levanter CE path behind
`--use_levanter_ce`; default MaxText CE behavior remains unchanged.

**Predicted signal**: Support requires clean compile/loss and either throughput
at or above the v230/v236 band (**59,749-59,750 tok/s**) or a profile showing
substantial HBM relief without the Tokamax CE custom-call regression.

**Falsification criterion**: Compile/runtime failure, invalid loss, silent
fallback, or clean completion materially below the current frontier with no
compelling memory/perf tradeoff.

## Setup

GKE workload `alekseyv-q3-v248-levanterce`.

- **Image**:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v248-levanter-ce`
- **Run dir**:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v248-no-scan-sparsecore-rs-ar-rs2-ar2-levanter-ce`
- **Mesh**: `fsdp=8,tp=1`
- **Environment**: `USE_TOKAMAX_SPLASH=1 TOKAMAX_MAX_LOGIT_CONST=30.0`
- **Command shape**: `--use_scan=False --use_remat=True --offload_remat=False
  --use_splash=True --use_levanter_ce=True --use_maxtext_ce=False
  --levanter_ce_b_block=1024 --levanter_ce_h_block=256
  --levanter_ce_v_block=512 --shard_acts=True --batch_size=4 --seqlen=8192
  --tp_parallelism=1`
- **Global batch**: 32
- **Profile window**: steps 12-14

## Equivalence

This changes the CE implementation only. The Levanter path computes
`hidden @ lm_head_weight.T` inside the fused CE kernel and uses fp32 CE compute
to match the existing MaxText/T5X custom-VJP CE precision intent. Correctness is
gated by finite loss matching the known synthetic-loss curve; no fallback path
is enabled because `implementation="pallas_tpu"` is passed explicitly.

## Result

Completed cleanly on both workers with `EXIT_CODE=0` and finite synthetic loss
ending at **12.0459**.

Worker0:

- Average steps 2-19: **56,088 tok/s**, **7,011 tok/s/chip**, about **40.2%
  nominal MFU**
- Representative late steps stayed around 56.1k tok/s; step 12 had the
  expected profile-start penalty.

Worker1:

- Average steps 2-19: **56,053 tok/s**, **7,007 tok/s/chip**, about **40.2%
  nominal MFU**
- Same finite-loss curve and same throughput band as worker0.

This is faster than the earlier Tokamax streamed-CE branch
([v244](2026-06-05-v244-no-scan-sparsecore-rs-ar-rs2-ar2-tokamax-ce.md) at
55,493 tok/s), but it remains far below the MaxText-CE RS2/AR2 frontier
([v230](2026-06-05-v230-no-scan-sparsecore-rs-ar-rs-latency2-ar-latency2-rerun.md)
/ [v236](2026-06-05-v236-no-scan-sparsecore-rs-ar-rs3-ar2-rerun.md) at
59,749-59,750 tok/s).

## Profile

XProf run:
`2026-06-05-qwen3-jax-v248-no-scan-sparsecore-rs-ar-rs2-ar2-levanter-ce/2026_06_05_11_00_51`.

Summary:

- Step time: **4702.9 ms**
- MXU utilization: **63.5%**
- Peak HBM: **29.37 GiB** of 31.25 GiB
- Memory profile: 21.51 GiB stack reservation, 7.86 GiB heap allocation,
  1.87 GiB free

Top op buckets:

- `convolution fusion`: **62,478.7 ms / 55.3%**
- `custom-call`: **30,572.1 ms / 27.1%**
- `loop fusion`: **10,569.5 ms / 9.4%**
- `data formatting`: **2,956.8 ms / 2.6%**
- `custom fusion`: **2,734.1 ms / 2.4%**
- `all-gather`: **1,331.3 ms / 1.2%**

The Levanter/Pallas CE forward custom-call is visible as
`linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu.1`, but it is only
**1,154.1 ms / 1.0%** of the profiled train-step time. The slower result is
therefore not a simple forward-CE custom-call bottleneck. The profile shifts
time and memory into generated CE-adjacent fusions, with `select_add_fusion.2`
alone at **6,913.5 ms / 6.1%** and **7,601.3 GiB** accessed.

## HLO Dump

Train-step HLO:

- GCS:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v248-no-scan-sparsecore-rs-ar-rs2-ar2-levanter-ce/hlo/xla-dump/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Local analysis copy:
  `/tmp/qwen3-v248-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256:
  `db06c168df683b818ae806fc4e7a34f0b695373398a18634219d10c243847ab0`
- Size: **19,562,405 bytes**
- Lines: **116,384**

The HLO stack frames include the vendored Levanter paths
`/app/levanter_src/levanter/kernels/pallas/fused_cross_entropy_loss/api.py` and
`pallas_tpu.py`, confirming that the intended Levanter/Pallas path compiled.
The HLO contains both
`linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu` and
`linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu` metadata.

The XLA memory-usage report for the same train-step HLO reports **26.09 GiB**
total allocation, dominated by a **20.34 GiB** preallocated temp.

## Verdict

**Refuted.** Levanter CE is a valid no-fallback implementation path and improves
on the prior Tokamax CE throughput branch, but it does not deliver enough HBM
relief and lands about **3.66k tok/s** below the current MaxText-CE frontier.
Do not replace the v230/v236 frontier with Levanter CE.

The useful follow-up signal is structural: CE/lm-head memory remains worth
attacking, but the next attempt should either reduce the generated backward
fusion/memory footprint or target a larger layout change. Pure Levanter CE
adoption at `b=1024,h=256,v=512` is closed.
