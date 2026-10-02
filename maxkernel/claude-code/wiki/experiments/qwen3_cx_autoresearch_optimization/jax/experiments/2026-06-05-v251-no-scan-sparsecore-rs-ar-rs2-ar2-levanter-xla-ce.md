---
title: "Qwen3 JAX v251 - Levanter XLA streaming CE"
type: experiment
hypothesis: "Levanter's XLA streaming CE custom-VJP may avoid the Pallas CE branch's generated fusion overhead while still avoiding full-logit materialization."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v251-levanter-xla-ce
status: refuted
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, levanter-ce, xla-ce, streaming-ce, ce, splash, tokamax-splash, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v251 - Levanter XLA streaming CE

## Hypothesis under test

**Hypothesis**: v248-v250 showed Levanter Pallas CE is valid but not a frontier.
The best Levanter Pallas point, v249, still trailed MaxText CE by about 1.3k
tok/s and retained high CE-adjacent fusion cost. Levanter also ships an XLA
streaming CE implementation with a custom VJP. It may avoid Pallas custom-call
and block-size pathologies while still avoiding full `[B,L,V]` logits.

**Mechanism**: Add a minimal `--levanter_ce_impl` switch to the existing v248
thin image. Default behavior remains `pallas_tpu`; this experiment runs
`--levanter_ce_impl=xla` explicitly.

- Base image: `qwen3-8b-jax:v248-levanter-ce`
- New image: `qwen3-8b-jax:v251-levanter-xla-ce`
- CE path: `--use_levanter_ce=True --use_maxtext_ce=False
  --levanter_ce_impl=xla`
- XLA CE block sizes: `b=1024,h=512,v=8192`
- Keep the v230 RS2+AR2 runtime stack, VMEM 100352, no-scan, remat,
  Tokamax Splash attention, activation sharding, and batch/sequence shape fixed.
- Use a concrete GCS path in `XLA_FLAGS` so the HLO dump lands in the run
  directory.

**No-fallback assumption**: The API call passes `implementation="xla"` as a
single explicit implementation. Levanter's dispatcher raises on explicit errors
rather than trying the next implementation, so this is a no-fallback test.

**Predicted signal**: Support requires clean compile/loss and improvement over
v249's **58,427 tok/s / 41.9% MFU**, ideally toward the current v230/v236
frontier (**59,749-59,750 tok/s / 42.9% MFU**), with lower CE-adjacent fusion
cost than Levanter Pallas.

**Falsification criterion**: Compile/runtime failure, invalid loss, or clean
completion below v249. A clean run below v249 closes Levanter CE as a near-term
replacement for MaxText CE.

## Setup

- Image: `<your-registry>/torchtitan-images/qwen3-8b-jax:v251-levanter-xla-ce`
- Digest: `sha256:d6f17afc476f1c8e741cf705217d6cb8b474d5a786e67e164a3b5f7f26ad7819`
- Thin image patch: add `--levanter_ce_impl` and pass it directly to
  Levanter fused CE instead of hard-coding `implementation="pallas_tpu"`.
- Syntax check: `python -m py_compile /app/trainer/train.py` passed in the
  v248 Levanter CE image.
- Workload: pending launch as `alekseyv-q3-v251-levxla-ce` on
  `<your-cluster>`.
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v251-no-scan-sparsecore-rs-ar-rs2-ar2-levanter-xla-ce`
- `XLA_FLAGS` used a concrete GCS HLO dump path.
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.

## Result

Completed cleanly on both workers with `EXIT_CODE=0` and finite synthetic loss
ending at **12.0487**.

Worker0:

- Average steps 2-19: **54,073 tok/s**, **6,759 tok/s/chip**, about **38.8%
  nominal MFU**
- Late/no-profile steps 15-19: 54,192, 54,095, 54,124, 54,055, 54,032 tok/s

Worker1:

- Average steps 2-19: **54,063 tok/s**, **6,758 tok/s/chip**, about **38.8%
  nominal MFU**
- Same finite-loss curve as worker0.

This is below v249's best Levanter Pallas result (**58,427 tok/s / 41.9% MFU**)
and far below the v230/v236 native-JAX frontier (**59,749-59,750 tok/s /
42.9% MFU**).

## Profile

XProf run:
`2026-06-05-qwen3-jax-v251-no-scan-sparsecore-rs-ar-rs2-ar2-levanter-xla-ce/2026_06_05_11_44_54`.

Summary:

- Step time: **4861.4 ms**
- MXU utilization: **61.8%**
- Peak HBM: **29.23 GiB** of 31.25 GiB
- Memory profile: 21.42 GiB stack reservation, 7.81 GiB heap allocation,
  2.02 GiB free

Top op buckets:

- `convolution fusion`: **57,451.7 ms / 49.3%**
- `custom-call`: **35,105.1 ms / 30.1%**
- `loop fusion`: **11,191.4 ms / 9.6%**
- `data formatting`: **4,112.2 ms / 3.5%**
- `custom fusion`: **3,949.7 ms / 3.4%**
- `pad`: **1,472.8 ms / 1.3%**
- `all-gather`: **1,149.6 ms / 1.0%**

The XLA streaming CE variant avoids the Pallas CE block-size/custom-call shape,
but it is not cheaper overall. Relative to v250/v249, device time shifts into
more custom-call, loop/data-formatting, and custom-fusion work, and MXU drops
from the mid/high-60s to **61.8%**. HBM is lower than the frontier, so memory
capacity is not the limiting signal.

## HLO Dump

Train-step HLO:

- GCS:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v251-no-scan-sparsecore-rs-ar-rs2-ar2-levanter-xla-ce/hlo/xla-dump/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Local analysis copy:
  `/tmp/qwen3-v251-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256:
  `b2fe8cf7500da79d467b0df5360d76f9788367a6afec3d2e84fa22f825ba75da`
- Size: **19,041,778 bytes**
- Lines: **114,165**

Compiled memory report:

- Total bytes used: **26.00 GiB**
- Dominant allocation: **20.25 GiB** preallocated temp

## Verdict

**Refuted.** Levanter XLA streaming CE is valid but much slower than the current
MaxText-CE frontier. It reduces compiled memory versus Pallas CE and the v230
frontier, but the device profile loses too much MXU and increases non-MXU work.

Close Levanter CE as the next replacement path for this frontier: the best
Levanter result is still v249, and v251 shows the XLA backend is not the missing
piece. The next attempt should move away from CE backend swaps toward source
layout/fusion changes or a Pallas kernel around the MLP/SwiGLU hotspot.
