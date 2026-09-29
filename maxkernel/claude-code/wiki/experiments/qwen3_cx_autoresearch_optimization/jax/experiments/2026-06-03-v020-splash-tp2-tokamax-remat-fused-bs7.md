---
title: "Qwen3 JAX v020 - splash TP2 tokamax remat fused bs7"
type: experiment
hypothesis: "Per-chip batch 7 in the v015 TP2 stack may recover useful HBM margin versus batch 8 while retaining closer MFU to the frontier than batch 6."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v6e8-qwen3-8b-jax-20260602-v015-splash-tp2-tokamax-remat-fused-rope-bs8
status: invalid
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax, remat, fused-rope, tp2, batch-scaling, invalid, oom, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v020 - splash TP2 tokamax remat fused bs7

## Hypothesis under test

**Hypothesis**: Per-chip batch 7 in the v015 TP2 stack may recover useful HBM
margin versus batch 8 while retaining closer MFU to the frontier than batch 6.

**Mechanism**: Reuse the v015 image and successful XLA flag stack. Keep
`fsdp=4,tp=2`, Splash attention, tokamax CE, remat, and fused RoPE, but set
`--batch_size=7`. Global batch becomes 28.

**Predicted signal**: The run should complete, land between v018 and v015 in
throughput/MFU, and leave roughly GiB-scale HBM margin. If it fits with >=30%
MFU and >1 GiB free HBM, it becomes the best short-sequence launch point for
longer-sequence probes.

**Falsification criterion**: Compile/runtime failure, peak HBM near the v015
ceiling with little margin, or MFU no better than v018.

## Setup

Planned GKE workload `alekseyv-qwen3-v020`; first launch was interrupted with
exit 143 before train-step compile finished, so the active retry uses workload
`alekseyv-qwen3-v020b`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v015-splash-tp2-tokamax-remat-fused-rope-bs8`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v020b-splash-tp2-tokamax-remat-fused-bs7`
- **Mesh**: `fsdp=4,tp=2`
- **Command shape**: `--batch_size=7 --seqlen=2048 --tp_parallelism=2 --use_splash=True`
- **Global batch**: 28
- **Profile window**: steps 12-14

## Results

Invalid: the retry workload `alekseyv-qwen3-v020b` failed during
`jit_train_step` compile with HBM OOM on both workers.

| Metric | Value |
|--------|-------|
| Exit code | 1 |
| Failure point | `jit_train_step` compile |
| Per-chip batch | 7 |
| Global batch | 28 |
| HBM used | 35.61 GiB |
| HBM capacity | 31.25 GiB |
| Over capacity | 4.37 GiB |
| Program HBM | 29.89 GiB |
| Arguments | 5.72 GiB |
| HLO temp | 29.86 GiB |

The result is counterintuitive but clear: the odd per-chip batch 7 shape is
worse than the known-fitting batch 8 shape. The compile report shows the largest
temp in tokamax backward (`fusion.4219.remat5`, 2.03 GiB), followed by a 593.50 MiB
all-gather, and many 168.00 MiB async collective fusion temps. The additional
argument/output footprint pushes total HBM to 35.61 GiB.

## Profile

No runtime profile was produced because compile failed before step 0.

## HLO Dump

The OOM dump is under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v020b-splash-tp2-tokamax-remat-fused-bs7/hlo/`

It contains 947 objects totaling 315.32 MiB, including
`module_0209.jit_train_step.cl_854318611.oom_intermediate_module.txt` and
`module_0209.jit_train_step.cl_854318611.oom_top_memory_consuming_instructions`.

## Verdict

Invalid.

Do not use odd per-chip batch 7 for this TP2 stack. It does not interpolate
between v018 batch 6 and v015 batch 8; instead it triggers a worse compile
layout with 35.61 GiB total HBM. Continue memory-shape exploration with even
per-chip batches.
