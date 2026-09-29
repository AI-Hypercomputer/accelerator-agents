---
title: "Qwen3 JAX v019 - phase2 bundle cost model off"
type: experiment
hypothesis: "Disabling the TPU bundle-aware fusion cost model in the v015 TP2 stack may reduce fusion/layout overhead without changing model semantics."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v6e8-qwen3-8b-jax-20260602-v015-splash-tp2-tokamax-remat-fused-rope-bs8
status: invalid
tags: [qwen3-cc, jax, flax-nnx, splash, tokamax, remat, fused-rope, tp2, xla-flags, invalid, oom, v6e-8]
created: 2026-06-03
updated: 2026-06-03
---

# Qwen3 JAX v019 - phase2 bundle cost model off

## Hypothesis under test

**Hypothesis**: Disabling the TPU bundle-aware fusion cost model in the v015 TP2
stack may reduce fusion/layout overhead without changing model semantics.

**Mechanism**: Reuse the v015 image, shape, and TP2 sharding, but set
`--xla_tpu_use_bundle_aware_cost_model_for_fusions=false` in `LIBTPU_INIT_ARGS`.
Everything else stays at the v015 frontier shape: `fsdp=4,tp=2`, per-chip batch
8, global batch 32, seq_len 2048.

**Predicted signal**: If bundle-aware cost modeling is hurting this graph, the
run should complete with equal or better MFU than v015 and lower loop/data
formatting share in XProf.

**Falsification criterion**: Compile/runtime failure, lower MFU with unchanged
memory, or an XProf profile that shows no reduction in fusion/layout overhead.

## Setup

Live GKE workload `alekseyv-qwen3-v019`.

- **Image**: `<your-registry>/torchtitan-images/qwen3-8b-jax:v6e8-qwen3-8b-jax-20260602-v015-splash-tp2-tokamax-remat-fused-rope-bs8`
- **Run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v019-phase2-bundle`
- **Mesh**: `fsdp=4,tp=2`
- **Command shape**: `--batch_size=8 --seqlen=2048 --tp_parallelism=2 --use_splash=True`
- **Global batch**: 32
- **Changed flag**: `--xla_tpu_use_bundle_aware_cost_model_for_fusions=false`
- **Profile window**: steps 12-14

## Results

Invalid: both workers failed during XLA compile with `RESOURCE_EXHAUSTED`.

| Metric | Value |
|--------|-------|
| Exit code | 1 |
| Failure point | `jit_train_step` compile |
| HBM used | 34.00 GiB |
| HBM capacity | 31.25 GiB |
| Over capacity | 2.75 GiB |
| Program HBM | 34.00 GiB |
| HLO temp | 33.98 GiB |
| Fragmentation | 108.95 MiB |

The largest allocation was a 2.32 GiB tokamax backward dot fusion:
`fusion.4184.remat5`, followed by a 593 MiB all-gather temp. The OOM list also
contains many 192 MiB async collective fusion temps, indicating the flag change
materially worsened scheduling/memory placement versus the v015 fit.

## Profile

No runtime profile was produced because compile failed before step 0.

## HLO Dump

The failed compile produced HLO/OOM artifacts under:

`gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-03-qwen3-jax-v019-phase2-bundle/hlo/`

The dump contains 964 objects totaling 194.44 MiB, including
`module_0210.jit_train_step.cl_854318611.oom_intermediate_module.txt` and
`module_0210.jit_train_step.cl_854318611.oom_top_memory_consuming_instructions`.

## Verdict

Invalid.

Disabling the bundle-aware fusion cost model makes the v015 frontier shape fail
compile with a 34.00 GiB HBM program, where v015 fits at 31.23 GiB peak HBM. Keep
the bundle-aware cost model enabled for this lane.
