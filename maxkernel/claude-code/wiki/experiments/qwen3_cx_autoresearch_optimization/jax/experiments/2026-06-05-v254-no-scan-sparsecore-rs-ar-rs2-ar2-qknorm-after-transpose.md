---
title: "Qwen3 JAX v254 - QK-norm after transpose"
type: experiment
hypothesis: "Applying Qwen3 Q/K RMSNorm after the q/k transpose may expose a better QK-norm+RoPE layout to XLA without changing math."
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: image-v254-qknorm-after-transpose
status: invalid
tags: [qwen3-cc, jax, flax-nnx, no-scan, remat, activation-sharding, sparsecore, collectives, reduce-scatter, all-reduce, qk-norm, rope, layout, splash, tokamax-splash, maxtext-ce, long-seq, v6e-8]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3 JAX v254 - QK-norm after transpose

## Hypothesis under test

**Hypothesis**: Qwen3's Q/K RMSNorm is currently applied on q/k tensors in
`(B,T,H,D)` layout, then transposed to `(B,H,T,D)` before RoPE and Splash
attention. RMSNorm reduces only over the last `D` dimension, so applying it
after the transpose is mathematically equivalent but presents QK-norm and RoPE
to XLA in the same layout. This may remove a layout/fusion barrier around the
Qwen3-specific QK-norm+RoPE surface.

**Mechanism**: Patch only `Qwen3Attention.__call__` in a thin image derived from
`qwen3-8b-jax:v169-no-mlp-sac`.

- Existing `q_norm` and `k_norm` modules and parameters are unchanged.
- Existing `checkpoint_name(..., "query_proj"|"key_proj")` placement on q/k
  projection outputs is preserved.
- Only the order changes:
  projection -> reshape -> transpose -> RMSNorm -> RoPE.
- Keep MaxText CE, no-scan, remat, Tokamax Splash, activation sharding, and the
  v230 RS2+AR2 runtime stack fixed.

**No-fallback assumption**: There is no alternate attention path in the image.
If the layout change compiles to the same HLO or regresses, record the signal
directly.

**Predicted signal**: Support requires a clean finite-loss run above the current
v230/v236 frontier band (**59,749-59,750 tok/s / 42.9% MFU**) or a clear XProf
reduction in QK-norm/RoPE-adjacent fusion time at in-band throughput.

**Falsification criterion**: Compile/runtime failure, invalid loss, changed
semantics signal, or clean completion below the frontier.

## Setup

- Image:
  `<your-registry>/torchtitan-images/qwen3-8b-jax:v254-qknorm-after-transpose`
- Digest:
  `sha256:53d1fd3f552af51368c14380109c4d7800578db5efffdd43ef5a11b755ec1ef5`
- Thin image patch: `Qwen3Attention.__call__` only, extracted from base image
  `qwen3-8b-jax:v169-no-mlp-sac`.
- Syntax check: `python -m py_compile /app/trainer/model/modeling_qwen3.py`
  passed in the base image.
- Workload: pending launch as `alekseyv-q3-v254-qknorm-xpose` on
  `<your-cluster>`.
- Run dir:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v254-no-scan-sparsecore-rs-ar-rs2-ar2-qknorm-after-transpose`
- `XLA_FLAGS` used a concrete GCS HLO dump path.
- Mesh: `fsdp=8,tp=1`; global batch 32; profile window steps 12-14.

## Result

Invalid launch. The workload completed cleanly on both workers with
`EXIT_CODE=0` and finite synthetic loss ending at **12.0460**, but the XPK
launch omitted the frontier-required `USE_TOKAMAX_SPLASH=1` /
`TOKAMAX_MAX_LOGIT_CONST=30.0` environment variables. The optimized HLO confirms
the wrong attention path: `tokamax` text count 0.

Worker0:

- Average steps 2-19: **56,438 tok/s**, **7,055 tok/s/chip**, about **40.5%
  nominal MFU**
- Late/no-profile steps 15-19: 56,549, 56,485, 56,479, 56,442, 56,400 tok/s

Worker1:

- Average steps 2-19: **56,417 tok/s**, **7,052 tok/s/chip**, about **40.5%
  nominal MFU**
- Same finite-loss curve as worker0.

Because the runtime stack did not match v230/v236, this is not a valid
refutation of the QK-norm layout hypothesis. Rerun as v257 with the same image
and Tokamax Splash env restored.

## Profile

XProf run:
`2026-06-05-qwen3-jax-v254-no-scan-sparsecore-rs-ar-rs2-ar2-qknorm-after-transpose/2026_06_05_12_20_35`.

Summary:

- Step time: **4656.9 ms**
- MXU utilization: **64.3%**
- Peak HBM: **30.87 GiB** of 31.25 GiB
- Memory profile: 23.09 GiB stack reservation, 7.79 GiB heap allocation,
  0.37 GiB free

Top op buckets:

- `convolution fusion`: **28,814.8 ms / 51.6%**
- `custom-call`: **17,555.2 ms / 31.4%**
- `loop fusion`: **5,535.4 ms / 9.9%**
- `data formatting`: **1,481.3 ms / 2.7%**
- `custom fusion`: **1,139.3 ms / 2.0%**
- `reduce`: **499.9 ms / 0.9%**
- `all-gather`: **367.5 ms / 0.7%**

The layout change did not expose a better QK-norm/RoPE schedule. MXU falls
relative to the frontier and the custom-call bucket rises to 31.4%.

## HLO Dump

Train-step HLO:

- GCS:
  `gs://<your-bucket>/autoresearch/qwen3_cc/2026-06-05-qwen3-jax-v254-no-scan-sparsecore-rs-ar-rs2-ar2-qknorm-after-transpose/hlo/xla-dump/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- Local analysis copy:
  `/tmp/qwen3-v254-hlo/module_0267.jit_train_step.cl_854318611.after_optimizations.txt`
- SHA256:
  `a5ce1755308298feb3ef35f7ae7d74b34ab2efe6d307e7101e68ed48f0946d84`
- Size: **19,013,836 bytes**
- Lines: **113,720**

Compiled memory report:

- Total bytes used: **27.67 GiB**

## Verdict

**Invalid.** The missing Tokamax Splash env changed the tested runtime/HLO, so
v254 is not evidence for or against the QK-norm-after-transpose layout change.
Do not compare this throughput to v230/v236. Rerun as v257 with the correct env.

The broader QK-norm+RoPE direction likely needs an actual fused kernel or deeper
HLO-guided rewrite; this simple layout reorder is not sufficient.
