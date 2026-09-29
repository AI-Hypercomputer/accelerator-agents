---
title: "Qwen3 MaxText rerun and MFU normalization"
date: 2026-06-12
type: analysis
models: [qwen3-cc]
lanes: [jax, torchax, maxtext]
hardware: v6e-8
---

# Qwen3 MaxText rerun and MFU normalization

## Result

Fresh MaxText rerun `alekseyv-qwen3-maxtext-v007-bs3r` reproduced the prior
bs3 recipe. A follow-up MaxText `base-b` run under the `qwen3_cc5` root
improved the runtime reference by reducing temp pressure. Both completed 20
steps on `<your-cluster>`.

- Output root: `gs://<your-bucket>/autoresearch/qwen3_maxtext/2026-06-12-qwen3-8b-v6e8-recipe-v007-bs3-rerun/alekseyv-qwen3-maxtext-v007-bs3r/`
- Profile: `tensorboard/plugins/profile/2026_06_12_06_07_07/`
- XProf summary: **3611.7 ms** average step, **62.2% MXU**, **0.3% idle**.
- Runtime: steady post-profile steps 15-19 at **409.2-410.0 TFLOP/s/device**,
  **6831-6845 tok/s/device**, **44.6% MFU** by MaxText's printed formula.
- HLO: `module_0103.jit_train_step.cl_759148519.after_optimizations.txt`,
  **1,498,186 bytes / 8,013 lines**, SHA256
  `598fe616e4031e66c7a0709c0e0dfd04bec2566041590378c0a4fb1ea931cf84`;
  byte-identical to the 2026-06-05 v006 MaxText HLO.
- HLO counts: `dot_general=387`, `fusion=1190`, `copy=667`,
  `all-gather=185`, `all-reduce=100`, `reduce-scatter=20`,
  `convolution=122`, `custom-call=68`, `splash=53`, `_one_hot=8`.
- Memory: XProf peak **25.81 GiB / 31.25 GiB**; XLA memory report shows
  **48.20 GiB** total including **23.62 GiB** host/offload color and
  **13.09 GiB** device temp.

Follow-up `alekseyv-qwen3-cc5-maxtext-v001-base-b`:

- Output root: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-maxtext-v001-base/mt-v001-base-b/`
- Profile: `tensorboard/plugins/profile/2026_06_12_06_18_19/`
- Pre-profile steady steps 1-9: **417.7-419.2 TFLOP/s/device**,
  **6973-6998 tok/s/device**, about **45.5-45.7% MFU**.
- Post-profile steps 15-19: **416.7-417.7 TFLOP/s/device**,
  **6956-6973 tok/s/device**, about **45.4-45.5% MFU**.
- XLA memory report: **22.3 GB** total device memory use, with **10.8 GB**
  temp and **23.6 GB** host temp. This is lower device temp than v007
  (**13.1 GB**) and likely explains the small throughput lift.

## Profile Shape

XProf op profile is dominated by the train step:

- `jit_train_step`: **99.7%** of profile compute time.
- Convolution fusion: **49.6%** of total profile time, all recorded model
  FLOPs.
- Splash custom-calls: **32.6%** total profile time; fwd residuals are about
  **18.8%**, DKV about **13.9%**.
- Idle is negligible (**0.3%**), so the recipe is compute/custom-call bound,
  not host-gap bound.

## MFU Formula Audit

The native Qwen3 JAX and torchax trainers previously used causal triangular
attention FLOPs:

```text
attention = 4 * B * L^2 * num_query_heads * head_dim / 2
```

The MaxText recipe image used for v006/v007 reports Qwen3-8B bs3 seq8192 as
**1472.20 TFLOPs/device-step** and splits that as **75.80% learnable** /
**24.20% attention**. The learnable term matches our code exactly:
**1115.96 TFLOPs/device-step**. The difference is attention: MaxText is
counting the full non-causal QK/AV matmul term, **356.24 TFLOPs**, not the
triangular causal term, **178.12 TFLOPs**.

Code changed in:

- `wiki/experiments/qwen3_cx_autoresearch_optimization/jax/train.py`
- `wiki/experiments/qwen3_cx_autoresearch_optimization/torchax/train.py`

The new reporting convention is:

```text
attention = 4 * B * L^2 * num_query_heads * head_dim
```

This is reporting-only. Runtime still uses causal masking. The print label now
says `MaxText-style` to make the convention explicit.

## Comparison After Normalization

The old native-JAX v273 full-Qwen3 frontier remains **60,346 tok/s aggregate /
7,543 tok/s/chip**. Under the MaxText-style FLOP counter, it is about
**49.2% MFU**, compared with MaxText's updated **45.4-45.7% MFU** reference
from `base-b`.

The newer `qwen3_cc5` lane is a different bs1 seq8192 memory-stability lane,
so it should not replace v273 as the full-throughput frontier. Its current best
is v017/v019/v022 in the **5,813-5,887 tok/s/chip** band. Under MaxText-style
MFU, v022's old **33.3%** becomes about **37.9%**.

## Gap Assessment

The apparent "native-JAX lower MFU than MaxText" gap was mostly formula
normalization. On the full-throughput lane, native-JAX already exceeds MaxText
in tokens/sec and exceeds it after using the same FLOP convention.

The remaining performance work should focus on the actual runtime lanes:

1. For the full v273 family, preserve the MaxText-style MFU counter and rerun a
   small validation once a fresh image includes the code change.
2. For `qwen3_cc5`, continue from v017/v019/v022, not v023. v023's SAVE_FFN
   probe regressed to **5,577 tok/s/chip**; v022 remains the best recent
   profiled same-lane result at **5,813 tok/s/chip**.
3. Do not chase MaxText HLO count differences as a primary gap yet: MaxText
   and native-JAX are running different objective shapes in `qwen3_cc5`, while
   the older full-throughput native v273 lane is already ahead by throughput.
