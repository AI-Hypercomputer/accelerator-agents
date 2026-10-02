---
title: "MaxText Qwen3-8B v6e-8 reference @ seq2048 (38.0% MFU / 7,116 tok/s/chip)"
type: experiment
hypothesis: qwen3-maxtext-reference-baseline
model: qwen3-cc-maxtext
variant: "8B/v6e-8"
commit: 9f1820b47
verdict: supported
status: supported
tags: [qwen3-cc, maxtext, v6e-8, reference, baseline, seq2048]
created: 2026-06-02
updated: 2026-06-02
---

# MaxText Qwen3-8B v6e-8 reference @ seq2048

> [!note] The **38.0% MFU is non-causal** — `tpu-recipes-v0.1.4` predates MaxText's causal-mask `/2` fix (commit `6288c233`, 2026-04-11). Causal-adjusted = **36.6% MFU** (same 7,116 tok/s/chip), the basis the jax/torchax lanes use. See the [qwen3-cc-maxtext model-page caveat](../../../../models/qwen3-cc-maxtext.md). Numbers below are left as-recorded (immutable).

## Hypothesis under test

**Hypothesis**: MaxText Qwen3-8B at **seq2048 bs4** (the exact shape of the jax v018 frontier) establishes
the reference ceiling for that shape, quantifying how close our hand-tuned jax frontier (35.8% MFU / 6,964
tok/s/chip) is to what MaxText achieves.

**Mechanism**: `benchmark_runner xpk --model_name=qwen3_8b_2048_ref` (= the seq8192 recipe with
`max_target_length=2048`, `per_device_batch_size=4`). Same worktree/venv/base-image as the
[seq8192 reference](2026-06-02-maxtext-qwen3-8b-v6e8-ref-seq8192.md).

## Setup

- **Workload**: `ale-qwen3-8b-2-1-060214-lgr`; pod `...-slice-job-0-0-q5zbn`. v6e-8, fsdp=8.
- **Config** `qwen3_8b_2048_ref`: `per_device_batch_size=4`, `max_target_length=2048`, attention=flash,
  remat custom + decoder/proj offloads, `sa_block_*=2048` + fused bwd, synthetic, profiler xplane
  (skip 10 / 5 steps), steps=20. Global batch = 4×8×2048 = 65,536 tokens/step (= `total_weights`).
- Same MaxText XLA flag stack as the seq8192 run.

## Baseline comparison

**Same shape (seq2048 bs4), same hardware — direct cross-stack:**

| Stack | MFU | tok/s/chip | step (ms) | jax % of MaxText (tok/s/chip) |
|-------|-----|-----------|-----------|-------------------------------|
| **MaxText (this run)** | **38.0%** | **7,116** | 1,151 | — |
| jax v018 (our frontier) | 35.8% | 6,964 | 1,154 | **97.9%** |

**The jax seq2048 frontier is at near-parity with MaxText — only ~2% / +152 tok/s/chip behind.** This
externally validates the [jax retrospective](../../../../analyses/2026-06-02-qwen3_cc-jax-retrospective-2.md)
conclusion that the jax **seq2048** frontier is at its practical ceiling. Contrast with seq8192, where the
gap is ~31% (jax 5,305 vs MaxText 6,942) because MaxText fits bs3 via host-offload and our lane was bs1.

## Results

Steady-state (steps 16–19; step 15 = 8.219 s profile-stop overhead, excluded):

| Metric | Value |
|--------|-------|
| TFLOP/s/device (median) | **348.9** (range 348.8–349.0) |
| **Per-chip MFU** | **38.0%** (348.9 / 918) |
| Tokens/s/device (median) | **7,116** (range 7,114–7,117) |
| Step time (median) | **1.151 s** |
| loss | 0.083 → 0.035 (synthetic, seq2048 memorizes fast) |
| exit | clean (`Completed`, ~2m46s wall) |

Note MaxText's own seq2048 MFU (38.0%) is **below** its seq8192 MFU (45.3%) — expected: at shorter seq,
attention is a smaller FLOP fraction and collective/overhead per token is relatively larger.

## Profile

- **Run name**: `ale-qwen3-8b-2-1-060214-lgr`
- **On-disk pointer**: [`raw/profiles/2026-06-02-maxtext-qwen3-8b-v6e8-ref-seq2048/`](../../../../../raw/profiles/2026-06-02-maxtext-qwen3-8b-v6e8-ref-seq2048/)
- **GCS**: `.../2026-06-02-qwen3-8b-2048-ref/ale-qwen3-8b-2-1-060214-lgr/tensorboard/plugins/profile/2026_06_02_14_02_35/` (`gke-tpu-964065d9-9vh8.xplane.pb` + `.trace.json.gz`, rank-0; steps 10–14).
- xprof: point `--logdir` at `.../ale-qwen3-8b-2-1-060214-lgr/tensorboard`. No HLO dump.

## Verdict

**Supported** — reference established. MaxText Qwen3-8B @ seq2048 bs4 = **38.0% MFU / 7,116 tok/s/chip**.
**Our jax v018 frontier (6,964 tok/s/chip) is at 97.9% of this — effectively at parity.** This is the
external confirmation that the jax seq2048 optimization is genuinely complete (the ~2% residual is within
noise + MaxText's heavier offload recipe). The actionable headroom for the jax lane is **only at seq8192**
(the ~31% gap, via the offload-enabled bs3 — [host-offload hyp](../../../../hypotheses/qwen3-jax-host-offload-bs3-seq8192.md)), not seq2048.

## Next hypotheses

- None for seq2048 — jax is at parity with the MaxText ceiling here; no further jax seq2048 work is warranted.
- [jax host-offload bs3 @ seq8192](../../../../hypotheses/qwen3-jax-host-offload-bs3-seq8192.md) — the seq8192 gap is the real remaining jax opportunity (this seq2048 result confirms seq2048 is closed).
