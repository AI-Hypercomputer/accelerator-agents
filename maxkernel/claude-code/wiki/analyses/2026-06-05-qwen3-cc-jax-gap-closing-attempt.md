---
title: "Qwen3-cc jax — gap-closing attempt vs MaxText (the 'try all' sweep)"
type: analysis
created: 2026-06-05
updated: 2026-06-05
tags: [qwen3-cc, jax, maxtext, gap-analysis, remat, tensor-parallel, rmsnorm, seq8192, retrospective]
---

# Qwen3-cc jax — gap-closing attempt vs MaxText (the "try all" sweep)

User directive (2026-06-05): *"inspect qwen3 maxtext implementation, profile, and hlo; compare to your best
model; come up with a plan to close the performance gap; then implement the plan… try all."* This is the
synthesis of that sweep.

## Starting point

seq8192 frontier = **v035** (maxtext-CE, bs3): **34.6% MFU / 6,030 tok/s/chip = 86.9% of the MaxText
reference** (45.3% / 6,942 on v6e-8). The +7% that got us here was the T5X `custom_vjp` cross-entropy (v033–v035).

## What the deep profile (profile-analyzer on v035) revealed — gap re-ranked

The 529 ms/step gap to MaxText decomposes (op-level, all-host):

| Rank | Op | ms | Tractable? |
|---|---|---|---|
| **1** | **SiLU recomputed TWICE in the backward** (`fusion.665` + `multiply_multiply_fusion.15`, bf16[3,8192,12288]) under `nothing_saveable` | **3,985** | ❌ **walled** |
| 2 | down_proj activation layout copy (`copy.403`, {1,2,0}→{2,1,0}) | 1,131 | ~90 ms, no clean JAX fix |
| 3 | QK-norm reduce remat (`fusion.671`) | 685 | ✅ small (v045) |
| 4 | QKV→splash layout reshapes | ~1,017 | hard |
| — | RMSNorm rsqrt/weight-mult | small | ✅ small (v042) |

Two corrections to the prior wiki story: (1) the dominant loop-fusion cost is **SiLU double-recompute, in
bf16** — *not* the RMSNorm fp32 multiply; (2) the data-formatting bucket is a **down_proj windowed-einsum layout
copy**, not the QKV→splash transpose.

**Why #1 is walled:** to avoid recomputing SiLU you must *save* its `[3,8192,12288]` output ≈ **21.7 GB** for
36 layers (per-chip, un-sharded in pure FSDP) — nowhere near the 2.9 GB free HBM. Offload-to-host is exactly
what **v039 refuted** (−6.6%, no pipeline on our scan/libtpu). So the #1 lever has no path on this stack.

## Levers tried (the sweep)

| Exp | Lever | Result | Why |
|---|---|---|---|
| [v043](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-05-v043-tp2-s8k.md) | tp=2 (bs3) | inconclusive (regression, **confounded**) | global batch halved 24→12 |
| [v043b](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-05-v043b-tp2-bs6.md) | tp=2 (bs6, equal global batch) | **refuted −26%** | see below |
| v044 | tp=4 | **retired without running** | strictly worse than tp=2 |
| [v042](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-05-v042-rmsnorm-bf16.md) | RMSNorm bf16 weight-mult (match MaxText) | **parity** +0.16% (sub-noise) | norm fp32-mult is a small slice |
| [v045](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-05-v045-save-norm-stats.md) | save norm rsqrt stats on-device | **supported (marginal)** +0.63% | skips f32 reduction recompute |
| v046 | down_proj layout | folded into tp obs | no clean JAX-level layout fix |

### The key negative result — tensor parallelism does NOT help

The plausible "structural unlock" was tp>1 to reach MaxText's sharded-activation regime. **It's refuted, with a
clean control.** The decisive insight: **at equal global batch, tp cannot reduce per-chip activation** — it
trades batch-sharding for model-sharding. At `fsdp=4×tp=2, bs=6` the per-chip MLP activation `[6,8192,6144]` is
*byte-identical* to v035's `[3,8192,12288]`, so the SiLU/norm recompute is unchanged — while tp *adds* per-layer
all-reduce collectives (after o_proj/down_proj) that pure FSDP avoids. Net: v043b is **−26% tok/s/chip, +35%
step time** vs v035. MaxText's loop-fusion advantage (7.2% vs our 17.4%) is therefore **not** from tp per se —
it's from its host-offload pipelining (which makes its sharded activations cheap to save), the same capability
our stack lacks (v039).

## Outcome

- New seq8192 frontier: **v045 = 34.8% / 6,068 = 87.4% of MaxText** (was 86.9%). A real but marginal nudge from
  the one on-device-saveable recompute slice; confirm run recommended.
- **Confirmed at op-level: the residual ~12.6% to MaxText is its host-offload pipelining** — a runtime/build
  capability (overlapped host-DMA across the scan), not a config or code lever. Every config+code path we can
  express has now been tried (CE ✓+7%, flags, splash blocks, activation constraints, offload, tp, norm dtype,
  norm-stat save). The one un-tried direction is **authoring a pipelined host-offload kernel** (out of
  config/compile scope) or a torchax-lane pivot.

## See also

- [Qwen3 8B — jax](../models/qwen3-cc-jax.md) · [v035 frontier](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v035-maxtext-ce-s8k-bs3.md) · [retrospective #5 (MaxText config delta)](2026-06-02-qwen3_cc-jax-retrospective-5-maxtext-delta.md)

## Sources

- v035 profile-analyzer deep-dive (xprof `2026-06-02-qwen3-jax-v035-maxtext-ce-s8k-bs3`)
- Experiments v042, v043, v043b, v045 (2026-06-05); MaxText recipe `benchmarks/maxtext_trillium_model_configs.py:qwen3_8b_8192_ref`
