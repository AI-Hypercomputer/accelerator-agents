---
title: "MaxText-CE climb: porting MaxText's cross-entropy cracked the Qwen3-8B jax seq8192 batch wall"
type: analysis
tags: [qwen3-cc, jax, maxtext, cross-entropy, custom-vjp, mfu-gap, closing, analysis]
model: qwen3-cc
created: 2026-06-02
updated: 2026-06-02
---

# MaxText-CE climb — the cross-entropy was the missing lever (Qwen3-8B jax, v6e-8)

> [!note] (2026-06-16) The MaxText MFU target referenced here (45.3% / 38.0%) is **non-causal** and overstates the gap — causal-adjusted it's **39.8% / 36.6%**. See the [qwen3-cc-maxtext caveat](../models/qwen3-cc-maxtext.md).

Closing synthesis of the arc that ran from the user directive **"use the same approach as MaxText for CE."**
It supersedes the [hard-wall closing](2026-06-02-qwen3-cc-jax-maxtext-closing.md): that page declared the
seq8192 batch a structural hard wall — **wrong**. Porting MaxText's exact cross-entropy proved the wall was a
cross-entropy-kernel artifact, cracked it, and moved the seq8192 frontier from 81% → **86.9%** of MaxText.

## What MaxText's CE actually is

Not a vocab-tiled Pallas kernel, and **not** the tokamax streamed CE we had been using. Reading
`MaxText/max_utils.py:557` + `train.py:291`: it is the **T5X `@jax.custom_vjp cross_entropy_with_logits`** over
the full logits with one-hot targets and an **explicit fused backward** (`deriv = softmax − onehot`, computed in
a single pass), z_loss optional (0 here). Ported verbatim into `train.py` as `--use_maxtext_ce`; CPU-verified
**bit-identical** to the prior `_ce` (|Δloss|=0, |Δgrad|=0) — semantics preserved.

## The arc (v033 → v037)

| v | shape / config | tok/s/chip | MFU | verdict |
|---|----------------|-----------|-----|---------|
| v028 | scan+overlap, bs1 (old seq8192 frontier) | 5,632 | 32.3% | (baseline) |
| v033 | maxtext-CE, **bs1** seq8192 | 5,656 | 32.4% | supported — **parity** (CE swap clean) |
| v034 | maxtext-CE, **bs2** seq8192 | **5,992** | 34.4% | supported — **WALL CRACKED** (+6.4% vs bs1) |
| v035 | maxtext-CE, **bs3** seq8192 (no offload) | **6,030** | **34.6%** | supported — **frontier**; fits w/o offload |
| v036 | maxtext-CE + named-offload, bs3 | 4,908 | 28.2% | refuted — offload −18.6% (not pipelined) |
| v037 | maxtext-CE, **seq2048** bs4 | 6,486 | 33.4% | refuted — **−6.9%** (seq8192-specific) |

## Why it cracked the wall

The prior closing's "hard wall" was the monotone seq8192 batch series bs1 5,632 > bs2 5,553 > bs3 4,595 — which
held only with the **tokamax/autodiff CE backward**. That CE backward (autodiff through `log_softmax`, or the
tokamax streamed kernel with its f32[H,V] weight-gather) carried a large memory **transient** that, at bs>1 with
the seq8192 logits, forced extra remat/spill and made batch *anti*-amortize. MaxText's custom_vjp computes the
logit gradient in one fused elementwise pass with no weight-gather → lower transient → **batch amortizes**
(v034 bs2 5,992 > v028 bs1 5,632; v035 bs3 6,030). It also let bs3 fit **without offload** (the tokamax 4.6G
weight-gather that forced [v030](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v030-scan-offload-ce-s8k-bs3.md)'s
offload is gone), beating v030's offloaded bs3 (4,595) by +31%.

**It is seq8192-specific** (v037): at seq2048 the custom_vjp's full one-hot `[BL,V]` materialization + saved
residuals aren't repaid when the logits are 4× smaller and the step is short → −6.9%. So the lane uses
`--use_maxtext_ce` at seq8192 and plain `_ce` at seq2048 — exactly parallel to scan (seq8192 win / seq2048
regress).

## Final frontiers vs MaxText

| shape | jax frontier (this lane) | MaxText ref | jax % of MaxText |
|-------|--------------------------|-------------|------------------|
| **seq8192** | **v035: 34.6% / 6,030 tok/s/chip** (maxtext-CE, scan, overlap, bs3) | 45.3% / 6,942 (bs3) | **86.9%** (was 81% / v028) |
| **seq2048** | v018: 35.8% / 6,964 (plain `_ce`, bs4) | 38.0% / 7,116 (bs4) | 97.9% |

## Residual seq8192 gap (6,030 → 6,942, ~13%) — honest attribution

Per the [v035 profile-analyzer pass](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v035-maxtext-ce-s8k-bs3.md):

1. **Remat recompute (~85%, +454 ms)** — our `nothing_saveable` recomputes every activation; RMSNorm/loop-fusion
   is 17.4% of the step vs MaxText's 7.2%. The obvious fix — named host-offload — **was refuted** (v036, −18.6%):
   our `pinned_host` offload is **not pipelined** (host round-trip on the critical path, vs MaxText's <0.1%
   overlapped copy), and it tags proj/mlpwi, not the norms that recompute. Closing this needs a **pipelined host
   offload** = XLA/kernel-authoring, **out of scope** for config/refactor iteration.
2. **MXU occupancy (~29%, +156 ms)** — 53.6% vs MaxText 61.2%, from per-matmul GeMM tile alignment
   (`logical_axis_rules`). A sharding-layout rewrite, medium-high effort, uncertain payoff — **deferred**
   ([qwen3-jax-seq8192-kernel-gap](../hypotheses/qwen3-jax-seq8192-kernel-gap.md)).
3. Reduce-scatter is **already better** than MaxText (windowed-einsum) — not a lever.

## Verdict

The user's CE directive was the right call and corrected my premature "hard wall." **maxtext-CE moved the
seq8192 frontier 5,632 → 6,030 (+7.1%), cracked the batch wall, and reached 86.9% of MaxText** — semantics
preserved. The remaining ~13% is genuinely kernel-authoring territory (a pipelined host offload) plus an
uncertain sharding lever (MXU/logical-axis, ~+3-4%). Both are beyond flag/refactor scope; the practical jax
frontier is **v035 (seq8192) + v018 (seq2048)**, with `--use_maxtext_ce` the seq8192 default.

## See also

- [Hard-wall closing (superseded)](2026-06-02-qwen3-cc-jax-maxtext-closing.md) · [MaxText gap teardown](2026-06-02-maxtext-vs-jax-qwen3-8b-mfu-gap.md)
- [Qwen3 8B — jax](../models/qwen3-cc-jax.md) · [maxtext-CE hypothesis](../hypotheses/qwen3-jax-maxtext-ce.md)
- Experiments: [v033](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v033-maxtext-ce-s8k-bs1.md) · [v034](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v034-maxtext-ce-s8k-bs2.md) · [v035](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v035-maxtext-ce-s8k-bs3.md) · [v036](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v036-maxtext-ce-offload-s8k-bs3.md) · [v037](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v037-maxtext-ce-s2k-bs4.md)

## Sources

- xprof traces: `raw/profiles/2026-06-02-qwen3-jax-v033..v037-*/` · MaxText reference per the gap teardown.
