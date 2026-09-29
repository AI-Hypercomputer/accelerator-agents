---
title: "qwen3-jax: chunked CE at seq-2048 as a batch unlock (bs6/bs8)"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: retired   # parity FAILED (v062: +0.006 systematic same-data offset; +4.1% CE cost at bs5); v061 bs8 +1.5% ruled invalid. Reopen only with a padding-masked kernel.
expected_gain: "5-15% tok/s/chip at seq2048 (via bs6/bs8 unlock)"
confidence: medium
effort: S
origin: 2026-06-12-v059-2kbs6
tags: [qwen3-cc5, jax, v6e-8, seq2048, chunked-ce, batch-scaling, charles-v6e]
created: 2026-06-12
updated: 2026-06-12
---

# Chunked CE at 2k — a memory lever, not a speed lever

The
[v059](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v059-2kbs6.md)
OOM proved the bs6 wall is the **plain-CE f32 logit pair** (fwd
`f32[6,2048,151936]` + log_softmax grad `f32[12288,151936]` ≈ 13.9 GiB,
~2.3 GiB/bs combined) — not remat residuals (non-CE slope ≈ 1.8 GiB/bs).
`tokamax chunked_xla` CE (f32-x rule, 8k-certified) eliminates the [BL,V]
materialization.

## Statement

On the v058 substrate + `--use_tokamax_ce=True --tokamax_ce_impl=chunked_xla`
(weights f32-cast at the boundary per the v008b numerics rule), bs6 fits and
yields tok/s/chip > 7,874 (the v058 bs5 frontier) — i.e. the CE step-cost
(+3.2% measured at bs2, v003-era) is smaller than the batch-amortization
gain of the unlocked rung(s).

## Rationale

- Envelope at bs6 + CE-cost: compute ~227 ms/bs × 6 + floor ~192 + CE ~3%
  ≈ 1,600 ms → ~7,680/chip — roughly v058-flat; **bs8** (if it fits once
  ~2.3 GiB/bs of logit pressure is gone: 27.28 − 2×2.3(CE part at bs5)
  ≈ 22.7 at bs5 → +2×~1.8 → bs8 ≈ ~28 GiB, plausible) ≈ 16,384 tokens /
  ~2,010 ms ≈ **8,150+/chip**. The win case is bs8; bs6 alone is likely
  ~flat.
- Risk: the +3.2% CE cost figure is from the old unrolled era at bs2; under
  scan+flags it may differ either way. The probe measures it directly.
- 8k precedent: chunked_xla f32-x is the certified 8k frontier CE — code
  already in trunk/image; launch-only.

## Proposed experiment

After the bs5 certification (v060): single run at **bs8** + chunked CE
(jump straight to the win case; if OOM, fall back bs6). 20 steps, profile,
loss-parity check vs the f32-x numerics rule (CE swap changes loss path —
verify series matches plain-CE within the v008b-established tolerance).

## Risks

CE implementation swap touches the loss computation — semantics check
mandatory (loss series + the f32-x rule). No other tracked-metric risk.

## Dependencies

[v060 bs5 certification](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v060-2kbs5val.md)
(frontier bookkeeping before the next axis).

## See also

- [v059 OOM breakdown](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v059-2kbs6.md) · [2k batch rescale](qwen3-jax-2k-batch-rescale.md) · [Model page](../models/qwen3-cc5-jax.md)
