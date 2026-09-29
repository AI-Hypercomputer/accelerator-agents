---
title: "Qwen3-jax: MaxText/T5X custom_vjp cross-entropy"
type: hypothesis
model: qwen3-cc-jax
variants: ["8B/v6e-8"]
status: supported
expected_gain: "breaks the bs>1 seq8192 wall: +6.4% (v034 bs2 vs v028 bs1)"
confidence: high
effort: S
origin: human
hlo_prefilter: n/a
tags: [qwen3-cc, jax, cross-entropy, maxtext, custom-vjp, seq8192]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-jax: MaxText/T5X custom_vjp cross-entropy

Replace our CE — autodiff through `jax.nn.log_softmax` over the full `[B,L,V]` logits (the `_ce`
path used by the v028 frontier) — with **MaxText's exact CE recipe**: the T5X-style
`@jax.custom_vjp cross_entropy_with_logits` with one-hot targets and an explicit, fused stable
backward, `z_loss=0`.

## Statement

On 8B/v6e-8 at seq8192, swapping in MaxText's `custom_vjp` CE (verbatim port of
`MaxText/max_utils.py:cross_entropy_with_logits`) leaves the loss/grad numerically identical
(verified: |Δloss|=0, |Δgrad|=0 on CPU) and **does not regress** step time at bs1 vs v028
(5,632 tok/s/chip); and — the real target — its lighter, single-pass backward (`softmax − onehot`
in one fused kernel, vs autodiff threading through `log_softmax`) reduces the CE-backward transient
peak enough that **bs2 amortizes** (beats v028 bs1 5,632), cracking the documented seq8192 batch wall
that plain-CE bs2 (v031, 5,553) and bs3 (v030, 4,595) could not.

## Rationale

- **This is what MaxText actually does.** The [gap teardown](../analyses/2026-06-02-maxtext-vs-jax-qwen3-8b-mfu-gap.md)
  named "custom-vjp CE" as a MaxText advantage; reading the source ([`train.py:291`](../../?), `max_utils.py:557`)
  confirms it is the T5X `cross_entropy_with_logits` over full logits — **not** a vocab-tiled Pallas kernel,
  and **not** the tokamax streamed CE that forced the f32 lm_head-weight all-gather (4.64G) which blocked our
  bs3 (v027/v029). So "use the same approach as MaxText" = drop tokamax CE, use this custom_vjp.
- **Why it might break the wall**: the [closing analysis](../analyses/2026-06-02-qwen3-cc-jax-maxtext-closing.md)
  showed batch anti-amortizes (bs1 > bs2 > bs3) on plain CE. One candidate driver is the CE backward's
  transient: at seq8192 the logit-gradient is f32[BL,V]≈5G, and autodiff of `log_softmax` may stage it through
  extra temporaries, raising the transient peak (→ more remat/spill at bs>1). MaxText's custom_vjp computes the
  gradient in a single fused elementwise pass (`exp_shifted/sum_exp − targets`), lowering that peak. If the
  bs>1 regression was CE-backward-transient-bound, this fixes it; if it's collective/attention-bound, it won't
  (→ refuted, the wall stands).

## Proposed experiment

- **v033** — scan + overlap + `--use_maxtext_ce`, **bs1**, seq8192. Parity/no-regression anchor vs v028
  (5,632). Falsify: < 5,632 beyond noise ⇒ the custom_vjp graph is worse at bs1 (unexpected, since identical math).
- **v034** — scan + overlap + `--use_maxtext_ce`, **bs2**, seq8192. The real test vs v031 plain-CE bs2 (5,553)
  and v028 bs1 (5,632). Support: bs2 > 5,632 (wall cracked). Refute: bs2 ≤ v031 5,553 (CE swap doesn't change the
  batch wall — it's not CE-backward-bound).

## Risks

- **Semantics**: none expected — z_loss=0 makes it mathematically identical to `_ce`; CPU check confirms
  |Δ|=0 on loss and grad. If the TPU loss trajectory diverges from v028, verdict is `invalid`.
- **Memory**: the custom_vjp fwd saves more residuals (`exp_shifted`, `log_softmax`), but under remat
  (nothing_saveable) they are recomputed, so peak-residual is unchanged; only the backward *transient* differs.

## Dependencies

- Image `qwen3-8b-jax:v033-maxtext-ce` (FROM v030-scan-full, swaps `train.py` only). Flag `--use_maxtext_ce`.
- Builds on the v028 scan+overlap frontier config.

## See also

- [Qwen3 8B — jax](../models/qwen3-cc-jax.md)
- [MaxText-gap closing analysis](../analyses/2026-06-02-qwen3-cc-jax-maxtext-closing.md)
- [MaxText vs jax MFU gap](../analyses/2026-06-02-maxtext-vs-jax-qwen3-8b-mfu-gap.md)
