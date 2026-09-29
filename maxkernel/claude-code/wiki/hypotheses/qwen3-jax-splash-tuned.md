---
title: "Qwen3 jax — Splash attention tuned (BTHD + Tokamax overlap)"
type: hypothesis
model: qwen3-cc-jax
variants: ["8B/v6e-8"]
status: open
expected_gain: "1-2% MFU"
confidence: medium
effort: S
---

# Hypothesis
The previous `v004` Splash Attention run successfully unblocked `bs=2`, but lost MFU due to Ring Attention cross-device sync overhead (28.5% of step time) and `data formatting` layout tax (6.4%). By changing the QKV inputs to natively emit the `(B, T, H, D)` layout expected by Splash Attention and enabling the Tokamax experimental scheduler (`TOKAMAX_USE_EXP_SCHED=1`) for better communication overlap, we can eliminate the layout tax and hide the ring collective overhead, increasing MFU.

## Mechanism
1. **Layout**: Modify `Qwen3Attention` in `modeling_qwen3.py` to keep the native `(B, T, H, D)` shape after rotary embeddings instead of transposing to `(B, H, T, D)`. Update the `q_sharding` to `P("fsdp", None, "tp", None)` correspondingly.
2. **Scheduler**: Set `USE_TOKAMAX_SPLASH=1` and `TOKAMAX_USE_EXP_SCHED=1` in `train.py` to enable Tokamax's improved ring communication overlap scheduling.

## Expected Signal
- The `data formatting` bucket should drop to ~0% in the xprof op profile.
- The `collective-permute` buckets (start and done) should either shrink or be better hidden under compute, reducing the overall step time.

## Falsification
If step time does not improve (remains >390ms) or MFU stays at ~25%, then the layout tax was not the primary bottleneck or the Tokamax scheduler failed to overlap the P2P communication for this shape.
