---
title: "Qwen3 jax — splash block-config sweep at the frontier shape (fused bwd already on)"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: refuted
expected_gain: "up to ~5% step (splash bucket is ~26% of frontier step; llama3 autotune measured +30% kernel-time at sym-1024 blocks)"
confidence: medium
effort: S
origin: 2026-06-12-v025-offload-scan-bs3
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — splash block-config sweep at the frontier shape

*Scope correction (2026-06-12)*: code inspection shows our `splash_attn.py`
**already runs the fused bwd kernel by default** (`SPLASH_FUSED_BWD=1`) and
all block sizes are env-var-driven (`SPLASH_BQ=2048, SPLASH_BKV=1024,
SPLASH_*_DKV=2048`) — so this is a **launch-only block sweep**, not a code
change.

*Hypothesis*: At the bs4 @ seq 8192 frontier shape, either the llama3
autotune winner (symmetric 1024 blocks everywhere) or the MaxText match
(2048 everywhere) beats our current 2048/1024-hybrid by ≥ 1% step
(tok/s/chip > 6,100 vs v027's 6,040).

*Rationale*: splash is 26.3% of the frontier step (1,088 ms at bs3,
profile-named top recoverable cost). llama3's kernel autotune: `block_q=
block_kv=1024, fused_bwd` = +30% kernel time vs production config. MaxText
runs `sa_block_*=2048` fused at this exact shape at 3,535 ms/step. Counters:
v016 (bkv 2048 fwd-only, bs1) and v022 (tokamax knobs, bs1) were both flat —
but at bs1 without scan/offload.

*Falsification criterion*: both sweep arms within ±1% of v027's 6,040
tok/s/chip → refuted (block config saturated on this stack — third flat
result; close the kernel-config direction).

> [!warning] Refuted by [v028](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v028-splash-blocks-bs4.md) on 2026-06-12
> Phase A (sym-1024): 5,892 tok/s/chip (−2.5%). Phase B (all-2048): 6,044 (+0.1%).
> Both within ±1% of the 6,040 bar. This is the third flat splash block-config
> result (v016, v022, v028). The splash cost (26.7% / 1,448 ms/step at bs4) is
> fundamental arithmetic — not a tile-scheduling artifact. Confirmed finding:
> bq=2048 beats bq=1024 by ~142 ms/step; v027's hybrid config (bq=2048, bkv=1024,
> dkv=2048) already had the optimal block sizes. Splash direction fully closed.

## See also

- [v028 (refuted — sym-1024 −2.5%, all-2048 +0.1%)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v028-splash-blocks-bs4.md)
- [v025 (profile: splash bucket 23.8%)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v025-offload-scan-bs3.md)
- [v016 (bkv 2048 flat at bs1)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v016-splash-bkv2048.md)
- [v022 (tokamax perf knobs neutral at bs1)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v022-tokamax-splash.md)
