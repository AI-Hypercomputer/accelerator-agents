---
title: "Qwen3-8B jax v029 — full stack (scan+offload+CE-bf16+overlap) @ seq8192 bs3"
type: experiment
hypothesis: qwen3-jax-host-offload-bs3-seq8192
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: jax-fullstack-2026-06-02
status: invalid
verdict: invalid
tags: [qwen3-cc, jax, scan, offload, tokamax-ce, overlap, splash, remat, seq8192, bs3, v6e-8]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-8B jax v029 — full MaxText-equivalent stack @ seq8192 bs3

## Hypothesis under test

**Hypothesis**: The full stack — scan (v028, makes overlap productive) + named-offload (frees activation
HBM) + tokamax-CE with **bf16 lm_head weight** (clears the v027 f32-weight 4.64G wall; matches plain-CE
precision) + the overlap flags — fits **bs3** at seq8192 AND amortizes the batch (now that collectives
overlap, unlike v016's sync-collective bs2<bs1), approaching the MaxText ceiling (45.3% / 6,942 tok/s/chip).

**Mechanism**: image `qwen3-8b-jax:v029-full`. `--use_remat --use_splash --use_scan --offload_remat
--use_tokamax_ce --tokamax_ce_impl=mosaic_tpu --batch_size=3 --seqlen=8192` + 7 scheduler + 4 overlap flags.
CE-weight fix: mosaic_tpu now gets the lm_head weight in bf16 (tiles V, accumulates lse in f32 internally)
→ no full f32[H,V] materialization. Scan validated (v028); CE-weight-bf16 validated on TPU by the loss
trajectory here (must be ~12.07 stable / no NaN — else the bf16 CE is wrong → invalid).

**Predicted signal**: bs3 **fits** (CE-bf16 saves ~2.49G + offload frees activations; v027 was +2.34G over);
loss ~12.07 stable; MFU/tok-s-chip > v028 5,632 (bs3 amortizes with overlap), toward 6,942.

**Falsification criterion**: OOM at bs3 (still over budget → attack next temp); loss NaN/divergence
(bf16-CE numerically wrong → invalid, revert CE-weight change); or MFU ≤ v028 5,632 (bs3 doesn't amortize
even with scan+overlap → batch genuinely doesn't help our stack; v028 bs1 is the seq8192 frontier).

## Setup

- Image `qwen3-8b-jax:v029-full`; full stack flags (scan + offload + CE-bf16-weight + bs3 + seq8192 + 7 sched + 4 overlap). scan/splash/offload all confirmed active in log; CE autotune started.

## Results

**Crashed at trace time (step 0), ~88 s — NOT OOM, NOT NaN.** The forward compiled (bs3 / global_batch=24 reached `[train] starting`), but the **tokamax `mosaic_tpu` CE backward Pallas kernel rejected the bf16 weight**:
```
ValueError: Invalid dtype for `swap`. Ref dtype: float32. Value dtype: bfloat16.
(pallas_mosaic_tpu.py ~L334: xw_scratch_ref[...] = x_ref[...] @ w_ref[...])
```
The kernel allocates its `xw_scratch_ref` accumulator as **float32** and requires the weight `w_ref` to match — it does **not** support a bf16 lm_head weight. So the v027 "bf16 weight clears the f32 wall" idea is **kernel-blocked**, independent of memory.

## Verdict

**Invalid** (trace-time dtype crash, no steps). Clean finding: **tokamax mosaic_tpu CE backward hardcodes an f32 weight scratch** → bf16 lm_head weight is impossible on this CE impl. Reverting `_ce_tokamax` to the f32 weight (kernel-compatible). The bs3 memory wall (v027, f32[H,V]=4.64G) therefore can't be cleared by weight-dtype; instead try **scan's program-size reduction** — scan compiles 1 layer body vs 36 unrolled, shrinking concurrent activations, which may free the **+2.34G** v026 was short (v026 had offload+CE+bs3 but NO scan).

## Next hypotheses

- **v030 = scan + offload + CE(f32) + bs3** — revert to f32 CE weight (kernel-compatible); add scan to v026's config. Scan's 1-body compile may close the +2.34G v026 lacked → fit bs3. **Dispatched next.**
- If still OOM: vocab-sharded CE (avoid the lm_head-weight all-gather) or bs2 fallback (v016 fit bs2 with f32 CE — run the full scan+overlap stack at bs2 to test amortization vs v028 bs1 5,632).
