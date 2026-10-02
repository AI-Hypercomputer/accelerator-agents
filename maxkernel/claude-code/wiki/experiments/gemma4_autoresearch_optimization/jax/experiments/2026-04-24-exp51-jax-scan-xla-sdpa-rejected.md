---
title: "Exp 51 — JAX scan-over-layers + XLA SDPA (REJECTED, −60%; splash cleared as scan-regression culprit)"
type: experiment
tags: [experiment, gemma4, jax, scan, xla-sdpa, isolation, rejected]
hypothesis: scan-regression-root-cause
model: gemma4-e4b-torchax-jax
created: 2026-04-24
updated: 2026-04-24
commit: pending
verdict: refuted
---

Isolation test: take exp 50's two-scan split implementation and swap **splash → XLA SDPA** (`JAX_ATTENTION_IMPL=xla JAX_SCAN_LAYERS=1`). **Result: 13,792 TPS / 9.19 % MFU — −60 % vs exp 36 and −52 % vs scan+splash (exp 50).** Splash is NOT the scan regression culprit; it's what keeps scan viable at all on this stack. Scan + XLA SDPA is catastrophically worse than no-scan XLA SDPA (exp 34's 30,285 TPS).

## Hypothesis

Exp 50 report suggested that "splash-under-shard_map-under-scan scheduling" might be the residual 11–16 % TPS gap under scan. Testing that hypothesis by running the same scan code with the plain XLA SDPA path instead of splash.

## Results

| Config | TPS | MFU | Step time | Δ vs exp 36 (no-scan splash) | Δ vs exp 34 (no-scan XLA SDPA) | Δ vs exp 50 (scan splash) |
|---|---:|---:|---:|---:|---:|---:|
| exp 34 baseline (no scan, XLA SDPA) | 30,285 | 20.17 % | 135 ms | — | — | — |
| exp 36 best (no scan, splash) | 34,614 | 23.05 % | 355 ms | — | — | — |
| exp 50 (scan, splash) | 29,044 | 19.34 % | 423 ms | −16.1 % | — | — |
| **exp 51 (scan, XLA SDPA)** | **13,792** | **9.19 %** | **891 ms** | **−60.1 %** | **−54.5 %** | **−52.5 %** |

## Mechanism

XLA SDPA at b=3 seq=1024 materializes `[B=3, H=8, S=1024, S=1024] × bf16 = 48 MiB` of attention scores per layer. **Outside scan**, XLA fuses this tensor with the surrounding Q@K^T and softmax+@V matmuls (the `convolution fusion` + `loop fusion` buckets in exp 34's profile) keeping it ephemeral in VMEM.

**Inside a scan body**, every layer iteration runs the same compiled subgraph — XLA can't fuse *across* scan iterations because iterations are symbolic (the scan body is one HLO region whose output is carried). So each of the 42 iterations materializes the `[3, 8, 1024, 1024]` score tensor to HBM, reads it back, runs softmax, writes the attention output, and reloads on the next iteration's residual-add. Total extra HBM traffic: ~42 layers × 48 MiB × 3 (fwd + 2 bwd halves under `jax.checkpoint`) = ~6 GiB extra bytes/step.

Splash's mosaic custom-call operates on per-chip Q/K/V directly without materializing the score tensor — the scan can't foil its internal SRAM-tiled attention. Hence splash preserves per-iteration cost under scan where XLA SDPA does not.

## Profile

- **xprof browser URL**: [2026-04-24-gemma4-jax-exp51-scan-xla-sdpa](http://localhost:8791/?run=2026-04-24-gemma4-jax-exp51-scan-xla-sdpa) — opens the interactive trace viewer.
- **Run name**: `2026-04-24-gemma4-jax-exp51-scan-xla-sdpa`
- **On-disk directory**: [`raw/profiles/2026-04-24-gemma4-jax-exp51-scan-xla-sdpa/`](../../../../../raw/profiles/2026-04-24-gemma4-jax-exp51-scan-xla-sdpa/) (gitignored; GCS mirror).
- **Steps captured**: 10, 11, 12.
- **What's inside**: xprof trace showing per-iteration HBM traffic explosion for the `[B, H, S, S]` score tensor.

## Verdict

**REJECTED.** Not merged — scan + XLA SDPA is unusable. Exp 36 remains JAX-stack best.

## Generalizable lesson

> **When using `jax.lax.scan` for transformer layers on TPU, an attention kernel that internally tiles (splash, flash) is MANDATORY — XLA SDPA cannot share the score-tensor memory across scan iterations.**

This is the third "scan-vs-fusion" finding on this wiki, generalizing from:
- Exp 49 baseline: scan costs ~21 % TPS even with splash.
- Exp 50 tuning: two-scan split + inner selective remat closes some gap but not all.
- Exp 51 (this): scan + XLA SDPA is ~60 % slower.

**Conclusion for the scan-over-layers direction**: the 21 % residual gap between exp 50 and exp 36 is likely a combination of (a) scan's inability to fuse across iterations, (b) per-iteration `checkpoint_dots_with_no_batch_dims` choosing slightly different remat points than XLA's unrolled for-loop chose, (c) FSDP all-gather-per-layer turning into all-gather-per-scan-iter which is identical in op count but may schedule worse. Closing this gap requires structural changes beyond kernel swaps — probably custom Pallas scan+attention fusion, out of scope.

**Final scan verdict**: **keep exp 49/50 behind `JAX_SCAN_LAYERS=1` env gate for HLO-invalidating dev-iteration cases** where the persistent compile cache (exp 45) misses. Use exp 36 (for-loop + splash) as trunk default.

## See also

- [exp 36 — JAX best](2026-04-23-exp36-jax-splash-batch3-accepted.md)
- [exp 49 — scan-over-layers baseline](2026-04-24-exp49-jax-scan-layers-potential.md)
- [exp 50 — scan tuning (two-scan split + nested-checkpoint fix)](2026-04-24-exp50-jax-scan-tuned-potential.md)
- [exp 34 — JAX baseline no scan XLA SDPA](2026-04-23-exp34-jax-baseline-accepted.md) — the non-scan XLA SDPA reference.

## Sources

- `/tmp/gemma4_jax_exp51.log`
- `raw/profiles/2026-04-24-gemma4-jax-exp51-scan-xla-sdpa/` — xprof run.
