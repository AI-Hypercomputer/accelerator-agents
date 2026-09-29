---
title: "int8 weight-only quantization (AQT) for Llama 3 8B JAX matmuls"
type: hypothesis
tags: [llama3, jax, aqt, qwix, int8, quantization, deep-work]
created: 2026-04-27
updated: 2026-04-27
model: llama3-8b-jax
status: open
expected_gain: "+15-30 % step time"
confidence: medium
effort: L
origin: jax-exp28b-profile-2026-04-26
---

Adopt int8-weight + bf16-activation matmuls (AQT or qwix) for the seven projection sites in `_decoder_call` so the TPU's int8 MXU pathway can run at ~2× the bf16 throughput, **shifting the critical batch downward** and breaking the 65.8 % bf16-MXU ceiling observed in [exp 28b's profile](../experiments/llama3_8B_autoresearch_optimization/jax/experiments/2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md#profile).

## Statement

Replacing all seven `jax.lax.dot_general` calls in `_decoder_call` (Q/K/V/O proj + gate/up/down) with int8-weight × bf16-activation matmul (weights pre-quantized at load time, activations quantized per-step on the fly) will reduce step time by **15–30 %** at bs=4 seq=8192 with **loss trajectory matching the bf16 baseline within bf16 noise**.

## Rationale

v6e MXU peak: **918 TFLOPs/sec bf16** vs **1,836 TOPS int8** (2×). At exp 28b, conv-fusion occupies 60.1 % of step at MXU-util 65.8 %, i.e. matmul-effective time = 39.5 %. If the matmul ran on int8 at the same MXU-util, it would take **half the time** for those FLOPs, dropping conv-fusion's step share toward ~30 % and lifting throughput proportionally. Realistic gain ~15–25 % once HBM-BW-bound layers (RMSNorm, residuals, splash) become the new bottleneck.

The win comes in two ways:
1. **Compute throughput**: int8 MXU is 2× bf16 MXU (hardware-level)
2. **HBM-traffic for weights**: int8 weights are half the size of bf16 (~4 GiB total at int8 vs 8 GiB at bf16). Sharded by 8 = 0.5 GiB on chip — fits more comfortably, may unlock bs=6.

Open-source references for the technique on TPU:
- **AQT** (`raw/code/aqt/`, deprecated) — Google's original quantization framework; supported int8 weight-only and full-int8 paths
- **qwix** (`raw/code/qwix/`) — successor to AQT; ships a `QArray` that is `pallas_call`-aware
- **MaxText** uses AQT (now qwix-track) and reports +5–8 % at int8-weight-only on similar shapes; full-int8 yields more
- The wiki concept page [int8-quantization](../concepts/int8-quantization.md) captures the canonical formulation

## Proposed experiment

Two-stage approach:

**Stage 1 (smoke)**: weight-only int8 (activations stay bf16):
1. Pre-quantize weights at load time using `qwix.QArray.quantize(scale_axis=-1, qtype=int8)` — channel-wise scale per output dim.
2. Replace `_matmul` to `dot_general(activation_bf16, weight_int8)` and de-quantize the matmul output by multiplying by the saved scale. This is the lowest-risk path; loss should match bf16 within 0.005/step.
3. Run with the exp 28b stack otherwise unchanged.

**Stage 2 (full int8)** if Stage 1 wins and is loss-clean:
1. Add per-step activation int8 quantization (compute scale per (B*T) batch row).
2. Run int8×int8 matmul, fp32 accumulator.
3. Validate loss trajectory; this stage is ~2× the throughput win but ~10× the precision risk.

## Measurement

| Metric | Stage 1 target | Stage 2 target |
|--------|---------------:|---------------:|
| tok/s/chip | ≥ 8,500 (+9 %) | ≥ 9,300 (+19 %) |
| MFU (bf16-formula) | ≥ 47 % | ≥ 51 % |
| Loss Δ vs bf16 baseline | ≤ 0.005/step | ≤ 0.01/step |
| HBM peak | should DROP (weights smaller) | drops further |
| Conv-fusion % in profile | ≤ 50 % | ≤ 35 % |

## Risks

- **Loss divergence**. The hardest risk. Stage 1 (weight-only) is well-known-safe in literature; Stage 2 (full-int8) is brittle on activations with outliers. Validate at 100+ steps, not 9.
- **Compile-time inflation**. AQT/qwix custom-calls add HLO bytes and break compile-cache hits; expect 2× compile cost on first run.
- **Tokamax CE compatibility**. The `chunked_xla` CE kernel requires fp32 inputs; we already cast at the boundary. The lm_head matmul (vocab projection) is part of the CE call — int8'ing the lm_head requires changes in tokamax/_src/ops/cross_entropy. Probably out-of-scope for the first cut; lm_head stays bf16.
- **Splash kernel** does not need int8 — it's not a matmul against weights, it's QK^T over activations. Leave splash bf16 and accept the matmul-only win.

## Dependencies

- Decide between AQT (deprecated, but well-tested) and qwix (successor, sparse documentation). Recommend **qwix** since AQT is in maintenance only.
- Need to validate `qwix.QArray` round-trips correctly through `jax.lax.scan` (our scan-over-layers stacks the quantized weights along a leading dim).
- A semantic-preservation gate: compare the first 100 steps' loss trajectory against the bf16 baseline; require correlation ≥ 0.999.

## See also

- [int8-quantization](../concepts/int8-quantization.md) — concept page
- [qwix](../codebases/qwix.md), [aqt](../codebases/aqt.md) — codebase pages
- [JAX exp 28b frontier writeup](../experiments/llama3_8B_autoresearch_optimization/jax/experiments/2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md)

## Sources

- `raw/code/qwix/` (commit `b966dc4`) — successor quantization framework.
- `raw/code/aqt/` (commit `9d1667e`) — deprecated reference.
- `raw/code/maxtext/` — uses AQT in production for int8-weight-only training.
- `raw/profiles/2026-04-26-jax-exp28b-sc-rsag-bs4/` — profile attesting to the bf16-MXU 65.8 % utilization ceiling that int8 would relax.
