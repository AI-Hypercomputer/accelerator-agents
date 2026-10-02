---
title: "Qwen3 jax — host-offload remat (the MaxText mechanism)"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: refuted
expected_gain: "bs3 @ seq8192 unlock → target MaxText's 6,953 tok/s/chip (+17.9% over our 5,898; ≈ +6 pp MFU in our accounting)"
confidence: high
effort: M
origin: 2026-06-12-mt-v001-base
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — host-offload remat (the MaxText mechanism)

*Hypothesis*: Offloading the remat-saved activations (our tagged
q/k/v/attn_out) to **pinned host memory** instead of HBM
(`save_and_offload_only_these_names`) plus the HOST_OFFLOAD scheduler flags
frees enough HBM that **bs3 @ seq 8192 fits and beats 5,898 tok/s/chip** —
the mechanism behind MaxText's 6,953 tok/s/chip on the identical
architecture ([mt-v001](../experiments/qwen3_cc5_autoresearch_optimization/maxtext/experiments/2026-06-12-mt-v001-base.md)).

*Mechanism*: MaxText's recipe runs `remat_policy=custom` with
`decoder_layer_input` + 4 projection outputs `=offload` — saved tensors cost
zero HBM, transfers overlap via the HOST_OFFLOAD scheduler features. Our
v010 refutation of those flags was at bs1–2 WITHOUT offload traffic — with
real host transfers they become load-bearing. Our prior walls: v014 bs3+CE
missed by 3.4 GB with HBM-saved attn tensors; offloading them (≈ 9 GiB at
bs3) closes that with margin.

*Falsification criterion*: bs3 still OOMs, or host-transfer stalls make bs3
tok/s/chip ≤ 5,898 + 1% (PCIe bandwidth insufficient on this VM shape) →
refuted; the MaxText gap is then attributable to deeper stack differences.

## Status — REFUTED (v025): bs3 fits under scan+offload but misses the throughput bar

[v025](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v025-offload-scan-bs3.md):
bs3 @ seq 8192 **fits** (27.34/31.25 G) and runs with exact loss parity —
the memory mechanism is fully validated — but lands at **5,381 tok/s/chip
< 5,898 bar** (falsification criterion met). Measured: bs1→bs3
amortization only +4.9%; offload DMA stalls 416 ms/step (62% overlap);
scan uniform-remat recompute 662 ms/step. The MaxText gap is therefore NOT
primarily the batch unlock — it lives in splash-bwd time, scan recompute,
and the unadopted recipe flag stack (→
[qwen3-jax-maxtext-flag-completion](qwen3-jax-maxtext-flag-completion.md),
[qwen3-jax-splash-fused-bwd](qwen3-jax-splash-fused-bwd.md)). Offload
remains the only known bs3-fitting substrate — a memory tool, not a
throughput win, on this stack.

### Earlier: v024 refuted the unrolled configuration; v025 retried under scan

[v024](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v024-offload-bs3.md)
(unrolled loop): offload mechanism **works** (tensors in `S(5)` host space,
loss parity exact at bs1) but bs3 compile-OOM'd at 59.41/31.25 G — the f32
rematted **backward temps across 36 unrolled layers** are co-live; the
offloaded tensors are innocent
([observation](../observations/host-offload-unrolled-backward-liveness.md)).
MaxText fits because its decoder is **scanned** — liveness bounded per
iteration. v025 = same image, `--use_scan=True` added (scan merged opt-in in
v020 with −8 GiB live-set).

## See also

- [mt-v001 (gap analysis)](../experiments/qwen3_cc5_autoresearch_optimization/maxtext/experiments/2026-06-12-mt-v001-base.md)
- [v014 (bs3 OOM with HBM-saved attn)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v014-bs3-ce.md)
- [v010 (scheduler bundle flat without offload traffic)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v010-host-offload-sched.md)
