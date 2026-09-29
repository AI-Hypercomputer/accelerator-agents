---
title: "qwen3-jax: batch rescale at seq-2048 on the remat substrate"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: supported   # CERTIFIED at bs5 (v060: 7,871/chip, 40.5% MFU); ladder closed (v059 bs6 OOM measured; bs8-via-chunked-CE ruled invalid in v061/v062)
expected_gain: "10-30% tok/s/chip at seq2048"
confidence: medium
effort: S
origin: 2026-06-12-v055-2kstack
tags: [qwen3-cc5, jax, v6e-8, seq2048, batch-scaling, remat, charles-v6e]
created: 2026-06-12
updated: 2026-06-12
---

# Batch rescale at 2k on the remat substrate

The 2k batch ladder froze at bs2 in the v007 era because bs3 needed ~5 GiB
that didn't exist (29.81 GiB peak of 31.25,
[observation](../observations/qwen3-2k-v6e8-hbm-headroom.md)). The
[v055](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v055-2kstack.md)
substrate (scan + save_qkv_ctx_hbm + context) bounds activation memory and
should reopen it.

## Statement

On the v055 substrate at seq2048, some bs ∈ {4, 6, 8} yields tok/s/chip
> 6,136 (the v052 bs2 unrolled baseline) — i.e. batch amortization of the
fixed-cost buckets (collectives, kernel launch) outweighs the remat tax that
made v055 −2.1% at bs2.

## Rationale

- Batch was the single biggest 2k lever on record: bs1→bs2 = **+40.5%
  tok/s/chip** ([v007](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v007-bs2-vmem.md)).
- v055's remat policy caps the per-bs activation slope (8k analog: bs2 fit in
  22.03 GiB with seq 4× longer); at 2k the slope should be ~¼ of that —
  bs8 plausibly fits.
- The original 20.5%-MFU baseline ran global batch 8 (bs1) — the lane has
  never measured bs>2 at 2k because of the (now-removed) memory wall.
- Risk: v055 showed the substrate costs ~2% at bs2; the bet is amortization
  beats it with headroom to spare.

## Proposed experiment

v057: v055 config with `--batch_size=4`; compare tok/s/chip vs both v052
(6,136) and v055 (6,005). If supported, probe bs5/bs6 with a fresh profile
first.

**HBM reading (v055 profile, 2026-06-12)**: peak 15.61 GiB (49.95%) at bs2 —
stack 9.83 GiB (scales ~linearly with batch; includes the 9.82 GiB stacked
scan activation buffer) + heap 5.78 GiB (fixed). Projection: bs4 ≈ 25.4 GiB
(~81%, safe — **opening rung**); bs6 ≈ 35.3 GiB (~113%, OOM on linear
extrapolation). The v055 decomposition strengthens the rationale: the 186 ms
collective floor (ARS 146.5 + AG 26.6 + AR 13.0) is weight-sized = fixed per
step, so 2× tokens halves its share; back-of-envelope bs4 step ≈ 486×2 + 186
≈ 1,158 ms → ≈7,070 tok/s/chip ≈ **+15% over v052**.

**Outcome at bs4 ([v057](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v057-2kbs4.md), 2026-06-12)**:
**supported — 1,094.5 ms / 7,479 tok/s/chip / ≈38.5% MFU (+21.9% vs v052)**;
collective floor amortized exactly (185.0 ms flat, 27.3% → 16.9%); MXU 48.8%.
Measured slope 4.10 GiB/bs (heap fixed 5.79): bs5 = 89.4% (fits), bs6 =
102.5% (**OOM — batch ceiling is bs5**). Throughput model projects bs5 ≈
8,196 tok/s/chip (+9.6%) — v058 probing.

## Risks

None semantic (batch is a throughput config; loss-series stability check
only). Possible per-step latency increase (fine — tok/s/chip is the metric).

## Dependencies

v055 profile-analyzer memory reading (sets the opening rung).

## See also

- [Model page](../models/qwen3-cc5-jax.md) · [v055](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v055-2kstack.md) · [HBM observation](../observations/qwen3-2k-v6e8-hbm-headroom.md)
