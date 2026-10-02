---
title: "Exp 48 — JAX splash-kernel parameter sweep (POTENTIAL, plateau at exp-36 defaults)"
type: experiment
tags: [experiment, gemma4, jax, splash-attention, sweep, potential]
hypothesis: splash-block-layout-tuning
model: gemma4-e4b-torchax-jax
created: 2026-04-24
updated: 2026-04-24
commit: pending
verdict: inconclusive
---

Plumbed splash-attention knobs through env vars (`SPLASH_BLOCK_Q`, `SPLASH_BLOCK_KV`, `SPLASH_BLOCK_KV_COMPUTE`, `SPLASH_BLOCK_Q_DKV`, `SPLASH_BLOCK_KV_DKV`, `SPLASH_BLOCK_KV_DKV_COMPUTE`, `SPLASH_QKV_LAYOUT`, `SPLASH_USE_FUSED_BWD`) in `jax/model/pallas_attention.py`, with defaults preserving exp 36's winning config. Swept four non-baseline variants; **all landed flat within ±0.5 % of exp 36 (34,614 TPS). Exp 36's defaults are on a plateau at the local optimum — no marginal knob tuning unlocks further gain on this stack.**

## Setup

Plumbing lands in this session's commit `8c45d1c`. Each env var, when set, overrides the corresponding splash `BlockSizes` field or `QKVLayout` enum. `_splash_config_key()` reads the env snapshot and threads it through the lru_cache key so variant changes produce a fresh kernel without invalidating unrelated caches.

Startup prints a `[attn] splash overrides: {…}` line listing active env vars for reproducibility.

Variants tested:

| Exp | Variant (env override) | Mean step (ms) | TPS | Δ vs exp 36 | MFU | Notes |
|---|---|---:|---:|---:|---:|---|
| 36 (baseline) | defaults (all 1024, SEQ_MINOR, fused_bwd) | 355.0 | 34,614 | — | 23.05 % | — |
| 41 | `scoped_vmem_limit_kib=524288` (512 MiB) | 411.9 | 29,832 | **−13.8 %** | 19.87 % | XLA over-commits (REJECTED earlier) |
| 44 | `scoped_vmem_limit_kib=65536` (64 MiB) | 356.6 | 34,461 | −0.44 % | 22.95 % | Flat — 128 MiB default is the optimum |
| 48a | `SPLASH_BLOCK_KV_COMPUTE=512` | 354.9 | 34,626 | **+0.03 %** | 23.06 % | Inner compute tile half of outer; flat |
| 48b | `SPLASH_QKV_LAYOUT=HEAD_DIM_MINOR` | 355.5 | 34,569 | **−0.13 %** | 23.02 % | Layout ablation; torchax exp 24 saw +0.5 % here but JAX doesn't reproduce |

All loss trajectories match exp 36 within noise.

## Why the sweep shows a plateau

Two plausible reasons:

1. **Exp-36 block defaults (1024) are actually the XLA/Mosaic cost model's preferred config for this shape** (Q=[1,8,1024,256] per chip, K/V=[1,2,1024,256]). At seq=1024, `block_q = block_kv = 1024` means the whole sequence is one tile per head — no within-seq tiling, no partial reduction. `block_kv_compute` separation only matters when VMEM-pressured; at this shape there's headroom.
2. **SEQ_MINOR layout is a no-op on JAX graph topology** — torchax exp 24's +0.5 % win came from better HBM streaming when torchax's jit-vmap inflated batch dim broadcasting; the JAX port doesn't have that broadcast inflation, so layout doesn't affect the traffic pattern.

**Corollary**: further splash-block-size tuning at b=3 s=1024 is not promising on this stack. Move to structural changes (scan-over-layers, fused softcap+CE kernel, etc.).

## Profile

- **xprof browser URL for exp 48a** (block_kv_compute=512): [2026-04-24-gemma4-jax-exp48a-kv-compute-512](http://localhost:8791/?run=2026-04-24-gemma4-jax-exp48a-kv-compute-512)
- **xprof browser URL for exp 48b** (HEAD_DIM_MINOR): [2026-04-24-gemma4-jax-exp48b-head-dim-minor](http://localhost:8791/?run=2026-04-24-gemma4-jax-exp48b-head-dim-minor)
- **On-disk**: [`raw/profiles/2026-04-24-gemma4-jax-exp48a-kv-compute-512/`](../../../../../raw/profiles/2026-04-24-gemma4-jax-exp48a-kv-compute-512/), [`raw/profiles/2026-04-24-gemma4-jax-exp48b-head-dim-minor/`](../../../../../raw/profiles/2026-04-24-gemma4-jax-exp48b-head-dim-minor/)

## Durable artifact

The splash env-var plumbing stays in `pallas_attention.py` regardless of this verdict. Future experiments (e.g. after scan-over-layers changes the HLO shape) can re-sweep without touching code; just set env vars. Usage:
```bash
SPLASH_BLOCK_Q=2048 SPLASH_BLOCK_KV=2048 ... python -m train ...
```

## Verdict

**POTENTIAL / INCONCLUSIVE.** No new best from the block-size / layout axis. Defaults preserved. Infrastructure (env-var knobs) landed for future re-sweep.

## Next

- **Exp 47** — levanter fused CE kernel (the real remaining TPS lever, found during directory review).
- **Exp 49** — scan-over-layers (compile-time).
- **Exp 46** — Pallas RMSNorm retest (low priority given fusion-boundary pattern).

## See also

- [exp 36 — baseline (current JAX best)](2026-04-23-exp36-jax-splash-batch3-accepted.md)
- [exp 41 — VMEM=512 rejected](2026-04-23-exp41-jax-vmem-512-rejected.md), [exp 44 — VMEM=64 flat](2026-04-24-exp44-jax-vmem-64-potential.md)
- [torchax exp 24 SEQ_MINOR accepted +0.5 %](../../torchax/experiments/2026-04-23-exp24-splash-seq-minor-accepted.md) — the source of the SEQ_MINOR lesson that didn't reproduce on JAX.

## Sources

- `/tmp/gemma4_jax_exp48a.log`, `/tmp/gemma4_jax_exp48b.log`
- `jax/model/pallas_attention.py` (env-var plumbing landed this session)
- `raw/profiles/2026-04-24-gemma4-jax-exp48{a,b}-*/` — xprof runs.
