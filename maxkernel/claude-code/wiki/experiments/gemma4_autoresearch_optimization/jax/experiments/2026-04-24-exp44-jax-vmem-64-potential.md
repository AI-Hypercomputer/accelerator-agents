---
title: "Exp 44 — JAX VMEM limit 64 MiB (POTENTIAL, −0.44% flat)"
type: experiment
tags: [experiment, gemma4, jax, xla-flags, vmem, potential]
hypothesis: vmem-shrink-mirror-of-exp41
model: gemma4-e4b-torchax-jax
created: 2026-04-24
updated: 2026-04-24
commit: pending
verdict: inconclusive
---

Symmetric to [exp 41](2026-04-23-exp41-jax-vmem-512-rejected.md) which bumped VMEM to 512 MiB and regressed −13.8 %. This shrinks to 64 MiB. **Result: 34,461 TPS, 22.95 % MFU, −0.44 % flat (within noise).**

Combined with exp 41, confirms **the XLA default `scoped_vmem_limit_kib=131072` (128 MiB) is the local optimum** on v6e-4 for Gemma 4 E4B splash-attention + b=3 + s=1024. Both directions — halving and quadrupling — produce either a regression or noise.

## Results

| Config | TPS | MFU | Step time | Δ vs exp 36 |
|---|---:|---:|---:|---:|
| Exp 36 (default 128 MiB) | 34,614 | 23.05 % | 355.0 ms | — |
| Exp 41 (512 MiB) | 29,832 | 19.87 % | 411.9 ms | **−13.8 %** |
| **Exp 44 (64 MiB)** | **34,461** | **22.95 %** | **356.6 ms** | **−0.44 %** |

## Profile

- **xprof browser URL**: [2026-04-24-gemma4-jax-exp44-vmem64](http://localhost:8791/?run=2026-04-24-gemma4-jax-exp44-vmem64) — opens the interactive trace viewer.
- **Run name**: `2026-04-24-gemma4-jax-exp44-vmem64`
- **On-disk directory**: [`raw/profiles/2026-04-24-gemma4-jax-exp44-vmem64/`](../../../../../raw/profiles/2026-04-24-gemma4-jax-exp44-vmem64/) (gitignored; GCS mirror).
- **Steps captured**: 10, 11, 12.

## Verdict

**POTENTIAL / INCONCLUSIVE.** Flat within noise. Not merged. Close the VMEM-tuning avenue — default is optimal in BOTH directions.

## See also

- [exp 41 — VMEM 512 MiB rejected](2026-04-23-exp41-jax-vmem-512-rejected.md) — opposite direction.
- [exp 36 — JAX best (default VMEM)](2026-04-23-exp36-jax-splash-batch3-accepted.md)

## Sources

- `/tmp/gemma4_jax_exp44.log`
- `raw/profiles/2026-04-24-gemma4-jax-exp44-vmem64/` — xprof run `2026-04-24-gemma4-jax-exp44-vmem64` at http://localhost:8791/?run=2026-04-24-gemma4-jax-exp44-vmem64
