---
title: "Attention Block Sizes"
type: concept
tags: [stub, kernel, tuning]
created: 2026-04-22
updated: 2026-04-22
sources: 3
---

Tunable tile sizes (`block_q`, `block_kv`, `block_kv_compute`) for TPU flash/splash attention; backward-pass block sizes (`block_q_dkv`, etc.) exposed via `SplashConfig` but currently hidden from tokamax's autotuner.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [Splash Attention](splash-attention.md)
- [Flash Attention](flash-attention.md)
- [Autotuning](autotuning.md)
- [VMEM (Vector Memory)](vmem.md)
- [Base-2 Softmax](base2-softmax.md)

## Sources

- [tokamax splash attention](../sources/2026-tokamax-splash-attention.md) — `raw/code/tokamax/docs/splash_attention.md`
- [tokamax autotuning](../sources/2026-tokamax-autotuning.md) — `raw/code/tokamax/docs/autotuning.md`
- [Ultra-Scale Playbook](../sources/2025-ultrascale-playbook.md) — FlashAttention tiling of Q/K/V (GPU analogue); notes backward block sizes (`block_q_dkv`, `block_kv_dkv`, `block_kv_dkv_compute`, `block_q_dq`, `block_kv_dq`) are **not** surfaced to the tokamax autotuner — hidden tuning surface
