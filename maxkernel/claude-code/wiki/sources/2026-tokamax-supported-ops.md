---
title: "tokamax docs — supported ops and hardware"
type: source
tags: [docs, kernels, pallas, attention, glu, layer-norm, rms-norm, ragged-dot, moe, tpu, gpu]
created: 2026-04-22
updated: 2026-04-22
---

Short doc page from the tokamax repository enumerating which kernels ship today and which accelerator families they target. It is the authoritative "what is actually implemented" statement for tokamax, as opposed to the ambition in the README. For the TPU performance wiki, it pins down exactly which tokamax entry points are usable on TPU and which are GPU-only.

## Overview

The doc lists the public kernel entry points and, for each, the hardware family the implementation targets. It splits the list in two: kernels that are **GPU only**, and kernels that run on **both GPU and TPU**. No TPU-only entries are called out in this doc, although other tokamax subsystems (splash attention, linear-softmax cross-entropy loss) are TPU-only — those are documented elsewhere and not mentioned here.

The doc is intentionally terse: it is a support-matrix, not a usage guide.

## Key claims

- `tokamax.dot_product_attention` (FlashAttention-style) is currently a **GPU-only** public kernel per this doc. In practice the tokamax codebase ships a Pallas-Mosaic-TPU backend for the same entry point via the Splash Attention kernel, so the doc understates TPU coverage (see Gaps & caveats).
- `tokamax.gated_linear_unit` (SwiGLU / GEGLU / REGLU / GLU) is **GPU-only** — there is no TPU-specific kernel, and TPU callers fall back to the XLA reference lowering.
- `tokamax.layer_norm` — covering both LayerNorm and RMSNorm (`subtract_mean=False`) — is **GPU-only**. Same caveat: TPU falls back to XLA.
- `tokamax.ragged_dot` (grouped matmul for Mixture-of-Experts / Megablocks-style MoE) is supported on **both GPU and TPU**.
- The doc cites the canonical paper for each op:
  - Attention → [FlashAttention](https://arxiv.org/abs/2205.14135) (Dao 2022).
  - GLU → [GLU variants](https://arxiv.org/abs/2002.05202) (Shazeer 2020).
  - LayerNorm → [Ba et al. 2016](https://arxiv.org/abs/1607.06450); RMSNorm → [Zhang & Sennrich 2019](https://arxiv.org/abs/1910.07467).
  - Ragged-dot → [Megablocks / Fedus et al. 2022](https://arxiv.org/abs/2211.15841).

## Key data points

Kernel × platform matrix as stated by this doc:

| Kernel | GPU | TPU |
|---|---|---|
| `tokamax.dot_product_attention` | yes | not listed (but implemented — see caveats) |
| `tokamax.gated_linear_unit` | yes | no |
| `tokamax.layer_norm` (+ RMSNorm) | yes | no |
| `tokamax.ragged_dot` | yes | yes |

The [tokamax codebase page](../codebases/tokamax.md) expands this with exact Pallas backends per op — not repeated here.

## Techniques referenced

- FlashAttention (tiled, SRAM-resident attention that avoids materialising the `QK^T` matrix).
- Gated Linear Unit family — fused `activation(x @ W_gate) * (x @ W_up)` — SwiGLU is the load-bearing variant for modern LLMs.
- Layer Normalization and Root-Mean-Square Normalization — single-pass reductions over the feature axis.
- Mixture-of-Experts grouped matrix multiplication via ragged-dot / Megablocks formulation.

## Gaps & caveats

- **The doc understates TPU coverage for attention.** The supported_ops list places `dot_product_attention` under "GPU kernels", but the repo does contain a Pallas-Mosaic-TPU backend that wraps the Splash Attention kernel (see [tokamax codebase page](../codebases/tokamax.md) — `_src/ops/attention/pallas_mosaic_tpu.py`). TPU v5+ models can use `tokamax.dot_product_attention` today. Treat the doc's GPU-only claim as stale.
- **TPU-only kernels are omitted.** The repo ships `tokamax.linear_softmax_cross_entropy_loss`, a TPU-only fused linear + log-softmax + NLL kernel. The supported_ops doc does not mention it.
- **No version / generation gating.** The doc does not say which TPU generations (v4? v5p? v6e? v7?) each kernel targets. In the code, the TPU ragged-dot and attention backends gate on `pltpu.get_tpu_info().generation >= 5`.
- **No performance claims.** The doc is a support matrix only — it does not quantify speedup over XLA.
- **No links into source files.** Users have to read the codebase page or the repo to find the backends.

## Connections

Concept slugs this source informs:

- `flash-attention` — entry point `tokamax.dot_product_attention`.
- `splash-attention` — the TPU backend of the same entry point (see [splash-attention source](2026-tokamax-splash-attention.md)).
- `gated-linear-unit` / `swiglu` — `tokamax.gated_linear_unit`; no TPU kernel yet.
- `layer-norm` / `rms-norm` — `tokamax.layer_norm`; no TPU kernel yet.
- `ragged-dot` / `moe-grouped-matmul` — `tokamax.ragged_dot` is the TPU + GPU implementation.

Model pages using any of these ops on TPU should list "swap the reference op to the `tokamax.*` entry point" as a candidate hypothesis.

## See also

- [tokamax](../codebases/tokamax.md) — full repo page with per-backend file references and Config dataclasses.
- [tokamax basic usage](2026-tokamax-basic-usage.md) — how to actually call these kernels, including `implementation=` dispatch.
- [tokamax splash attention](2026-tokamax-splash-attention.md) — the TPU attention backend that this support matrix omits.
- [tokamax autotuning](2026-tokamax-autotuning.md) — tuning surface for each `Config`.
- [tokamax benchmarking](2026-tokamax-benchmarking.md) — how to measure speedup over XLA.

## Sources

- [`raw/code/tokamax/docs/supported_ops.md`](../../raw/code/tokamax/docs/supported_ops.md)
