---
title: "ringattention (haoliuhl/ringattention)"
type: codebase
tags: [ring-attention, pallas, mosaic-tpu, sequence-parallelism, long-context, paper-companion, liu-2023]
repo: haoliuhl/ringattention
commit: d2ea1af
created: 2026-04-23
updated: 2026-04-23
---

> **Grounded front door → [ringattention overview](ringattention/overview.md).** This hand page is the thin, on-demand companion: cross-ecosystem perf verdicts, loop-experiment backlinks, and Pallas-kernel-directory cross-links that the code-regenerated overview intentionally omits. For architecture, mechanisms, and per-symbol detail, start at the overview.

Paper companion for [Liu et al. 2023 "Ring Attention with Blockwise Transformers"](https://arxiv.org/abs/2310.01889) — **the canonical public Pallas TPU ring-attention implementation**. Single-file kernel (`ringattention_pallas_tpu.py`) directly implementing the paper. Low-maintenance (770★, last updated 2025-10-13) but stable and verbatim. **Unidirectional ring; straight `below_or_on_diag` causal mask — no Zig-Zag / load-balanced variant.** Per-kernel detail in [§6.4 of the Pallas kernel directory](../analyses/pallas-kernel-directory/06-community-research.md#64-haoliuhlringattention).

## Overview

Ingestion scope is the whole repo — it's tiny, single-purpose, and the **reference implementation** of a technique still broadly open in the ecosystem. This closes a gap previously flagged in the wiki: from [sources/2025-ultrascale-playbook.md](../sources/2025-ultrascale-playbook.md) Gap #3, "Ring Attention Pallas — no public TPU Pallas implementation found" — that claim was incorrect; haoliuhl ships it. **The remaining gap**, Zig-Zag / load-balanced causal variant from Brandon et al. 2023, is confirmed still absent here and elsewhere.

## Architecture

```
ringattention/
  ringattention_pallas_tpu.py    ← the Pallas TPU kernel (canonical paper impl)
  ringattention_jax.py           ← pure-JAX reference (oracle)
  ringattention/                 ← package init + utility helpers
  scripts/, benchmarks/, tests/  ← minor
```

Single design principle: across `ring_axis` mesh axis, shift K/V via `lax.ppermute` in a ring; at each step, call the local flash-attention inner kernel; combine partial results with online softmax. The inner kernel is a from-scratch flash-attention Pallas implementation (does **not** compose with upstream splash the way maxdiffusion's `ring_attention_kernel.py` does).

## Key abstractions

| Kernel | File | Backend | Stability | Notes |
|---|---|---|---|---|
| `ring_flash_attention_tpu` (fwd + bwd) | `ringattention_pallas_tpu.py` | `mosaic_tpu` | research (paper companion) | Canonical ring-attention Pallas. Unidirectional. Straight `below_or_on_diag` causal. No zig-zag |
| Reference JAX implementation | `ringattention_jax.py` | XLA | stable | Oracle for correctness |

## Entry points

```bash
cd raw/code/ringattention

# reference impl (requires TPU for kernel path)
python -m pip install -e .

# single-file kernel
python -c "from ringattention import ring_flash_attention_tpu"
```

## Dependencies

- `jax` / `jaxlib` with TPU runtime.
- No framework wrapper — raw research-grade single-file usage.

## Performance-relevant surfaces

### 1. Canonical ring-attention Pallas TPU implementation
This is the reference. If a hypothesis needs to port ring-attention to a different codebase (e.g., gemma4), this file is authoritative for the **orchestration pattern** (ring step via `lax.ppermute`, online-softmax combine, segment-id rotation). Note: the inner kernel is from-scratch flash — not necessarily the fastest; composing with a modern splash (the [maxdiffusion](maxdiffusion.md) approach) is likely better.

### 2. Causal pattern is unidirectional
`below_or_on_diag` check. **Not load-balanced.** On long sequences with causal mask, work imbalance grows linearly with sequence length — early-ring devices do progressively less work. Brandon et al. 2023's Zig-Zag (striped) remedy is **not implemented here, nor in any surveyed repo** (confirmed in [§6.4](../analyses/pallas-kernel-directory/06-community-research.md#64-haoliuhlringattention)). Remains an open port.

### 3. No measured numbers in the repo
README: *"train with tens of millions of tokens in context size without adding any communication or computation overhead"* — qualitative. No microbench numbers. Citable only with an accompanying measurement.

### 4. Three ring-attention implementations exist — pick one
Per the [pallas kernel directory](../analyses/2026-04-23-pallas-kernel-directory.md):
- **This repo** — canonical paper impl; from-scratch flash inner kernel; unidirectional.
- **[maxdiffusion](maxdiffusion.md)** — splash-integrated; announced 2026-04-16; shares the splash tile kernel.
- **[ejkernel](ejkernel.md)** — wraps splash with ring orchestration (not a novel kernel; orchestration only).

For a gemma4 TP-to-context-parallel hypothesis, **maxdiffusion's pattern is likely better** because it reuses splash's tuned bf16 performance; **haoliuhl is better** as a reference for understanding the paper.

## Connections

- **[Pallas kernel directory §6.4](../analyses/pallas-kernel-directory/06-community-research.md#64-haoliuhlringattention)** — per-kernel row.
- [maxdiffusion](maxdiffusion.md) — splash-integrated ring-attention variant.
- [ejkernel](ejkernel.md) — ring-attention as splash wrapper.
- [tokamax](tokamax.md) — has a `ring_attention_kernel` in its experimental tree that is **not reachable via public API**; closing that reachability gap is a candidate hypothesis (see [log.md](../log.md) Wave 1 tokamax findings).
- [sources/2025-ultrascale-playbook.md](../sources/2025-ultrascale-playbook.md) Gap #2 — this repo reduces that hypothesis from "open research" to "port + adapt".
- Paper: [Liu et al. 2023](https://arxiv.org/abs/2310.01889).
- Follow-up paper (Zig-Zag): [Brandon et al. 2023](https://arxiv.org/abs/2311.09431) — implementation still open.

## See also

- [ring-attention](../concepts/ring-attention.md)
- [splash-attention](../concepts/splash-attention.md)
- [context-parallelism](../concepts/context-parallelism.md)
- [sequence-parallelism](../concepts/sequence-parallelism.md)
- [pallas-kernel](../concepts/pallas-kernel.md)

## Sources

- `raw/code/ringattention/ringattention_pallas_tpu.py`
- `raw/code/ringattention/ringattention_jax.py`
- Upstream: <https://github.com/haoliuhl/ringattention>
