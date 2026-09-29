---
title: "EasyDeL (erfanzar/EasyDeL)"
type: codebase
tags: [EasyDeL, ejkernel, pallas, mosaic-tpu, training, serving, operations-registry, ssm, gated-delta-rule, kimi-linear]
repo: erfanzar/EasyDeL
commit: 090a03b2
created: 2026-04-23
updated: 2026-04-23
---

> **Grounded front door → [easydel overview](easydel/overview.md).** This hand page is the thin, on-demand companion: cross-ecosystem perf verdicts, loop-experiment backlinks, and Pallas-kernel-directory cross-links that the code-regenerated overview intentionally omits. For architecture, mechanisms, and per-symbol detail, start at the overview.

Training / serving framework that **consumes [ejkernel](ejkernel.md)**. TPU Pallas usage lives in `easydel/operations/kernels/`, which re-exports ejkernel implementations through an operations registry. Unique EasyDeL-side content: `glm_moe_dsa_indexer.py` (GLM MoE DSA index op). Otherwise the TPU kernels here are thin adapters. Ingested as the companion to ejkernel per the Wave 4 ingestion recommendation (they're meant to be read together). Per-kernel detail in [§6.2 of the Pallas kernel directory](../analyses/pallas-kernel-directory/06-community-research.md#62-erfanzareasydel).

## Architecture

```
easydel/
  operations/
    kernels/                          ← TPU Pallas consumer surface (wrapping ejkernel)
      flash_attention.py              ← registry wrapper → ejkernel.flash_attention
      ring_attention.py               ← wraps ejkernel ring (Splash-based)
      blocksparse_attention.py        ← Splash / blocksparse wrapper
      ragged_page_attention.py        ← paged-attention wrapper
      ssm1.py, ssm2.py                ← SSM1Op / SSM2Op wrappers (axlearn-style Mamba via ejkernel)
      gated_delta_rule.py             ← GatedDeltaRuleOp
      kernel_delta_attn.py            ← KernelDeltaAttnOp (Kimi Linear)
      glm_moe_dsa_indexer.py          ← UNIQUE to EasyDeL (not in ejkernel)
      ...
  ...
```

Kernel folder lists (per directory analysis): `BlockSparseAttn`, `FlashAttn`, `RingAttn`, `UnifiedAttn` ("vLLM-style" continuous-batching prefill+decode), `RaggedPageAttnV2`, `RaggedPageAttnV3`, `AutoRegressiveDecodeAttn`, `SSM1Op`, `SSM2Op`, `GatedDeltaRuleOp`, `KernelDeltaAttnOp`.

## Key abstractions

| Wrapper | File | Backs into |
|---|---|---|
| `FlashAttn` | `operations/kernels/flash_attention.py` | `ejkernel.flash_attention` |
| `RingAttn` | `operations/kernels/ring_attention.py` | `ejkernel.ring_attention` (Splash wrapper) |
| `BlockSparseAttn` | `operations/kernels/blocksparse_attention.py` | `ejkernel.blocksparse_attention` |
| `RaggedPageAttnV2/V3` | `operations/kernels/ragged_page_attention.py` | `ejkernel.ragged_page_attention_v2/v3` |
| `SSM1Op` / `SSM2Op` | `operations/kernels/ssm1.py` / `ssm2.py` | (via ejkernel; lineage ties to [axlearn](axlearn.md) SSM work) |
| `GatedDeltaRuleOp` | `operations/kernels/gated_delta_rule.py` | `ejkernel.gated_delta_rule` |
| `KernelDeltaAttnOp` | `operations/kernels/kernel_delta_attn.py` | (Kimi-linear-specific) |
| **`GLM MoE DSA indexer`** | `operations/kernels/glm_moe_dsa_indexer.py` | **unique — not re-exported from ejkernel** |

## Entry points

```bash
cd raw/code/EasyDeL
pip install -e .

python -c "from easydel.operations.kernels.flash_attention import *"
```

## Dependencies

- `ejkernel` (the main source of TPU kernels).
- `jax`, `libtpu`, `flax`.
- Heavier framework stack (trainers, serving runtime).

## Performance-relevant surfaces

### 1. Operations-registry as a wrapper pattern
Every TPU kernel wrapper in `operations/kernels/` follows a consistent registration/dispatch pattern. If this wiki ever needs an analogous registry for the gemma4 program, this is a reference template.

### 2. `glm_moe_dsa_indexer.py` — the sole EasyDeL-unique kernel
GLM MoE Deep-State-Aware indexer. Other kernels all pass through to ejkernel; this one is here.

### 3. Pairing with ejkernel
Reading EasyDeL on its own gives only wrapper-layer detail. For the actual Pallas code, cross-reference each wrapper to the ejkernel file it imports.

### 4. Breadth of model support
Framework supports Mamba (SSM1/SSM2), GDN, Kimi-Linear, MoE — the wrappers reflect that breadth. A model-availability matrix rather than a novel-kernel source.

## Connections

- **[Pallas kernel directory §6.2](../analyses/pallas-kernel-directory/06-community-research.md#62-erfanzareasydel)** — full wrapper detail.
- **[ejkernel](ejkernel.md)** — the backing library. Read ejkernel for kernel code, EasyDeL for wrapper/registration patterns.
- [axlearn](axlearn.md) — Mamba / SSM lineage that EasyDeL's SSM1/SSM2 wrappers trace back to.

## See also

- [pallas-kernel](../concepts/pallas-kernel.md)
- [continuous-batching](../concepts/continuous-batching.md)
- [splash-attention](../concepts/splash-attention.md)

## Sources

- `raw/code/EasyDeL/easydel/operations/kernels/`
- Upstream: <https://github.com/erfanzar/EasyDeL>
