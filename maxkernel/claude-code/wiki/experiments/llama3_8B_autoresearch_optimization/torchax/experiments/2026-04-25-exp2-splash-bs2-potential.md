---
title: "Exp 2 — Splash attention at bs=2 seq=1024 (POTENTIAL, +1.6% within noise)"
type: experiment
tags: [llama3, torchax, splash, fsdp, v6e, gke, potential]
hypothesis: llama3-torchax-splash-attention
model: llama3-8b-torchax
created: 2026-04-25
updated: 2026-04-25
commit: "v6e8-llama3-8b-torchax-20260425-exp2-splash-bs2"
branched_from: v6e8-llama3-8b-torchax-20260425-baseline
verdict: potential
---

Replaced HF SDPA's default backend with the canonical TPU splash-attention
kernel via `env.override_op_definition(F.scaled_dot_product_attention, …)`,
keeping the baseline shape `bs=2 seq=1024 fsdp=8`. Result: **+1.6 % TPS within
noise**, but the override unlocks higher-density configurations that the
baseline cannot fit (see exp 3 / exp 5).

## Setup

Same as [baseline](2026-04-25-baseline.md) plus `--use_splash=True`. Image
`hf-v2`. New code in [`train.py`](../train.py) (≈25 lines) installs the
splash kernel via the canonical pattern from
[raw/code/torchax/examples/train_llama_torchtitan/](../../../../../raw/code/torchax/examples/train_llama_torchtitan/train_llama.py):

```python
attn_partition = P("fsdp", "tp", None, None)
_splash = jax.jit(functools.partial(splash_attn.tpu_splash_attention, mesh, attn_partition, True))
def _custom_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                      is_causal=False, scale=None, enable_gqa=False):
    jq, jk, jv = jax_view((query, key, value))
    return torch_view(_splash(jq, jk, jv, None))
torchax.default_env().override_op_definition(F.scaled_dot_product_attention, _custom_attention)
```

GQA (Hq=32, Hkv=8) handled natively by `make_splash_mha`.

## Results

| Metric | Baseline | Exp 2 (splash) | Δ |
|---|---|---|---|
| Cold compile | 92 s | 92 s | flat |
| Steady step time | 446 ms | 437 ms | **−2.0 %** |
| Throughput (aggregate) | 36,729 TPS | **37,299 TPS** | **+1.6 %** |
| Per-chip TPS | 4,591 | 4,662 | +1.5 % |
| MFU | 22.9 % | 23.3 % | +0.4 pp |

Δ is below the run-to-run noise band (~±2 %).

## Profile

Profile not pulled — the marginal gain at this shape is not where the win
lives. Captured trace tarball was discarded with the workload cleanup
following exp 3's win.

## Verdict + reasoning

**Potential** — the override itself is **not the win at this shape**, but
the change is the precondition for both:

1. [Exp 3 (`splash bs=4`)](2026-04-25-exp3-splash-bs4-accepted.md) — splash
   frees enough HBM that bs=4 (which OOMs on the baseline by 1 GiB on v6e-8)
   now fits. **+55.6 % TPS / +12.8 pp MFU.**
2. [Exp 5 (`splash bs=2 seq=2048`)](2026-04-25-exp5-splash-seq2k-accepted.md)
   — splash makes attention `O(L)` so `seq=2048` becomes feasible. **+45.6 %
   TPS / +11.1 pp MFU.**

Splash itself is fast at this shape (B·L = 2,048 tokens-per-chip) — there is
not enough attention work to overlap or compress meaningfully — but the
attention-memory savings unblock larger compute density, which is where the
gain lives.

## See also

- [Baseline](2026-04-25-baseline.md).
- [Exp 1 (XLA recipe flags)](2026-04-25-exp1-xla-recipe-flags-rejected.md) — the prior pivot point.
- [Exp 3 (splash bs=4)](2026-04-25-exp3-splash-bs4-accepted.md) — the splash-enables-this win.

## Sources

- [`train.py` on this branch](../train.py).
- [`splash_attn.py`](../splash_attn.py) — splash kernel wrapper.
