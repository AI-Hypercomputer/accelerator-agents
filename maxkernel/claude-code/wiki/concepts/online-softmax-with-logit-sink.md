---
title: "online softmax with logit sink"
type: concept
tags: [pallas, attention, softmax, flash-attention, splash-attention, axlearn, stub]
created: 2026-04-23
updated: 2026-04-23
---

Extension to the online-softmax step in flash/splash attention that adds a learnable "sink" logit to the denominator — widely used as a streaming-attention regularizer. *Stub — expand when more sources are available.*

## Definition

Standard online softmax tracks `m = max(scores)` and `l = sum(exp(scores - m))` in a running fashion across KV tiles. The logit-sink variant initializes `m` with the sink value and adds `exp(sink - m_final)` to `l` at the normalize step — so the sink "attracts" attention mass that otherwise floats to tokens.

## Why it matters for TPU perf

The extension is essentially free at the kernel level — one extra `exp` per head per final-normalize. But it changes the numerical stability envelope enough that the stable-softmax path must be aware of it; sink-aware variants of splash attention exist in production.

## Mechanism

1. In the forward online-softmax loop over KV tiles, initialize `m` with the sink logit (not `-inf`).
2. Accumulate `l` and `acc` as usual.
3. At normalize, add `exp(sink - m_final)` to `l` before the final divide.

Applies equally to flash-attention and splash-attention variants.

## When it applies / when it doesn't

- **Applies** to models trained with a sink logit (streaming-LLM family, some xAI/Grok variants, EAGLE-style trees with target-only spec decoding).
- **Does not apply** to vanilla attention — the sink term is zero and the math is unchanged.

## Known results

| Reference | Repo | Notes |
|---|---|---|
| `axlearn/common/flash_attention/tpu_splash_attention.py` | [axlearn](../codebases/axlearn.md) | Splash extension; sink-aware online softmax documented in-source |
| `ragged_paged_attention_v3` (`attention_sink` knob) | [tpu-inference](../codebases/tpu-inference.md) + [sglang-jax](../codebases/sglang-jax.md) | SGLang-specific RPA v3 extension for streaming |

## Connections

- [flash-attention](flash-attention.md)
- [splash-attention](splash-attention.md)
- [base2-softmax](base2-softmax.md) — related softmax-rewrite concern.

## Sources

- [axlearn codebase](../codebases/axlearn.md) "Performance-relevant surfaces §5".
- [Pallas kernel directory §4.1](../analyses/pallas-kernel-directory/04-research-labs.md#41-appleaxlearn).
