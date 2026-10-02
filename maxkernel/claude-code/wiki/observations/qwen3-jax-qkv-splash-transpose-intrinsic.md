---
title: "QKV→splash (B,T,H,D)→(B,H,T,D) transpose is intrinsic — not a removable lever (qwen3-jax)"
type: observation
tags: [qwen3-cc, jax, splash, data-formatting, layout]
created: 2026-06-02
updated: 2026-06-02
---

# QKV→splash transpose is intrinsic on qwen3-jax

The v018 frontier profile flagged a **data-formatting bucket = 4.5% of step** (GQA layout bridges
between the QKV projections and the splash kernel). Investigation (2026-06-02) shows this is **not a
removable lever** by any reshape/sharding change.

## What was observed

- `model/modeling_qwen3.py` Qwen3Attention.__call__: projections emit `(B, T, H·D)`, reshaped to
  `(B, T, H, D)`, then **transposed `(0,2,1,3)` → `(B, H, T, D)`** (lines 217–222) before `_attn_splash`.
- `splash_attn.py` `tpu_splash_attention`: the kernel block sizing uses `query.shape[2]` as the
  **sequence** length (`block_q = min(global_block_q, query.shape[2])`), confirming splash expects
  `(B, H, T, D)` layout. (The function's `(batch, length, heads, kv_dim)` docstring comment is **stale**.)

## Implications

- The `(B,T,H,D) → (B,H,T,D)` transpose moves the head axis ahead of the sequence axis. A reshape
  **cannot** reorder T and H (they are distinct axes); only an explicit transpose can. So the layout
  bridge is **structurally required** to feed any `(B,H,T,D)` attention kernel from `(B,T,H·D)` matmul
  output. XLA already partially fuses it (only 4.5% residual).
- The **only** way to remove it would be a custom Pallas attention kernel that ingests `(B,T,H,D)`
  directly (fusing the transpose into the kernel prologue) — effort L, and qwen3-jax's established
  pattern is that kernel-swap levers regress (tokamax-splash −8.5%, v020). Low prior; not pursued.
- **Data-transformation overhead topic = CLOSED for this lane** (no clean lever; the residual 4.5% is
  near-irreducible without deep kernel work).

## Seen in

- [v018](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v018-xla-flag-stack.md) (frontier profile, data-formatting 4.5%).
- Investigation: retrospective rec #1 follow-up, 2026-06-02.
