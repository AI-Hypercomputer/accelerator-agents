---
title: "Qwen3-jax: splash bkv=2048 (MaxText value) at the maxtext-CE frontier"
type: hypothesis
model: qwen3-cc-jax
variants: ["8B/v6e-8"]
status: refuted
expected_gain: "REFUTED — v040 parity (+0.05%); bkv=1024 already optimal"
confidence: low-medium
effort: S
origin: 2026-06-02-qwen3_cc-jax-retrospective-5-maxtext-delta
hlo_prefilter: n/a
tags: [qwen3-cc, jax, splash, splash-block-tuning, maxtext, seq8192]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-jax: splash bkv=2048

## Statement

MaxText's qwen3-8b reference uses `sa_block_kv=2048` (all splash blocks 2048); our frontier uses bkv=1024
(bq already 2048). Matching MaxText's all-2048 at the maxtext-CE frontier (v035) lifts throughput > 6,030.

## Rationale

[Retrospective #5](../analyses/2026-06-02-qwen3_cc-jax-retrospective-5-maxtext-delta.md) — the splash block
config is the last cheap config-level delta vs MaxText. Larger KV blocks may reduce kernel-boundary crossings.
Prior block sweeps (v021 bq1024 refuted, v022 bq4096 inconclusive) were **pre-CE** and didn't isolate bkv=2048
at the current frontier; low-medium prior (our bkv=1024 was the v009 autotune winner, but never re-checked
against MaxText's 2048 at the maxtext-CE bs3 frontier).

## Proposed experiment

- v040: v035 config + `SPLASH_BKV=2048 SPLASH_BKV_COMPUTE=2048`. Support: > 6,030. Refute: ≤ 6,030 / VMEM-OOM.

## See also

- [Retrospective #5](../analyses/2026-06-02-qwen3_cc-jax-retrospective-5-maxtext-delta.md) · [Qwen3 8B — jax](../models/qwen3-cc-jax.md)
