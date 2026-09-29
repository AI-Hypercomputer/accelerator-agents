---
title: "Qwen3-jax: tensor-parallel activation sharding (tp>1) to shrink per-chip recompute"
type: hypothesis
model: qwen3-cc-jax
variants: ["8B/v6e-8"]
status: refuted
expected_gain: "REFUTED — v043b (tp2 bs6, equal global batch) −26% tok/s/chip; tp adds all-reduce w/o reducing recompute"
confidence: low
effort: S
origin: human
hlo_prefilter: n/a
tags: [qwen3-cc, jax, tensor-parallel, tp, activation-sharding, megatron, seq8192]
created: 2026-06-05
updated: 2026-06-05
---

# Qwen3-jax: tensor-parallel activation sharding (tp>1)

## Statement

Running `--tp_parallelism=N>1` (Megatron-style: q/k/v/gate/up sharded on the `tp` axis, o/down all-reduced)
shards the MLP/attention activations across the `tp` axis, which — per the v035 profile gap analysis —
*could* reduce the per-chip SiLU/QK-norm recompute that inflates our loop-fusion bucket (17.4% vs MaxText 7.2%),
moving the seq8192 frontier above v035 (34.6% / 6,030).

## Rationale

The [v035 profile](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v035-maxtext-ce-s8k-bs3.md)
attributes ~85% of the MaxText gap to the `nothing_saveable` remat recomputing the SiLU twice
(3,985 ms) over the full per-chip MLP activation `bf16[3,8192,12288]`. MaxText's loop-fusion is 2.4×
smaller partly because it shards `activation_mlp`/`activation_length`. The sharding plan
([model/sharding.py](../../?)) already encodes full Megatron TP, and the trainer plumbs `--tp_parallelism`,
so this is a flag, not a code change.

> [!warning] Counter-argument surfaced by [v043](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-05-v043-tp2-s8k.md) on 2026-06-05
> At **equal global batch**, tp does NOT reduce per-chip activation: it trades batch-sharding for
> model-sharding. With `fsdp=4×tp=2, bs=6` the per-chip MLP activation `[6,8192,6144]` equals v035's
> `[3,8192,12288]` — same bytes, same SiLU recompute. tp only reshapes the matmul N tile and adds
> all-reduce collectives. So tp is unlikely to shrink the recompute gap; the clean control is v043b
> (tp=2 at bs=6, equal global batch).

## Proposed experiment

- v043: tp=2 at bs=3 — REGRESSED (24.7%), but confounded (global batch halved 24→12).
- v043b: tp=2 at **bs=6** (global batch 24, equal to v035) — the clean control. Support: > 6,030. Refute: ≤ 6,030.
- v044: tp=4 — only if v043b shows promise.

## See also

- [Qwen3 8B — jax](../models/qwen3-cc-jax.md) · [v035](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v035-maxtext-ce-s8k-bs3.md)
