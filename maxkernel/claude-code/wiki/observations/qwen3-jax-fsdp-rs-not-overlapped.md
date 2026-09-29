---
title: "FSDP gradient reduce-scatter runs synchronously on qwen3-jax v6e-8 (12.9% of step)"
type: observation
tags: [qwen3-cc, jax, fsdp, collective, reduce-scatter, profile]
created: 2026-06-02
updated: 2026-06-02
---

# FSDP gradient reduce-scatter runs synchronously on qwen3-jax v6e-8

On the v018 frontier (remat + splash + bs4 + seq2048 + MaxText HOST_OFFLOAD scheduler flags), the
FSDP **backward gradient reduce-scatter** is the single largest non-matmul cost: **12.9% of step time
(~149 ms at 1,154 ms/step)**, ICI-bandwidth-bound, and **not pipelined with the backward matmuls** —
it runs effectively synchronously, 362 per-tensor instances (one per weight per layer).

## What was observed

- Profile bucket attribution (v018, profile-analyzer, steps 12–14): matmul 51.4%, **all-reduce-scatter
  fusion 12.9%**, splash 12.3%, loop-fusion 11.0%, data-formatting 4.5%.
- The reduce-scatter lowers as the classic XLA `all-reduce → dynamic-slice` pattern wrapped in a kCustom
  fusion. The `--xla_tpu_enable_ag_backward_pipelining` flag (active in v018) overlaps the *forward
  all-gather*, but there is no corresponding overlap of the *backward reduce-scatter* in the current
  schedule.
- MXU util is 48.3% — there is idle MXU time that overlapped collectives could fill.

## How to see it

- xprof: run `2026-06-02-qwen3-jax-v018-xla-flag-stack`, Op Profile → `all-reduce-scatter fusion` bucket.
- HLO: `module_0209.jit_train_step` ENTRY — 362 `all-reduce`→`dynamic-slice` instances; no async wrapper
  around the reduce-scatter (unlike the all-gathers which carry `AsyncCollectiveStart/Done`).

## Implications

- This is the **#1 addressable lever** on the seq2048 frontier (~7–13 pp MFU potential per the analyzer).
- The fix is a **compiler/scheduler lever** (async-collective-fusion to overlap RS with compute), which is
  the lever category that *transfers* to qwen3-jax — unlike collective *offload* (SparseCore, refuted
  v003/v017, which moved the collective off-TC but added net overhead). Overlap ≠ offload.
- Tested by [v023](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v023-async-collective-fusion.md)
  (`qwen3-jax-async-collective-fusion` hypothesis).

## Seen in

- [v018](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v018-xla-flag-stack.md) (frontier profile).
