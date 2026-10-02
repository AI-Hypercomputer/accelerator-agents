---
title: "compile-time-aware autotune candidate filtering"
type: concept
tags: [autotuning, xla-compile-time, marin, levanter, stub]
created: 2026-04-23
updated: 2026-04-23
---

Discard autotune candidates whose XLA compile time alone dominates the training-step wall-clock. marin/levanter constant: `_AUTOTUNE_COMPILE_HIT_THRESHOLD_S = 0.20` seconds. *Stub — expand when more sources are available.*

## Definition

A Pallas kernel config that is 5% faster at steady state but takes 30 seconds to compile is strictly worse than a slower config that compiles in 2 seconds — at least until the compile cache is warm. Compile-time-aware filtering evicts candidates whose XLA compile cost exceeds a threshold (marin: 0.20 s over baseline compile).

## Why it matters for TPU perf

At autoresearch scale, hundreds of training jobs hit PJRT compile cache cold. A tuner that ignores compile cost picks the fastest-at-steady-state config even when that config's cold-start cost eats weeks of kernel-optimization gain. The filter maps compile-cost into the optimization objective.

## Mechanism

1. For each candidate config, call `jax.jit(fn).lower(*args).compile()` and measure wall-clock.
2. Compute `compile_excess = compile_time - baseline_compile_time`.
3. If `compile_excess > threshold` (0.20 s in marin), discard.
4. Among the remaining candidates, pick the fastest by kernel wall-time.

Works with off-thread compile (`ThreadPoolExecutor(max_workers=1)`) so the main JIT / mesh-bound context isn't blocked.

## When it applies / when it doesn't

- **Applies** to any autotune harness used across multiple shape / flag permutations (cold-start is common).
- **Does not apply** when the tuner's output is pinned / cached persistently and compile cost is amortized across many runs.

## Known results

| Reference | Repo | Notes |
|---|---|---|
| `_AUTOTUNE_COMPILE_HIT_THRESHOLD_S = 0.20` in `fused_cross_entropy_loss/api.py` | [marin](../codebases/marin.md) | Canonical constant; paired with `should_offload_compile` and `compile_benchmark_fn` |

## Connections

- [autotuning](autotuning.md)
- [jaxpr-hash-cache-keys](jaxpr-hash-cache-keys.md)
- [vmem-oom-fallthrough](vmem-oom-fallthrough.md)

## Sources

- [marin codebase](../codebases/marin.md) "Performance-relevant surfaces §1".
- [Pallas kernel directory §5.8](../analyses/pallas-kernel-directory/05-frameworks-quant.md#58-marin-communitymarin-vendors-levanter).
