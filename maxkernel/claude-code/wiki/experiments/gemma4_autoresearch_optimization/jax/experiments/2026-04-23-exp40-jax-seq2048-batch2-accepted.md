---
title: "Exp 40 — JAX seq=2048 b=2 splash (ACCEPTED; dominated by b=3 s=1024 but long-seq capability unlocked)"
type: experiment
tags: [experiment, gemma4, jax, long-seq, batch-growth, capability-unlock]
hypothesis: jax-seq2048-b2-fits
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: pending
verdict: supported
---

Seq=2048 batch=2 runs on the JAX stack. **Torchax exp 10 / 11 / 23 all compile-OOM'd at this config** (all hit the ~1.25 GiB XLA compile-time planner margin). JAX stack fits cleanly — capability unlocked. TPS 31,809 (MFU 21.18 %), −8.1 % vs exp 36's b=3 s=1024 best; **dominated** for the current optimization target but preserved as the long-seq reference for the JAX stack.

## Hypothesis

Torchax exp 22 / 23 compile-OOM'd with a reproducible ~1.25 GiB margin, suggesting the XLA compile-time peak was dominated by torchax-specific scaffolding (JittableModule dedup, HF cache-class pytrees, torch→jax view dtypes, etc.). The JAX port has none of that. Hypothesis: **compile fits at seq=2048 b=2 on JAX**.

Secondary: what TPS does this config deliver? Expected: dominated by b=3 s=1024 (exp 36) since both have 16,384 global tokens/step but s=2048 has 4× attention N² work.

## Setup

- Branch: `main`.
- Zero code changes from exp 36. Config: `JAX_ATTENTION_IMPL=splash`, `--batch_size 2 --seq_len 2048 --steps 20`. Global batch = 8 (2 × fsdp=4), tokens/step = 16,384.
- Command:
  ```
  cd wiki/experiments/gemma4_autoresearch_optimization/jax
  JAX_ATTENTION_IMPL=splash python -m train --steps 20 --batch_size 2 --seq_len 2048 \
    --profile_dir raw/profiles/2026-04-23-gemma4-jax-exp40-seq2048-batch2 --profile_steps 10 11 12
  ```

## Results

| Metric | Exp 36 (b=3 s=1024) | Exp 39 (b=4 s=1024) | **Exp 40 (b=2 s=2048)** |
|---|---:|---:|---:|
| Tokens / step (global) | 12,288 | 16,384 | 16,384 |
| Step time (mean, steps 2–19) | 355.0 ms | 505.64 ms | **515.07 ms** |
| **TPS** | **34,614** | 32,402 | **31,809** |
| **MFU** | **23.05 %** | 21.58 % | **21.18 %** |
| Per-token cost | 28.88 µs | 30.86 µs | **31.44 µs** |
| Compile step 0 | 132 s | 169 s | **165.57 s** |
| Loss descent | clean | clean | **3.28 → 2.08** clean |

Compile **FIT** at 165 s — no `CompileTimeHbmOom`. That's the structural data point.

## Mechanism — JAX port has smaller compile-time peak than torchax

The torchax exp 10 / 11 / 22 / 23 pattern was "peak HBM at compile time = actual-use + 1.25 GiB of XLA planner overhead that doesn't materialize at runtime." The fixed 1.25 GiB margin across four different torchax configs implicated the torchax pytree (HF cache classes, JittableModule dedup duplicates during compile trace). The native-JAX port's pytree is leaner (Flax NNX `nnx.Param` tree, no PyTorch scaffolding layer) — enough to close the ~1.25 GiB gap.

This is the same-kind-of-win as exp 36's +13.9 %: removing a layer of framework overhead that doesn't participate in actual compute or storage.

## Why TPS is lower than b=3 s=1024

Same 16,384 tokens/step as exp 39 b=4 — both sit at the memory-pressure regime. At s=2048, attention N² = 4× vs s=1024 (2048² vs 1024²) per token, but batch is halved so total attention FLOPs per step are 2× (not 4×) higher. Meanwhile MLP / matmul work scales linearly with token count (same between the two configs). The extra 2× attention FLOPs at s=2048 don't amortize as well as the extra batch slots at s=1024 b=4, explaining why both configurations land at ~500 ms step vs exp 36's 355 ms.

Per-token cost progression:
- exp 36 (b=3 s=1024, optimal): 28.88 µs / token
- exp 39 (b=4 s=1024, HBM-pressured): 30.86 µs / token (+6.9 %)
- exp 40 (b=2 s=2048, long-seq): 31.44 µs / token (+8.8 %)

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-jax-exp40-seq2048-batch2](http://localhost:8791/?run=2026-04-23-gemma4-jax-exp40-seq2048-batch2) — opens the interactive trace viewer.
- **Run name**: `2026-04-23-gemma4-jax-exp40-seq2048-batch2`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-jax-exp40-seq2048-batch2/`](../../../../../raw/profiles/2026-04-23-gemma4-jax-exp40-seq2048-batch2/) (gitignored; mirrored at `gs://<your-bucket>/autoresearch/2026-04-23-gemma4-jax-exp40-seq2048-batch2/`).
- **Steps captured**: 10, 11, 12.
- **What's inside**: xprof trace at seq=2048 b=2 — the long-seq profile for the JAX stack. Splash's N² elimination visible here; comparing with exp 36's b=3 s=1024 trace shows where attention cost scales vs where MLP / other scales.

## Verdict

**ACCEPTED.** Not a new TPS best, but supports the structural hypothesis (JAX port compile-time planner has less overhead than torchax → long-seq configs fit that torchax couldn't). Preserved as the JAX-stack long-seq reference point — parallel to torchax exp 28 (also accepted-but-dominated at seq=2048 b=1).

## Next hypotheses

- **exp 41 — scan-over-layers**. JAX port owns the layer loop directly (no `ScannedModule` blocker). Biggest remaining compile-time lever — 150 s step 0 → ~5 s. Step-time neutral expected but still worth filing. Infrastructure win for iteration speed.
- **exp 42 — Pallas RMSNorm on JAX** (retest of torchax exp 33). Same −8 % fusion-boundary risk, but JAX graph differs; worth a bounded test.
- **exp 43 — persistent JAX compile cache** (`JAX_COMPILATION_CACHE_DIR`). Not TPS; 30× iteration speed on cache hits. Trivial.
- **exp 44 — step-1 recompile root cause**. Pinning `out_shardings` or donation annotations. On JAX stack no tied-weight issue (no torchax dedup); might be simpler than on torchax exp 2.

## See also

- [exp 36 — current JAX best (b=3 s=1024)](2026-04-23-exp36-jax-splash-batch3-accepted.md)
- [exp 39 — b=4 s=1024 rejected -6.4%](2026-04-23-exp39-jax-splash-batch4-rejected.md)
- [torchax exp 10 — seq=2048 b=2 compile-OOM (the gap this exp closes)](../../torchax/experiments/2026-04-23-exp10-seq2048-batch2-bf16ce-rejected.md)
- [torchax exp 28 — accepted-but-dominated at seq=2048 b=1 (analog)](../../torchax/experiments/2026-04-23-exp28-seq2048-exp25config-accepted.md)

## Sources

- `/tmp/gemma4_jax_exp40.log`
- `raw/profiles/2026-04-23-gemma4-jax-exp40-seq2048-batch2/` — xprof run `2026-04-23-gemma4-jax-exp40-seq2048-batch2` at http://localhost:8791/?run=2026-04-23-gemma4-jax-exp40-seq2048-batch2
