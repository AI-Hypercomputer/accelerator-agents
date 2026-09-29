---
title: "Exp 39 — JAX splash + batch=4 (REJECTED, −6.4%; memory-pressure regression)"
type: experiment
tags: [experiment, gemma4, jax, batch-growth, memory-ceiling, rejected]
hypothesis: jax-batch4-fits-and-wins
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: pending
verdict: refuted
---

Bumped batch from 3 (exp 36, the JAX-stack best) to 4. **Batch=4 FITS on the JAX stack** where torchax exp 22 compile-time OOM'd at the same config — but **TPS regresses −6.4 % (34,614 → 32,402)**. Memory pressure at the HBM ceiling slows per-token execution: step time 355 → 506 ms for 4/3 tokens, giving per-token 28.88 → 30.86 µs.

## Hypothesis

If JAX port's HBM footprint is lower than torchax's (as the 27.11 GiB / 86.75 % utilization at exp 36 suggested vs torchax's 30.87 GiB / 98.78 % at b=3), batch=4 might fit and deliver +~10 % TPS by amortizing the fixed per-step overhead over more tokens.

Torchax analog: [`exp22 batch=4 fused_bwd`](../../torchax/experiments/2026-04-23-exp22-batch4-fused-bwd-rejected.md) **compile-OOM'd** by ~1.25 GiB.

## Setup

- Branch: `main`.
- Zero code changes. `JAX_ATTENTION_IMPL=splash` + `--batch_size 4 --seq_len 1024 --steps 20`. Global batch = 16 (4 × fsdp=4).
- Command:
  ```
  cd wiki/experiments/gemma4_autoresearch_optimization/jax
  JAX_ATTENTION_IMPL=splash python -m train --steps 20 --batch_size 4 --seq_len 1024 \
    --profile_dir raw/profiles/2026-04-23-gemma4-jax-exp39-splash-batch4 --profile_steps 10 11 12
  ```

## Results

| Metric | Exp 36 (b=3) | **Exp 39 (b=4)** | Δ |
|---|---|---|---|
| Tokens / step (global) | 12,288 | 16,384 | +33 % |
| Step time (mean, steps 2–19) | 355.0 ms | **505.64 ms** | +42 % |
| **TPS** | **34,614** | **32,402** | **−6.4 %** |
| **MFU** | **23.05 %** | **21.58 %** | **−1.47 pts** |
| Per-token cost | 28.88 µs | 30.86 µs | +6.9 % |
| Compile step 0 | 132 s | 168.58 s | +28 % |
| Loss descent | clean | clean (3.67 → 2.17) | match within noise |

Step time 505.6 ms → TPS 32,402 = 1.42× step for 1.33× tokens. Per-token cost is **worse**, not better.

## Mechanism

Two forces at batch=4:
1. **Good**: amortization of fixed per-step overhead (FSDP all-gather, collective-permute, launch latency). Would give +TPS if memory were free.
2. **Bad**: activation memory scales linearly with batch, pushes HBM past the optimal region where live-set fits comfortably in the compute-friendly portion of HBM. Effect = more spilling, more reshuffle, larger working set for the scheduler → slower per-step.

At b=3 on JAX we had 27.11 GiB peak (86.75 %). b=4 should have pushed ~36 GiB — but it **fit** rather than compile-OOM'd. That's a win over torchax's planner (torchax exp 22 saw the same 1.25-GiB-over compile-time OOM pattern as exp 10/11/23). Possible reasons:
- Flax NNX param tree has slightly lower peak-instantaneous HBM than torchax's (torchax's `JittableModule` duplicates during dedup).
- The JAX port's selective-remat policy has different compile-time accounting.

But the fit-but-slower result confirms **batch=3 is the sweet spot** on v6e-4 for Gemma 4 E4B on BOTH stacks — just like the torchax exp 7 → exp 15 arc (b=3 with bf16 CE was the high-water mark on torchax before splash; same pattern here).

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-jax-exp39-splash-batch4](http://localhost:8791/?run=2026-04-23-gemma4-jax-exp39-splash-batch4) — opens the interactive trace viewer.
- **Run name**: `2026-04-23-gemma4-jax-exp39-splash-batch4`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-jax-exp39-splash-batch4/`](../../../../../raw/profiles/2026-04-23-gemma4-jax-exp39-splash-batch4/) (gitignored; mirrored at `gs://<your-bucket>/autoresearch/2026-04-23-gemma4-jax-exp39-splash-batch4/`).
- **Steps captured**: 10, 11, 12.
- **What's inside**: xprof trace of b=4 splash — use to see which op bucket grew most (expected: loop_fusion and memory-traffic ops).

## Verdict

**REJECTED / REFUTED.** Not merged. Exp 36 (b=3) remains JAX-stack best at 34,614 TPS / 23.05 % MFU.

This confirms: **b=3 is the sweet spot on v6e-4 for Gemma 4 E4B** — both stacks converge on this.

## Next hypotheses

- **exp 40 — seq=2048 batch=2** (a different shape of batch growth). Torchax exp 10 / 11 / 23 all compile-OOM'd at this config; JAX's lower compile-time HBM might close that gap. If it fits, global tokens/step = 16,384 (same as b=4) but the attention N² work is larger per token, possibly shifting the compute/memory balance.
- **exp 41 — scan-over-layers**. Compile-time win, not TPS — but valuable for iteration speed. JAX port owns the layer loop directly, so torchax's ScannedModule blockers don't apply.
- **exp 42 — Pallas RMSNorm with hand-rolled `custom_vjp`** on JAX. Torchax exp 33 refuted it as fusion-boundary tax. JAX graph may behave differently. Bounded downside (kernel is numerically correct; risk is the same -X% pattern).

## See also

- [exp 36 — JAX splash + b=3 (current best)](2026-04-23-exp36-jax-splash-batch3-accepted.md)
- [torchax exp 22 — b=4 compile-OOM](../../torchax/experiments/2026-04-23-exp22-batch4-fused-bwd-rejected.md)
- [torchax exp 7 — b=3 without bf16 CE, HBM 97.6 %, regression](../../torchax/experiments/2026-04-23-exp7-selective-batch3-rejected.md) — the other end of the "fit but slow" pattern.

## Sources

- `/tmp/gemma4_jax_exp39.log`
- `raw/profiles/2026-04-23-gemma4-jax-exp39-splash-batch4/` — xprof run `2026-04-23-gemma4-jax-exp39-splash-batch4` at http://localhost:8791/?run=2026-04-23-gemma4-jax-exp39-splash-batch4
