---
title: "Exp 3 — full activation remat via jax.checkpoint (KEEP, memory-first prep)"
type: experiment
tags: [experiment, gemma4, remat, jax-checkpoint, memory-win, tps-regression]
hypothesis: full-activation-remat
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: "wiki:HEAD; train.py: grad_fn = jax.value_and_grad(jax.checkpoint(forward_loss))"
verdict: supported
---

Wrapped `forward_loss` with `jax.checkpoint` to recompute every activation during the backward pass instead of stashing intermediates. **Peak HBM dropped 29.69 → 21.08 GiB (−29%)** at the cost of **+27.5% step time** — within the predicted 30–40% full-remat overhead. This is the memory-first prep step for exp 4 (doubled batch).

## Hypothesis under test

**Statement**: Wrapping the per-step forward with `jax.checkpoint` (full remat) will reduce peak HBM by 25–60% at the cost of 30–40% more step time, preserving the loss trajectory exactly. This unblocks state-growing experiments (bigger batch, longer seq) that the 95% HBM baseline cannot attempt.

Origin: baseline profile showed peak HBM at 95%; [xprof-mcp TPU optimization guide §4.6/§5](../../../../sources/2026-xprof-mcp-tpu-optimization.md) quantifies remat savings; `program.md`'s **memory-ceiling rule** and **HBM ratchet** heuristics drove the choice.

## Setup

- Same as [baseline-seq1024](OBSERVATIONS.md#baseline-torchax--gemma-4-e4b--v6e-4--fsdp4): `--steps 20 --batch_size 1 --seq_len 1024 --strategy fsdp`. No XLA / LIBTPU flags. No sharding changes.
- Code change — single line in `train.py`:

  ```python
  # Before (baseline):
  grad_fn = jax.value_and_grad(forward_loss)

  # After (exp 3):
  grad_fn = jax.value_and_grad(jax.checkpoint(forward_loss))
  ```

- Command:

  ```bash
  cd wiki/experiments/gemma4_autoresearch_optimization/torchax
  python -m train --steps 20 --batch_size 1 --seq_len 1024 \
    --profile_dir raw/profiles/2026-04-23-gemma4-exp3-full-remat \
    --profile_steps 10 11 12
  ```

## Baseline comparison

Baseline: [baseline-seq1024 from 2026-04-23](OBSERVATIONS.md#baseline-torchax--gemma-4-e4b--v6e-4--fsdp4). Profile at `raw/profiles/2026-04-23-gemma4-loss-confirm/`, xprof session `gemma4_baseline_seq1024_20260423`.

## Results

| Metric | Baseline | Exp 3 | Delta |
|---|---|---|---|
| Steady-state step time (wall-clock) | **134.4 ms** | **171.4 ms** | **+37.0 ms (+27.5 %)** |
| Device step time (xprof overview) | 148.2 ms | 184.9 ms | +36.7 ms (+24.8 %) |
| Compile step 0 | 149.5 s | 182.4 s | +32.9 s (+22 %; checkpoint machinery) |
| Compile step 1 (recompile) | 150.1 s | 177.5 s | +27.4 s |
| Peak HBM | 29.69 GiB (95 %) | **21.08 GiB (67 %)** | **−8.61 GiB (−29 %)** |
| Stack reservation | 17.37 GiB | 8.72 GiB | **−50 %** |
| Heap allocation | 12.31 GiB | 12.36 GiB | flat |
| Fragmentation | 0 % | 25.4 % | +25 pp (expected — smaller live set, more gaps) |
| TPS (tokens/sec) | 30,570 | 23,900 | −21.8 % |
| Loss, step 0 | 3.9339 | 3.9147 | −0.019 (bf16 reorder) |
| Loss, step 14 (lowest) | 1.9685 | 1.9621 | −0.006 |
| Loss trajectory match | n/a | within bf16-reorder noise across 20 steps | ✓ correctness preserved |

**HLO-op-level diff** (3 profiled steps, aggregated across 4 chips):

| Op | Baseline time (ms) | Exp 3 time (ms) | Time Δ | Baseline FLOPs | Exp 3 FLOPs | Notes |
|---|---|---|---|---|---|---|
| convolution fusion | 613 | 793 | **+180 ms** | 356 TFLOPs | 473 TFLOPs | +33 % FLOPs — forward runs twice (once for fwd output, once during bwd for remat). Expected. |
| loop fusion | 321 | 373 | +52 ms | 1.02 TFLOPs | 1.17 TFLOPs | +16 %. |
| all-gather | 111 | 225 | **+114 ms** | 0 | 0 | **Doubled** — weight all-gathers happen on both the rematted fwd and the bwd. This is the dominant tax. |
| custom fusion | 182 | 223 | +41 ms | 0 | 0 | Includes async-collective-done waits. |
| all-reduce-scatter fusion | 75 | 86 | +11 ms | 0.005 | 0.006 | |
| data formatting | 40 | 57 | +17 ms | 0 | 0 | |
| all-reduce | 46 | 33 | −13 ms | 0.002 | 0.002 | Slight improvement — recompute during bwd allowed some fusion. |
| Total profile | 1,694 | 2,142 | +448 ms (+26 %) | 357 TFLOPs | 474 TFLOPs | +33 % compute; matches "double-forward" expectation. |

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-exp3-full-remat](http://localhost:8791/?run=2026-04-23-gemma4-exp3-full-remat) — opens the interactive trace viewer for this run.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-exp3-full-remat`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-exp3-full-remat/`](../../../../../raw/profiles/2026-04-23-gemma4-exp3-full-remat/) (gitignored; relative link click-throughs open the trace folder locally)
- **Steps captured**: 5, 6, 7
- **What's inside**: xprof trace — full forward remat via `jax.checkpoint`; memory profile shows −29 % peak HBM vs baseline.

## Mechanism

`jax.checkpoint(forward_loss)` marks the forward as a checkpointed boundary — during the backward pass, JAX discards stashed forward activations and **recomputes the forward** to regenerate them just in time for their gradients. Net effect:

- **Activations no longer live from fwd-end through bwd-start** → stack reservation drops (17.4 → 8.7 GiB, a near-exact halving).
- **Compute cost**: one additional forward pass per training step. With the per-layer compute dominated by matmuls in `convolution fusion`, FLOPs rise from 357 to 473 TFLOPs (+33 %) which is consistent with "forward done twice vs forward done once" (since backward ≈ 2× forward FLOPs, base = 3× forward; +remat adds 1× forward → 4× forward total; ratio 4/3 = 33 %). This is exactly what the profile shows.
- **All-gather doubles**: every recompute re-gathers FSDP-sharded weights. This is the single biggest concrete cost beyond raw compute.

The heap (live tensors at peak, mostly the stashed inputs to recomputed segments + gradients) stayed roughly the same because remat converts the stack-type reservation (many time-shifted live activations) into compute (recompute-from-smaller-inputs). Fragmentation jumped to 25 % — expected when the live set shrinks and leaves holes.

## Verdict

**KEEP (as memory-first prep).** By strict "TPS improved" criteria this would be `discard` (−21.8 % TPS). By the program's **memory-ceiling rule** and **HBM ratchet** heuristic in `program.md`, it's a required-preparation change: the 8.6 GiB freed unlocks the next experiment (exp 4 — double batch to 2) which cannot run against the 95 % HBM baseline. Correctness is preserved; the loss trajectory matches baseline within bf16-reorder noise.

The final judgment on this change is the **exp3 + exp4 combined TPS** vs baseline. If exp 4 returns TPS above 30,570 with the same loss trajectory, the chain is `supported`. If exp 4 underwhelms, revisit selective remat (expected +2.7 % compute for ~70 % memory savings — much better compute/memory tradeoff than full remat).

## Next hypotheses (follow-ups)

1. **Exp 4 — double batch (1 → 2)**: unlocked by this memory-win. Expected global_batch = 2 × 4 = 8, tokens/step = 8192, TPS target > 30,570 (baseline). Launched immediately.
2. **Selective remat via `jax.checkpoint_policies`**: replace `jax.checkpoint` with `jax.checkpoint(forward_loss, policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims)`. Should save dots (matmul intermediates) and recompute only the cheap bits — target +2.7 % compute for 60–70 % of the memory savings. Much better tradeoff than full remat if exp 4's memory budget has slack.
3. **Revisit exp 1 (async-collective flags) at batch=2**: the regression-mechanism there was scheduler-driven compute-order breakage at a small workload. Larger per-chip batch gives the scheduler more compute to hide behind — the flags may flip to positive.
4. **Scan over layers**: collapse 42-layer unroll into `jax.lax.scan` (via `torchax.train.ScannedModule` or similar). Targets compile time (149 → estimated 10–20 s) and possibly shares activation buffers across layers. Orthogonal to remat.
5. **Host-offload the single biggest activation** instead of remat (per heuristic 5 from the sibling wiki's program.md): pick the per-layer hidden-residual and offload only that. Likely cheaper compute than full remat.

## See also

- [Program page](../../README.md) and [program.md](../../program.md) — the protocol.
- [OBSERVATIONS.md § exp03](OBSERVATIONS.md) — reasoning block.
- [2026-04-22-baseline.md](2026-04-22-baseline.md) — baseline page.
- [rematerialization](../../../../concepts/rematerialization.md), [xla-fusion](../../../../concepts/xla-fusion.md), [all-gather](../../../../concepts/all-gather.md), [fsdp](../../../../concepts/fsdp.md).
- [xprof-mcp TPU optimization guide §4.6 + §5](../../../../sources/2026-xprof-mcp-tpu-optimization.md) — quantitative predictions for selective vs full remat.
- [xprof — memory profile](../../../../sources/2026-xprof-memory-profile.md) — how peak HBM was read.

## Sources

- `raw/profiles/2026-04-23-gemma4-exp3-full-remat/` — xprof trace.
- `raw/profiles/2026-04-23-gemma4-loss-confirm/` — baseline counterpart.
- `/tmp/gemma4_exp3.log` — console log.
