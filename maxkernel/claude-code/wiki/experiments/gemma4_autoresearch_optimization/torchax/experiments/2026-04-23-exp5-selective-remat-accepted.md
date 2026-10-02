---
title: "Exp 5 — selective remat (checkpoint_dots_with_no_batch_dims) (KEEP, memory-win low tax)"
type: experiment
tags: [experiment, gemma4, remat, selective, jax-checkpoint-policies]
hypothesis: selective-activation-remat
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: "wiki:HEAD; train.py: jax.checkpoint(forward_loss, policy=checkpoint_dots_with_no_batch_dims)"
verdict: supported
---

Replaced exp 3's full remat with the **selective** policy `checkpoint_dots_with_no_batch_dims`: save the matmul outputs (dots), recompute only the cheap elementwise intermediates. Mechanism-wise this is a much better trade — +8.7 % step time (vs full remat's +27.5 %), no forward-doubling of all-gather, and **better** peak HBM (19.3 vs 21.1 GiB). Unlocks exp 6 (batch=2) which becomes the first TPS win.

## Hypothesis under test

**Statement**: Full remat (exp 3) paid two costs — +33 % FLOPs (forward runs twice) and doubled all-gather (FSDP weight gather runs twice). Selective remat via `jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims` keeps the expensive dots and recomputes only the cheap bits. Predicted: +5–10 % step time, 50–70 % of exp 3's memory savings, **no forward-doubling**.

Origin: [xprof-mcp TPU optimization guide §4.6/§5](../../../../sources/2026-xprof-mcp-tpu-optimization.md); [rematerialization concept](../../../../concepts/rematerialization.md); exp 3's mechanism analysis pointed at the doubled-gather as the dominant cost.

## Setup

- Config same as [baseline-seq1024](OBSERVATIONS.md#baseline-torchax--gemma-4-e4b--v6e-4--fsdp4): `--steps 20 --batch_size 1 --seq_len 1024`.
- Code change — single spot in `train.py`:
  ```python
  from jax import checkpoint_policies as _ckpt_policies
  grad_fn = jax.value_and_grad(
      jax.checkpoint(forward_loss, policy=_ckpt_policies.checkpoint_dots_with_no_batch_dims)
  )
  ```
- Command: `python -m train --steps 20 --batch_size 1 --seq_len 1024 ...`

## Results

| Metric | Baseline | Exp 3 (full remat) | Exp 5 (selective) | Δ vs baseline | Δ vs exp 3 |
|---|---|---|---|---|---|
| Step time (wall) | 134.4 ms | 171.4 ms | **146.1 ms** | +8.7 % | −14.8 % |
| Compile step 0 | 149.5 s | 182.4 s | 156.1 s | +4.4 % | −14.4 % |
| TPS | 30,570 | 23,900 | 28,035 | −8.3 % | +17.3 % |
| Peak HBM | 29.69 GiB (95 %) | 21.08 GiB (67 %) | **19.32 GiB (62 %)** | −35 % | −8 % (better!) |
| Stack reservation | 17.37 GiB | 8.72 GiB | **7.01 GiB** | **−60 %** | −20 % |
| Heap allocation | 12.31 GiB | 12.36 GiB | 12.31 GiB | flat | flat |
| Free memory | 1.56 GiB | 10.16 GiB | **11.93 GiB** | +10.4 GiB | +1.8 GiB |
| Fragmentation | 0 % | 25 % | 22 % | | |
| Loss trajectory | 3.93 → 1.97 | 3.91 → 1.96 | 3.93 → 1.94 | match | match |

**HLO-op diff vs baseline**:

| Op | Baseline (ms) | Exp 5 (ms) | Baseline FLOPs | Exp 5 FLOPs | Notes |
|---|---|---|---|---|---|
| convolution fusion | 613 | 649 | 356 TFLOPs | **358 TFLOPs** | **No forward-doubling** (exp 3 was 473 TFLOPs). Only a +2 TFLOPs tweak — matches the "save dots, don't recompute" mechanism. |
| loop fusion | 321 | 375 | 1.02 | 1.18 | +54 ms — the recompute cost is the cheap elementwise ops. |
| custom fusion | 182 | 164 | 0 | 0 | **Surprise: −18 ms.** Fewer async-collective-done waits. |
| all-gather | 111 | 123 | 0 | 0 | +11 ms (not doubled like exp 3's 225). |
| all-reduce-scatter fusion | 75 | 82 | 0.005 | 0.006 | +7 ms. |
| data formatting | 40 | 58 | 0 | 0 | +18 ms. |

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-exp5-selective-remat](http://localhost:8791/?run=2026-04-23-gemma4-exp5-selective-remat) — opens the interactive trace viewer for this run.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-exp5-selective-remat`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-exp5-selective-remat/`](../../../../../raw/profiles/2026-04-23-gemma4-exp5-selective-remat/) (gitignored; relative link click-throughs open the trace folder locally)
- **Steps captured**: 5, 6, 7
- **What's inside**: xprof trace — selective remat (`checkpoint_dots_with_no_batch_dims`); peak HBM 62 %, +8.7 % step cost.

## Mechanism

`checkpoint_dots_with_no_batch_dims` is a JAX-documented policy that marks **dot-product outputs without a batch-dim** as "keep" during the checkpointed forward's backward pass. In practice this saves all the matmul intermediates (the expensive stuff) while letting JAX discard and recompute the surrounding elementwise operations (the cheap stuff) during bwd.

Key wins over full remat:
- **No forward-doubling of matmuls**: convolution-fusion FLOPs essentially unchanged (356 → 358 TFLOPs, noise). Full remat went 356 → 473 (+33 %).
- **No doubled all-gather**: 111 → 123 ms (+11). Full remat had 111 → 225 ms (+114).
- **Better memory savings**: stack 17.4 → 7.0 GiB (−60 %) — even better than full remat's 17.4 → 8.7 (−50 %). The policy avoids materializing layout-churn intermediates that full remat was preserving as "input"; selective's saved-dots are contiguous and dense, enabling fewer live buffers overall.

The −18 ms on `custom fusion` is an unexpected minor improvement — likely because the policy reshapes the graph in a way that lets more async-collectives overlap (fewer sync points at the checkpoint boundary). Worth confirming if it replicates; otherwise noise.

## Verdict

**Supported as a memory-win prep.** +8.7 % step-time is inside the predicted range; −35 % peak HBM and 11.9 GiB free memory unlocks batch growth. The selective policy is **strictly better** than full remat on every metric except raw step time (where it's still only mildly worse than baseline). Loss trajectory identical. Kept in the main `train.py` going forward.

Final judgment requires the paired state-growing experiment ([exp 6 — batch=2](2026-04-23-exp6-selective-batch2-accepted.md)), which IS where the TPS win materialized (+1.2 % over baseline).

## Next hypotheses (followed up in exp 6+)

1. **Exp 6 — batch=2 + selective remat**: direct HBM-ratchet follow-up. *Result: +1.2 % over baseline — first real win.*
2. **Exp 7 — batch=3**: push further; at this config the sweet spot is batch=2 because HBM at 97 % starts hurting.
3. **Offload variant**: `jax.checkpoint_policies.offload_dots_with_no_batch_dims` would move saved dots to host memory instead of keeping them in HBM — frees even more HBM. Pending experiment.

## See also

- [program.md](../../program.md), [program page README](../../README.md).
- [OBSERVATIONS.md § exp05](OBSERVATIONS.md#exp05--selective-remat-checkpoint_dots_with_no_batch_dims--keep-memory-first-prep-low-tax).
- [2026-04-23-exp3-full-remat-accepted.md](2026-04-23-exp3-full-remat-accepted.md) — the comparison point.
- [2026-04-23-exp6-selective-batch2-accepted.md](2026-04-23-exp6-selective-batch2-accepted.md) — the paid-off follow-up.
- [rematerialization concept](../../../../concepts/rematerialization.md).

## Sources

- `raw/profiles/2026-04-23-gemma4-exp5-selective-remat/`
- `raw/profiles/2026-04-23-gemma4-loss-confirm/` — baseline counterpart.
- `/tmp/gemma4_exp5.log`.
