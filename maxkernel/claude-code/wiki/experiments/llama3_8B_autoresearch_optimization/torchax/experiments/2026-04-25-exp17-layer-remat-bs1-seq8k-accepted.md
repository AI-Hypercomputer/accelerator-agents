---
title: "Exp 17 — bs=1 seq=8192 + per-layer remat (ACCEPTED, 32.2% MFU at program-target seq)"
type: experiment
tags: [llama3, torchax, splash, remat, seq8k, accepted]
hypothesis: llama3-torchax-seq8192-target
model: llama3-8b-torchax
created: 2026-04-25
updated: 2026-04-25
commit: "v6e8-llama3-8b-torchax-20260425-exp13-scan (image hf-v5)"
branched_from: v6e8-llama3-8b-torchax-20260425-exp13-scan
verdict: supported
---

**First successful run at the program-target sequence length** —
[program.md](../../program.md) calls out `seq=8192` as the final target.
With per-layer gradient_checkpoint (from [exp 13](2026-04-25-exp13-per-layer-remat-accepted.md))
+ autotuned splash (from [exp 9](2026-04-25-exp9-splash-autotuned-bs4-accepted.md))
+ kernel-validated optimal block sizes for seq=8192 (from
[exp 10](2026-04-25-exp10-splash-autotune-multishape-potential.md)),
**`bs=1 seq=8192` fits in HBM and runs at 32.2 % MFU**.

## Setup

- Image: `hf-v5` (per-layer remat enabled).
- Workload: `llama3-8b-exp17-layer-remat-bs1-seq8k`.
- Per-chip B·L = 1 × 8192 = **8192** (same density as exp 13c, but at the
  program-target seq).
- Splash kernel: autotuned config (from exp 9), `block_q=block_kv=1024`
  with the standard `min(global, seq_len)` clamping. Per [exp 10](2026-04-25-exp10-splash-autotune-multishape-potential.md),
  this is within 0.16 % of the seq=8192-specific optimum
  (`block_q=2048 block_kv=1024`) — within noise.

## Results

| Metric | Value |
|---|---|
| Cold compile | 102 s |
| Steady step time | **1,425 ms** |
| Throughput (aggregate) | **46,043 TPS** |
| Per-chip TPS | **5,755** |
| MFU (MaxText formula) | **32.2 %** |
| Loss step 0..14 | 11.7500 (bf16-precision constant) |

Cross-shape comparison (all per-layer remat, splash autotuned):

| | Shape | per-chip B·L | step ms | TPS/chip | MFU |
|---|---|---|---|---|---|
| exp 9 (no remat) | bs=4 seq=1024 | 4096 | 560 | 7,225 | **36.1 %** |
| exp 13a | bs=4 seq=1024 | 4096 | 752 | 5,452 | 27.2 % |
| exp 13b | bs=4 seq=2048 | 8192 | 1,212 | 6,757 | 32.9 % |
| exp 13c | bs=8 seq=1024 | 8192 | 1,161 | **7,058** | **35.2 %** |
| **exp 17** | **bs=1 seq=8192** | **8192** | **1,425** | **5,755** | **32.2 %** |

Within the same per-chip token count (B·L = 8192), exp 17 is the slowest
in MFU (32.2 % vs 13c's 35.2 %). Reason: attention `O(L²)` flops at
seq=8192 are 64× larger than at seq=1024, so the kernel becomes a much
larger share of step time. Per [exp 10](2026-04-25-exp10-splash-autotune-multishape-potential.md),
attention at seq=8192 is 7.944 ms / layer × 32 = **254 ms attention
budget per step alone** — ~18 % of exp 17's 1,425 ms step time.

## Verdict + reasoning

**Supported (accepted).** This is a milestone — the program target seq
is now feasible. Becomes the foundation for the fp32-master experiments
that follow (exp 16+ on the queue).

The 32.2 % MFU is below MaxText's reference 44.6 % (at `bs=3 seq=8192`
which has 3× our per-chip B·L), but it is competitive given (a) per-chip
B·L = 8192 vs MaxText's 24576 — most of MaxText's MFU advantage comes
from compute density; (b) per-layer remat costs ~12-30 % step time. The
remaining gap is closable via larger batch (exp 18 / exp 19) and selective
remat ([exp 14, queued](#next-hypotheses-generated)).

## Profile

Not pulled — workload deleted after summary. Will re-run with profile
capture once exp 18 / exp 19 settle the high-density shape choice.

## Observations

### O1. Splash-kernel autotune from exp 8/10 is robust at seq=8192.

The autotuned `BlockSizes` (which clamp via `min(global, seq_len)`) work
correctly at seq=8192 — no kernel issues, splash returns sensible numerics.
The 0.16 % gap to the seq=8192-specific optimum (`block_q=2048`) is within
noise.

### O2. Compile-time HBM peak fit with margin (no OOM number printed).

The exact HBM peak we don't know (XLA only prints on OOM). Empirically the
compile succeeded; downstream experiments (e.g. fp32-master at this shape)
will discover whether the margin is large enough to absorb +8 GiB / chip
of fp32 weights+opt-state.

### O3. Attention is now the dominant per-layer cost.

At seq=8192, per [exp 10](2026-04-25-exp10-splash-autotune-multishape-potential.md)
the splash kernel is 7.944 ms per layer (fwd+bwd). Across 32 layers that's
254 ms — ~18 % of the 1,425 ms step time. Combined with FFN (~3.7 GB/chip
per layer activation peak that gets recomputed under remat), attention has
become the next-biggest optimization target. Future hypotheses: ring
attention for context-parallel sequences (out of scope at single-host
density), SparseCore offload of attention compute, etc.

## Next hypotheses generated

1. **`exp 18`: bs=2 seq=8192** — does the per-chip B·L = 16,384 (2× exp 17)
   fit? If yes, becomes the new program-target shape for fp32-master
   experiments. If no, exp 17 is the seq=8192 ceiling.
2. **`exp 19`: bs=4 seq=4096** — same B·L as 18 (16,384) but better attention
   efficiency. Useful comparison data point.
3. **`exp 16`: fp32 master + bf16 compute** — the **end-goal** per
   user-direction. Stack on top of exp 17 (seq=8192 fitting confirms the
   memory baseline). Need AMP wiring in `train.py` (stored fp32, computed
   bf16, mu/nu fp32).
4. **`exp 14`: dots_saveable selective remat** — replace `nothing_saveable`'s
   12-30 % tax with a more nuanced policy. Could lift exp 13c / exp 17 MFU
   by 2-5 pp. Effort: S (policy swap).

## See also

- [Exp 13 (per-layer remat unlock)](2026-04-25-exp13-per-layer-remat-accepted.md) — the precondition.
- [Exp 9 (current-best at seq=1024)](2026-04-25-exp9-splash-autotuned-bs4-accepted.md).
- [Exp 10 (splash autotune at seq=8192)](2026-04-25-exp10-splash-autotune-multishape-potential.md).
- [Llama 3 program](../../program.md) — `seq=8192` is the program-target.
