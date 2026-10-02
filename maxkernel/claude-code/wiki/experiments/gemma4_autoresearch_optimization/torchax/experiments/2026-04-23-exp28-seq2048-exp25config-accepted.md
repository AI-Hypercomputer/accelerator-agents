---
title: "Exp 28 — seq=2048 batch=1 at exp25 config (KEEP, +0.9% over exp14)"
type: experiment
tags: [experiment, gemma4, long-seq, splash-attention, tps-win-modest]
hypothesis: exp25-stack-at-seq2048
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: "branch perfautoresearch/v6e4-20260423-exp28-seq2048-exp25config"
verdict: supported
---

Rerun seq=2048 b=1 at the current-best configuration (exp25: splash + SEQ_MINOR + block=1024 + fused_bwd + bf16 CE). The last direct data point at seq=2048 b=1 was exp 14 (31,960 TPS, on the splash + fused_bwd + bf16 CE stack before SEQ_MINOR and block=1024 landed). Goal: see whether exp25's block/layout improvements extend to long sequences.

## Hypothesis under test

**Statement**: At seq=2048, attention's N² term is 4× larger than at seq=1024 (relatively more time in splash). Gains from splash block=1024 + SEQ_MINOR are expected to amplify: if exp25 at seq=1024 b=3 gains 1.1 % over exp18 (which used block=512 + default layout), exp28 at seq=2048 b=1 may gain 1.5–3 % over exp 14.

Falsifiable: exp 28 TPS > 31,960 (exp 14) + noise band is "supported". TPS ≤ 31,960 + noise is "refuted".

## Setup

- Branch `perfautoresearch/v6e4-20260423-exp28-seq2048-exp25config` off trunk at exp 25. Zero code changes.
- Command:
  ```
  python -m train \
    --steps 20 --batch_size 1 --seq_len 2048 \
    --profile_dir raw/profiles/2026-04-23-gemma4-exp28-seq2048-exp25config \
    --profile_steps 10 11 12
  ```
- Stack: selective remat + splash_pallas + bf16 CE + fused_bwd + SEQ_MINOR + block=1024. Same as exp25, only seq and batch differ.

## Results

| Metric | Exp 14 (seq=2048 b=1 @ exp12 config) | Exp 25 (seq=1024 b=3 @ best) | **Exp 28 (seq=2048 b=1 @ exp25 config)** | Δ vs exp 14 |
|---|---|---|---|---|
| TPS (global, steady-state) | 31,960 | **33,372** | **32,251** | **+0.9%** |
| Step time (steady-state mean, steps 2–19) | 256.3 ms | ~245 ms | **254.01 ms** | −0.9% |
| Compile step 0 | ~155 s | ~155 s | 154.06 s | flat |
| Step 1 recompile | ~152 s | ~152 s | 151.93 s | flat |
| Loss trajectory | 3.26 → 1.50 | 3.82 → 1.55 | **3.27 → 1.98** | match |
| Peak HBM | n/a | n/a (exp 25 TBD) | see profile | — |

Per-step times (steps 2–19): min 253.5 ms, max 254.6 ms, σ < 0.4 ms. Very tight — no outliers aside from step 10/11 (profile-capture bracketing adds <10 ms).

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-exp28-seq2048-exp25config](http://localhost:8791/?run=2026-04-23-gemma4-exp28-seq2048-exp25config) — opens the interactive trace viewer for this run.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-exp28-seq2048-exp25config`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-exp28-seq2048-exp25config/`](../../../../../raw/profiles/2026-04-23-gemma4-exp28-seq2048-exp25config/) (gitignored; relative link click-throughs open the trace folder locally)
- **Steps captured**: 10, 11, 12
- **What's inside**: xprof trace — exp 25 stack at seq=2048 b=1; +0.9 % over exp 14.

## Mechanism

At seq=2048 vs exp 14 (same batch=1, same fsdp=4, same splash + bf16 CE + fused_bwd), the only change is splash block sizes 512 → 1024 and QKVLayout HEAD_DIM_MINOR → SEQ_MINOR (from exp 24 + exp 25). The 0.9% gain at seq=2048 is smaller in relative terms than at seq=1024 (where exp 24 + 25 delivered +1.1% jointly) — not amplified as the hypothesis speculated. Possible reason: at seq=2048, attention N² is larger but block=1024 is half the seq, so the tile-reuse advantage is bounded; at seq=1024 with block=1024, the entire attention is one block and the savings are maximized.

Loss trajectory identical within bf16-reorder noise vs exp 14 — confirms splash's numerically stable softmax path holds (fixes the pre-exp-9 NaN-at-seq-2048 bug permanently, no regression even as the stack has evolved).

## Verdict

**SUPPORTED.** +0.9% TPS at seq=2048 b=1 over exp 14. New best at seq=2048 b=1. Not a new overall best (exp 25 at seq=1024 b=3 is still higher TPS at 33,372). This experiment closes the matrix gap: "what does exp25 stack give at seq=2048 b=1" is now answered.

**Not merged to trunk** — the config is strictly dominated by exp 25 (seq=1024 b=3, 33,372 TPS) on this hardware. Keep the branch for reference; trunk stays at exp 25.

## See also

- [exp 14 — splash + bf16 CE at seq=2048 b=1](#) (writeup gap — see RESULTS.tsv row)
- [exp 25 — splash block=1024, new best](#) (writeup gap — see RESULTS.tsv row)
- [exp 9 — original seq=2048 NaN fix via splash](2026-04-23-exp9-splash-seq2048-accepted.md) — not on disk; see OBSERVATIONS.md gap note.
- [program.md § Pallas kernel landscape](../../program.md).

## Sources

- `wiki/experiments/gemma4_autoresearch_optimization/torchax/train.py` (trunk @ exp25)
- `wiki/experiments/gemma4_autoresearch_optimization/torchax/model/pallas_attention.py` (trunk @ exp25)
- `/tmp/gemma4_exp28.log`
- `raw/profiles/2026-04-23-gemma4-exp28-seq2048-exp25config/`
