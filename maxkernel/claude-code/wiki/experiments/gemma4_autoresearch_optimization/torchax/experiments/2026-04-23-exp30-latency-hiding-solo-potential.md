---
title: "Exp 30 — latency_hiding_scheduler flag in isolation (INCONCLUSIVE/no-op; clears a culprit)"
type: experiment
tags: [experiment, gemma4, xla-flags, scheduler, culprit-isolation]
hypothesis: latency-hiding-scheduler-alone
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: "branch perfautoresearch/v6e4-20260423-exp30-latency-hiding-solo"
verdict: inconclusive
---

Enabled **only** `--xla_tpu_enable_latency_hiding_scheduler=true` via `LIBTPU_INIT_ARGS`, with no other flag changes. Goal: decide whether latency-hiding was the bad actor in the exp 1 / exp 13 / exp 21 collective-flag-bundle regressions, or if the async-collective-fusion flags were. **Result: TPS 33,371 vs exp 25's 33,372 (−0.003 %, i.e. statistically identical).** Latency-hiding-scheduler is a no-op at this workload size — neither helps nor hurts. The async-collective-fusion flags are therefore the bundle regression's culprit.

## Hypothesis under test

**Statement**: The exp 1 / 13 / 21 bundle regressions (−12–25 % step time) came from one specific flag, not all five. Candidate culprit: `async_collective_fusion` + `async_collective_fusion_fuse_all_gather` + `async_collective_fusion_multiple_steps` — they reorder collectives aggressively, which we know broke compute-fusion locality at this workload. Candidate innocent: `latency_hiding_scheduler`. If we enable ONLY latency-hiding, expect ≈0 % Δ.

Falsifiable: TPS within ±0.5 % of exp 25 → innocent (hypothesis supported → this experiment itself is inconclusive for the main optimization target, but it narrows the investigation).

## Setup

- Branch `perfautoresearch/v6e4-20260423-exp30-latency-hiding-solo` off trunk at exp 25.
- Zero code changes. Only `LIBTPU_INIT_ARGS="--xla_tpu_enable_latency_hiding_scheduler=true"` prepended to the command.
- Stack: selective remat + splash_pallas + bf16 CE + fused_bwd + SEQ_MINOR + splash block=1024. Same as exp 25.

## Results

| Metric | Exp 25 (no flag) | **Exp 30 (latency_hiding alone)** | Δ |
|---|---|---|---|
| TPS | 33,372 | **33,371** | **−0.003 %** |
| Step time (steady-state mean, steps 2–19) | ~368 ms | **368.23 ms** | +0.06 % |
| Per-step σ | tight | min 367.40 / max 368.80 (n=18) | — |
| Compile step 0 | ~155 s | 178.68 s | +15 % |
| Step 1 recompile | ~152 s | 170.44 s | +12 % |
| Loss trajectory | 3.82 → 1.55 | 3.83 → 1.84 | match |

Compile-time penalty is notable (+15 %) — the scheduler has more search space to explore — but runtime is indistinguishable.

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-exp30-latency-hiding-solo](http://localhost:8791/?run=2026-04-23-gemma4-exp30-latency-hiding-solo) — opens the interactive trace viewer for this run.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-exp30-latency-hiding-solo`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-exp30-latency-hiding-solo/`](../../../../../raw/profiles/2026-04-23-gemma4-exp30-latency-hiding-solo/) (gitignored; relative link click-throughs open the trace folder locally)
- **Steps captured**: 10, 11, 12
- **What's inside**: xprof trace — `latency_hiding_scheduler=true` alone at exp 25 stack; flat (−0.003 %). Use for HLO-diff to see where the scheduler actually reordered anything.

## Mechanism

The XLA scheduler's latency-hiding pass tries to overlap collectives with compute by *allowing* out-of-order scheduling, but doesn't by itself *force* collective fusion. On this workload the scheduler already has good locality with the default pass; giving it the additional "you may reorder for latency-hiding" hint produces a schedule within noise of the default. The pathology in exp 1 / 13 / 21 required the downstream flags (`async_collective_fusion*`) to actually collapse multi-collective operations into combined async ops — which is what broke compute-fusion locality.

**Inference for future experiments**:
- `latency_hiding_scheduler`: benign at this scale. Can leave on if the scheduler has more work to do (e.g. after other optimizations create more collective parallelism).
- `async_collective_fusion`, `async_collective_fusion_fuse_all_gather`, `async_collective_fusion_multiple_steps`: culprits. Park permanently at batch=3 / seq=1024.
- `async_collective_fusion_fuse_all_reduce` and `overlap_compute_collective_tc`: not tested in isolation yet. Future sub-hypothesis if we return to collective-overlap work.

## Verdict

**INCONCLUSIVE** for the primary metric (flat TPS). **INFORMATIVE** for the investigation: clears latency-hiding-scheduler from the exp 1 / 13 / 21 regressions. No production change recommended (no win, and +15 % compile-time cost).

Not merged. Branch preserved as reference.

## Next hypotheses

1. **`--xla_tpu_scoped_vmem_limit_kib=524288`** — larger scoped VMEM limit may let splash-block=1024 tiles fit more instances in flight. Simple env flag.
2. **2D mesh (fsdp=2, tp=2)** — biggest remaining structural lever; eng-medium.
3. **Persistent JAX compile cache** — not TPS but 30× iteration speed; pending since exp 2.
4. **Pallas RMSNorm with hand-rolled custom_vjp** — targets the ~57 ms loop-fusion bucket per step.

## See also

- [exp 1 — async-collective-flag bundle (discard)](2026-04-23-exp1-async-collective-flags-rejected.md) — the regression that motivated this isolation.
- [exp 25 — splash block=1024 (current best)](../..) (writeup gap)
- [async-collectives concept](../../../../concepts/async-collectives.md).

## Sources

- `wiki/experiments/gemma4_autoresearch_optimization/torchax/train.py` (trunk @ exp 25, unchanged)
- `wiki/experiments/gemma4_autoresearch_optimization/torchax/model/pallas_attention.py` (trunk @ exp 25, unchanged)
- `/tmp/gemma4_exp30.log`
- `raw/profiles/2026-04-23-gemma4-exp30-latency-hiding-solo/`
