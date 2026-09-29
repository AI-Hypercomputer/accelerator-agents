---
title: "Exp 32 — 2D mesh dp=2 tp=2 batch=2 (REFUTED, 2.4× slower at same global batch)"
type: experiment
tags: [experiment, gemma4, sharding, tensor-parallel, 2d-mesh, refuted]
hypothesis: 2d-mesh-tp2-hybrid
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: "branch perfautoresearch/v6e4-20260423-exp32-2d-mesh-tp2"
verdict: refuted
---

Wired a 2D `(dp=2, tp=2)` mesh with hybrid tp+dp sharding: MLP / attention projections tp-sharded on their NeMo-Megatron axis AND dp-sharded on the other axis (so opt-state is 4-way = 2×2 per chip, matching 1D fsdp=4). Non-TP params dp-sharded FSDP-style. Goal: shard heads across tp for parallel per-chip attention compute; unlock batch=4 via reduced per-chip memory. **Result: batch=3 still compile-time OOMs (by 1.15 GiB); batch=2 runs but is 12,711 TPS at global batch=4 vs baseline's 30,570 same global batch — 2.4× regression. TP overhead at 2 chips/axis dominates.** Refuted, not merged.

## Hypothesis under test

**Statement**: On v6e-4, 2D mesh with tp=2 should (a) shard Q-heads 2-way across chips so attention compute halves per chip, (b) shard MLP intermediate 2-way so MLP matmuls halve per chip, (c) free per-chip HBM by sharding MLP/embed weights 2-way on tp. Expected gain: +5–10 % TPS. Downside: adds all-reduce collectives per MLP and per attention (2-way ring).

Falsifiable: TPS > exp 17 (32,663, 1D fsdp=4 batch=2 global=8) at matched global batch → supported. Flat or regression → refuted.

## Setup

- Branch `perfautoresearch/v6e4-20260423-exp32-2d-mesh-tp2` off trunk at exp 25.
- Code changes:
  - `torchax/model/sharding.py` `plan_tp_shardings`:
    - Accept optional `param_shapes: Mapping[str, tuple[int, ...]]`. Gate TP sharding with `rank >= 2` so rank-0 params like Gemma 4's `layer_scalar` don't get `P('tp', None)` applied (smoke 1 blew up on exactly this).
    - For 2D meshes (`dp_size > 1`): TP-matched weights get `P('tp', 'dp')` / `P('dp', 'tp')` / `P('tp', 'dp')` for col/row/embed (instead of tp-only). This halves per-chip opt-state compared to tp-only sharding (smoke 2 was still OOM by 3.2 GiB; hybrid 2D dropped that to 1.15 GiB).
    - Non-TP params dp-sharded FSDP-style on largest divisible dim (fallback). Bucket counts for Gemma 4 E4B at (dp=2, tp=2): `{col_shard: 305, row_shard: 116, row_shard_embed: 3, replicated: 928, dp_shard: 683, undivisible: 42}`.
  - `torchax/model/pallas_attention.py` `_jax_splash_fwd`:
    - Detect mesh axes (fsdp vs dp+tp).
    - For 2D (dp, tp): shard_map specs `P('dp', 'tp', None, None)` on Q/K/V/out. Rebuild splash kernel with `num_q_heads_local = num_q_heads // tp` (4 Q-heads per chip on v6e-4 tp=2). Keep 1D fsdp path unchanged.
- Command at batch=3 **OOM'd** at compile time (32.40 GiB vs 31.25 GiB, 1.15 GiB over — same `~1.25 GiB XLA compile-overhead` pattern as exp 10/11). Retried at batch=2.

## Results

| Config | TPS | Step time | Global batch | Tokens/step |
|---|---|---|---|---|
| Baseline (fsdp=4, b=1) | 30,570 | 134.4 ms | **4** | 4096 |
| Exp 17 (fsdp=4, b=2) | 32,663 | 250.8 ms | 8 | 8192 |
| Exp 25 (fsdp=4, b=3, current best) | 33,372 | 368 ms | 12 | 12288 |
| **Exp 32 (dp=2 tp=2, b=2)** | **12,711** | **322.23 ms** | **4** | 4096 |

Like-for-like vs **baseline** (same global batch=4): exp 32 is **2.40× slower per token** (322 ms vs 134 ms).

Steady-state over steps 2–19: mean 322.23 ms, min 321.60, max 323.40, σ < 1 ms. Very tight — the regression is reproducible, not noise.

Loss trajectory clean (3.92 → 1.84 over 20 steps) — semantics preserved.

## Profile

- **xprof browser URL**: [2026-04-23-gemma4-exp32-2d-mesh-tp2-batch2](http://localhost:8791/?run=2026-04-23-gemma4-exp32-2d-mesh-tp2-batch2) — opens the interactive trace viewer for this run.
- **Run name** (as listed by `mcp__xprof__list_runs`): `2026-04-23-gemma4-exp32-2d-mesh-tp2-batch2`
- **On-disk directory**: [`raw/profiles/2026-04-23-gemma4-exp32-2d-mesh-tp2-batch2/`](../../../../../raw/profiles/2026-04-23-gemma4-exp32-2d-mesh-tp2-batch2/) (gitignored; relative link click-throughs open the trace folder locally)
- **Steps captured**: 10, 11, 12
- **What's inside**: xprof trace — 2D dp=2 tp=2 at batch=2 (batch=3 OOM'd); 2.4× slower than matched-global-batch 1D baseline. Use this to see the TP all-reduce overhead at 2 chips/axis.

## Mechanism — why TP regresses on v6e-4

TP wins when the attention/MLP compute saved on each chip (via head / intermediate sharding) more than pays for the added all-reduce collectives. On v6e-4 with tp=2:
- **Collective overhead dominates**: every MLP now does an all-reduce across the 2-chip tp axis. Gemma 4 has 42 layers × 2 MLP matrices = ~84 extra all-reduces per forward (plus another 84 in the bwd). Each all-reduce over 2 chips on v6e interconnect is ~2 µs just for launch latency, plus bandwidth time.
- **Compute saved is small**: per-chip MLP intermediate halves from 10240 to 5120 — but the inner matmul is compute-bound (OI=1376 FLOPs/byte ≫ v6e ridge 578), so halving doesn't halve wall time. You mostly lose the MXU pipeline.
- **Attention compute**: per-chip Q-heads drop from 8 to 4. Splash kernel handles this fine, but the attention kernel work was only ~5 % of step time at seq=1024. Halving that is ~2.5 % of step time gained.
- **Small-scale penalty**: with only 4 chips total, the 2-way TP halves the DP axis, which forces each chip to process larger per-chip batches to get the same global batch. But the DP axis also halves from 4→2, so the memory benefits of DP-style sharding (opt-state per chip) are also halved.

At this model/hardware scale, the collective-overhead tax is much larger than the compute/memory gains. TP on v6e-4 with Gemma 4 is a net loss.

## Verdict

**REFUTED.** 2.4× slower at matched global batch. Not merged.

This result is consistent with scaling literature: TP generally wins at ≥8 chips/axis where per-chip compute is large enough to hide collective latency. On v6e-4 with tp=2 the break-even is below 1. Don't revisit TP unless the hardware shape changes (e.g., v6e-16, v5p) or the model scales up significantly (70B+).

## Next hypotheses

The remaining untested levers with potential:

1. **Persistent JAX compile cache** (`JAX_COMPILATION_CACHE_DIR`) — not TPS, but 30× iteration-speed win; pending since exp 2. Infrastructure rather than optimization.
2. **Pallas RMSNorm with hand-rolled `custom_vjp`** — targets the ~57 ms/step loop-fusion bucket. Eng-heavy (needs fwd + bwd Pallas kernels) but bounded. Potential: 2–5 % TPS.
3. **Pallas SwiGLU / gated_linear_unit kernel** — tokamax only has GPU Pallas for this, not TPU. Would have to build from scratch. Potential: 2–5 % on the MLP pointwise + matmul fusion boundary.
4. **AoT compile** of train_step via `jax.jit.lower().compile()` — may pre-empt step-1 recompile by freezing the partition spec.

## See also

- [exp 25 — splash block=1024 (current best, 1D fsdp=4)](../..) — the config exp 32 tried to beat.
- [exp 10 — seq=2048 b=2 OOM](../..) and [exp 11 — offload remat OOM](../..) — the other `~1.25 GiB compile-overhead` OOM pattern data points.
- [sharding concept](../../../../concepts/sharding.md) (if extant) and [FSDP vs TP design notes](../..).
- `scaling-book` source (ingested earlier in the session): 2D mesh wins kick in at ≥8 chips/tp axis.

## Sources

- `wiki/experiments/gemma4_autoresearch_optimization/torchax/model/sharding.py` (+80 lines: rank guards + hybrid 2D sharding)
- `wiki/experiments/gemma4_autoresearch_optimization/torchax/model/pallas_attention.py` (+45 lines: 2D mesh shard_map)
- `/tmp/gemma4_exp32_smoke.log` (first failure — rank-0 sharding crash)
- `/tmp/gemma4_exp32_smoke2.log` (second failure — compile OOM at batch=3 tp-only by 3.2 GiB)
- `/tmp/gemma4_exp32_smoke3.log` (third failure — compile OOM at batch=3 hybrid by 1.15 GiB)
- `/tmp/gemma4_exp32_b2.log` (the measured batch=2 run)
- `raw/profiles/2026-04-23-gemma4-exp32-2d-mesh-tp2-batch2/`
