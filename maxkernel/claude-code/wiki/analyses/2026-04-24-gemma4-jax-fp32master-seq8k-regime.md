---
title: "Gemma 4 JAX stack — fp32-master + bf16-compute AMP regime ceiling (seq=8192 target)"
type: analysis
tags: [analysis, gemma4, jax, mixed-precision, fp32-master, seq8192, ceiling, memory-wall]
created: 2026-04-24
updated: 2026-04-24
---

User shifted the optimization target to a **new regime** on 2026-04-24: fp32 master weights (for the optimizer), bf16 matmul/conv compute (standard AMP), and `seq_len=8192` as the new default. This analysis records the findings from the first optimization loop at that regime on the JAX stack (exps 52–53), the memory wall reached at seq=8192, and the recommended path forward.

Compare against:
- [Old-regime (bf16 single-dtype) JAX-stack ceiling (exp 43–49 arc)](2026-04-24-gemma4-jax-stack-ceiling-exp43-49.md) — if that page exists.
- [2026-04-23 JAX-stack ceiling (exp 34–42 arc)](2026-04-23-gemma4-jax-stack-ceiling.md) — if that page exists.

## Headline

| regime | config | TPS | step time | notes |
|---|---|---|---|---|
| **New regime baseline (exp 52)** | **seq=2048 b=1 fp32-master + bf16-compute, splash** | **26,807** | **305.6 ms** | **Largest feasible fp32-master config on v6e-4.** |
| Old regime best (exp 36) | seq=1024 b=3 bf16-single-dtype, splash | 34,614 | 355.0 ms | reference |
| Old regime long-seq ref (exp 40) | seq=2048 b=2 bf16-single-dtype, splash | 31,809 | 515.1 ms | closest old-regime sibling |
| User's target (exp 52 attempt) | seq=8192 b=1 fp32-master | — | — | **compile OOM, 35.18 GiB / 31.25 GiB per-chip (−3.93 GiB)** |
| seq=8192 bf16-legacy (sanity) | seq=8192 b=1 bf16-single-dtype | — | — | **also OOM, 36.16 GiB (−4.91 GiB)** — the seq=8192 wall is not fp32-master-specific |

**Bottom line.** seq=8192 is a hardware-level wall on v6e-4 for Gemma 4 E4B, independent of the dtype regime. fp32 master weights add ~1 GiB of peak HBM relative to bf16 single-dtype and make the wall slightly steeper, but the fundamental blocker is the aggregate model state (params + opt state + activations) exceeding 4 × 31.25 GiB when the sequence length pushes activations over ~7 GiB/chip at b=1. The largest feasible new-regime configuration on the current hardware is **seq=2048 b=1 fp32-master at 26,807 TPS**, and that is exp 52, the new baseline.

## The ceiling probe (full table)

Compile-time peak HBM across the new-regime OOM-probe matrix (all b=1 where not noted; all on v6e-4 fsdp=4; compile-time estimates, not actual allocations):

| seq | dtype | remat policy | scan | peak HBM | runs? | TPS |
|---|---|---|---|---|---|---|
| 2048 | fp32-master + bf16 | dots_no_batch (default) | off | fits (~27 GiB est.) | **yes** | **26,807 (exp 52)** |
| 2048 | fp32-master + bf16 | dots_no_batch | off, splash BLOCK=512 | fits | **yes** | **26,807 (exp 53b)** — flat |
| 2048 | fp32-master + bf16 | dots_no_batch | off, splash BLOCK=2048 | — | no | CompileTimeScopedVmemOom (VMEM 32.14/32.00 MiB, exp 53a) |
| 2048 | fp32-master + bf16 (b=2) | dots_no_batch | off | 39.37 GiB | no | OOM −8.13 GiB |
| 4096 | fp32-master + bf16 | dots_no_batch | off | 39.58 GiB | no | OOM −8.33 GiB |
| 4096 | fp32-master + bf16 | dots_no_batch | **on** | 39.66 GiB | no | OOM −8.41 GiB (scan doesn't help here) |
| 6144 | fp32-master + bf16 | dots_no_batch | off | 49.66 GiB | no | OOM −18.41 GiB (!) |
| 8192 | fp32-master + bf16 | dots_no_batch | off | 35.18 GiB | no | OOM −3.93 GiB |
| 8192 | fp32-master + bf16 | dots_no_batch | **on** | 34.83 GiB | no | OOM −3.58 GiB (scan saves 0.35 GiB) |
| 8192 | fp32-master + bf16 | **nothing_saveable** | off | 39.66 GiB | no | OOM −8.41 GiB (full remat is WORSE) |
| 8192 | fp32-master + bf16 | **offload_dot_no_batch** | off | 38.17 GiB | no | OOM −6.93 GiB (offload not HBM-accounted at compile) |
| 8192 | bf16 (legacy) | dots_no_batch | off | 36.16 GiB | no | OOM −4.91 GiB |
| 8192 | bf16 (legacy) | dots_no_batch | **on** | 32.46 GiB | no | OOM −1.22 GiB (closest to fitting across all tries) |

### Three durable observations from the table

1. **seq=8192 isn't primarily an AMP-cost problem.** Legacy bf16-single-dtype at seq=8192 b=1 also OOMs (36.16 GiB, exceeded by 4.91 GiB). The fp32-master overhead narrows the gap by ~1 GiB (35.18 vs 36.16) but neither dtype configuration fits. The wall is the model's aggregate HBM footprint at this sequence length on 4 chips.

2. **XLA's compile-time peak is non-monotonic in seq_len.** seq=4096 and seq=6144 report worse peak HBM than seq=8192 under the same fp32-master + bf16 + default remat config (39.58, 49.66, 35.18 GiB respectively). Intermediate seq_lens are not stepping stones between a known-good (seq=2048) and a near-miss (seq=8192) — the compiler picks qualitatively different schedules, and some are far worse than others. **Heuristic for the program**: when probing a new shape-space, don't assume the HBM curve is monotonic; measure at least three points (the low, the target, and an intermediate) before committing.

3. **`nothing_saveable` and `offload_dot_with_no_batch_dims` remat policies make peak HBM *worse* at seq=8192.** `nothing_saveable` serializes more live tensors (every saved intermediate must be kept) to avoid recomputing. `offload_dot_with_no_batch_dims` stashes to pinned host RAM, but the XLA compile-time planner does not credit the offload as freed HBM (re-confirming torchax exp 11's lesson). On this workload, the default `dots_with_no_batch_dims` is already the best among the canned policies. **Durable takeaway**: when a problem is at the memory wall, don't reach for "more remat" reflexively — the XLA planner's behavior can flip the sign. Measure.

## Throughput delta vs old regime

| shape | old regime bf16 | new regime fp32-master + bf16 | delta |
|---|---|---|---|
| s=2048, b=2 | 31,809 TPS (exp 40) | **infeasible** (b=2 OOMs under fp32 master) | — |
| s=2048, b=1 | not measured | 26,807 TPS (exp 52) | — |
| s=1024, b=1 | 30,285 TPS (exp 34) | not measured (exp 54 proposed) | — |
| s=1024, b=3 | 34,614 TPS (exp 36) | **infeasible** (b=3 s=1024 projected OOM under fp32 master, not tested) | — |

The clean apples-to-apples comparison (exp 40 s=2048 b=2 → exp 52 s=2048 b=1) combines two confounders: dtype regime change **and** batch halving. Ballpark: at fixed b=1 s=2048, old-regime (bf16 single-dtype) would probably do ~25–28K TPS too (since exp 52's 26,807 is already close to the per-b=1 limit at this seq). **The "AMP tax" at a fixed shape is expected to be small (<5 %)** because the matmuls still run in bf16; only the param/grad/opt-state storage is fp32, and those costs are dominated by HBM capacity, not matmul throughput.

**Exp 54 (pure-AMP isolation)** is queued to measure this cleanly: run both `--weights-dtype fp32 --compute-dtype bf16` and `--dtype bf16` at `--batch_size 1 --seq_len 1024`, same splash config. Expect ≲5 % delta on TPS; the value is in confirming the mental model.

## Recommended path forward

The program's mandate at this ceiling has three branches, ranked by expected value:

### Branch A — accept seq=2048 as the new long-seq baseline and continue the optimization loop there

Proceed with exps 54–57 (pure-AMP isolation, scan at new regime, 2D mesh, PLE host-offload) at seq=2048 b=1 fp32-master. Each of these is a hypothesis we can test today without additional hardware. Expected aggregate gain to the new-regime baseline: **+5 to +15 %** if any land; none is a sure win given the plateau history from the old regime.

### Branch B — grow the mesh

v6e-8 (or v5p-8, v5e-8) would roughly halve the per-chip HBM pressure and almost certainly fit seq=8192 b=1 fp32-master. This is the **correct** path for the user's target config but requires hardware change; out of scope for this session.

### Branch C — memory-saving code changes before trying seq=8192 again

Three candidates, in order of effort × probability:
1. **PLE host-offload (exp 57)**: `embed_tokens_per_layer` is 11 GiB fp32 / 5.5 GiB bf16 — the largest single tensor. Streaming it from host at the first-layer boundary could free 2–3 GiB/chip. Risky code change (custom dispatch, host-device barrier). Effort L. If it lands, **might** close the 3.93-GiB seq=8192 gap.
2. **Offload opt state to host (not dot outputs)**: optax has optimizer-state offload patterns. Trade PCIe bandwidth for HBM. Effort M. Probably saves 4–6 GiB. **Would close the gap at seq=8192.**
3. **Selective layer offload**: pipeline-style overlap with host transfers for activations at the layer boundary. Effort L–XL.

None of these is trivial. The pragmatic path is **Branch A now, Branch B when hardware is available**. Branch C options go on the queue as stretch targets.

## Durable artifacts from this session

- **`--weights-dtype` / `--compute-dtype` CLI flags** on both trainers (commit 517a689). Legacy `--dtype` kept for pre-exp-52 invocations.
- **Flax NNX AMP plumbing** (`weights_dtype` + `compute_dtype` threaded through every module constructor; commit 176fd2c). Downcast-weight-in-matmul pattern verified correct.
- **Weight-loader scatter-shard path** (new `shardings=` kwarg) avoids single-device OOM on fp32 PLE embedding init.
- **`JAX_REMAT_POLICY` env var** (`dots_with_no_batch_dims` / `nothing_saveable` / `offload_dot_with_no_batch_dims` / `dots` / `everything_saveable`) — surface for probing the HBM wall without editing code per variant.
- **Non-monotonic XLA compile-time peak observation** and **`nothing_saveable` / offload counter-intuitive regression observation** — filed in OBSERVATIONS.md and the exp 52 writeup.
- **torchax CLI parity** (stub/warn mode; full AMP deferred).

## See also

- [exp 52 — new-regime baseline](../experiments/gemma4_autoresearch_optimization/jax/experiments/2026-04-24-exp52-jax-fp32master-seq2k-accepted.md)
- [exp 53 — splash block sweep flat/rejected](../experiments/gemma4_autoresearch_optimization/jax/experiments/2026-04-24-exp53-jax-splash-block-sweep-fp32master-rejected.md)
- [program.md](../experiments/gemma4_autoresearch_optimization/program.md) — fixed bindings, hardware, protocol
- [`jax/experiments/README.md`](../experiments/gemma4_autoresearch_optimization/jax/experiments/README.md) — current state: new-regime + old-regime baselines

## Sources

- `raw/profiles/2026-04-24-gemma4-jax-exp52-baseline-seq2k-fp32master/`
- `raw/profiles/2026-04-24-gemma4-jax-exp53-splash-block512-seq2k-fp32master/`
- commits 517a689, 176fd2c, 732c0b3 on main
- `/tmp/gemma4_jax_exp52.log`, `/tmp/gemma4_jax_exp53.log`
