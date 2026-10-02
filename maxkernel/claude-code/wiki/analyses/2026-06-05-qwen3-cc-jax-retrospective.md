---
title: "Qwen3 8B JAX retrospective update"
type: analysis
model: qwen3-cc
lane: jax
created: 2026-06-05
updated: 2026-06-05
mode: incremental
tags: [qwen3-cc, jax, retrospective, v6e-8, autoresearch]
---

# Qwen3 8B JAX retrospective update

Mode: INCREMENTAL with current-state refresh. This update supersedes the 2026-06-04 full retrospective for the active frontier state. It focuses on experiments v160-v247, plus the lane log, current model page, MaxText comparison notes, and the prior full retrospective.

Current confirmed frontier: v230/v236 band, 8B on v6e-8, seq8192, global batch 32, native JAX/Flax NNX, no-scan + remat + activation sharding, tokamax Splash, MaxText CE, no MLP intermediate `_sac`, SparseCore reduce-scatter/all-reduce offload with aggregator, all-gather kept on async collective fusion, RS latency multiplier 2/3 same-band, AR latency multiplier 2, scoped VMEM 100352 KiB. Repeated throughput is about 59,749-59,750 tok/s, with nominal MFU about 42.9%.

## Executive Readout

The lane is not exhausted, but the current v230 HLO neighborhood is. The durable post-v160 gains came from structural changes: removing the MLP intermediate `_sac`, moving to no-scan, enabling SparseCore collective offload, keeping the aggregator, and tuning the RS/AR split. Since v230, the probes have mostly been local scheduler, VMEM, shared-memory, latency-multiplier, tracing, and rerun variants. v242, v246, and v247 in particular validate that multiple knobs now compile to the same material HLO/profile family.

The next experiment should not be another scalar flag probe on the v230 stack. The search should move to one of three structural directions:

- A real streaming/fused CE or lm-head+CE implementation that avoids full-logit materialization without the tokamax CE backward custom-call overhead.
- Code-level collective/layout changes that preserve the SparseCore RS/AR benefit without relying only on backend flag interactions.
- A Pallas MLP/SwiGLU/gate-up epilogue feasibility probe, because convolution fusions remain the largest device-time bucket.

## Premature-Exhaustion Signals

- Frontier-stuck duration: v231-v247 have not produced a material gain beyond the v230/v236 band.
- Single-axis imbalance: post-frontier work has become dominated by scalar local probes: RS/AR latency multipliers, VMEM, rerun count, shared memory, tracing, memory-pressure tracking, and async all-gather caps.
- Same-HLO recurrence: v242, v246, and v247 all validated that several accepted runtime flags are diagnostic/no-op knobs for this compiled program.
- CE false starts: v200-v203, v213-v216, and v244-v245 show that plain CE, naive tiled CE, custom-VJP tiled CE, tokamax CE, and direct MaxText CE batch-5 paths are invalid, OOM, or slower on this stack.
- Batch-5 remains near-fit but not solved: scalar memory changes did not open global batch 40; this should only be reopened after structural memory relief.

## Mechanism Tree Update

### Sharding / `_sac` Layout

Frontier-shifting: v169/v170 removed the MLP intermediate `_sac` and produced a durable gain over the v153/v154 family. v182/v183 then moved that idea into the no-scan stack and opened the next frontier branch.

Refuted or bounded: v161-v168 lm-head, logits, final-hidden, RMSNorm replication, and related constraints did not carry a durable new frontier. v172-v176 and v193-v196 show that removing other `_sac` boundaries is not equivalent to removing the MLP intermediate boundary; most variants regress or tie below the frontier.

Open: inspect the HLO diff around v153 -> v169 -> v183 to identify the exact layout/collective side effect. This is more useful now than additional blind `_sac` removals.

### No-Scan + SparseCore Collectives

Frontier-shifting: v182/v183 established the no-scan + no-MLP-`_sac` branch. v205/v206 showed SparseCore all-collective offload was a real gain. v208/v210 confirmed the aggregator is needed. v220/v221 showed the best split is to keep all-gather on async collective fusion and offload reduce-scatter/all-reduce. v224-v230 tuned the RS/AR latency multipliers into the current frontier.

Refuted or bounded: v211 concurrent offload, v218/v219 partial offload isolations, v222 no aggregator, v223 ND reduce-scatter, v231 AR latency 3, v232/v233 all-gather async caps, v234 RS1, v237 RS4, and v239-v243 VMEM/shared-memory/rerun variants did not improve the frontier.

Open: the current profile still spends large time in custom-call and convolution-fusion buckets. The next collective work should be source/layout work informed by HLO diffs, not another backend flag sweep.

### CE / Logits Memory

Supported historically: MaxText CE made the target seq8192 lane viable, and remains part of the current valid stack.

Refuted or invalid: v181/v201/v202/v244/v245 show tokamax CE fits but is slower due to backward/custom-call cost on the current stack. v200 plain CE and v203 direct MaxText CE batch-5 paths OOM or fail the memory envelope. v213-v216 show the naive/remat/chunk/custom-VJP tiled CE family is not correct or not performant enough.

Open: this is still the highest-value structural target. The required mechanism is not "use tokamax CE"; it is a streaming/fused loss/lm-head path with an equivalence gate, reduced HBM pressure, and no new large backward custom-call bucket.

### VMEM / Scheduler / Diagnostics

Frontier-shifting historically: scoped VMEM tuning mattered before v160 and again as part of the pre-SparseCore frontier.

Refuted or no-op now: v231-v247 make the local flag surface look exhausted for the current material HLO. v246 disabled scheduler memory-pressure tracking and v247 disabled active offload tracing; both were same-HLO, same-profile ties.

Open: reopen these knobs only after a structural code change changes HLO shape, memory pressure, or collective topology.

### Splash Attention

Supported historically: tokamax Splash and max-logit control were necessary to reach the current target-sequence family.

Refuted or bounded: v188-v190 and earlier BKV/DKV/DQ tile probes do not point to another tile-only gain. qseq2 remains an invalid/correctness branch because it produced NaNs in earlier runs.

Open: attention should only be attacked via a correctness branch for qseq2 or a genuine kernel change, not another tile sweep.

### MaxText Comparison

The MaxText lane produced useful baselines and profiles, but it did not reveal a missing 45% MFU recipe that directly transfers. The same-shape native JAX comparison already beats MaxText token throughput per chip, while MFU accounting is not directly comparable across the two codepaths. Further MaxText experiments are lower priority unless a new recipe or exact shape target is introduced.

## Ranked Next Directions

1. Implement a real streaming/fused CE or lm-head+CE path with local equivalence tests before cluster launch. Success signal: batch 4 stays at or above the v230 band while CE/logit HBM/custom-call cost falls materially, or batch 5/global 40 fits without the tokamax CE slowdown.

2. HLO-driven collective/layout source change. Diff the v183, v205/v210, v220/v230 families and identify the source-level sharding/layout constraints responsible for the SparseCore RS/AR benefit. Success signal: same or better throughput with a cleaner HLO contract and fewer backend-flag dependencies.

3. Pallas MLP/SwiGLU/gate-up epilogue feasibility probe. The current profile is still dominated by convolution fusion, so this is the next kernel-shaped opportunity after CE. Success signal: reduced convolution-fusion self-time without raising HBM pressure or breaking remat/no-scan behavior.

4. qseq2 correctness branch. Treat this as a separate correctness/debug lane; do not count nominal throughput until loss is valid.

5. Retire scalar probes on the current v230 HLO. VMEM, shmem, rerun count, tracing, memory-pressure tracking, and nearby latency multipliers should only be revisited after a structural change.

## Cross-Lane Brief

Native JAX is the active performance lane. The MaxText lane is useful as a reference implementation and profile comparator, but current evidence says the remaining native-JAX gap is not "copy one MaxText recipe." The actionable transfer point is CE/lm-head implementation detail, not more MaxText scheduler sweeps.

## Incremental Experiment Ledger

| Experiments | Topic | Outcome | Frontier shift |
|---|---|---|---|
| v160 | VMEM 100864 on v153 stack | refuted/tie | no gain over v153/v154 |
| v161-v168 | lm-head/logits/final-hidden/RMSNorm sharding | refuted or provisional then refuted | no durable frontier |
| v169-v170 | remove MLP intermediate `_sac` | supported | moved frontier to about 57.94k tok/s |
| v171 | VMEM 100864 on no-MLP stack | refuted/tie | no gain |
| v172-v176 | remove other `_sac` boundaries / RMSNorm combo | refuted | MLP intermediate boundary is the special case |
| v177-v178 | lm-head / CE target-gradient variants | refuted/no-op | no gain |
| v179-v180 | no-remat / remat offload | invalid or refuted | memory/correctness not viable |
| v181 | tokamax CE on no-MLP stack | refuted | slower CE path |
| v182-v183 | no-scan + no-MLP `_sac` | supported | opened new no-scan frontier branch |
| v184-v187 | VMEM/shared-memory/host-transfer on no-scan | refuted/tie | no durable gain |
| v188-v190 | Splash tile variants on no-scan | refuted/invalid | no attention tile gain |
| v191 | experimental scheduler | refuted | no gain |
| v192-v196 | no-shard-acts and more `_sac` removals | refuted/OOM | no durable gain |
| v197-v199 | collective-matmul auto variants | invalid/OOM | not viable |
| v200-v204 | plain/MaxText/tokamax CE and batch-5 probes | invalid or refuted | CE memory wall unresolved |
| v205-v206 | SparseCore all-collective offload | supported | moved frontier to about 59.34k-59.35k tok/s |
| v207-v210 | SparseCore aggregator validation | supported | aggregator branch to about 59.37k tok/s |
| v211 | concurrent offload | refuted | no gain |
| v212 | same-shape MaxText comparison | analysis/supporting | native JAX faster token throughput per chip |
| v213-v217 | tiled/custom-VJP CE prototypes | invalid/refuted | naive CE family closed |
| v218-v219 | AG/RS/AR offload isolations | refuted | RS and AR offload both needed |
| v220-v221 | RS/AR only, AG on async fusion | supported | improved over all-collective offload |
| v222 | no aggregator | refuted | aggregator needed |
| v223 | ND reduce-scatter | invalid | not viable |
| v224-v230 | RS/AR latency tuning | supported | current material frontier, about 59.75k tok/s |
| v231-v238 | nearby RS/AR, AG cap, tracing variants | refuted/tie | RS2/AR2 and RS3/AR2 same-band only |
| v239-v243 | VMEM, shared-memory, rerun retunes | refuted/tie | scalar surface exhausted |
| v244-v245 | tokamax CE post-SparseCore / batch 5 | refuted | CE backward custom-call cost too high |
| v246 | disable memory-pressure tracking | refuted/same-HLO tie | no gain |
| v247 | disable offload tracing | refuted/same-HLO tie | no gain |
