---
title: "Qwen3 8B JAX retrospective"
type: analysis
model: qwen3-cc
lane: jax
created: 2026-06-04
updated: 2026-06-04
mode: full
tags: [qwen3-cc, jax, retrospective, v6e-8, autoresearch]
---

# Qwen3 8B JAX retrospective

Mode: FULL. No prior Qwen3 JAX retrospective existed in `wiki/analyses`, so this pass read all 145 experiment pages in `wiki/experiments/qwen3_cx_autoresearch_optimization/jax/experiments/`, plus the lane log, model page, model-optimization index, and Pallas kernel directory.

Current confirmed frontier: v153/v154, 8B on v6e-8, seq8192, global batch 32, native JAX/Flax NNX, scan+remat+activation sharding, tokamax Splash, MaxText CE, collective-matmul modes disabled, scoped VMEM 100352 KiB. Repeated throughput is 57,754-57,757 tok/s average with 57,888 best observed, 41.4% MFU.

## Executive readout

The lane is not exhausted, but the current local-search strategy is. Since v153/v154, probes v155-v160 stayed inside the same local schedule neighborhood: VMEM midpoints, Splash block-size tweaks, and scheduler flag no-ops. v155, v158, and several earlier flag probes emitted byte-identical or effectively unchanged optimized HLO; v156/v157/v159 changed HLO but regressed. This is the stuck-frontier signature the retrospective skill is designed to catch.

The largest durable wins came from structural transitions, not scalar tuning:

- Baseline native JAX beat torchax at seq2048: 31,955 tok/s, 20.5% MFU.
- Splash/TP/tokamax/remat/fused RoPE moved the lane to 47,846 tok/s, 30.8% MFU.
- Scan+activation sharding+MaxText CE made seq8192 viable at 48,312 tok/s, 34.6% MFU.
- Tokamax Splash max-logit and scheduler/VMEM/resource flags moved the bs3 frontier to about 51,033 tok/s, 36.6% MFU.
- Disabling collective-matmul modes moved the batch-4 frontier to 56,445 tok/s, 40.5% MFU.
- Scoped VMEM bracketing moved it to 57,754-57,757 tok/s, 41.4% MFU.

By contrast, the recent local probes after the 41.4% frontier have not produced a new mechanism. The next experiment should be structural: a real kernel/loss/sharding rewrite, not another nearby XLA flag.

## Premature-exhaustion signals

- Frontier-stuck duration: v155-v160 are all post-frontier local probes around v153/v154 with no durable gain.
- Single-axis imbalance: the lane contains many XLA/libtpu/VMEM/Splash parameter probes and very little kernel-level implementation work after the current frontier.
- No-op recurrence: multiple flags are accepted but compile to the same HLO/profile family, especially after collective-matmul-none.
- False-fast invalid path: qseq2 probes v097-v100 report attractive nominal throughput but all produce NaN loss, so they must not be counted as viable frontier signals.
- Near-fit memory wall: batch 5/global 40 remains near-fit OOM in v119-v122; scalar shared-memory and VMEM caps did not make it fit.

## Mechanism Tree

### VMEM / Scratch Scheduling

Frontier-shifting: v068, v072/v073, v144/v145, v146/v147, v148/v149, v150/v151, v153/v154. The final confirmed optimum is 100352 KiB on the batch-4 collective-matmul-none stack.

Refuted: v069-v071, v111, v121-v122, v143, v152, v155-v157, v159. Lower VMEM often changes schedule but loses MXU/throughput; high-side 102400 and 101376 are slower or no-gain; smaller Splash KV/DKV tiles are not the next step.

Open: VMEM is now a local optimum, not a promising search axis unless coupled to a structural memory reduction that changes the batch/shape envelope.

### FSDP / Collectives / Collective-Matmul Modes

Frontier-shifting: v116/v117 disabled collective-matmul modes for batch 3 and produced the largest late-stage jump; v118/v123 carried it to batch 4/global 32.

Refuted/no-gain: v101, v113-v115, v131-v142 show base async collective fusion and all-gather fusion remain required; reduce-scatter subtype and host-transfer/sendrecv/rerun/shared-memory variants are ties or regressions.

Open: understand why collective-matmul-none helped structurally, then seek a code-level sharding/layout rewrite that preserves the benefit while reducing the remaining convolution-fusion/custom-call buckets.

### CE / Logits Memory

Supported: MaxText CE plus scan/activation sharding made the seq8192 lane viable; v041 established the 34.6% target-sequence frontier. Tokamax max-logit control in v052 produced a durable attention/loss-path gain.

Refuted/invalid: v026 OOMs on full logits at batch 4; v030/v031 prove tokamax CE removes the memory wall but is too slow; v032 MaxText offload fits but regresses; v043/v044 full-logit custom VJP is invalid because it materializes/promotes the full vocabulary dot.

Open: this remains the best structural target. A genuinely streamed/fused CE path or a different vocab/lm-head sharding layout could attack both the batch-5 OOM wall and the residual loss/logit overhead. Reusing tokamax CE as-is is already refuted.

### Splash Attention / Tokamax

Supported: Splash/tokamax was essential for moving beyond the baseline, and max-logit control is part of the current valid stack.

Refuted: forward BKV512, DKV1024, DQ2/DQ3, unfused backward, bq/dkv variations, and save-residual variants did not improve the frontier. v156/v157 repeat the block-size failures on the batch-4 frontier.

Invalid: qseq2 is not usable until the NaN root cause is fixed; v097-v100 closed the runtime numeric-control probes.

Open: if attention is attacked again, it should be kernel-level or correctness-level work, not tile-only sweeps.

### Sharding / Batch / Sequence Envelope

Supported: TP4 at short sequence, TP2 tokamax/remat/fused RoPE, scan+activation-sharded target sequence, and batch-4/global-32 after collective-matmul-none.

Invalid/refuted: TP2/TP4 variants at target sequence often OOM; no-scan and TP2 paths fail through full-logit/unrolled temporaries; batch 5 remains OOM after scalar memory knobs.

Open: batch 5/global 40 is a high-value frontier only after a structural memory reduction of roughly the remaining OOM margin. It should not be retried with another scalar VMEM cap alone.

### Pallas Kernel Catalog Coverage

Explored directly: tokamax Splash and tokamax CE.

Unexplored for this lane:

- TPU fused CE/lm-head variant beyond tokamax's current implementation.
- TPU fused GLU/SwiGLU or MLP epilogue pattern. The catalog identifies TPU as a hole for GLU, with GPU references in tokamax/AlphaFold-style code.
- TPU all_gather_matmul / reduce_scatter_matmul kernels from ejkernel or tpu-inference as an alternative to compiler collective-matmul lowering.
- Ring/context-parallel attention is cataloged, but this seq8192 single-slice training shape does not yet point to it as the first next step.

## Ranked Next Directions

1. Structural CE/lm-head rewrite. Goal: avoid full-logit materialization without tokamax CE's current overhead. Success signal: batch 4 stays at or above 41.4% MFU and either batch 5 fits or the CE/logit-related profile/HBM pressure falls materially.

2. Code-level collective-matmul-none investigation. Diff the v115/v116 and v123/v127 HLO families and identify the actual layout/collective boundary that changed. Then encode the beneficial layout in model code or sharding constraints instead of relying only on backend flags.

3. Pallas MLP/GLU or fused epilogue feasibility probe. The current profile is dominated by convolution fusion and custom-call attention; a fused gate/up/SwiGLU path is one of the few cataloged kernel gaps not yet tried on this lane.

4. Batch-5 after memory relief only. Do not rerun v119-v122 variants directly. Reopen global batch 40 only after a structural memory change frees GiB-scale HBM or reduces the offending temps.

5. qseq2 correctness root cause as a separate branch. It has attractive nominal speed but invalid loss. It should be treated as a correctness/debug lane, not mixed into frontier throughput iteration.

## Cross-Lane Brief

Native JAX already beat torchax at the identical seq2048 baseline shape: 20.5% MFU vs torchax 19.2%, mainly from lower collective/dispatch overhead. The JAX lane is therefore a good vehicle for kernel/loss-path work, not merely a port-validation lane.

No sibling Qwen3 lane retrospective exists yet. The torchax lane should be mined only for structural ideas that transfer across frameworks; XLA/libtpu flag no-op lessons from this JAX lane should not be blindly repeated there.

## Experiment Ledger

The ledger below is compact but covers every experiment page read in FULL mode.

| Experiments | Topic | Outcome | Frontier shift |
|---|---|---|---|
| baseline | Baseline/profile | baseline | 31,955 tok/s, 20.5% MFU at seq2048 |
| v011 | Splash/TP/remat | invalid | TP2 bs8 compile OOM |
| v012 | Splash/TP/remat | supported | 39,438 tok/s, 25.4% MFU |
| v013-v014 | Splash/tokamax/remat | inconclusive observations | 44,638 then 47,558 tok/s, backfilled |
| v015 | Splash/tokamax/remat/fused RoPE | supported | 47,846 tok/s, 30.8% MFU |
| v016 | Sharding/batch | invalid | TP1 path invalid/OOM |
| v017 | scan-over-layers | refuted | no gain over v015 |
| v018 | batch relief | supported | 46,526 tok/s, 29.9% MFU with HBM relief |
| v019-v021 | XLA flags / long seq / batch | invalid | compile OOM |
| v022 | long seq | supported | seq4096 valid, 41,592 tok/s, 27.8% MFU |
| v023 | long seq | supported | seq8192 valid, 37,148 tok/s, 26.6% MFU |
| v024-v026 | target-seq batch scaling / logits | invalid | compile OOM |
| v027-v029 | Splash block/backward variants | refuted | below v041 target frontier |
| v030-v032 | CE/logit memory | refuted | tokamax CE/offload fit but slower |
| v033-v034 | TP2/no-scan sharding | invalid | invalid/OOM |
| v041 | scan+remat+MaxText CE+activation sharding | supported | 48,312 tok/s, 34.6% MFU |
| v042 | no-scan | invalid | full-logit/unrolled temp OOM |
| v043-v044 | label custom VJP CE | invalid | full-vocab temp materialization |
| v045 | tokamax Splash target stack | supported | 49,069 tok/s, 35.2% MFU |
| v046 | experimental scheduler bundle | refuted | no gain |
| v047-v048 | tokamax softmax toggles | refuted | no gain |
| v049 | DQ3 | refuted | no gain |
| v050-v051 | qseq2 | invalid | NaN/invalid despite nominal speed |
| v052 | max-logit 30 | supported | 50,376 tok/s, 36.1% MFU |
| v053-v055 | max-logit/block variants | refuted | no gain over v052 |
| v056 | scheduler rerun3 | supported | 50,383 tok/s, 36.1% MFU |
| v057-v061 | rerun/max-logit/block variants | refuted | no durable gain |
| v062 | DQ2 | invalid | invalid |
| v063 | max-logit 32.5 | refuted | no gain |
| v064 | qseq2 sharded mask | invalid | invalid |
| v065-v066 | save residuals | invalid | invalid |
| v067 | no AG backward pipeline | refuted | no gain |
| v068 | scoped VMEM 81920 | supported | 50,790 tok/s, 36.4% MFU |
| v069-v071 | VMEM bracket | refuted | lower/upper points slower |
| v072-v073 | custom-call VMEM adjustment | supported | 50,793 tok/s, repeatable tiny gain |
| v074-v075 | selective resources | supported | 51,021 tok/s, 36.6% MFU |
| v076-v080 | scheduler/resource/overlap variants | refuted/tie | no durable gain |
| v081 | collective-compute fusion | refuted | hard regression |
| v082 | multiple async collectives | supported/tie | non-harmful, no durable frontier |
| v083 | kloop async collective fusion | invalid | libtpu crash |
| v084 | start/done-only ACF | refuted | regression |
| v085 | staged collective-compute pipelining | invalid | libtpu crash |
| v086-v087 | ILP/BRKGA LHS | supported/tie | valid but no durable gain |
| v088 | layer scheduler alone | invalid | requires sparse-core aggregator |
| v089-v090 | layer scheduler + sparse-core aggregator | supported | 51,026-51,033 tok/s frontier |
| v091-v096 | layeragg variants | refuted/tie | no gain; pair interaction bounded |
| v097-v100 | qseq2 correctness probes | invalid | NaN loss; retire runtime-control probes |
| v101-v115 | ACF/scheduler/host-transfer subtype sweep | refuted/tie | base ACF and all-gather fusion required; others no gain |
| v116-v117 | collective-matmul-none, batch 3 | supported | 56,009-56,017 tok/s, 40.2% MFU |
| v118 | collective-matmul-none, batch 4 | supported | 56,445 tok/s, 40.5% MFU |
| v119-v122 | batch 5 memory attempts | invalid/refuted | near-fit OOM persists |
| v123 | batch-4 rerun | supported | confirms 56,445 tok/s band |
| v124-v130 | simplify/no-op scheduler flags | supported ties/refuted ties | same HLO/profile family, no new gain |
| v131-v134 | post-cmnone ACF ablations | refuted | ACF/all-gather/multistep remain required |
| v135-v142 | host-transfer/sendrecv/rerun/shared-memory sweep | refuted/tie | no gain over v123/v127 |
| v143 | VMEM 73728 | refuted | slower, 55,810 tok/s |
| v144-v145 | VMEM 86016 + rerun | supported | 56,603-56,609 tok/s |
| v146-v147 | VMEM 90112 + rerun | supported | 56,635-56,644 tok/s |
| v148-v149 | VMEM 94208 + rerun | supported | 56,768-56,773 tok/s |
| v150-v151 | VMEM 98304 + rerun | supported | 57,572-57,578 tok/s, 41.3% MFU |
| v152 | VMEM 102400 | refuted | 57,410 tok/s, slower |
| v153-v154 | VMEM 100352 + rerun | supported | confirmed frontier, 57,754-57,757 tok/s, 41.4% MFU |
| v155 | VMEM 101376 | refuted/no-gain | byte-identical HLO, slower |
| v156 | forward BKV512 on frontier | refuted | 56,541 tok/s |
| v157 | DKV BKV1024 on frontier | refuted | 55,258 tok/s |
| v158 | explicit layeragg pair on frontier | refuted/tie | byte-identical HLO, no gain |
| v159 | VMEM 99328 | refuted | changed HLO, slower |
| v160 | VMEM 100864 | running at retrospective time | high-side midpoint check in progress |
