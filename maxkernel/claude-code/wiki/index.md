# TPU Model Performance Auto-optimization — Index
*Last updated: 2026-05-06 — 193 pages (27 codebases + 45 sources + 97 concepts + 2 model-programs + 4 analyses + 6 analysis subpages + jax-exp47 rejected page + 2 maxtext baseline experiments + 1 torchax llama3-8b baseline + 1 torchax llama3-8b splash kernel-only autotune + 1 jax-llama3-8b SparseCore-offload frontier)*

*Methodology: autoresearch (see [README](../README.md) + [SCHEMA](../SCHEMA.md)).*

## ⭐ Featured reference — Pallas kernel directory

**[Pallas kernel directory](analyses/2026-04-23-pallas-kernel-directory.md)** — the single most load-bearing reference document in this wiki. A repo-by-repo catalog of **~200 Pallas kernels across ~30 open-source repositories**, each row with a clickable link to the source code, backend (Mosaic-TPU / Mosaic-GPU SM90/SM100 / Triton / XLA fallback), stability tier, any performance claim quoted verbatim from source, application use case, and known callers. Cross-cutting tables group kernels by function (attention, paged-KV, ring, MoE grouped matmul, normalization, GLU, matmul, collectives, SSM/linear-recurrence, cross-entropy, PRNG, non-ML).

> **Machine-generated grounded companion:** `wikify connect` maps the shared concept vocabulary (`concepts/`) to each repo's grounded, SCIP-cited implementation page — **inline**: a connected concept page carries an `## In this wiki's repos` block linking each ingested repo's implementation, and each silo page links back up (**147 implementations across 22 ingested repos**). Where the kernel directory is the curated wide catalog, this is the always-fresh, click-through-to-grounding view of *which repos implement each concept and how they differ*. Refresh with `wikify connect --refresh`.

Use this page to answer:
- *"Is there a public Pallas implementation of X?"* — look up X in the [functional-category tables](analyses/2026-04-23-pallas-kernel-directory.md#kernel-inventory-by-functional-category).
- *"Where is the canonical / production version of X?"* — every row has a `[file](https://github.com/…)` link to the definition.
- *"Which repos should this wiki ingest next?"* — see [recommended Wave-4 ingestion order](analyses/2026-04-23-pallas-kernel-directory.md#recommended-next-ingestion-wave-wave-4-proposal) ranked by novelty-vs-tokamax and relevance.
- *"What tuning priors already exist?"* — see [Autotune / perf-tuning artifacts — crown jewels](analyses/2026-04-23-pallas-kernel-directory.md#autotune--perf-tuning-artifacts--crown-jewels) (sglang-jax ~2,000-entry RPA table; tpu-inference 600+ quantized_matmul table; marin/levanter deployment-time autotune harness).

**Six subpages** (split per SCHEMA ~500-line rule) hold the full per-kernel detail:

1. [§1 Upstream JAX + tokamax](analyses/pallas-kernel-directory/01-upstream-jax-tokamax.md) — ~55 kernels. The root of the vendoring graph.
2. [§2 AI-Hypercomputer stacks](analyses/pallas-kernel-directory/02-ai-hypercomputer.md) — MaxText, MaxDiffusion, JetStream (archiving 2026-02-01).
3. [§3 Inference engines](analyses/pallas-kernel-directory/03-inference-engines.md) — vLLM tpu-inference (authoritative author), SGLang-JAX, Aphrodite.
4. [§4 Research labs](analyses/pallas-kernel-directory/04-research-labs.md) — Apple AxLearn (uniquely novel Mamba/SSD/RAttention SSM Pallas), DeepMind repos (RecurrentGemma LRU scan, AlphaFold3 v3.0.1 fused GLU).
5. [§5 Frameworks & quantization](analyses/pallas-kernel-directory/05-frameworks-quant.md) — Tunix, Qwix, AQT, Jaxite (non-ML FHE), PyTorch/XLA, Marin/Levanter autotune harness.
6. [§6 Community & research companions](analyses/pallas-kernel-directory/06-community-research.md) — erfanzar/ejkernel (broadest community surface), haoliuhl/ringattention (canonical Ring Attention), plus flagged GPU-only repos misadvertised as "Pallas".

Companion document: [Pallas kernel source survey](analyses/2026-04-23-pallas-kernel-source-survey.md) — the predecessor repo-level inventory this directory refines to kernel-level.

**Status of open Pallas-related wiki hypotheses after this directory closed**:
- Ring Attention Pallas impl → **found** (maxdiffusion, haoliuhl, ejkernel — three patterns to choose among).
- Fused GLU Pallas reference → **found** (AlphaFold3 @ tag v3.0.1; pin the tag).
- Mamba / SSM Pallas → **found** (axlearn — only public source).
- **Zig-Zag ring attention on TPU → confirmed absent from every surveyed repo.** Remains an open algorithmic port from Brandon et al. 2023.
- TPU Pallas RMSNorm/LayerNorm → **confirmed-absent with external evidence**: maxtext/tpu-inference/axlearn/upstream all skip it — XLA-fusion is sufficient. Consistent with the Gemma 4 exp 33 −8.1% empirical result; don't build.

---

## Models (4)
- [Qwen3 8B — torchax](models/qwen3-cc-torchax.md) — `Qwen/Qwen3-8B` on TPU v6e-8 via torchax. Status: **live — baseline captured 2026-06-02**: 519 ms/step, 3,724 tok/s/chip, **19.2% MFU** @ seq2048 bs8 (66% TC idle → big headroom). 3 ranked open hyps (batch-scaling, splash, tokamax CE). [baseline exp](experiments/qwen3_cc_autoresearch_optimization/torchax/experiments/2026-06-02-qwen3-torchax-v6e8-baseline.md).
- [Qwen3 8B — maxtext](models/qwen3-cc-maxtext.md) — `Qwen/Qwen3-8B` on TPU v6e-8 via **upstream MaxText** (reference-ceiling lane). Status: **reference baselines measured 2026-06-02 — seq8192 bs3: 45.3% MFU / 6,942 tok/s/chip; seq2048 bs4: 38.0% / 7,116** (seq8192 on par with llama3-8b MaxText 44.6%). Cross-stack: jax is at **97.9% of MaxText @ seq2048** (near parity → seq2048 ceiling confirmed) but only **76.4% @ seq8192** (+31% gap = MaxText's offload-enabled bs3, which **reopens the jax seq8192 frontier**). [seq8192](experiments/qwen3_cc_autoresearch_optimization/maxtext/experiments/2026-06-02-maxtext-qwen3-8b-v6e8-ref-seq8192.md) · [seq2048](experiments/qwen3_cc_autoresearch_optimization/maxtext/experiments/2026-06-02-maxtext-qwen3-8b-v6e8-ref-seq2048.md).
- [Qwen3 8B — jax](models/qwen3-cc-jax.md) — `Qwen/Qwen3-8B` on TPU v6e-8, native-JAX (Flax NNX). Status: **live — seq2048 frontier 35.8% / 6,964 tok/s/chip @ bs4** ([v018](experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v018-xla-flag-stack.md)); **seq8192 frontier 34.8% / 6,068 @ bs3** ([v045](experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-05-v045-save-norm-stats.md): + on-device norm-stat save, +0.63% over [v035](experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v035-maxtext-ce-s8k-bs3.md) maxtext-CE). At MaxText **87.4%**. The 2026-06-05 "try all" gap-closing sweep: profiler re-ranked the gap (#1 = SiLU double-recompute 3,985ms, **walled**); **tp refuted** ([v043b](experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-05-v043b-tp2-bs6.md) −26%), **RMSNorm bf16 parity** ([v042](experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-05-v042-rmsnorm-bf16.md)), norm-stat save marginal win (v045). Residual ~12.6% confirmed = MaxText's host-offload pipelining (runtime/build wall). [gap-closing retrospective](analyses/2026-06-05-qwen3-cc-jax-gap-closing-attempt.md) · [MaxText-CE climb](analyses/2026-06-02-qwen3-cc-jax-maxtext-ce-climb.md).
- **Cross-agent demonstration lanes** — the same Qwen3-8B v6e-8 baseline, independently optimized by different agents and kept separate to show each approach side by side. *Best-effort index; see each lane folder + model page for detail.*
  - [Qwen3 8B — jax (ag / antigravity)](models/qwen3-ag-jax.md) · [maxtext](models/qwen3-ag-maxtext.md) · [torchax](models/qwen3-ag-torchax.md) — lane folder `experiments/qwen3_ag_autoresearch_optimization/`.
  - [Qwen3 8B — jax (cx / codex)](models/qwen3-cx-jax.md) · [maxtext](models/qwen3-cx-maxtext.md) · [torchax](models/qwen3-cx-torchax.md) — lane folder `experiments/qwen3_cx_autoresearch_optimization/`.
  - [Qwen3 8B — jax (cc5)](models/qwen3-cc5-jax.md) · [torchax](models/qwen3-cc5-torchax.md) — lane folder `experiments/qwen3_cc5_autoresearch_optimization/`.
- [Gemma 4 E4B — TPU autoresearch optimization](experiments/gemma4_autoresearch_optimization/README.md) — program page for `google/gemma-4-E4B` on TPU v6e via torchax/JAX. Status: **active, baseline not yet captured**. 16 open hypotheses consolidated from Wave 1/2 findings. *Note: filed under `experiments/<program>/` rather than `models/` — see schema-note in the page and the 2026-04-22 log entry.*
- [Llama 3 8B — TPU autoresearch optimization](experiments/llama3_8B_autoresearch_optimization/README.md) — program page for `meta-llama/Meta-Llama-3-8B` (and Llama 3.1-8B) on TPU v6e via torchax / native-JAX / MaxText. Status: **🚀 EXCEEDED MaxText reference (+9.9 % per-chip throughput) — [JAX exp 28b: native-JAX (Flax NNX) + scan + tokamax CE + tokamax-splash + MaxText XLA flag stack with **SparseCore offload of all-reduce + reduce-scatter + all-gather** at bs=4 seq=8192 = 62,142 TPS (7,768/chip), 43.6 % reported MFU](experiments/llama3_8B_autoresearch_optimization/jax/experiments/2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md)** vs MaxText reference 7,069/chip = **+9.9 % throughput per chip**. Reported MFU gap is now **1.0 pp** (43.6 % vs MaxText 44.6 %), much of which is FLOP-counter normalization. **Cumulative climb 2026-04-26: 4,591 → 7,768 tok/s/chip = +69.2 %** vs the morning AMP-only torchax baseline (exp 20). Path: [exp 13 (HOST_OFFLOAD breakthrough)](experiments/llama3_8B_autoresearch_optimization/jax/experiments/2026-04-26-jax-exp13-maxtext-xla-stack-bs5-accepted.md) → exp 18 (bkv=2048 match): 7,471/chip 41.9 % → 🏆 **exp 27 (SC offload of RS+AG added to AR): 7,724/chip 43.3 %** at bs=5 → 🏆 **exp 28b (bs=4 density tweak): 7,768/chip 43.6 %**. Torchax frontier 2026-04-26: [exp 74b 6,559/chip 36.8 % MFU at bs=3 seq=8192](experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-26-exp72a-tokamax-splash-bs3-seq8k-accepted.md#follow-up--max_logit_const-exp-74). Key wins: (1) tokamax CE (chunked_xla impl) unlocks bs=3+ at seq=8192; (2) tokamax-shipped splash with perf knobs (+4.4 %); (3) MaxText's HOST_OFFLOAD scheduler-flag bundle — the breakthrough lever (+10 % alone); (4) splash `bkv=2048` matched to MaxText (+0.7 %); (5) **SparseCore offload of all three FSDP collectives — relays RS+AG off the TC** (+3.4 %); (6) bs=4 sweet-spot with full SC offload (+0.6 % over bs=5).

## Hypotheses — ranked, open only (1)

| # | Hypothesis | Model | Expected | Confidence | Effort |
|---|---|---|---|---|---|
| 1 | [int8 weight-only quantization (AQT/qwix)](hypotheses/llama3-jax-int8-weight-quantization.md) | llama3-8b-jax | +15-30 % step time | medium | L |

**Refuted via HLO inspection 2026-04-27** (after the fact, before any Pallas kernel was written):
- ~~[Pallas RMSNorm + matmul-prologue fusion](hypotheses/llama3-jax-rmsnorm-matmul-prologue-fusion.md)~~ — XLA already inlines RMSNorm into each matmul's `kind=kOutput` Mosaic kernel (`fused_computation.47` for Q-proj, `.48` for K-proj, `.31`/`.32` for gate/up; all consume `add_rsqrt_fusion` + matmul → output, single kernel).
- ~~[Pallas SwiGLU + down_proj fusion](hypotheses/llama3-jax-pallas-swiglu-downproj-fusion.md)~~ — XLA already fuses silu+mult+down_proj+residual into a single `kind=kOutput` kernel (`fused_computation.40` containing `fused_computation.8`'s silu*u body, `convolution.111`, and the residual add).

These were the candidates expected to recover the 9.2 % loop-fusion line in the [exp 28b profile](experiments/llama3_8B_autoresearch_optimization/jax/experiments/2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md#profile); since both are already done by the XLA fuser, the loop-fusion bytes must come from other ops (residual scratchpads, embedding-grad, FSDP all-gather staging, etc.) where Pallas-replacing the XLA fusion is unlikely to help. **Only int8/AQT remains as a meaningful per-chip throughput lever**; the bf16-MXU regime is saturated.

## Observations (5)
- [qwen3-jax-sharding-matches-maxtext](observations/qwen3-jax-sharding-matches-maxtext.md) — our FSDP sharding already equals MaxText's `logical_axis_rules` (embed/D on fsdp, pure 1D fsdp=8 in both); the MXU occupancy gap (53.6% vs 61.2%) is NOT a sharding lever but recompute-downstream. Closed the "MXU/logical-axis" lever without a dispatch.
- [qwen3-jax-fsdp-rs-not-overlapped](observations/qwen3-jax-fsdp-rs-not-overlapped.md) — FSDP grad reduce-scatter ran synchronous on the unrolled graph (the #1 non-matmul cost at v018); overlap needs scan (v028).
- [qwen3-jax-qkv-splash-transpose-intrinsic](observations/qwen3-jax-qkv-splash-transpose-intrinsic.md) — the QKV→splash layout transpose is intrinsic (investigated + closed).
- [qwen3-maxtext-config-exists](observations/qwen3-maxtext-config-exists.md) — MaxText ships a qwen3-8b model config (reference lane enabler).
- [llama3-8b-torchax-converged-stack-bottleneck-breakdown](observations/llama3-8b-torchax-converged-stack-bottleneck-breakdown.md) — xprof breakdown of the 2026-04-26 program-target frontier. Captured twice: (a) exp 56 mosaic_tpu CE stack via exp 61b — conv fusion 46.0 %, splash 24.4 %, CE 7.5 %, loop fusion 13.6 %, MXU util 51.9 %, MFU 34.8 %; (b) exp 74b chunked_xla + tokamax-splash final frontier via exp 79 — conv fusion **54.4 %**, splash 21.8 %, **CE collapsed to ~0 %**, loop fusion 15.2 %, MXU util **55.0 %**, MFU **36.8 %**. Matmul achieves 73.9 % MXU efficiency when running; the gap to MaxText 44.6 % MFU is matmul time-share, not matmul-tile efficiency.

## Analyses (5 +)
- [2026-07-08 **qwen3 four-lane cross-model comparison** — why Fable-5/Codex crossed MaxText where Opus-4.8/Antigravity stalled](analyses/2026-07-08-qwen3-four-lane-cross-model-comparison.md) — synthesis over the cc/cc5/cx/ag session retrospectives. Thesis: the winners **changed the memory budget then grew batch or saved activations**; the losers assumed the budget was fixed. Both winners independently found the same ~9 GiB HBM headroom (cx via `DISABLE_COLLECTIVE_MATMUL`, cc5 by *measuring* splash residuals) and used it **on-chip** — the opposite of the host-offload path cc chased and declared "out of scope." Two ways to win (cc5 measure-and-target, cx brute-breadth), two ways to lose (cc reason-a-wall-and-quit, ag find-the-lever-but-crash-loop). Plus the causal-vs-non-causal MFU artifact.
- [2026-07-08 qwen3-cx-jax (Codex/GPT) **session** process retrospective](analyses/2026-07-08-qwen3-cx-jax-session-retrospective.md) — audits the Codex/GPT-5.5 session `019e8ab6` (155 MB / 76,776-line rollout, **349 jax experiments**, 116 compactions). **Beat MaxText**: seq8192 7,543 tok/s/chip / ~49.2% MaxText-style MFU @ global-batch-32 (v273). Won by breadth: `DISABLE_COLLECTIVE_MATMUL` (v116/117, −9 GiB HBM → reopened batch 32) + on-chip SparseCore collective offload (v205→v273). Pushed the batch scaling cc's early OOMs declared dead. Strengths: persistence, rerun-before-carry HLO gating, honest invalids. Fixable: queue-refill/lineage collapse (Next-hyps 3/349, Sources 2/349), all frontmatter mis-slugged `model: qwen3-cc-jax`, 241/349 sub-noise reruns with no pre-filter.
- [2026-07-08 qwen3-ag-jax (Antigravity/Gemini) **session** process retrospective](analyses/2026-07-08-qwen3-ag-jax-session-retrospective.md) — audits the Antigravity/Gemini session (06-02→06-15, ≥4 "brain" sessions, no Claude Stop hook → self-enforced never-stop). **Stalled at ~77% (seq8192 5,329) / ~90% (seq2048 4,515)** of MaxText. Crux: *not* idea-poverty — ag touched **both** winning lever families (SparseCore offload + context-checkpointing) but every run crashed (VMEM/HBM/shard_map / unsupported flag), so it never got a clean measurement. Primarily a **harness-persistence failure** (self-enforcement failed, multi-session fragmentation, dead/mis-slugged model page) amplified by weak discipline (v057-d…u letter-suffix thrashing, no CPU gate). Structural compliance collapsed: 33/90 Profile, 4/90 Next-hyps, 2/90 Sources.
- [2026-07-08 qwen3-cc5-jax (Fable 5) **session** process retrospective](analyses/2026-07-08-qwen3-cc5-jax-session-retrospective.md) — audits the Fable-5 session `3e7df36d` (~29 h burst, 62 experiments, 21 supported). **Matched/beat MaxText** (seq8192 6,959 tok/s/chip / 39.9% MFU; seq2048 tie 7,871). Won on the exact on-device selective-remat lever Opus-4.8/cc dismissed as "walled": v044/45 splash context-checkpointing → v047/48 *measured* residuals fit in HBM (~9 GiB headroom, no host offload) → v050/51 (in,out) weight layout past MaxText. Used discrete audit gates (17 profile-analyzer, formulate-hypothesis, 50-step certification). Fixable: chunked-CE numerics bug reached the certified frontier before the parity arm caught it.
- [2026-07-08 qwen3-cc-jax autonomous **session** process retrospective](analyses/2026-07-08-qwen3-cc-jax-session-retrospective.md) — audits the Opus-4.8 session `5d03ff24` against the SCHEMA/program.md loop. Strong science (confound control, contradiction discipline, honest failure accounting, premature-exhaustion catch) but operational waste: ~¼ of GKE runs were avoidable no-step crashes (AOT pre-flight would catch), the MaxText MFU denominator was unpinned/non-causal for two weeks, and loop+ops ran in one session that compacted 5×. Prioritized fix backlog A–F.
- [2026-06-02 qwen3_cc-jax retrospective #5 — MaxText config delta](analyses/2026-06-02-qwen3_cc-jax-retrospective-5-maxtext-delta.md) — diff vs MaxText's exact qwen3-8b reference config. **Re-opens the offload lever**: v036 was MIS-TESTED — it offloaded mlpwi (giant) + missed decoder_layer_input, the OPPOSITE of MaxText (offload decoder_layer_input + qkvo, recompute mlpwi). Also surfaces ~9 missing XLA flags (incl `*_in_chain` host-offload-in-loop flags = the scan-offload pipelining mechanism) + splash bkv 1024→2048 + a DISABLE_COLLECTIVE_MATMUL MXU candidate. Ranked next directions; loop resumed (v038 flags-probe, v039 offload-recipe).
- [2026-06-02 qwen3_cc-jax retrospective #4](analyses/2026-06-02-qwen3_cc-jax-retrospective-4.md) — INCREMENTAL (+13 exp, v025–v037). **Overturns #3's "lane done, pivot to torchax"**: the maxtext-CE arc moved the seq8192 frontier 30.4%→34.6% (+13.7% tok/s/chip) — the premature-exhaustion failure this skill exists to catch. Lane is NOT exhausted: top remaining levers are MXU/logical-axis tile alignment (+3-4%, needs user opt-in — sharding rewrite) and norm-inclusive selective-save remat (+2-4%). Anti-recs: named-offload (refuted ×2), maxtext-CE/scan @ seq2048 (seq-specific), bs4+.
- [2026-06-02 MaxText-CE climb — CE cracked the seq8192 batch wall](analyses/2026-06-02-qwen3-cc-jax-maxtext-ce-climb.md) — **supersedes the hard-wall closing below.** Porting MaxText's exact CE (T5X `@jax.custom_vjp cross_entropy_with_logits`, NOT vocab-tiled, NOT tokamax) as `--use_maxtext_ce` **cracked the seq8192 batch wall** (the prior "hard wall" was the tokamax/autodiff CE-backward transient). Arc v033→v037: bs1 parity → bs2 5,992 (+6.4%) → **bs3 6,030 = 86.9% of MaxText** (was 81%), fits w/o offload. seq8192-specific (regresses −6.9% at seq2048, v037). Residual ~13% = remat-recompute (pipelined offload refuted v036 −18.6%, needs kernel-authoring) + MXU/logical-axis (+3-4%, deferred).
- [2026-06-02 Qwen3-8B jax — MaxText-gap closing (SUPERSEDED)](analyses/2026-06-02-qwen3-cc-jax-maxtext-closing.md) — scan+named-offload+overlap arc (v025–v032); scan+overlap = real seq8192 win (+6.2%). **Its "documented hard wall" conclusion is refuted by the maxtext-CE climb above** — kept for history with a contradiction block.
- [2026-06-02 MaxText vs jax Qwen3-8B MFU gap](analyses/2026-06-02-maxtext-vs-jax-qwen3-8b-mfu-gap.md) — code+profile teardown of why MaxText hits 45.3% (MXU 61.2%) vs our jax 30.4% (48.3%) @ seq8192. Gap = collective overlap (RS 12.9% sync → 4.0% async, ~+9pp) + MXU occupancy (fewer/larger tiled dots), both downstream of **scan-over-layers** + **named-offload** (fits bs3). At seq2048 we're at parity. Ranked 3-lever package (named-offload + scan + overlap-flags) to close it; revises the "structural ceiling" + "async-fusion refuted" reads (those held only on the unrolled graph).
- [2026-06-02 qwen3-cc-jax-retrospective #3](analyses/2026-06-02-qwen3_cc-jax-retrospective-3.md) — FULL `--full-redo` audit: independent full-disk re-read of all 24 experiments **confirms #2 with zero discrepancies** (verdicts↔numbers consistent, frontier lineage intact, LINT clean, all hypothesis links resolve). Frontier verified v018 35.8% / v009 30.4%; lane at ceiling; torchax pivot is rec #1.
- [2026-06-02 qwen3-cc-jax-retrospective #2](analyses/2026-06-02-qwen3_cc-jax-retrospective-2.md) — INCREMENTAL (post-v018 arc, 23 exp): **v018 XLA flag stack landed +3.4 pp → 35.8% frontier** (validated prior rec #1), then 5 consecutive non-wins; profile-analyzer found RS reduce-scatter (12.9%) is #1 cost but async-fusion to overlap it backfired (over-subscribes scheduler). Verdict: seq2048 near practical ceiling for non-deep-kernel levers; best remaining = QKV→splash layout opt (medium prior).
- [2026-06-02 qwen3-cc-jax-retrospective](analyses/2026-06-02-qwen3_cc-jax-retrospective.md) — FULL lane retrospective (17 exp): frontier 32.4% (seq2048) / 30.4% (seq8192); remat 🏆 + splash 🏆 the wins; batch/CE/SparseCore closed; top unexplored lever = MaxText HOST_OFFLOAD XLA flag stack; ~11 pp headroom to llama3-jax 43.3%.
- [2026-04-26 llama3-8b-torchax-day-summary](analyses/2026-04-26-llama3-8b-torchax-day-summary.md) — full day climb (4,591 → 6,559 tok/s/chip, +42.9 %); validated wins (chunked_xla CE, tokamax-splash, base2 / fuse_reciprocal / max_logit_const), refutations and invalidations (bf16-native CE invalid, q_seq_shards invalid, recipe XLA flags neutral, TP=2 -14 %, splash block sweeps refuted); analysis of the 7.8 pp gap to MaxText 44.6 % (deep-work levers required: faster splash bwd kernel, custom matmul prologue, MaxText-style layer fusion).

## Experiments (19)
- 🏆 **[2026-04-26 jax-llama3-8b-exp27/28b SparseCore RS+AG offload + bs=4](experiments/llama3_8B_autoresearch_optimization/jax/experiments/2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md)** — **accepted, NEW PROGRAM-TARGET BEST** — **62,142 TPS (7,768/chip), 43.6 % MFU** at `bs=4 seq=8192 fsdp=8`. Add `xla_tpu_enable_sparse_core_collective_offload_{reduce_scatter,all_gather}=true` (already had `…_all_reduce`) to relay all three FSDP collectives off the TensorCore onto the SparseCore — frees TC cycles for matmul. **+4.0 % per-chip vs prior frontier (exp 18: 7,471/chip 41.9 %)**; **+9.9 % per-chip vs MaxText reference (7,069/chip 44.6 %)**. Reported MFU gap to MaxText narrowed from 2.7 pp → 1.0 pp. bs=4 hits the sweet-spot once collectives stop hogging TC; bs=6 OOMs with bkv=2048 splash floor.
- 🏆 **[2026-04-26 jax-llama3-8b-exp13/15/18 MaxText XLA flag stack + bkv=2048 match (chronicle)](experiments/llama3_8B_autoresearch_optimization/jax/experiments/2026-04-26-jax-exp13-maxtext-xla-stack-bs5-accepted.md)** — **accepted, prior frontier** — **59,765 TPS (7,471/chip), 41.9 % MFU** at `bs=5 seq=8192 fsdp=8` (exp 18 with bkv=2048). The HOST_OFFLOAD scheduler bundle from `maxtext/benchmarks/xla_flags_library.py` is the breakthrough lever (+10 % alone over JAX baseline). Includes density sweep + kernel-tune sub-experiment (tokamax-splash kernel-only +13.5 % over jax-experimental). Reproduces tokamax-splash perf knobs (`base2/fuse_recip/max_logit_const=30`) as +4.4 % over MaxText's `attention="flash"` path.
- 🏆 **[2026-04-26 torchax-llama3-8b-exp72a tokamax-shipped splash attention](experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-26-exp72a-tokamax-splash-bs3-seq8k-accepted.md)** — **accepted, NEW PROGRAM-TARGET BEST** — **51,139 TPS (6,392/chip), 35.8 % MFU** at `bs=3 seq=8192 fsdp=8`. Replace `jax.experimental.pallas.ops.tpu.splash_attention` with `tokamax._src.ops.experimental.tpu.splash_attention` and turn on its perf knobs (`use_base2_exp=True`, `fuse_reciprocal=True`). +1.3 % per-chip vs exp 65 frontier; +1.8 % at bs=2; +1.2 % at bs=4. Both knobs independently contribute (~+0.7-0.8 %) and compound. Surfaced by reading MaxText `attention_op.py` at `use_tokamax_splash` mode.
- ⭐ **[2026-04-26 torchax-llama3-8b-exp62b → 66b chunked_xla CE iteration](experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-26-exp62b-chunkedxla-ce-bs3-seq8k-accepted.md)** — **accepted, prior frontier**. Path:
  - exp 62b — `tokamax_ce_impl=chunked_xla` instead of `mosaic_tpu`: 50,424 TPS (6,303/chip), 35.3 % MFU — +1.6 % per-chip vs exp 56. Directly motivated by the xprof bottleneck breakdown (CE bwd was the asymmetric Pallas-recompute cost).
  - exp 65 — + tokamax autotune over chunked_xla b/v block sizes: 6,313/chip 35.4 % (+0.16 %).
  - exp 66 — + skip the fp32-cast at the CE boundary that was only required for mosaic_tpu's hardcoded fp32 grad output: 6,329/chip 35.5 % (+0.3 %).
  - 🏆 **exp 66b — combined (chunked_xla + bf16 native + autotune): 50,647 TPS (6,331/chip), 35.5 % MFU**. bs=2: 6,212/chip 34.8 %; bs=4: 6,170/chip 34.6 %; bs=5: OOM by 2.91 GiB.
- ⭐ **[2026-04-26 torchax-llama3-8b-exp61b-profile-bs3](experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-26-exp55b-fp32-master-tokamax-ce-bs3-seq8k-accepted.md#profile)** — **profile capture** at the prior frontier (mosaic_tpu CE) shape. Trace persisted to `gs://<your-bucket>/autoresearch/profiles/llama3-8b-exp61b-profile-bs3/`. Used to author the bottleneck-breakdown observation above.
- 🏆 **[2026-04-26 torchax-llama3-8b-exp55b-fp32-master-tokamax-ce-bs3-seq8k](experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-26-exp55b-fp32-master-tokamax-ce-bs3-seq8k-accepted.md)** — **accepted, NEW PROGRAM-TARGET BEST** — **49,427 TPS (6,178/chip), 34.6 % MFU** at `bs=3 seq=8192 fsdp=8` with **scan + splash + tokamax CE (shard_map'd) + fp32 master + bf16 compute**. Tokamax mosaic-TPU `linear_softmax_cross_entropy_loss` removes the materialized `(B*L, V)` logits buffer (~6 GiB savings), unlocking bs=3 at seq=8192 (was OOM in exp 44). Includes MFU-formula bug fix (data loader exits early; `avg_step_time` divisor now uses actual measured-step count). Branch `v6e8-llama3-8b-torchax-20260426-exp55b-fp32-master-tokamax`. **+35 % per-chip TPS over exp 20.**
- [2026-04-25 torchax-llama3-8b-exp9-splash-autotuned-bs4](experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-25-exp9-splash-autotuned-bs4-accepted.md) — **accepted** — **57,799 TPS (7,225/chip), 36.1 % MFU** at `bs=4 seq=1024 fsdp=8`. Validates exp 8 autotune end-to-end: +1.1 % TPS / +0.4 pp MFU vs exp 3. Branch `v6e8-llama3-8b-torchax-20260425-exp9-splash-autotuned-bs4`. **Best non-program-target peak (short-seq).**
- [2026-04-25 torchax-llama3-8b-exp3-splash-bs4](experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-25-exp3-splash-bs4-accepted.md) — **accepted** — 57,154 TPS (7,144/chip), 35.7 % MFU at `bs=4 seq=1024 fsdp=8`. Splash attention + `bs=4` (which OOM'd on baseline by 1 GiB). +55.6 % TPS / +12.8 pp MFU vs baseline.
- ⭐ [2026-04-25 torchax-llama3-8b-exp8-splash-kernel-autotune](experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-25-exp8-splash-kernel-autotune-potential.md) — **potential** — first **kernel-only autotune** in this program. 171-config sweep across 3 shapes, **3 minutes wall-clock**. Universal winner: `block_q=block_kv=1024, fused_bwd=True` — **+30-32 % fwd+bwd kernel time** vs production. Validated by exp 9.
- [2026-04-25 torchax-llama3-8b-exp10-splash-autotune-multishape](experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-25-exp10-splash-autotune-multishape-potential.md) — **potential** — autotune extended to seq=2048 / 4096 / 8192. Same winner pattern; +29-34 % vs production at every shape. Awaiting full-training fit at seq>1024 bs=4 (blocked on memory unlock — see exp 11 / exp 13).
- [2026-04-25 torchax-llama3-8b-exp11-remat](experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-25-exp11-remat-rejected.md) — **refuted** — enabling `nothing_saveable` remat: 11a slower at fitting shape (52 % step-time tax, MFU 24.0 %); 11b/c still OOM at non-fitting shapes (only 50 MiB lower than exp 4/6 OOMs). **Discovered: torchax.train.make_train_step's `remat_policy` arg is silently ignored** — the canonical line is commented out. Net: `nothing_saveable` is too aggressive AND doesn't fix compile-time HBM peak. Path forward = scan-over-layers (next).
- [2026-04-25 torchax-llama3-8b-exp5-splash-seq2k](experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-25-exp5-splash-seq2k-accepted.md) — **accepted** — 53,472 TPS (6,684/chip), 34.0 % MFU at `bs=2 seq=2048`. Same per-chip B·L as exp 3 reached via longer seq; valid alternate path toward `seq=8192` program target. −1.7 pp MFU vs exp 3 (attention `O(L²)` FLOPs not amortized).
- [2026-04-25 torchax-llama3-8b-exp7-splash-xla-bs4](experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-25-exp7-splash-xla-bs4-rejected.md) — **refuted** — 56,059 TPS, 35.0 % MFU. Recipe XLA flags on top of exp 3: flat (−0.7 pp MFU within noise). Heuristic: recipe flags require per-chip B·L ≳ 10,000 to pay; we are at 4,096.
- [2026-04-25 torchax-llama3-8b-exp2-splash-bs2](experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-25-exp2-splash-bs2-potential.md) — **potential** — 37,299 TPS, 23.3 % MFU. Splash override at baseline shape: +1.6 % TPS within noise. Code change is the precondition for exp 3 + exp 5 wins.
- [2026-04-25 torchax-llama3-8b-exp6-splash-bs4-seq2k](experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-25-exp6-splash-bs4-seq2k-rejected.md) — **invalid (OOM)** — `bs=4 seq=2048` (per-chip B·L = 8,192) doubles the FFN intermediate vs exp 3; compile-time crash. Same OOM mechanism as exp 4. Path forward: selective remat or scan-over-layers.
- [2026-04-25 torchax-llama3-8b-exp4-splash-bs8](experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-25-exp4-splash-bs8-rejected.md) — **invalid (OOM 7.36 GiB)** — splash compresses attention but not FFN intermediates; bs=8 doubles them.
- [2026-04-25 torchax-llama3-8b-exp1-xla-recipe-flags](experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-25-exp1-xla-recipe-flags-rejected.md) — **refuted** — 36,224 TPS, 22.6 % MFU. Recipe XLA flags at baseline shape: flat ±noise. At per-chip B·L = 2,048 the FSDP all-gather is already fully hidden behind compute.
- [2026-04-25 torchax-llama3-8b-baseline (v6e-8)](experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-25-baseline.md) — **supported** — 36,729 TPS (4,591/chip), 22.9 % MFU at `bs=2 seq=1024 fsdp=8`. First end-to-end run of the HF-`transformers`-based Llama 3 8B torchax trainer on multi-host GKE; baseline that all subsequent experiments measure against.
- [2026-04-25 maxtext-gemma4-e4b-v6e8-baseline](experiments/gemma4_autoresearch_optimization/maxtext/experiments/2026-04-25-maxtext-gemma4-e4b-v6e8-baseline.md) — **supported (approximation)** — 282.9 TFLOP/s/device, 10,003 TPS, 30.8 % MFU at `bs=2 seq=8192 fsdp=8`. Authored wiki-local `gemma4-e4b.yml` (no upstream); approximation: missing `num_kv_shared_layers=18` → +47M extra params (~0.6 % vs true E4B).
- [2026-04-25 maxtext-llama3-1-8b-v6e8-baseline](experiments/llama3_8B_autoresearch_optimization/maxtext/experiments/2026-04-25-maxtext-llama3-1-8b-v6e8-baseline.md) — **supported** — 409.4 TFLOP/s/device, 7,069.7 TPS, 44.6 % MFU; reproduces tpu-recipes-v0.1.4 README within −1.0 %.

## Sources (45)

### xprof documentation — capture & deployment (6)
- [xprof — capturing profiles](sources/2026-xprof-capturing-profiles.md) — how XProf traces are captured (programmatic, on-demand gRPC, continuous snapshot).
- [xprof — JAX profiling](sources/2026-xprof-jax-profiling.md) — `jax.profiler` API (`start_trace`/`stop_trace`/`trace`/`TraceAnnotation`, `ProfileOptions`, TPU trace modes).
- [xprof — PyTorch/XLA profiling](sources/2026-xprof-pytorch-xla-profiling.md) — `xp.start_trace` / `xp.Trace` surface and mark-step pitfalls.
- [xprof — TensorFlow profiling](sources/2026-xprof-tensorflow-profiling.md) — `tf.profiler` APIs (experimental.Trace, client.trace, Keras callback).
- [xprof — Docker deployment](sources/2026-xprof-docker-deployment.md) — running the XProf UI in Docker with GCS-backed logdirs.
- [xprof — Kubernetes deployment](sources/2026-xprof-kubernetes-deployment.md) — aggregator/worker topology, headless gRPC discovery, round-robin LB.

### xprof documentation — UI & viewers (7)
- [xprof — overview page](sources/2026-xprof-overview-page.md) — high-level performance summary: step-time breakdown, MFU, goodput, precisions, duty cycle.
- [xprof — trace viewer](sources/2026-xprof-trace-viewer.md) — event timeline across host/device tracks (Steps, XLA Ops, Framework Ops, TraceMe, Host Offload, SparseCore, Launch Stats).
- [xprof — memory profile](sources/2026-xprof-memory-profile.md) — HBM allocation/fragmentation analysis with timeline and peak-instant breakdown.
- [xprof — memory viewer](sources/2026-xprof-memory-viewer.md) — per-buffer HBM view with program-order axis and on-chip memory tiers.
- [xprof — graph viewer](sources/2026-xprof-graph-viewer.md) — interactive optimized-HLO graph with cost overlays, fusion toggle, DOT export.
- [xprof — utilization viewer](sources/2026-xprof-utilization-viewer.md) — per-HW-unit utilization: MXU, VPU, XU, RPU, DMA paths, ICI/HIB.
- [xprof — terminology](sources/2026-xprof-terminology.md) — vocabulary: profile, session, run, host, device, step.

### xprof documentation — HLO & op stats (5)
- [xprof — HLO op stats](sources/2026-xprof-hlo-op-stats.md) — ranked HLO ops by time; total vs self time; fusion, collective-op, replica-group breakdowns.
- [xprof — HLO op profile](sources/2026-xprof-hlo-op-profile.md) — per-op detail: FLOPs/bytes, arithmetic intensity, roofline classification, wasted-time sort.
- [xprof — framework op stats](sources/2026-xprof-framework-op-stats.md) — JAX/TF framework-level op breakdown; host vs device time; call-stack attribution.
- [xprof — perf counters](sources/2026-xprof-perf-counters.md) — raw HW performance counters (TPU issue counters, GPU CUPTI kernel fingerprints).
- [xprof — custom call profiling](sources/2026-xprof-custom-call-profiling.md) — profiling Pallas/Mosaic custom calls; LLO utilization tracks.

### xprof documentation — roofline & megascale (4)
- [xprof — roofline model](sources/2026-xprof-roofline-model.md) — roofline tool: per-memory-tier rooflines, OI/FLOPs axes, cost-model vs HW-counter FLOPs.
- [xprof — megascale stats](sources/2026-xprof-megascale-stats.md) — aggregated cross-slice collective stall statistics (slack, stall, required bandwidth).
- [xprof — megascale viewer](sources/2026-xprof-megascale-viewer.md) — Perfetto-embedded per-collective action graph (D2H / NetworkSend / NetworkReceive / H2D).
- [xprof — megascale viewer SQL](sources/2026-xprof-megascale-viewer-sql.md) — PerfettoSQL queries over megascale traces; p99/mean heavy-tail diagnostics.

### xprof-mcp documentation (1)
- [xprof-mcp — TPU optimization guide](sources/2026-xprof-mcp-tpu-optimization.md) — **crown-jewel practical guide**: roofline, dimension alignment, dtype strategy, fusion, rematerialization, KV cache, XLA flags, decision trees.

### tokamax documentation (5)
- [tokamax — supported ops](sources/2026-tokamax-supported-ops.md) — kernel matrix by platform (TPU / GPU / both) and backend (Mosaic / Triton / XLA).
- [tokamax — basic usage](sources/2026-tokamax-basic-usage.md) — API surface: `implementation=`, autotune, jax-export caveats.
- [tokamax — splash attention](sources/2026-tokamax-splash-attention.md) — TPU flash-attention variant: tunables (block sizes, QKV layouts), soft-cap, base-2 softmax.
- [tokamax — autotuning](sources/2026-tokamax-autotuning.md) — autotune API, on-disk cache, non-determinism, all-implementations sweep.
- [tokamax — benchmarking](sources/2026-tokamax-benchmarking.md) — kernel-only timing (`xprof_hermetic`, CUPTI); fwd / fwd_res / vjp / fwd_and_vjp modes.

### jax-huggingface tutorial series (4)
- [jax-huggingface part 1 — single-device forward + jit](sources/2026-jax-huggingface-part-1.md) — Llama-2-7B bf16 forward on TPU v6e: 4.37s cold → 13ms cached via torchax + jax.jit; pytree + static-arg cookbook.
- [jax-huggingface part 2 — 8-way tensor parallelism](sources/2026-jax-huggingface-part-2.md) — NeMo-Megatron TP recipe for Llama on 8 TPU v6e chips; 3.8× cached speedup (13ms → 3.4ms); sub-linear scaling flagged.
- [jax-huggingface part 3 — StaticCache + jax.jit decode](sources/2026-jax-huggingface-part-3.md) — Llama-2-7B 50-token decode 130.9s → 14.77s (8.9×) via StaticCache + functional_call; 13.48GB captured-constants warning.
- [jax-huggingface part 4 — SD2 via torchax.compile](sources/2026-jax-huggingface-part-4.md) — Stable Diffusion 2-base on **A100 GPU** (not TPU): 5.9s → 1.07s/image after fixing `methods_to_compile=['decode']` for VAE.

### External publications (1)
- [Ultra-Scale Playbook (2025)](sources/2025-ultrascale-playbook.md) — Tazi et al., Hugging Face, Feb 2025. GPU-cluster LLM-training playbook (5D parallelism, FlashAttention, FP8, kernel engineering). Ingested with GPU↔TPU contrast emphasis; 90 figures saved under `raw/assets/ultrascale-playbook/`.

### Methodology (1)
- [LLM Wiki (Karpathy)](sources/2026-karpathy-llm-wiki.md) — the idea file this wiki's SCHEMA.md descends from. Raw/wiki/schema layers; ingest/query/lint ops; index+log navigation pair; contradiction-flag convention.

### Scaling-book chapters (11, ingested 2026-04-23 from [scaling-book](codebases/scaling-book/overview.md) @ `6cda371`, book dated 2025-02-04)
- [Ch 1 — Rooflines](sources/2025-scaling-book-ch1-roofline.md) — three-constraint model + critical intensity thresholds (v5e 240, v5p 164, H100 296).
- [Ch 2 — How to Think About TPUs](sources/2025-scaling-book-ch2-tpus.md) — TPU hardware (MXU / VPU / VMEM / HBM / ICI / DCN / PCIe); canonical per-chip specs for v5e/v5p.
- [Ch 3 — Sharded Matrices](sources/2025-scaling-book-ch3-sharding.md) — four-case sharded-matmul taxonomy; collective-cost formulas.
- [Ch 4 — Transformer Math](sources/2025-scaling-book-ch4-transformers.md) — FLOPs/params/KV-cache formulas; attention dominates FLOPs only when T > 8D.
- [Ch 5 — Parallelize for Training](sources/2025-scaling-book-ch5-training.md) — DP/FSDP/TP/PP compute-comm thresholds on v5p.
- [Ch 6 — Training LLaMA-3 on TPUs](sources/2025-scaling-book-ch6-applied-training.md) — LLaMA-3-70B on 8960-chip v5p pod, 44 days @ 40% MFU.
- [Ch 7 — Transformer Inference](sources/2025-scaling-book-ch7-inference.md) — prefill/decode split; decode always bandwidth-bound.
- [Ch 8 — Serving LLaMA-3-70B](sources/2025-scaling-book-ch8-applied-inference.md) — min topology by dtype (bf16 4×4, int8 4×2, int4 2×2).
- [Ch 9 — Profiling TPU Programs](sources/2025-scaling-book-ch9-profiling.md) — HLO/LLO/xprof; matmul roofline matched to 0.4%.
- [Ch 10 — Programming TPUs in JAX](sources/2025-scaling-book-ch10-jax.md) — Auto/Explicit/shard_map; collective matmul 1.27× via ppermute overlap.
- [Ch 11 — How to Think About GPUs](sources/2025-scaling-book-ch11-gpus.md) — H100/B200 vs v5p/v5e; FLOPs/$ ranking.

## Codebases (27)

### Wave 4 follow-up — deferred Pallas ecosystem (5, added 2026-04-23)
- [graphcast](codebases/graphcast/overview.md) — commit `08cf736` — DeepMind weather model; splash wrapper + `WeatherMeshMask` (non-LLM block-sparse reference).
- [simply](codebases/simply/overview.md) — commit `f40b81e` — DeepMind serving framework; RPA wrapper that documents the DMA-overhead-bytes autotune heuristic (0.5 MiB virtual bytes).
- [jaxite](codebases/jaxite/overview.md) — commit `e4a3351` — Google FHE library; only non-ML Pallas TPU kernel in this wiki (CGGI bootstrap).
- [qwix](codebases/qwix/overview.md) — commit `b966dc4` — Google quantization framework; `QArray`-aware `pallas_call`; successor to AQT.
- [aqt](codebases/aqt/overview.md) — commit `9d1667e` — deprecated; superseded by qwix.

### Wave 4 — Pallas kernel ecosystem (11, added 2026-04-23)
- [axlearn](codebases/axlearn/overview.md) — commit `b479714` — Apple's training framework; **only public TPU Pallas Mamba1/Mamba2/RAttention SSM kernels**, plus splash extensions (dropout + logit sink); GPU Triton megablox (`arXiv:2507.05411`).
- [tpu-inference](codebases/tpu-inference/overview.md) — commit `a657060d` — vLLM's TPU backend; **broadest novel Pallas surface** (RPA v2/v3, MLA v1/v2, fused_moe v1, quantized_matmul blockwise, all_gather_matmul, GDN, SparseCore, structured-sparse-matmul); crown-jewel tuning tables (1,200+ RPA v2 + 600+ quantized_matmul entries; v6 96 MiB / v7 48 MiB VMEM).
- [maxtext](codebases/maxtext/overview.md) — commit `532c8b3d8` — AI-Hypercomputer reference JAX trainer for Gemma/Llama/DeepSeek/Qwen/Mistral/Kimi; splash + ragged-paged-attention + megablox GMM + MLIR-dialect SparseCore. Closest public analogue of the gemma4 program.
- [maxdiffusion](codebases/maxdiffusion/overview.md) — commit `c98002fe` — AI-Hypercomputer reference diffusion trainer; **only repo where ring-attention is wired as first-class splash-integrated kernel** (2026-04-16).
- [ringattention](codebases/ringattention/overview.md) — commit `d2ea1af` — haoliuhl's canonical Pallas TPU ring-attention (Liu et al. 2023 paper companion); unidirectional, no zig-zag. Closes ultrascale-playbook Gap #2.
- [alphafold3](codebases/alphafold3/overview.md) — commit `231efc9` (**tag v3.0.1**, removed from main) — only public production-grade **Pallas fused GLU** (GPU via Triton-on-Pallas); reference for a future TPU port.
- [recurrentgemma](codebases/recurrentgemma/overview.md) — commit `2efa84d` — Google DeepMind's canonical public Mosaic-TPU LRU Pallas scan (Griffin RG-LRU); ancestor of axlearn Mamba; real+complex accumulators + multi-shard correction.
- [ejkernel](codebases/ejkernel/overview.md) — commit `f2289a0` — single-author community Pallas library (erfanzar); broadest community TPU surface (17 kernels), Apache-2.0.
- [EasyDeL](codebases/easydel/overview.md) — commit `090a03b2` — training/serving framework wrapping ejkernel via an operations registry (same author).
- [sglang-jax](codebases/sglang-jax/overview.md) — commit `7907875a` — SGLang JAX port; mostly vendored from tpu-inference; **novel EAGLE speculative-decoding tree kernels** + ecosystem's largest tuning table (~2,000+ RPA entries v4/v5/v6e/v7).
- [marin](codebases/marin/overview.md) — commit `7a56e016d` — vendors levanter; **deployment-time autotune harness** (kernel-agnostic, shard-aware, compile-cost-aware, GCS-persistent) — the autotune pattern this wiki should emulate.
- [tpu-recipes](codebases/tpu-recipes/overview.md) — commit `e284e361` — AI-Hypercomputer's per-(model, hardware, topology) reproduction recipes for Trillium (v6e) and Ironwood (v7x); **canonical reference for MaxText `tuning_params` blocks** (`remat_policy`, `decoder_layer_input: offload`, per-projection offload, FSDP sharding) on Llama 3.1 / Gemma 3-4 / Mixtral / DeepSeek 3 / Qwen 3 / GPT-OSS / GPT-3 175B, plus matmul + HBM microbenchmarks.

### Wave 1–3 (10)
- [jax](codebases/jax/overview.md) — commit `feb5ba0` — The JAX library itself: transformations, sharding, `jax.profiler`, **`jax.experimental.roofline`**, Pallas DSL, and the first-party reference TPU kernel tree at `jax.experimental.pallas.ops.tpu.*` (splash_attention, paged_attention, ragged_paged_attention, megablox, flash_attention, matmul, all_gather, threefry). Ground-truth for every other codebase in this wiki.
- [pallas-forge](codebases/pallas-forge/overview.md) — commit `090510b` — Pallas kernel auto-tuning framework (tiled matmul, fused RMSNorm+residual, SwiGLU/GeGLU); **forward-only — no custom_vjp** so unusable in training as-is. Already evaluated via gemma4 exp 20 (rejected).
- [jax-huggingface](codebases/learning-machine/overview.md) — commit `93328b2` (subfolder of `qihqi/learning_machine`) — 4-part tutorial + scripts running HuggingFace Llama-2-7B and Stable Diffusion under JAX via torchax.
- [xprof](codebases/xprof/overview.md) — commit `2e33c01` — OpenXLA profiler + TensorBoard plugin; canonical metric vocabulary and profile-capture surface for every experiment.
- [xprof-mcp](codebases/xprof-mcp/overview.md) — commit `9970d65` — MCP server exposing 18 tools for agent-driven profile analysis; wraps a local xprof HTTP server and `.xplane.pb` files.
- [torchax](codebases/torchax/overview.md) — commit `8f957d1` — PyTorch backend that runs torch programs on TPU via JAX; op lowering + graph-compile boundary for torch-origin models.
- [tokamax](codebases/tokamax/overview.md) — commit `54bdd95` — Pallas kernel library (splash/flash attention, GLU, layer_norm, ragged_dot, cross-entropy); **direct optimization toolbox**.
- [stablehlo](codebases/stablehlo/overview.md) — commit `9793611` — MLIR op-set + pass-vocabulary reference (grounded docs wiki: 11 topics / 27 sources); consulted when reading HLO dumps + xprof traces. Perf-lens companion: [stablehlo.md](codebases/stablehlo.md).
- [scaling-book](codebases/scaling-book/overview.md) — commit `6cda371` — "How To Scale Your Model" book (DeepMind); 11 chapters to be ingested as sources in Wave 3.
- [autoresearch](codebases/autoresearch/overview.md) — commit `228791f` — Karpathy's autoresearch reference impl (single H100, `val_bpb`); methodological model for this wiki's loop (grounded: `train` time-budgeted loop + FA3/MFU, `prepare` packed dataloader + frozen BPB). Companion: [autoresearch.md](codebases/autoresearch.md).

## Concepts (97)

> **Cross-repo view:** a connected concept page carries an `## In this wiki's repos` section linking each ingested repo's grounded implementation (`wikify connect`). The concept page is the *definition + hub*; the links reach *who implements it, where*.

### Pallas-authoring patterns (11, stubs — added 2026-04-23)
- [online-softmax-with-logit-sink](concepts/online-softmax-with-logit-sink.md) — axlearn splash extension; add `exp(sink - m_final)` to normalizer.
- [in-kernel-dropout](concepts/in-kernel-dropout.md) — generate dropout mask from prng_key + block indices (avoids HBM mask materialization).
- [two-level-chunk-recomputation](concepts/two-level-chunk-recomputation.md) — SSD / Mamba2 pattern for linear-recurrence kernels.
- [block-sparse-offset-masks](concepts/block-sparse-offset-masks.md) — precomputed `(n_q_blocks, n_kv_blocks)` offset table for paged-attention sliding-window.
- [multi-shard-sequence-parallel-correction](concepts/multi-shard-sequence-parallel-correction.md) — recurrentgemma pattern for sequence-parallel linear recurrences.
- [custom-splash-masks](concepts/custom-splash-masks.md) — subclass `splash_attention_mask.Mask` for any structured adjacency (LLM, weather, etc.).
- [manual-mlir-dialect-pallas](concepts/manual-mlir-dialect-pallas.md) — drop below `pallas_call` to raw MLIR for SparseCore / TCGEN05 / TMEM features.
- [pallas-on-triton-fused-gemm-activation-gemm](concepts/pallas-on-triton-fused-gemm-activation-gemm.md) — GPU fused-GLU pattern; alphafold3 v3.0.1 reference.
- [grouped-program-ids-for-l2](concepts/grouped-program-ids-for-l2.md) — GPU-side PID reordering for L2 cache reuse (not applicable on TPU).
- [dma-overhead-heuristic](concepts/dma-overhead-heuristic.md) — ~0.5 MiB virtual-bytes DMA-setup cost in block-size autotune.
- [nvidia-weight-tile-bytes-limit](concepts/nvidia-weight-tile-bytes-limit.md) — 101,376-byte per-SM shared-memory weight-tile cap (H100/GB10/A100).

### Autotune-harness patterns (3, stubs — added 2026-04-23)
- [jaxpr-hash-cache-keys](concepts/jaxpr-hash-cache-keys.md) — pin autotune-cache to stringified jaxpr (marin/levanter pattern).
- [compile-time-aware-autotune-filtering](concepts/compile-time-aware-autotune-filtering.md) — discard candidates whose compile cost > 0.20 s over baseline.
- [vmem-oom-fallthrough](concepts/vmem-oom-fallthrough.md) — catch `resource_exhausted … vmem` at autotune, demote candidate.

### Hardware facts (1, added 2026-04-23)
- [vmem-budget](concepts/vmem-budget.md) — per-generation VMEM: v4 32, v5e 32, v5p 95, **v6e 96**, **v7 48** MiB (v6e→v7 halves).

### Architecture & hardware (12)
- [memory-hierarchy](concepts/memory-hierarchy.md) — TPU memory stack (HBM, VMEM, SMEM, CMEM, host RAM).
- [hbm](concepts/hbm.md) — on-package high-bandwidth memory.
- [vmem](concepts/vmem.md) — on-chip vector scratchpad.
- [cmem](concepts/cmem.md) — TPU v4-only on-chip memory tier.
- [mxu](concepts/mxu.md) — systolic matrix unit (128×128 or 256×256 depending on generation).
- [vpu](concepts/vpu.md) — vector programmable unit; gateway to MXU/XU/RPU.
- [sparsecore](concepts/sparsecore.md) — TPU v5p/v6e units for embedding-style sparse lookups.
- [tensor-node](concepts/tensor-node.md) — compute unit on a TPU chip (two per chip).
- [ici](concepts/ici.md) — intra-slice Inter-Chip Interconnect.
- [dcn](concepts/dcn.md) — inter-slice Data Center Network.
- [megascale](concepts/megascale.md) — Google multi-slice collective-communication layer over DCN.
- [multi-slice](concepts/multi-slice.md) — workload spanning multiple TPU slices.

### Performance metrics & roofline (13)
- [mfu](concepts/mfu.md) — Model FLOPs Utilization.
- [step-time](concepts/step-time.md) — wall-clock per training step.
- [mxu-utilization](concepts/mxu-utilization.md) — matrix-unit achieved/peak throughput.
- [memory-bandwidth-utilization](concepts/memory-bandwidth-utilization.md) — HBM bandwidth achieved/peak.
- [tpu-duty-cycle](concepts/tpu-duty-cycle.md) — fraction of wall time the TPU was busy.
- [arithmetic-intensity](concepts/arithmetic-intensity.md) — FLOPs per byte; x-axis of roofline.
- [peak-flops](concepts/peak-flops.md) — device's peak FLOPs/s ceiling.
- [hbm-bandwidth](concepts/hbm-bandwidth.md) — peak HBM throughput per TPU generation; sets the slope of the roofline.
- [roofline-model](concepts/roofline-model.md) — compute-vs-memory-bound performance model.
- [ridge-point](concepts/ridge-point.md) — arithmetic intensity at which peak FLOPs becomes reachable.
- [memory-bound](concepts/memory-bound.md) — BW-limited regime (below ridge point).
- [compute-bound](concepts/compute-bound.md) — FLOPs-limited regime (above ridge point).
- [ici-roofline](concepts/ici-roofline.md) — ICI-bandwidth ceiling for sharded ops.

### Compiler & HLO (13)
- [hlo](concepts/hlo.md) — XLA High-Level Optimizer IR.
- [hlo-op](concepts/hlo-op.md) — single HLO operation node.
- [xla-fusion](concepts/xla-fusion.md) — op-fusion compiler pass.
- [xla-flags](concepts/xla-flags.md) — `XLA_FLAGS` / `LIBTPU_INIT_ARGS` configuration surface.
- [custom-call](concepts/custom-call.md) — XLA's external-kernel mechanism.
- [pallas-kernel](concepts/pallas-kernel.md) — user-authored kernel in Pallas DSL.
- [mosaic-kernel](concepts/mosaic-kernel.md) — TPU kernel lowering backend used by Pallas.
- [autotuning](concepts/autotuning.md) — search over kernel-config space for fastest impl.
- [xla-cost-model](concepts/xla-cost-model.md) — compiler's static FLOPs/bytes estimates.
- [latency-hiding-scheduler](concepts/latency-hiding-scheduler.md) — XLA scheduling pass that overlaps collectives with compute.
- [async-collectives](concepts/async-collectives.md) — XLA flags for async all-reduce/all-gather fusion.
- [hlo-dumping-and-diffing](concepts/hlo-dumping-and-diffing.md) — workflow for capturing, inspecting, and diffing pass-by-pass HLO; xprof-mcp dump tools for offline CPU-only analysis.
- [aot-compilation](concepts/aot-compilation.md) — JAX AOT pipeline (`jit → lower → compile → cost_analysis`); HLO pre-filter for kernel-replacement hypotheses; pre-experiment screening; offline analysis.

### Kernels (9)
- [flash-attention](concepts/flash-attention.md) — tiled SRAM-resident attention algorithm.
- [splash-attention](concepts/splash-attention.md) — TPU-native flash-family attention (sparse-mask, soft-cap).
- [ring-attention](concepts/ring-attention.md) — ring-of-K/V streaming for long-sequence attention; tokamax kernel exists but not wired to public API.
- [ragged-dot](concepts/ragged-dot.md) — grouped matmul for MoE routing.
- [gated-linear-unit](concepts/gated-linear-unit.md) — fused SwiGLU/GEGLU/REGLU (TPU falls back to XLA).
- [layer-norm](concepts/layer-norm.md) — LayerNorm/RMSNorm fused kernel (TPU falls back to XLA).
- [memory-efficient-cross-entropy](concepts/memory-efficient-cross-entropy.md) — fused linear+log-softmax+NLL avoiding `[B,V]` logits.
- [attention-block-sizes](concepts/attention-block-sizes.md) — `block_q`/`block_kv` tunables for flash/splash attention.
- [base2-softmax](concepts/base2-softmax.md) — `exp(x) = 2^(x·log2 e)` rewrite for TPU's native base-2 exp.

### Optimization techniques (7)
- [rematerialization](concepts/rematerialization.md) — recompute activations to save HBM.
- [scan-over-layers](concepts/scan-over-layers.md) — `jax.lax.scan` pattern; O(N)→O(1) compile time.
- [dimension-alignment](concepts/dimension-alignment.md) — batch/hidden shape divisibility rules for MXU.
- [dtype-strategy](concepts/dtype-strategy.md) — bf16/fp32/fp8/int8 mixing; fp32 weights force per-matmul cast.
- [int8-quantization](concepts/int8-quantization.md) — AQT weight-only / full-int8; shifts critical batch.
- [host-offload](concepts/host-offload.md) — async host↔accelerator transfer to reduce HBM pressure.
- [training-memory-budget](concepts/training-memory-budget.md) — rule of thumb: ~16 bytes/param for bf16 + AdamW.

### Parallelism & collectives (12)
- [fsdp](concepts/fsdp.md) — Fully Sharded Data Parallelism (all-gather + reduce-scatter on ICI).
- [tensor-parallelism](concepts/tensor-parallelism.md) — op-level split; cheap within ICI island (≤8).
- [sequence-parallelism](concepts/sequence-parallelism.md) — TP companion that shards LayerNorm/Dropout along sequence; converts internal all-reduce to reduce-scatter + all-gather.
- [context-parallelism](concepts/context-parallelism.md) — parallelism axis that splits the sequence dim for attention.
- [pipeline-parallelism](concepts/pipeline-parallelism.md) — layer-stage parallelism; AFAB/1F1B/interleaved/zero-bubble/DualPipe schedule family; rarely used on a single TPU pod.
- [expert-parallelism](concepts/expert-parallelism.md) — MoE axis; all-to-all dispatch + tokamax `ragged_dot` compute; ICI-rich, DCN-cliff.
- [sharding](concepts/sharding.md) — GSPMD mesh partitioning.
- [collective-communication](concepts/collective-communication.md) — cross-replica ops umbrella.
- [all-gather](concepts/all-gather.md) — assembles one tensor from per-replica shards.
- [all-reduce](concepts/all-reduce.md) — sums and broadcasts across replicas.
- [reduce-scatter](concepts/reduce-scatter.md) — reduces across replicas and distributes the result as shards; FSDP's gradient-reduction half.
- [send-recv-done](concepts/send-recv-done.md) — XLA's four-op async-collective pattern on TPU.

### Inference (5)
- [kv-cache](concepts/kv-cache.md) — per-token key/value cache; makes decode bandwidth-bound at small batch.
- [static-cache](concepts/static-cache.md) — fixed-shape KV cache enabling `jax.jit`.
- [continuous-batching](concepts/continuous-batching.md) — paged-attention serving pattern.
- [decode-profile-signature](concepts/decode-profile-signature.md) — HBM-bound decode diagnostic pattern (dynamic-slice dominant, duty cycle <60%).
- [serving-warmup](concepts/serving-warmup.md) — pre-compile all shapes before serving traffic.

### Profiling (11)
- [profile-capture](concepts/profile-capture.md) — umbrella for XProf capture APIs.
- [jax-trace](concepts/jax-trace.md) — `jax.profiler` capture surface.
- [pytorch-xla-trace](concepts/pytorch-xla-trace.md) — `xp.start_trace` / `xp.Trace` surface.
- [custom-trace-annotations](concepts/custom-trace-annotations.md) — `xp.Trace` / `jax.profiler.TraceAnnotation` / `tf.profiler.experimental.Trace`.
- [trace-me](concepts/trace-me.md) — low-level annotation API underlying the framework-specific trace wrappers.
- [mark-step-sync](concepts/mark-step-sync.md) — PyTorch/XLA lazy-tensor sync boundary.
- [trace-viewer](concepts/trace-viewer.md) — xprof timeline UI.
- [trace-event-categories](concepts/trace-event-categories.md) — Trace Viewer track taxonomy.
- [perfetto](concepts/perfetto.md) — embedded trace-viewing UI used by megascale tools.
- [perf-counters](concepts/perf-counters.md) — raw HW performance counters.
- [llo-utilization](concepts/llo-utilization.md) — custom-call HW-resource usage track in the trace viewer.

## Observations (0)
*None yet.*

## Analyses (6)
- [2026-04-30 Bottleneck reverse-index — profile bucket → candidate levers](analyses/2026-04-30-bottleneck-reverse-index.md) — reverse-index page keyed by xprof bucket; for each bucket lists standing levers (concept/source links), tried-and-refuted hypotheses with verdict notes (so the agent doesn't re-propose disproved Pallas-replacements), and open hypotheses tagged for it. Bootstraps from the [llama3-8b converged-stack breakdown](observations/llama3-8b-torchax-converged-stack-bottleneck-breakdown.md). Explicitly non-exhaustive — meant as a starting shortlist for hypothesis generation, not a fence on exploration.
- [2026-04-24 Gemma 4 E4B — JAX fp32-master + seq=8192 regime ceiling (exp 52–53)](analyses/2026-04-24-gemma4-jax-fp32master-seq8k-regime.md) — new-regime baseline (exp 52 at 26,807 TPS seq=2048 b=1 fp32-master) + seq=8192 infeasibility probe on v6e-4. Legacy bf16 also OOMs at seq=8192 (memory wall is not AMP-specific). XLA compile-time peak HBM non-monotonic in seq_len (seq=6144 = 49.66 GiB, seq=8192 = 35.18 GiB). Three-branch forward path documented.
- [2026-04-23 Pallas kernel directory](analyses/2026-04-23-pallas-kernel-directory.md) — repo-by-repo catalog of ~200 Pallas kernels across ~30 repos, with source-code refs, stability, perf claims, use cases, and callers. Cross-cutting functional-category tables + 6 subpages ([§1 JAX+tokamax](analyses/pallas-kernel-directory/01-upstream-jax-tokamax.md), [§2 AI-Hypercomputer](analyses/pallas-kernel-directory/02-ai-hypercomputer.md), [§3 Inference engines](analyses/pallas-kernel-directory/03-inference-engines.md), [§4 Research labs](analyses/pallas-kernel-directory/04-research-labs.md), [§5 Frameworks & quant](analyses/pallas-kernel-directory/05-frameworks-quant.md), [§6 Community](analyses/pallas-kernel-directory/06-community-research.md)). Confirms Zig-Zag ring attention absent everywhere; identifies AlphaFold3 @ v3.0.1 fused GLU and apple/axlearn SSM kernels as the key novel content.
- [2026-04-23 Pallas kernel source survey](analyses/2026-04-23-pallas-kernel-source-survey.md) — web-research inventory of every public source of Pallas kernel code. Identifies 5 top ingest candidates (maxtext, tpu-inference, maxdiffusion, axlearn, sglang-jax) and updates 3 open hypothesis candidates on the Ultra-Scale Playbook page with reference implementations found in the wild.
- [2026-04-23 Gemma 4 E4B on v6e-4 — optimization ceiling (exp 1–33, torchax stack)](analyses/2026-04-23-gemma4-v6e4-optimization-ceiling.md) — synthesis of the 33-experiment torchax-stack loop. Best = exp 25 (33,372 TPS, +9.2% over baseline). Loop at diminishing returns; identifies Pallas-fuses-into-matmul lesson from exp 33 and proposes next levers (scan-over-layers Option B, compile cache, hardware scale-up).
- [2026-04-24 Gemma 4 E4B — JAX-stack ceiling + process retrospective (exp 34–42)](analyses/2026-04-24-gemma4-jax-ceiling-and-process-retrospective.md) — JAX-stack analog. Best = exp 36 (34,614 TPS / 23.05 % MFU, **+3.7 % over torchax session-best**). Explains why JAX beats torchax on same hardware (compile-time HBM ~1.25 GiB lower; no torchax dispatch overhead). Includes cross-stack generalizable lessons + **process retrospective** with concrete `program.md` / `SCHEMA.md` rule additions.
