---
title: "Retrospective — qwen3_cc5 / jax as of 2026-06-12"
type: analysis
tags: [retrospective, qwen3_cc5, jax]
model: qwen3_cc5
lane: jax
created: 2026-06-12
updated: 2026-06-12
---

# qwen3_cc5 / jax retrospective — 2026-06-12

**Scope**: 8B/v6e-8 (the lane's only variant row).
**Period**: 2026-06-02 (v000 baseline) → 2026-06-12 (v019), with v001–v019 all
run in the single 2026-06-12 session.
**Experiments**: 20 pages — 6 supported, 9 refuted, 3 invalid, 1 inconclusive,
1 baseline. 0 backfilled (every dispatch filed).
**Frontier**: **33.6% MFU @ seq 8192** (1389 ms, 47,182 TPS, validated 50
steps, [v019](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v019-val50.md));
seq-2048 best 31.4% ([v007](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v007-bs2-vmem.md)).
Baseline was 20.5% @ seq 2048 — **+13.1 pp at 4× the sequence length** in one
session.

**Mode**: FULL (no prior retrospective).

## Search-strategy assessment

- ✅ **Frontier NOT stuck**: it moved 4 times in 19 experiments (v002 splash
  +1.7 pp, v007 batch+vmem +8 pp, v015 seq-8192 unlock, v017 partial remat
  +6.3% tok/s), most recently 2 experiments ago.
- ✅ **No single-axis imbalance**: structural/kernel work (splash port, 4 CE
  kernel attempts, 4 remat experiments) ≈ 9 vs single-axis probes (3 flag,
  1 block, 4 batch) ≈ 8. Ratio ~1:1, well under the 5:1 alarm.
- ✅ **Closed topics are closed on evidence, not fatigue** (each with a
  measured endpoint or named mechanism — see anti-recommendations).
- 🔵 **Genuinely under-explored axes remain**: scan-over-layers (0 tried),
  custom Pallas authoring (0 tried — all kernel work so far was ports),
  host tensor offload (0 tried), data-formatting drilldown (0 tried).
- ⚠️ **Cross-model reference point**: llama3-8b-jax (same hardware, similar
  dense-GQA arch, V=128k vs our 152k) reached **43.6% MFU @ seq 8192 at bs4**
  in this wiki's sibling program. The dominant difference is **per-chip batch
  at the target seq** (they fit bs4; we fit bs1) on a stack that includes
  scan-over-layers. The 10 pp gap is mostly a memory-architecture gap, not a
  kernel-quality gap.

**Inference**: the lane is NOT exhausted. The highest-leverage unexplored
direction is the memory-architecture cluster (scan-over-layers + host
offload) that would raise per-chip batch at seq 8192 — exactly the lever the
llama3 evidence prices at several pp.

## Mechanism tree

- **Batch / sequence amortization** (7: v001, v004, v005, v007, v013, v014 + v019 validation)
  - bs2 + splash + vmem flag — 🏆 v007 +40.5% tok/s/chip (+8 pp MFU)
  - seq-8192 full stack — 🏆 v015 (the unlock; categorized under AC below)
  - bs4 / bs2 on baseline code — ❌ v001 (91.7 GB), ❌ v004 (43.1 GB) — cliffs measured
  - bs3 + CE — ❌ v014 (34.6 GB, −3.4 over)
  - seq-8192 without remat — ❌ v013 (44.9 GB, −13.6 over)
  - 50-step validation — ✅ v019 (frontier durable, zero drift)
  - 🔵 **bs2 @ seq 8192 — UNEXPLORED, memory-blocked** (needs ~12–15 GiB beyond current; the scan/offload cluster is the path)
- **Activation checkpointing / remat** (4: v012, v015, v017, v018)
  - dots-saveable policy — 💥 v012 (NaN from step 4 + 27% overhead; [observation](../observations/qwen3-jax-remat-splash-nan.md))
  - save_attn policy (checkpoint_name q/k/v/attn_out) — 🏆 v015 (seq-8192 enabler, no NaN)
  - partial remat skip-every-5 — 🏆 v017 (+6.3% tok/s; recompute tax linear, 0.43 GiB/exempt layer)
  - skip-every-4 — ❌ v018 (−2.4%: **non-monotonic near HBM ceiling**; optimum ≈30.7 GiB)
  - 🔵 finer remat scope (e.g. MLP-only / named-FFN policies) — unexplored, low expected value after v018's ceiling finding
- **CE / softmax loss** (4: v003, v006, v008(+b), v011)
  - tokamax mosaic_tpu + fp32 casts — ❌ v003 (2.49 GB f32 w-cast erases streaming win; +6.3%)
  - tokamax mosaic_tpu + bf16 inputs — 💥 v006 (kernel hardcodes f32 scratch Ref)
  - chunked_xla + bf16 inputs — 💥 v008b (lse follows x.dtype → −0.73 ≈ ln 2 bias)
  - chunked_xla + f32-x-only — ✅ v011 (parity ≤0.007; the validated recipe; +3.2% at small seq, pays at seq 8192)
  - 🔵 vocab-sharded CE — stub filed, **deprioritized** (no partial-vocab semantics in tokamax → hand-rolled kernel, effort L; v018 showed negative returns on the freed-memory use case)
- **Splash attention** (2: v002, v016)
  - splash port (bq2048/bkv1024, fused bwd) — 🏆 v002 (−11.5% step, −6 GiB; required q-prescale + output-transpose fixes vs the llama3 port basis — [observation](../observations/llama3-jax-splash-scaling-transpose-audit.md))
  - fwd bkv=2048 at seq 8192 — ❌ v016 (flat; llama3's +0.7% doesn't reproduce with QK-norm GQA)
- **libtpu / XLA flags** (2: v009, v010)
  - SparseCore collective offload — ❌ v009 (−12.7%; offloaded collectives became async-done waits at seq2048/bs2)
  - HOST_OFFLOAD scheduler bundle — ❌ v010 (flat; +40% compile)
  - Both are **shape-gated**: queued for re-test at seq 8192, then deprioritized when v015's profile showed collectives collapse to ~4.5% there (sub-5% rule)
  - ✅ `xla_tpu_scoped_vmem_limit_kib=81920` — lane-standard (v005→v007; mandatory for bs≥2/seq8192 splash fused-bwd)
- **VMEM / scratch** (1: v005)
  - splash dkv tile vs 32 MB default scoped limit — ⚠️ v005 (inconclusive; root-caused and fixed by the flag)
- **torch.compile / scan-over-layers** (0)
  - 🔵 **scan-over-layers — UNEXPLORED** (catalog: `concepts/scan-over-layers.md`; llama3-jax frontier stack uses it; benefits: O(36)→O(1) compile (~2×130 s per run today), smaller HLO, and the memory-scheduling regularity that may unlock bs2 @ seq 8192)
- **Pallas kernel authoring** (0 — all kernel work so far was ports)
  - 🔵 **fused QK-norm+RoPE — UNEXPLORED** (the model program's named Qwen3-specific candidate; no existing kernel in the catalog — would be written fresh; targets part of the 14.4% loop-fusion bucket)
  - 🔵 NSA / ring attention — not applicable at seq 8192 (splash sufficient; these are ≥16k-seq levers)
- **HBM / host offload** (0)
  - 🔵 **host offload of decoder-layer inputs — UNEXPLORED** (catalog: tpu-recipes `decoder_layer_input: offload` tuning_params on this exact hardware class; would free activation memory toward bs2 @ seq 8192)
- **Profile-driven attribution** (cross-cutting; every verdict carried op_profile + HLO evidence)
  - 🔵 data-formatting bucket (4.9% at seq 8192) — borderline sub-5%; drilldown unexplored

## Coverage gaps — what's NOT been explored

| Topic | Catalog options | Tried | Unexplored |
|---|---|---|---|
| torch.compile / scan | scan-over-layers | 0 | **scan-over-layers** |
| HBM / offload | host tensor offload (tpu-recipes), TPU_PREMAPPED_BUFFER | 0 | both |
| Pallas authoring | fused QK-norm+RoPE (program CAN list) | 0 | the kernel |
| CE restructure | vocab-sharded CE | stub only | hand-rolled kernel (deprioritized) |
| Batch @ target seq | bs2 @ 8192 | 0 (memory-blocked) | via scan/offload |
| Splash | block sweep, tokamax-shipped knobs (base2/fuse_recip) | blocks only | tokamax splash perf knobs (llama3: +4.4% — but its splash impl differs; medium transferability) |

## Recommended next directions

| # | Direction | Topic | Gain | Conf | Effort | Score | Source |
|---|---|---|---|---|---|---|---|
| 1 | **Scan-over-layers** (Flax `nnx.scan`/`jax.lax.scan` over the 36 decoder layers, composed with save_attn remat) | scan | +1–3 pp indirect (compile ~4 min/run → seconds; HLO shrink; memory regularity → bs2@8192 path) | medium-high (llama3-jax frontier uses it on same hw) | M–L | ~0.6 | sibling-model: llama3-8b-jax frontier stack |
| 2 | **Host offload of decoder-layer inputs** (tpu-recipes `decoder_layer_input: offload` pattern) | hbm-offload | frees ~2–4 GiB → more remat-exempt layers or bs2@8192 piece | medium (tpu-recipes on v6e, llama3 adjacency) | M | ~0.4 | local-gap: tpu-recipes catalog |
| 3 | **bs2 @ seq 8192** (after 1 and/or 2 land) | batch-seq | +3–6 pp (llama3 evidence: batch at target seq is the big lever) | medium | S once unblocked | ~0.4 (gated) | cross-model: llama3 bs4@8192 = 43.6% |
| 4 | **tokamax-shipped splash perf knobs** (base2_exp, fuse_reciprocal — `USE_TOKAMAX_SPLASH=1` path already in splash_attn.py) | splash | +1–4% step (llama3: +4.4%) | medium (different splash internals; v016 showed weak transfer of block configs) | S | ~0.35 | sibling-model knobs: llama3 exp72a |
| 5 | **Fused QK-norm+RoPE Pallas kernel** | pallas-author | +1–2% (part of 14.4% loop fusion) | low-medium (no existing kernel; fresh authoring) | L | ~0.1 | local-gap: model program CAN list |
| 6 | Data-formatting drilldown (top HLO ops at seq 8192) | profile | diagnostic | high (cheap) | S | diag | local-gap: v015/v017 profiles |

## Anti-recommendations (don't pursue these)

- **Flag-bundle retests at current shapes** — SC offload (−12.7%) and the
  scheduler bundle (flat) both refuted at seq2048/bs2, and their target
  bucket (collectives) is ~4.5% at seq 8192. Structurally closed on this
  topology.
- **Splash block sweeps** — bq2048/bkv1024 confirmed optimal twice (v002
  autotuned port, v016 flat at 2048).
- **Batch probes without a new memory lever** — four cliffs measured
  (v001/v004/v013/v014); the walls are known to the GB.
- **Skip-every remat ladder** — closed with a non-monotonic optimum (v017
  supported, v018 refuted); don't ride closer to the HBM ceiling.
- **Vocab-sharded CE** — L-effort hand-rolled kernel; primary payoff
  (freed HBM → more exempt layers) measured negative near the ceiling (v018).
- **bf16 inputs to either tokamax CE path** — mosaic_tpu crashes (f32 Ref),
  chunked_xla biases the loss (lse in x.dtype). The f32-x rule is load-bearing.

## Cross-lane brief

### Sibling lane summaries (model pages)

- **qwen3-cc5-torchax**: frontier 19.2% MFU @ seq2048 bs1 (2026-06-02
  baseline; no experiments since). No retrospective. Nothing to import — the
  jax lane leads by +14.4 pp. Reverse transfer (splash port, vmem flag, CE
  recipe, remat policy) is available to that lane when it wakes.

### Cross-model reference (same wiki, same hardware — context only)

- **llama3-8b-jax**: 43.6% MFU @ seq8192 **bs4** (scan + tokamax CE
  chunked_xla + tokamax splash + HOST_OFFLOAD bundle + SC offload). Two
  caveats discovered THIS session when borrowing from that lane: (a) its
  splash dispatch lacks the q-prescale + output transpose
  ([audit observation](../observations/llama3-jax-splash-scaling-transpose-audit.md));
  (b) its flag-bundle wins are shape/batch-gated and did not reproduce here
  at bs1–2. Borrow mechanisms, re-verify per-stack.

### Conflicts with our prior work

- llama3's SC-offload / scheduler-bundle wins vs our v009/v010 refutations —
  resolved as op-point gating (their wins were at bs4-5 @ seq8192 with the
  full stack; our collectives are already 4.5% at bs1 @ seq8192). Re-test
  there ONLY if bs2+ @ 8192 lands and collectives grow back above ~10%.

## Experiment ledger (machine-readable — used by future incremental retrospectives)

| v-ID | Date | Variant | Topic | Verdict | Frontier-shift |
|---|---|---|---|---|---|
| v000 | 2026-06-02 | 8B/v6e-8 | baseline | supported | yes (initial: 20.5% @ seq2048) |
| v001 | 2026-06-12 | 8B/v6e-8 | batch-seq | refuted | no |
| v002 | 2026-06-12 | 8B/v6e-8 | splash-attention | supported | yes (+1.7 pp → 22.3%) |
| v003 | 2026-06-12 | 8B/v6e-8 | ce-softmax-loss | refuted | no |
| v004 | 2026-06-12 | 8B/v6e-8 | batch-seq | refuted | no |
| v005 | 2026-06-12 | 8B/v6e-8 | vmem-scratch | inconclusive | no |
| v006 | 2026-06-12 | 8B/v6e-8 | ce-softmax-loss | invalid | no |
| v007 | 2026-06-12 | 8B/v6e-8 | batch-seq | supported | yes (31.4% @ seq2048) |
| v008 | 2026-06-12 | 8B/v6e-8 | ce-softmax-loss | invalid | no |
| v009 | 2026-06-12 | 8B/v6e-8 | libtpu-xla-flags | refuted | no |
| v010 | 2026-06-12 | 8B/v6e-8 | libtpu-xla-flags | refuted | no |
| v011 | 2026-06-12 | 8B/v6e-8 | ce-softmax-loss | supported | no (enabler) |
| v012 | 2026-06-12 | 8B/v6e-8 | activation-checkpointing | invalid | no |
| v013 | 2026-06-12 | 8B/v6e-8 | batch-seq | refuted | no |
| v014 | 2026-06-12 | 8B/v6e-8 | batch-seq | refuted | no |
| v015 | 2026-06-12 | 8B/v6e-8 | activation-checkpointing | supported | yes (seq-8192 unlock, 31.8%) |
| v016 | 2026-06-12 | 8B/v6e-8 | splash-attention | refuted | no |
| v017 | 2026-06-12 | 8B/v6e-8 | activation-checkpointing | supported | yes (33.6% settled @ seq8192) |
| v018 | 2026-06-12 | 8B/v6e-8 | activation-checkpointing | refuted | no |
| v019 | 2026-06-12 | 8B/v6e-8 | measurement-validation | supported | no (validates frontier) |

**Methodology note**: categorization 100% Wave-1/2 (tags + slugs — every page
was filed with topic tags this session). Dictionary additions proposed:
`prem|skip-every` → Activation checkpointing; `cxla|cxf32|tkmce` → CE/softmax
loss; `scoff|hosch` → libtpu/XLA flags; `val50|validation` →
measurement-validation (new minor topic for ledger-locking runs).

## Sources

- Per-lane log: `wiki/experiments/qwen3_cc5_autoresearch_optimization/jax/log.md`
- Per-model refuted-patterns: none yet (proposed: create from this ledger's 9 refutations)
- Model page: `wiki/models/qwen3-cc5-jax.md`
- Topic taxonomy: `wiki/model-optimization-index.md`
- Pallas catalog: `wiki/analyses/2026-04-23-pallas-kernel-directory.md`
- All 20 experiment pages in `wiki/experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/`

## See also

- `wiki/models/qwen3-cc5-jax.md`
- `.claude/skills/formulate-hypothesis/SKILL.md` — downstream consumer
