---
title: "Retrospective — qwen3_cc / jax as of 2026-06-02"
type: analysis
tags: [retrospective, qwen3-cc, jax]
model: qwen3_cc
lane: jax
created: 2026-06-02
updated: 2026-06-02
---

# qwen3_cc / jax retrospective — 2026-06-02

**Scope**: 8B/v6e-8 (only variant). **Mode**: FULL (first retrospective).
**Period**: 2026-06-02 (single-day lane, ~17 experiments).
**Experiments**: 17 — 6 supported, 2 refuted, 7 invalid (integration/OOM, no frontier impact), 2 in-flight (v016/v017).
**Frontier**: seq2048 = **32.4% MFU / 6,299 tok/s/chip** (v008 remat+splash bs4, MXU util 44.6%); seq8192-target = **30.4% MFU / 5,305 tok/s/chip** (v009 remat+splash bs1).

## Search-strategy assessment

✅ **NOT exhausted — actively climbing.** Frontier moved **4 times in one day**: 20.5% (baseline) → 22.0% (v004 remat) → 25.1% (v005 remat bs3) → **32.4%** (v008 splash bs4); +seq8192 target reached (v009). Cumulative **+58% tok/s/chip** over baseline.
✅ **Kernel work is being done** — single-axis-probe : kernel-port ratio ≈ **1:1** (batch/sparsecore probes vs splash+CE kernel ports). No single-axis imbalance; the canonical premature-exhaustion failure mode is absent.
✅ **No frontier-stuck signal** — longest gap between frontier moves is 1 experiment.
🔵 **Clear headroom to the same-architecture sibling**: llama3-jax reaches **43.3% MFU** with HOST_OFFLOAD XLA flags + SparseCore-offload + scan + CE + splash. qwen3-jax is at 32.4% having done remat+splash+CE-correctness; **the XLA-flag-stack and scan levers that drove llama3 from ~34% → 43% are UNEXPLORED here.** ~11 pp of catalogued upside.

**Inference**: the lane is early and healthy. The next wins are the *compiler/runtime* levers (MaxText XLA flag stack, host-offload, scan) that llama3-jax proved on the identical hardware/regime — not more batch/attention probing.

## Mechanism tree

- **Activation checkpointing / remat** (3 exp) — the pivotal unlock
  - per-layer jax.checkpoint — 🏆 v004 (bs2, +1.5pp→22.0%); 🏆 v005 (bs3, +3.1pp→25.1%). Unlocks batch scaling (without it, bs≥2 OOMs).
  - 🔵 selective remat (`save_dot_except_mlp`, offload policies) — UNEXPLORED (index AC catalog).
- **Splash attention** (4 exp) — the big frontier mover
  - GQA-native splash + scoped-vmem=96M — 🏆 v008 (bs4, +7.3pp→32.4%, MXU 44.6%); 🏆 v009 (seq8192 target, 30.4%).
  - splash w/o vmem flag — 💥 v006/v007 (scoped-VMEM OOM in backward dkv kernel; fixed by `--xla_tpu_scoped_vmem_limit_kib=98304`).
  - 🔵 splash block-size tuning (bkv/bq sweep), 🔵 sliding-window mask — UNEXPLORED.
- **tokamax CE / streamed loss** (5 exp) — validated, memory-only
  - mosaic_tpu CE — ✅ v013 (numerically correct, MFU 21.0% ~neutral at bs2); ❌ v014 (splash+CE bs6 = 30.5% < 32.4% — overhead not repaid at seq2048).
  - integration bugs (all fixed) — 💥 v010 (chunked_xla impl absent), 💥 v011/v012 (absl-flags collision → added pre-parse guard).
  - 🔵 CE at seq8192 (bigger logits — where it should pay) — IN FLIGHT (v016).
- **Batch/seq amortization** (2 exp, refuted pre-remat)
  - bs2/bs4 without remat — 💥 v001/v002 (compile OOM 43G/92G). REFUTED standalone; subsumed by remat.
- **FSDP / collective offload** (2 exp)
  - SparseCore collective-offload — ❌ v003 (bs1, −4pp; async overhead > TC relief); 🔵 retry at bs4-frontier scale — IN FLIGHT (v017).
- **libtpu / XLA flags** (1 lever used: scoped_vmem) — **🔵 MAJOR GAP**
  - 🔵 MaxText HOST_OFFLOAD flag bundle — UNEXPLORED (llama3-jax: +10% alone).
  - 🔵 collective-matmul, latency-hiding-scheduler flags — UNEXPLORED.
- **Dimension alignment / dtype (AMP)** — **🔵 GAP**
  - 🔵 fp32-master + bf16-compute true mixed precision — UNEXPLORED (trainer has weights_dtype but no separate compute_dtype). Often +MFU.
- **torch.compile / scan-over-layers** — **🔵 GAP**
  - 🔵 scan-over-36-layers — UNEXPLORED (llama3-jax used it; cuts compile-time HBM, enables denser shapes).
- **Tensor parallelism** — 🔵 tp=2 mesh UNEXPLORED (llama3 found tp=2 −14%; low priority).
- **Pallas kernels (Qwen3-specific)** — 🔵 fused QK-norm+RoPE kernel UNEXPLORED (Qwen3 has q_norm/k_norm+RoPE in sequence — a fusion candidate llama3 doesn't have).

## Coverage gaps — what's NOT been explored

| Topic | Tried | Unexplored (catalog) |
|-------|-------|----------------------|
| libtpu/XLA flags | scoped_vmem only | **HOST_OFFLOAD bundle, collective-matmul, scheduler flags** (llama3 evidence) |
| AMP / dtype | bf16 single-dtype | **fp32-master + bf16-compute** |
| scan / compile | none | **scan-over-layers** |
| CE | seq2048 (neutral) | **seq8192 + bigger batch** (in-flight v016) |
| Pallas kernels | splash, CE | **fused QK-norm+RoPE** (Qwen3-specific) |
| Splash knobs | default blocks | block-size sweep, sliding-window |
| seq scaling | 2048, 8192 | seq16384 |

## Recommended next directions

| # | Direction | Topic | Gain | Conf | Effort | Score | Source |
|---|---|---|---|---|---|---|---|
| 1 | MaxText HOST_OFFLOAD XLA flag bundle | libtpu/XLA flags | +5–10% | high | S | high | sibling-model: llama3-jax (+10% alone) |
| 2 | seq8192 + CE + bs2/4 (CE's real payoff) | CE | unlock seq8192 batch | medium | S | high | local-gap (in-flight v016) |
| 3 | AMP fp32-master / bf16-compute | dtype | +2–5% | medium | M | med | local-gap: index dtype topic |
| 4 | scan-over-layers | compile/scan | compile-HBM ↓ → denser shapes | medium | M | med | sibling-model: llama3-jax used it |
| 5 | Fused QK-norm+RoPE Pallas kernel | Pallas | +1–2% | low | L | low | local-gap: Qwen3-specific |
| 6 | Splash block-size sweep (bkv/bq) | splash | +0.5–1% | medium | S | med | local-gap: splash knobs |
| 7 | seq16384 (splash scale) | seq | target-seq+ | low | S | low | local-gap |

## Anti-recommendations (don't pursue)

- **Batch-scaling without memory wins** — v001/v002 OOM'd; only works stacked on remat (already the baseline). Don't re-probe raw batch.
- **SparseCore-offload at small/bs1 shape** — v003 −4pp. (v017 is testing the bs4 shape; if that also refutes, close the topic.)
- **CE as a seq2048 throughput lever** — v014 refuted (30.5% < 32.4%). CE is a *memory enabler* only; use it where logits bind HBM (seq8192+batch), not seq2048.
- **tp=2** — low priority (llama3 −14%); revisit only if fsdp saturates.

## Cross-lane brief

### Sibling lane (model pages)
- **qwen3-cc-torchax**: baseline 19.2% MFU (minimal trainer, no optimization loop run yet). The jax lane is **+13.2 pp ahead**. Transferable TO torchax once it iterates: remat, splash, CE (all proven here).
- **llama3-8b-jax** (same architecture class, same v6e-8): frontier **43.3% MFU** at bs4/seq8192 via HOST_OFFLOAD flags + SC-offload(3 collectives) + scan + tokamax-CE + tokamax-splash. **This is the target regime** — qwen3-jax has the same model shape (32q/8kv hd128) and hardware.

### Transferable levers from llama3-jax (not yet on this lane)
1. **MaxText HOST_OFFLOAD XLA flag stack** — the single biggest llama3 lever (+10%). Rec #1.
2. **scan-over-layers** — llama3 used it for compile-HBM. Rec #4.
3. **SC-offload of all 3 FSDP collectives at bs4+seq8192** — helped llama3 (+3.4%) at the *big* shape, unlike qwen3 v003 at bs1. v017 tests this.
4. **tokamax-splash perf knobs** (use_base2_exp, fuse_reciprocal, max_logit_const) — UNEXPLORED here.

### Conflicts with our prior work
- SC-offload: llama3-jax says +3.4% (at bs4/seq8192); qwen3 v003 says −4% (at bs1). **Resolution: shape-dependent** — offload helps only when collectives are a big absolute cost (large batch/seq). v017 re-tests at the qwen3 frontier shape; don't treat v003 as closing the topic.

## Experiment ledger (machine-readable)

| v-ID | Date | Variant | Topic | Verdict | Frontier-shift |
|---|---|---|---|---|---|
| baseline | 2026-06-02 | 8B/v6e-8 | batch-seq | supported | yes (20.5% base) |
| v001 | 2026-06-02 | 8B/v6e-8 | batch-size | invalid | no (OOM) |
| v002 | 2026-06-02 | 8B/v6e-8 | batch-size | invalid | no (OOM) |
| v003 | 2026-06-02 | 8B/v6e-8 | fsdp-collective | refuted | no |
| v004 | 2026-06-02 | 8B/v6e-8 | activation-checkpointing | supported | yes (→22.0%) |
| v005 | 2026-06-02 | 8B/v6e-8 | activation-checkpointing | supported | yes (→25.1%) |
| v006 | 2026-06-02 | 8B/v6e-8 | splash-attention | invalid | no (VMEM) |
| v007 | 2026-06-02 | 8B/v6e-8 | splash-attention | invalid | no (VMEM) |
| v008 | 2026-06-02 | 8B/v6e-8 | splash-attention | supported | yes (→32.4%) |
| v009 | 2026-06-02 | 8B/v6e-8 | splash-attention | supported | yes (seq8192 30.4%) |
| v010 | 2026-06-02 | 8B/v6e-8 | tokamax-ce | invalid | no (impl) |
| v011 | 2026-06-02 | 8B/v6e-8 | tokamax-ce | invalid | no (absl) |
| v012 | 2026-06-02 | 8B/v6e-8 | tokamax-ce | invalid | no (absl) |
| v013 | 2026-06-02 | 8B/v6e-8 | tokamax-ce | supported | no (canary 21.0%) |
| v014 | 2026-06-02 | 8B/v6e-8 | tokamax-ce | refuted | no (30.5%) |
| v016 | 2026-06-02 | 8B/v6e-8 | tokamax-ce | in_progress | — |
| v017 | 2026-06-02 | 8B/v6e-8 | fsdp-collective | in_progress | — |

## Sources

- Per-lane log: `wiki/experiments/qwen3_cc_autoresearch_optimization/jax/log.md`
- Model page: `wiki/models/qwen3-cc-jax.md`
- Topic taxonomy: `wiki/model-optimization-index.md`
- Pallas catalog: `wiki/analyses/2026-04-23-pallas-kernel-directory.md`
- Cross-lane ref: `wiki/models/qwen3-cc-torchax.md`, `wiki/experiments/llama3_8B_autoresearch_optimization/jax/README.md`
- All 17 experiment pages in `wiki/experiments/qwen3_cc_autoresearch_optimization/jax/experiments/`

## See also

- [Qwen3 8B — jax](../models/qwen3-cc-jax.md)
- `.claude/skills/formulate-hypothesis/SKILL.md` — downstream consumer
