---
title: "Gemma 4 E4B on v6e-4: optimization ceiling analysis (exp 1–33)"
type: analysis
tags: [analysis, gemma4, v6e4, ceiling, pallas, fsdp, xla-fusion]
created: 2026-04-23
updated: 2026-04-23
---

Synthesis of the 33-experiment autoresearch loop optimizing Gemma 4 E4B pretraining throughput on TPU v6e-4 with torchax + HuggingFace + JAX/Pallas. **Current best (exp 25): 33,372 TPS at seq=1024 batch=3 fsdp=4 bf16**, +9.2 % over the initial baseline-seq1024 reference of 30,570 TPS. The optimization loop has reached diminishing returns on this hardware/model combo — 8 experiments since exp 25 (exp 26–33) produced no further wins. This page explains why, and what would move the needle next.

## Trajectory

```
30,570  baseline-seq1024   (exp 0)
 31,387  splash attention  (exp 8,  +2.7 %)
 32,340  + bf16 CE         (exp 12, +5.8 %)
 32,717  + batch=3         (exp 15, +7.0 %)
 33,016  + fused_bwd       (exp 18, +8.0 %)
 33,193  + SEQ_MINOR layout (exp 24, +8.6 %)
 33,372  + splash block=1024 (exp 25, +9.2 %)  ← ceiling
  ──── exp 26–33: no further win ────
 32,251  seq=2048 b=1 at exp25 stack (exp 28, kept but dominated)
 30,657  Pallas RMSNorm+custom_vjp (exp 33, refuted −8.1 %)
 12,711  2D mesh dp=2 tp=2 (exp 32, refuted −58 %)
```

## What worked (kept in trunk)

| Knob | Win | Mechanism |
|---|---|---|
| Splash attention (Pallas) | +2.7 % | XLA couldn't express online-max softmax; Pallas kernel gets fusion + numerical stability + N² elimination. **Fixed seq=2048 NaN** as a side-effect (exp 9). |
| bf16 cross-entropy | +2.5 % | Saves ~2 GiB fp32 logits materialization. Gemma 4's `final_logit_softcapping=30.0` keeps logits bounded → bf16 log-softmax is stable. |
| Selective remat (`checkpoint_dots_with_no_batch_dims`) | memory-enabling | Only rematerializes linear projections, not the full forward. Unlocks batch=3 without the +27 % step cost of full remat. |
| Batch=3 | +3.3 % (vs batch=1) | Amortizes weight-loading + collective overhead over more work. HBM at 98.78 % is tight but stable. |
| Splash `use_fused_bwd_kernel` | +0.9 % | One kernel call for Q/K/V gradients instead of 3. |
| Splash `QKVLayout.SEQ_MINOR` | +0.5 % | Better HBM streaming pattern for batch-dim. |
| Splash `block_q = block_kv = 1024` | +0.6 % | Whole-seq one-block attention at seq=1024. |

## What didn't work (refuted / parked)

| Category | Experiments | Why not |
|---|---|---|
| **Async-collective XLA flag bundle** | exp 1, 13, 21 | `async_collective_fusion*` flags reorder collectives aggressively → break compute-fusion locality for small workload (~14 % of step in collectives, but compute + memory traffic blows up +30 %). Isolated tests (exp 30, 31) cleared `latency_hiding_scheduler` and `overlap_compute_collective_tc` — only the `async_collective_fusion*` family is culpable. Park those flags permanently at this scale. |
| **out_shardings pinning** | exp 2 | `tie_word_embeddings=True` ⇒ lm_head↔embed_tokens share storage, torchax `JittableModule` dedupes one of them, jit sees mismatched output-sharding spec. Step-1 recompile remains open. |
| **Full activation remat** | exp 3 | −22 % TPS; traded for memory, enabled exp 5's selective remat. |
| **Batch growth under full remat** | exp 4, 7 | +16 % over exp 3 but net regressions vs baseline. Memory-first approach only pays when paired with selective remat + splash. |
| **Splash block=256 / asymmetric (1024, 512)** | exp 19, 29 | Smaller kv tile adds reload cost (softmax accumulator is sequential across kv tiles, no concurrency gain). Symmetric 1024 is optimal at seq=1024. |
| **pallas-forge RMSNorm** | exp 20 | No `custom_vjp` → grad fails. Parked. |
| **Scan-over-layers** | exp 26 | Gemma 4's 42-layer stack is heterogeneous (18 kv-shared layers), has Python-int `layer_idx`, side-effect `shared_kv_states` dict, kwargs-only call. Option A (`torchax.train.ScannedModule`) blocked on 5 independent reasons. Option B (custom scan + stacked weights + kv-carry) is 300–500 LOC — deferred. |
| **tokamax `dot_product_attention`** | exp 27 | mosaic_tpu kernel rejects `mask.k_start`; 21/42 Gemma 4 layers are sliding-window → fallback to XLA, defeating the purpose. Would need upstream fix. |
| **seq=2048 b=1 at exp25 stack** | exp 28 | Kept at 32,251 TPS (+0.9 % over exp 14, the prior seq=2048 point) but strictly dominated by exp 25's higher batch throughput. |
| **2D mesh (dp=2 tp=2)** | exp 32 | At 2 chips/tp axis, collective overhead (~84 per-MLP + 84 per-attn all-reduces in fwd alone) **swamps** the per-chip compute savings. **2.4× slower at matched global batch.** Revisit only at ≥8 chips/axis. |
| **Hand-written Pallas RMSNorm + `custom_vjp`** | exp 33 | Kernel is numerically correct (bf16 fwd abs_err ≤ 1.6e-2, bwd rel_err ≤ 2 %), but **−8.1 % TPS**. XLA was already fusing RMSNorm into the matmul→norm→matmul seams via the `loop_fusion` bucket; introducing a Pallas custom-call breaks that fusion → 2× extra HBM round-trips per norm + ~25 ms/step shard_map launch overhead across ~250 norm sites per step. |

## The generalizable lesson from exp 33

> **Pallas kernels are a net win only when XLA wasn't already exploiting the pattern via fusion.**

Exp 8 (splash attention) succeeded because XLA can't express online-softmax attention — it materializes `[B, H, S, S]` and loses memory bandwidth. Exp 33 (Pallas RMSNorm) failed because XLA was already fusing RMSNorm with neighboring matmuls (visible in the `loop_fusion` op bucket); replacing a fused segment with an opaque custom-call breaks neighbor fusion and costs more memory traffic than it saves.

**Corollary**: Pallas SwiGLU almost certainly loses for the same reason (SwiGLU is a simple 3-op pattern XLA fuses). Don't build it.

**Pallas is still the right tool for**:
- Ops with internal sequential reduction that XLA can't fuse (online softmax ✓, online variance? Probably ✓ at very long seq where tile-wise works; ✗ at seq=1024 where XLA fits whole tensor in vmem).
- Ops with custom block layouts / non-standard tensor shapes.
- Ops that *force* a specific memory layout XLA's fusion planner can't reach.

## Step-time decomposition at exp 25 (estimated from exp 8 profile + later deltas)

Per step (~368 ms @ batch=3 seq=1024 fsdp=4):
- convolution fusion (MLP + attention matmuls, MXU compute-bound): ~160 ms (~43 %)
- loop fusion (RMSNorm + SwiGLU + residual + elementwise): ~60 ms (~16 %)
- custom-call (splash attention): ~30 ms (~8 %)
- all-gather (FSDP weight gather): ~45 ms (~12 %)
- all-reduce-scatter (FSDP grad reduce): ~20 ms (~6 %)
- custom fusion: ~40 ms (~11 %)
- other (formatting, send/recv-done, idle): ~15 ms (~4 %)

**Compute-bound with OI=1376 FLOPs/byte on v6e's ridge point 578** — the matmul bucket is already near-optimal per chip. At exp 25 config, per-chip effective throughput is ~210 TFLOPS out of v6e's 946 TFLOPS peak = **~22 % MFU**. Most of the "missing" 78 % is:
- bf16 dot-precision overhead and dispatch latency (~15 %)
- FSDP all-gather per layer (~12 % of step)
- Python + trace + launch overhead not captured in peak-rate math (~5 %)
- Activation memory traffic (~5 %)

## What could actually move the needle

Ordered by expected upside, with a note on cost:

1. **Scale up to v6e-8 or v5p-4**. More chips → sharding is cheaper per parameter, collective overlap actually pays, 2D mesh becomes viable at ≥4 chips/axis. Expected: +30–60 % TPS (including 2× raw device count). **Out of scope this session (hardware).**
2. **`scan_over_layers` Option B** (exp 26 deferred). 300–500 LOC custom scan with stacked weights + explicit kv-carry. Expected compile-step-0 drop from ~150 s → ~10 s (iteration speed ×15) and possibly 2–5 % step-time via shared activation buffers. **Eng-heavy, bounded outcome.**
3. **Persistent JAX compile cache** (`JAX_COMPILATION_CACHE_DIR`). Cuts compile-step-0 (~150 s) to ~0.5 s on cached runs. **Not TPS**; pure iteration speed. **Trivial (env var).**
4. **AoT compile** via `jit.lower().compile()`. May eliminate step-1 recompile (~152 s). Requires pinning output shardings — previously blocked on tied-weight dedup (exp 2). Potentially unblocks with a `dedup_parameters=False` + manual tie. **Not TPS (steady-state unchanged); ~1-day eng.**
5. **Reach broader hypothesis set via newer kernels** (tokamax `linear_softmax_cross_entropy_loss` — fused LM head + CE, potentially frees 1–1.5 GiB fp32 logits). **Medium eng, uncertain — may OOM the compile planner again.**
6. **Quantization / fp8 for matmuls**. Out of scope per behavioral rule #8 (model-semantic change).

## Verdict

**Session-level: exp 25 is the practical ceiling on this stack (torchax + HF Gemma 4 E4B + v6e-4, FSDP-only).** Further wins require either a hardware change, a structural model change, or the 300–500 LOC scan-over-layers refactor. The flag-space has been well-characterized, the Pallas-kernel approach has been tested on its two likely remaining candidates, and the 2D-mesh structural change is refuted at this device count.

The wiki has documented the full trajectory; future sessions should pick up from either (a) scan-over-layers Option B, (b) the moment better hardware is available, or (c) different model scale (E2B or 27B+) where different bottlenecks will dominate.

## See also

- [program.md](../experiments/gemma4_autoresearch_optimization/program.md) — session protocol.
- [RESULTS.tsv](../experiments/gemma4_autoresearch_optimization/torchax/experiments/) — master ledger (gitignored).
- [OBSERVATIONS.md](../experiments/gemma4_autoresearch_optimization/torchax/experiments/OBSERVATIONS.md) — per-experiment threaded notes.
- Per-experiment writeups (all under `wiki/experiments/gemma4_autoresearch_optimization/`): [baseline](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-22-baseline.md), [exp 1](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp1-async-collective-flags-rejected.md), [exp 2](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp2-pin-out-shardings-rejected.md), [exp 3](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp3-full-remat-accepted.md), [exp 4](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp4-batch2-with-remat-rejected.md), [exp 5](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp5-selective-remat-accepted.md), [exp 6](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp6-selective-batch2-accepted.md), [exp 7](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp7-selective-batch3-rejected.md), [exp 8 splash](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp8-splash-attention-accepted.md), [exp 9](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp9-splash-seq2048-accepted.md), [exp 10](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp10-seq2048-batch2-bf16ce-rejected.md), [exp 11](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp11-offload-remat-rejected.md), [exp 12 bf16 CE](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp12-bf16-ce-accepted.md), [exp 13](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp13-async-collective-flags-revisit-rejected.md), [exp 14](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp14-splash-seq2048-bf16ce-accepted.md), [exp 15 batch=3](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp15-splash-bf16ce-batch3-accepted.md), [exp 16](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp16-fused-bwd-broken-rejected.md), [exp 17 fused_bwd](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp17-splash-fused-bwd-accepted.md), [exp 18 fused_bwd+batch=3](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp18-fused-bwd-batch3-accepted.md), [exp 19](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp19-splash-block256-rejected.md), [exp 20](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp20-pallas-forge-rmsnorm-rejected.md), [exp 21](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp21-latency-hiding-alone-rejected.md), [exp 22](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp22-batch4-fused-bwd-rejected.md), [exp 23](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp23-seq2048-batch2-fused-bwd-rejected.md), [exp 24 SEQ_MINOR](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp24-splash-seq-minor-accepted.md), [**exp 25 block=1024 (current best)**](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp25-splash-block1024-accepted.md), [exp 26 scan](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp26-scan-over-layers-potential.md), [exp 27 tokamax](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp27-tokamax-dpa-potential.md), [exp 28 seq=2048](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp28-seq2048-exp25config-accepted.md), [exp 29 asymmetric splash](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp29-splash-asymmetric-rejected.md), [exp 30 latency-hiding](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp30-latency-hiding-solo-potential.md), [exp 31 overlap-compute-collective](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp31-overlap-compute-collective-tc-potential.md), [exp 32 2D mesh](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp32-2d-mesh-tp2-rejected.md), [exp 33 Pallas RMSNorm](../experiments/gemma4_autoresearch_optimization/torchax/experiments/2026-04-23-exp33-pallas-rmsnorm-rejected.md).
- [splash-attention concept](../concepts/splash-attention.md) — the one big win's mechanism.

## Sources

- All exp branches under `perfautoresearch/v6e4-20260423-*` (trunk at exp 25, `ebb00ec`).
- All profiles under `raw/profiles/2026-04-{22,23}-gemma4-*/` (gitignored).
- All per-experiment `.md` writeups under `wiki/experiments/gemma4_autoresearch_optimization/`.
