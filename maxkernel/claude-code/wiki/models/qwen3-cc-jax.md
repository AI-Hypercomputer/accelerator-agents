---
title: "Qwen3 8B — jax"
type: model
architecture: qwen3-cc
lane: jax
sizes: ["8B"]
hardwares: ["v6e-8"]
tags: [qwen3-cc, jax]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3 8B — jax

Qwen3 8B (`Qwen/Qwen3-8B`) as a native-JAX (Flax NNX) port on **TPU v6e-8**.
**Scaffold only** — the trainer is not yet written. This lane lights up once the
[torchax](qwen3-cc-torchax.md) baseline is stable and a from-scratch JAX port
becomes a ranked hypothesis; it then serves as the reference ceiling for
cross-lane comparison.

## Target metrics

- Primary: MFU at seq_len=8192 (v6e bf16 peak ≈ 918 TFLOPS/chip).
- Secondary: tokens/sec (and tok/s/chip), step time, peak HBM.

## How to run

```
TODO: native-JAX trainer not yet written.
```

## Variant matrix

| Size | Hardware | Status | Baseline (step / TPS / MFU) | Current best (step / TPS / MFU) | Open hyps | Frontier exp |
|------|----------|--------|-----------------------------|----------------------------------|-----------|--------------|
| 8B | v6e-8 | live | 512 ms / 31,955 TPS / 20.5% MFU @ seq2048 bs8 | **seq2048: 35.8% MFU / 6,964 tok/s/chip (v018)** · **seq8192: 34.8% MFU / 6,068 tok/s/chip (v045, +norm-stat save)** | 1 | [v045 save-norm-stats](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-05-v045-save-norm-stats.md) (seq8192) · [v018](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v018-xla-flag-stack.md) (seq2048) |

*Climb (seq2048): baseline 20.5% → remat bs2 22.0% (v004) → remat bs3 25.1% (v005) → remat+splash bs4 32.4% (v008) → **+MaxText HOST_OFFLOAD XLA flag stack 35.8%** (v018), **+15.3 pp / +74% tok/s/chip**. Stable seq2048 config: `--use_remat --use_splash --batch_size=4` + `LIBTPU_INIT_ARGS` = scoped-vmem(98304) + the 7-flag scheduler bundle (all_experimental_scheduler_features, scheduler_memory_pressure_tracking, ag_backward_pipelining, host_transfer_overlap_limit=24, scheduler_percent_shared_memory_limit=100, latency_hiding_scheduler_rerun=2, max_concurrent_host_send_recv=100), on qwen3-8b-jax:v006-splash. (Note: `xla_jf_spmd_threshold_for_windowed_einsum_mib` is invalid on this XLA build — v019 — and is NOT part of the config; the +3.4 pp is from the 7 scheduler flags.) Remaining gap to llama3-jax sibling (43.3%): ~7.5 pp.*

***Seq-8192 target reached** (v009): splash+remat bs1 seq8192 = **30.4% MFU / 5,305 tok/s/chip** — the program-target seq, which the dense path can't run at all. Best-for-target-seq.*

***Seq-8192 frontier moved to v028** (2026-06-02): **scan-over-layers + collective-overlap flags = 32.3% MFU / 5,632 tok/s/chip** (+1.9 pp / +6.2% over v009 bs1). scan (stacked-param `lax.scan`) makes the async-collective-fusion overlap flags productive (they regressed on the unrolled graph, v023); compile dropped to 35 s. Stable: `--use_remat --use_splash --use_scan --batch_size=1` + 7 scheduler + 4 async-collective-fusion flags on qwen3-8b-jax:v028-scan.*

> [!warning] Superseded by [v034](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v034-maxtext-ce-s8k-bs2.md) (maxtext-CE) on 2026-06-02 — the "hard wall" was CE-backward-bound, not structural
> The note below called the seq8192 batch a documented hard wall (bs1>bs2>bs3 anti-amortization, residual = MaxText's bs3 kernel-pipeline efficiency). **v034 refutes it**: porting MaxText's T5X custom_vjp CE (`--use_maxtext_ce`) made bs2 **amortize** — 34.4% / 5,992 > v028 bs1 5,632. The wall was the tokamax/autodiff CE-backward transient, not a structural batch limit. Frontier line updated below.

~~***Seq-8192 batch is a documented HARD WALL** (2026-06-02): batch anti-amortizes — bs1 5,632 (v028) > bs2 5,553 (v031) > bs3 4,595 (v030)... residual is MaxText's bs3 per-token kernel-pipeline efficiency, not a config lever.*~~ — **superseded by v034**.

***Seq-8192 frontier moved to v034** (2026-06-02): **MaxText T5X custom_vjp CE (`--use_maxtext_ce`) + scan + overlap, bs2 = 34.4% MFU / 5,992 tok/s/chip** (+6.4% over v028 bs1 5,632; +7.9% over v031 plain-CE bs2 5,553). The CE swap is the lever that makes **batch amortize** at seq8192 — identical config to v031 except the CE path. [v033](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v033-maxtext-ce-s8k-bs1.md) confirmed bs1 parity (5,656 ≈ v028 5,632). Stable: `--use_remat --use_splash --use_scan --use_maxtext_ce --batch_size=2` + 7 scheduler + 4 overlap on qwen3-8b-jax:v033-maxtext-ce.*

***Seq-8192 frontier → v035** (2026-06-02): **maxtext-CE bs3 (NO offload) = 34.6% MFU / 6,030 tok/s/chip** (+0.6% over v034 bs2). Two findings: (1) **bs3 fits without offload** — maxtext-CE dropped the ~4.6G tokamax f32[H,V] weight-gather that forced [v030](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v030-scan-offload-ce-s8k-bs3.md)'s offload (and beats it +31%); (2) **batch amortization plateaued** (near-linear bs2→bs3 step scaling) — bs2–bs3 is the sweet spot. Gap to MaxText (6,942) = **86.9%**; the residual ~13% is NOT memory/batch but a scheduling/fusion/kernel-config difference ([profile-driven hypothesis](../hypotheses/qwen3-jax-seq8192-kernel-gap.md), profile-analyzer attributing the bucket). Frontier `--batch_size=3` on qwen3-8b-jax:v033-maxtext-ce.*

***scan+overlap is seq8192-specific, NOT universal** ([v032](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v032-scan-overlap-s2k-bs4.md), refuted): the same scan+overlap package **regresses at seq2048** (34.6% / 6,731, −3.4% vs v018) — the collective tail is too small at the shorter seq to amortise the scan-body overhead. **seq2048 frontier stays v018 (35.8% / 6,964, no scan).***

***Seq-8192 frontier → v045** (2026-06-05, the "try all" gap-closing sweep): **+ save norm rsqrt stats on-device = 34.8% MFU / 6,068 tok/s/chip** (+0.63% over v035), `SAVE_NORM_STATS=1` → `save_only_these_names("norm_rsqrt")` on qwen3-8b-jax:v045-save-norm-stats. Keeping the f32 norm reduction stat on-device skips its backward recompute (works where offload can't — tiny + on-device). Gap to MaxText (6,942) = **87.4%**. Marginal/single-run; confirm recommended. This was the ONLY frontier-moving lever of the sweep — see [gap-closing retrospective](../analyses/2026-06-05-qwen3-cc-jax-gap-closing-attempt.md). Profiler re-ranked the gap: #1 = SiLU recomputed twice in bwd (3,985 ms, bf16) — **walled** (can't save on-device 21.7 GB; can't offload, v039). **Tensor-parallel REFUTED** ([v043b](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-05-v043b-tp2-bs6.md): tp=2 at equal global batch = −26%; at equal global batch tp can't reduce per-chip recompute, only adds all-reduces). **RMSNorm bf16 weight-mult = parity** ([v042](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-05-v042-rmsnorm-bf16.md), +0.16% sub-noise). Residual ~12.6% confirmed = MaxText's host-offload pipelining (runtime/build wall).*

> [!note] Earlier (superseded by v028): seq8192 explored vs MaxText — initially read as "structural ceiling," but scan+overlap (v028) reopened it
> The [MaxText reference](qwen3-cc-maxtext.md) (45.3% MFU [**non-causal — causal-adjusted 39.8%**; see [caveat](qwen3-cc-maxtext.md)] / 6,942 tok/s/chip @ seq8192 bs3, +31% over v009) reopened the question. Explored the offload-enabled-bs3 lever ([v025](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v025-offload-remat-s8k-bs2.md), [v026](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v026-offload-ce-s8k-bs3.md)): added a working `--offload_remat` host-offload policy (equivalence-passing), got **within 2.34G of fitting bs3** (CE drops the logit wall, offload frees no-batch dots; last blocker = batch-dim layer activations needing MaxText's invasive *named* offload). **But the throughput thesis is undermined**: v016 shows bs2 (29.5%) < bs1 (30.4%) — our stack doesn't amortize batch at seq8192, so fitting bs3 likely won't beat bs1. The ~31% gap to MaxText is **per-token efficiency** (its integrated offload+loss+matmul-fusion pipeline), not a batch-fitting lever we can port. seq8192 frontier **stays v009 (30.4%)**; [hypothesis](../hypotheses/qwen3-jax-host-offload-bs3-seq8192.md) → inconclusive. (seq2048 ceiling separately confirmed at parity with MaxText.)

*tokamax CE: validated numerically correct (v013) but **not** a seq2048 frontier-mover — CE+splash bs6 (v014) = 30.5% < 32.4% (bs4 is the seq2048 occupancy sweet spot; CE is a memory-enabler, overhead not repaid where logits already fit). CE's live value is at seq8192 (larger logits) → next: seq8192+CE+bs2.*

*Baseline captured at seq 2048 (global batch 8). **Cross-lane: jax 20.5% MFU /
3,994 tok/s/chip beats torchax 19.2% / 3,724 (+7.3% tok/s/chip, +1.3 pp)** at the
identical shape — the native-JAX path has lower collective/dispatch overhead
(profile: 21.7% collective vs torchax 31.3%). See the
[jax baseline](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-qwen3-jax-v6e8-baseline.md)
and [torchax baseline](../experiments/qwen3_cc_autoresearch_optimization/torchax/experiments/2026-06-02-qwen3-torchax-v6e8-baseline.md).*

## Cross-variant open hypotheses

Re-ranked after [v018](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v018-xla-flag-stack.md) (XLA flag stack SUPPORTED, new frontier 35.8%; batch/CE/SparseCore closed). The XLA flag stack is now part of the frontier config — remaining open levers stack on it:

From [retrospective #5](../analyses/2026-06-02-qwen3_cc-jax-retrospective-5-maxtext-delta.md) (MaxText config delta), post-probe status:
1. [**MaxText exact offload recipe**](../hypotheses/qwen3-jax-maxtext-offload-recipe.md) — **REFUTED** ([v039](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v039-maxtext-offload-recipe-s8k-bs3.md), −6.6% vs no-offload). The corrected set (decoder_input+qkvo, recompute mlpwi) + the full MaxText flag superset incl `*_in_chain` STILL regresses — host-offload genuinely doesn't pipeline on our scan/libtpu (beats the wrong-set v036 4,908 → 5,630, but < v035 6,030). The MaxText offload advantage is **unreachable by config** on this stack.
2. **MaxText XLA flag superset** — **PARITY** ([v038](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v038-maxtext-flags-s8k-bs3.md)); all 25 flags valid, neutral without offload.
3. [**Splash bkv 1024→2048**](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v040-splash-bkv2048-s8k-bs3.md) — **REFUTED** ([v040](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v040-splash-bkv2048-s8k-bs3.md) parity +0.05%; bkv=1024 already optimal).
4. `DISABLE_COLLECTIVE_MATMUL` (MXU 54%→61% candidate) — `jf_spmd_threshold` form crashed v019; no working alternate found. Catalogued, low prior.

**CONFIG + CODE SEARCH vs MaxText EXHAUSTED** (offload refuted v039, flags parity v038, sharding match, splash parity v040, **activation sharding constraints parity v041**). seq8192 frontier = v035 (34.6% / 6,030 = 86.9% of MaxText). [Activation `with_logical_constraint`](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-03-v041-shard-acts-s8k-bs3.md) — the obvious code gap — is a **no-op in pure FSDP** (XLA already infers the batch-sharded layout); our inflated loop-fusion/data-formatting buckets are intrinsic recompute + the known QKV→splash transpose, not layout conversion. The residual ~13% is confirmed = MaxText's host-offload PIPELINING (runtime/build wall, v039) — out of config/compile scope. See [retrospective #5 outcome](../analyses/2026-06-02-qwen3_cc-jax-retrospective-5-maxtext-delta.md).

Earlier-closed: **named-offload** all forms refuted (v030/v036/v039 — even MaxText's exact recipe; host-offload not pipelined on our stack). **MXU/logical-axis sharding** — our sharding == MaxText's ([observation](../observations/qwen3-jax-sharding-matches-maxtext.md)). **maxtext-CE @ seq2048** ([refuted v037](../hypotheses/qwen3-jax-maxtext-ce-seq2048.md), seq8192-specific).
2. [scan-over-layers](../hypotheses/qwen3-jax-scan-layers.md) — **RESOLVED**: at **seq8192** scan makes the overlap flags productive → +6.2% (v028, new frontier); at **seq2048** scan+overlap regresses −3.4% (v032). seq8192-specific win, not universal. Closed.
3. [AMP fp32-master / bf16-compute](../hypotheses/qwen3-jax-amp-mixed-precision.md) — **deprioritized**: trainer already full-bf16; true AMP adds fp32 master (~+16 GB → OOM risk at bs4), matmuls stay bf16, so quality lever (SCHEMA rule 8) not MFU. Effort M.

Profile-driven (v018 frontier, profile-analyzer): NOT matmul-bound (MXU 48.3%, 48.6% non-matmul). Top non-matmul = grad reduce-scatter 12.9% (→ async fusion, #1), splash 12.3%, loop-fusion/norms 11.0% (HBM-bound). QK-norm+RoPE already XLA-fused (custom kernel → xla-already-fuses; deprioritized).

Validated this lane: [XLA flag stack](../hypotheses/qwen3-jax-xla-flag-stack.md) (**supported** seq2048 +3.4 pp / v018; **neutral seq8192** / v019). Retired kernel-swap levers: tokamax-splash (v020 −8.5%), splash-block-tuning (v021/v022 neutral). Catalogued: fused QK-norm+RoPE kernel (deprioritized), seq16384, XLA flag-ablation.

Also catalogued (not yet filed as pages): scan-over-layers (compile-HBM), fused QK-norm+RoPE Pallas kernel (Qwen3-specific), seq16384 (splash scale).

## Variant-specific open hypotheses

(none yet)

## Retired hypotheses

- [Tensor-parallel activation sharding (tp>1)](../hypotheses/qwen3-jax-tp-activation-sharding.md) — **refuted** for 8B/v6e-8 ([v043b](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-05-v043b-tp2-bs6.md): tp=2 at equal global batch 24 = 25.6% / 4,455, −26% vs v035). At equal global batch tp trades batch-sharding for model-sharding → per-chip activation unchanged → can't reduce the SiLU/norm recompute, while adding per-layer all-reduce collectives pure FSDP avoids. tp=4 retired without running (strictly worse).
- [RMSNorm bf16 weight-multiply](../hypotheses/qwen3-jax-rmsnorm-bf16.md) — **parity** for 8B/v6e-8 ([v042](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-05-v042-rmsnorm-bf16.md): +0.16%, sub-noise). Matching MaxText's bf16 norm weight-mult (fp32-identical) trims only a small slice of loop-fusion (SiLU-dominated, already bf16). Keep as no-cost reference-alignment, not a frontier win.
- [Per-chip batch scaling](../hypotheses/qwen3-jax-batch-scaling.md) — **refuted** for 8B/v6e-8 (v001/v002 compile-OOM on the dense path; subsumed by remat, which is now the baseline).
- [tokamax streamed cross-entropy](../hypotheses/qwen3-jax-tokamax-ce.md) — **refuted** for 8B/v6e-8 (v013 correct, but v014 seq2048 = 30.5% and v016 seq8192 = 29.5%, both below the splash frontier; memory enabler, not a throughput lever).
- [SparseCore collective-offload](../hypotheses/qwen3-jax-sparsecore-offload.md) — **refuted** for 8B/v6e-8 at every shape (v003 bs1 −4 pp, v017 bs4 −0.7 pp; collective share already low at 6.3%).

## Knobs translation matrix

| Technique | 8B/v6e-8 | Notes |
|-----------|----------|-------|
| remat (jax.checkpoint, nothing_saveable) | ✅ universal | baseline-enabling at both seq; required to fit any batch at seq8192 |
| splash attention (Pallas GQA) | ✅ universal | frontier component at both seq2048 (v008/v018) and seq8192 (v009) |
| MaxText 7-flag XLA scheduler bundle | ✅ seq2048 (+3.4 pp, v018) · neutral seq8192 (v019) | scoped-vmem + host-transfer-overlap scheduler flags |
| scan-over-layers + 4 async-collective-fusion flags | ✅ **seq8192 only** (+6.2%, v028) · ❌ seq2048 (−3.4%, v032) | NOT universal — needs a large collective tail to amortise the scan-body overhead; productive only on the long-seq graph |
| named host-offload remat (save_and_offload_only_these_names → pinned_host) | ❌ refuted (v030 −18% w/ tokamax-CE; **v036 −18.6% w/ maxtext-CE**) | our pinned_host round-trip is NOT pipelined (lands on critical path, opposite MaxText's <0.1%); and it tags proj/mlpwi, not the norms that recompute. Pipelined offload = kernel-authoring, out of scope |
| tokamax streamed CE (mosaic_tpu, f32 weight) | ⚠️ memory-enabler, not throughput | correct (v013); below frontier at both seq (v014/v016); CE backward kernel requires f32 lm_head weight (v029); its f32[H,V] weight-gather (4.64G) blocked bs3 |
| **MaxText T5X custom_vjp CE** (`--use_maxtext_ce`) | 🏆 **seq8192 wall-cracker** (v034 bs2 5,992 > bs1; v035 bs3 6,030) · ❌ **seq2048 regress** (v037 −6.9%) | seq8192-SPECIFIC (like scan): makes batch amortize @ seq8192 (lower CE-backward transient); at seq2048 the one-hot[BL,V] materialization isn't repaid. Use `--use_maxtext_ce` @ seq8192, plain `_ce` @ seq2048. CPU bit-identical math to `_ce` |
| SparseCore collective-offload | ❌ refuted all shapes | collective share already low (v003/v017) |
| per-chip batch scaling @ seq8192 | 🏆 **amortizes WITH maxtext-CE** (v034 bs2 > bs1); ❌ anti-amortizes with tokamax/autodiff CE (v030/v031) | the prior "hard wall" was CE-backward-transient-bound, NOT structural — cracked by v034 |
| tensor parallelism (tp>1, Megatron) | ❌ refuted (v043b tp2-bs6 −26%) | at equal global batch tp can't shrink per-chip recompute (trades batch-shard for model-shard); only adds o/down all-reduces. NOT a path to MaxText's sharded-activation regime on this stack |
| save norm rsqrt stats on-device (`save_only_these_names`) | ✅ marginal seq8192 (+0.63%, v045) | skips the f32 norm-reduction recompute in the backward; works where host-offload can't (tiny + on-device). Only frontier-mover of the 2026-06-05 sweep. Single-run; confirm pending |
| RMSNorm bf16 weight-multiply | ⚪ parity (+0.16% sub-noise, v042) | matches MaxText rms_norm (fp32-identical); norm fp32-mult is a small slice of SiLU-dominated loop-fusion |

## Iteration ladder

1. Port Qwen3 8B to native JAX (Flax NNX), matching the torchax baseline number.
2. Apply the MaxText-style XLA flag stack + collective offload.
3. Cross-lane comparison vs the torchax frontier.

## See also

- [Qwen3 8B — maxtext](qwen3-cc-maxtext.md) — reference-ceiling lane (MaxText `qwen3-8b.yml`; reference run dispatched 2026-06-02).
- [Qwen3 8B — torchax](qwen3-cc-torchax.md) — primary lane (live frontier).
- Program: [`../experiments/qwen3_cc_autoresearch_optimization/program.md`](../experiments/qwen3_cc_autoresearch_optimization/program.md)
- Lane README: [`../experiments/qwen3_cc_autoresearch_optimization/jax/README.md`](../experiments/qwen3_cc_autoresearch_optimization/jax/README.md)

## Sources

- `wiki/experiments/qwen3_cc_autoresearch_optimization/jax/` (scaffold).
