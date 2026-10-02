---
title: "Qwen3 8B — jax"
type: model
architecture: qwen3-cc5
lane: jax
sizes: ["8B"]
hardwares: ["v6e-8"]
tags: [qwen3-cc5, jax]
created: 2026-06-02
updated: 2026-06-12
---

# Qwen3 8B — jax

Qwen3 8B (`Qwen/Qwen3-8B`) as a native-JAX (Flax NNX) port on **TPU v6e-8** —
the **live frontier lane** (trainer + model in
`wiki/experiments/qwen3_cc5_autoresearch_optimization/jax/`,
equivalence-verified vs HF). Current best **39.9% MFU / 6,959 tok/s/chip — MATCHES AND EXCEEDS the MaxText reference (6,953)**
(v050/v051: bs2 + scan + save_qkv_ctx + splash + chunked CE + 21-flag stack;
context checkpointing offloads splash out+lse to host, eliminating the backward
splash-forward re-run; +7.3% vs v043), up from the 20.5% 2026-06-02 baseline.
Serves as the reference ceiling over the [torchax](qwen3-cc5-torchax.md) lane.

## Target metrics

- Primary: MFU at seq_len=8192 (v6e bf16 peak ≈ 918 TFLOPS/chip).
- Secondary: tokens/sec (and tok/s/chip), step time, peak HBM.

## How to run

Current target-shape frontier recipe (v044/v045; trainer in
`wiki/experiments/qwen3_cc5_autoresearch_optimization/jax/` trunk,
image `qwen3-8b-jax:v6e8-qwen3-8b-jax-20260612-v044-ctx`, dispatched via
gke-cluster-runner on `<your-cluster>`):

```bash
export JAX_COMPILATION_CACHE_DIR=gs://<your-bucket>/autoresearch/qwen3_cc5/jax_lane_cache
# 21-flag stack (v035: ACF flags worth ~0; use this trimmed form):
export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 --xla_tpu_assign_all_reduce_scatter_layout=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_enable_all_experimental_scheduler_features=true --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_host_transfer_overlap_limit=24 --xla_tpu_aggressive_opt_barrier_removal=ENABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_tpu_enable_ag_backward_pipelining=true --xla_should_allow_loop_variant_parameter_in_chain=ENABLED --xla_should_add_loop_invariant_op_in_chain=ENABLED --xla_max_concurrent_host_send_recv=100 --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_latency_hiding_scheduler_rerun=2 --xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000"
# NEW (v044): splash context checkpointing — removes bwd splash-fwd re-run
export SPLASH_RESIDUAL_CKPT_NAME=context
export JAX_SCAN_UNROLL=1    # guards against XLA S(5) unroll×offload bug
# TARGET FRONTIER (v050/v051 = 6,959/chip / 39.9% MFU, 50-step certified — EXCEEDS MaxText 6,953):
export JAX_WEIGHT_LAYOUT=io   # (in,out) Linear kernels, transpose-free x@W (lm_head excluded)
# bs2 + scan + save_qkv_ctx + splash + chunked CE f32-x + 21-flag + context offload.
python -u train.py --model_id=Qwen/Qwen3-8B --use_real_data=False \
  --batch_size=2 --seqlen=8192 --tp_parallelism=1 \
  --train_steps=50 --weights_dtype=bf16 --use_splash=True \
  --use_tokamax_ce=True --tokamax_ce_impl=chunked_xla --remat_policy=save_qkv_ctx_hbm \
  --use_scan=True \
  --profile_dir=gs://…/qwen3_cc5/<run-name> --profile_start_step=12 --profile_steps=3
# alternate (save_qkv NO offload, v043 = 6,265): --remat_policy=save_qkv (unset SPLASH_RESIDUAL_CKPT_NAME)
# alternate (bs4 + offload_attn, v027 = 6,040): --batch_size=4 --remat_policy=offload_attn (full 25-flag stack)
# seq 2048: LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=81920" only; CE/remat/scan OFF (v007 recipe)
```

## Variant matrix

| Size | Hardware | Status | Baseline (step / TPS / MFU) | Current best (step / TPS / MFU) | Open hyps | Frontier exp |
|------|----------|--------|-----------------------------|----------------------------------|-----------|--------------|
| 8B | v6e-8 | live | 512 ms / 31,955 TPS / 20.5% MFU @ seq2048 bs8 | **2,354.4 ms / 55,671 TPS (6,959/chip) / 39.9% MFU @ TARGET seq8192 bs2 — ≥ MaxText 6,953 (v050/v051, 50-step certified: scan + save_qkv_ctx_hbm + context checkpointing + (in,out) weight layout + splash + chunked CE f32-x + 21-flag stack); peak HBM 22.03 GiB (70.5%)**; prior: save_qkv 6,265 ([v043](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v043-bs2-svqkv-val.md)), save_attn 6,221 ([v038](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v038-bs2-saveattn-val.md)), bs4+offload 6,040 ([v027](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v027-bs4-mtfl.md)); **seq-2048 best (charles-v6e phase): 1,301 ms / 62,971 TPS / 7,871 tok/s/chip / 40.5% MFU @ bs5 — 50-step CERTIFIED ([v058](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v058-2kbs5.md)/[v060](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v060-2kbs5val.md); recipe = bs5 + scan + save_qkv_ctx_hbm + context + plain CE + 21-flag stack; plain-CE batch ceiling measured at bs5 [v059](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v059-2kbs6.md); prior rung bs4 7,479 [v057](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v057-2kbs4.md))**; prior 2k: 31.4% ([v007](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v007-bs2-vmem.md)) | 3 | [2026-06-12 v051-wioval](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v051-wioval.md) |

*Baseline captured at seq 2048 (global batch 8). **Cross-lane: jax 20.5% MFU /
3,994 tok/s/chip beats torchax 19.2% / 3,724 (+7.3% tok/s/chip, +1.3 pp)** at the
identical shape — the native-JAX path has lower collective/dispatch overhead
(profile: 21.7% collective vs torchax 31.3%). See the
[jax baseline](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-02-qwen3-jax-v6e8-baseline.md)
and [torchax baseline](../experiments/qwen3_cc5_autoresearch_optimization/torchax/experiments/2026-06-02-qwen3-torchax-v6e8-baseline.md).*

## Cross-variant open hypotheses

Post-v044/v045 state (2026-06-12, after context checkpointing; reference =
[mt-v001](../experiments/qwen3_cc5_autoresearch_optimization/maxtext/experiments/2026-06-12-mt-v001-base.md)
6,953 tok/s/chip; **gap now −3.3%** at the v044/v045 frontier):

1. [fused gate-up MLP projection](../hypotheses/qwen3-jax-fused-gate-up.md) — targets the 52.2% convolution fusion bucket (MXU gap 55.7% vs MaxText's implied 65%+); **HLO pre-filter required before running**; M effort. This is now the primary MaxText-gap lever.
2. [Context DMA scheduling](../hypotheses/qwen3-jax-ctx-dma-pipeline.md) — improve XLA scheduler placement of context DMA async-start; targets the 49 ms async-done bucket (−25 ms residual); S effort.
3. [Context residuals in HBM](../hypotheses/qwen3-jax-ctx-hbm-only.md) — skip the host offload; HBM has ~9.2 GiB headroom; may eliminate the 25 ms DMA cost entirely; OOM risk at compile time; S effort.
4. [int8 weight-only AQT](../hypotheses/qwen3-jax-int8-aqt.md) — targets conv-fusion bandwidth + weight memory; loss-parity validation mandatory; L effort.
5. [Vocab-sharded CE](../hypotheses/qwen3-jax-vocab-sharded-ce.md) — deprioritized (CE overhead small vs frontier); L effort.

Resolved since retrospective #3: context checkpointing **supported** (v044/v045, +7.3%, new frontier, merged); MaxText gap −9.9% → −3.3%.

**Gap decomposition** (remaining ~85 ms after v044): convolution fusion (52.2%, 1,272 ms) dominates — MXU layout difference vs MaxText is the primary candidate (inventory #2); async-done residual DMA 25 ms (2.0%); smaller buckets (CE, pad) sum to <40 ms.

Resolved prior phases: TP **refuted** (v033 −23.6%); manual-weight-prefetch **refuted** (v034–v037); ACF flags ~0 (v035, 21-flag standard); bs3/bs5 batch **refuted** on every substrate; save_qkv ablation **supported** (v041/v043, merged `23aa5e7`); bs2+save_attn **supported** (v038, superseded by v043); context-checkpoint **supported** (v044/v045, new frontier).

**Gap decomposition** ([analysis](../analyses/2026-06-12-maxtext-vs-jax-bucket-diff.md)): AG exposure 74% / MXU-layout 29% (partly stall feedback) / chunked-CE 12% / splash is OUR advantage (−89 ms). Batch CLOSED at bs4 ([v029](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v029-bs5-probe.md)); AG flags REFUTED ([v030](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v030-ag-overlap.md)/[v031](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v031-ag-pipeline-flags.md)/[v032](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v032-loop-unroll.md): absent / fire-and-hurt −1.8% / silent no-op).

Landed this phase: [MaxText flag completion](../hypotheses/qwen3-jax-maxtext-flag-completion.md) — **supported (v026), FRONTIER**: +10.3% vs v025; windowed-einsum disable worth −341 ms alone; flags now lane-standard.
Landed this phase: [Batch scaling bs4](../hypotheses/qwen3-jax-batch-scaling.md) — **supported (v027)**: +1.8% tok/s/chip; no OOM; actual HBM 23.72 GiB (well under 30.1 GiB projection). New frontier 6,040/chip.
Retired this phase: [Splash block config](../hypotheses/qwen3-jax-splash-fused-bwd.md) — **refuted (v028)**: sym-1024 −2.5%, all-2048 +0.1%; both within ±1% of baseline. Third flat splash config result (v016, v022, v028). bq=2048 confirmed optimal over bq=1024 by 142 ms/step; splash direction fully closed. Splash cost (26.7% / 1,448 ms) is fundamental arithmetic, not tile-scheduling artifact.
Retired this phase: [Host-offload remat](../hypotheses/qwen3-jax-host-offload-remat.md) — **refuted as standalone** (v024/v025) but **part of the v026 frontier config** (with the completed flags its DMA is fully hidden, 0.7 ms/step); [Scheduler shared-mem limit](../hypotheses/qwen3-jax-sched-shared-mem-limit.md) — retired (mooted: scan+offload+flags fits and leads).

**Landed this cycle (v044/v045)**: [Splash context checkpoint](../hypotheses/qwen3-jax-splash-context-checkpoint.md) — **supported, NEW FRONTIER (50-step certified)**: bs2 + save_qkv_ctx = **2,437 ms / 6,715 tok/s/chip / ≈38.6% MFU** (+7.3% vs v043; HBM 22.03 GiB / 70.5%; splash custom-call −206 ms, async-done +25 ms residual DMA). MaxText gap −9.9% → −3.3%.

### Landed / retired this cycle

- ✅ [Splash attention](../hypotheses/qwen3-jax-splash-attention.md) — **supported** ([v002](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v002-splash.md)): −11.5% step, +1.7 pp MFU, −6.0 GiB HBM. Merged to trunk.
- ✅ [Per-chip batch scaling](../hypotheses/qwen3-jax-batch-scaling.md) — **supported at bs2** ([v007](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v007-bs2-vmem.md)): +40.5% tok/s/chip; bs3+ blocked on HBM; **supported at bs4** ([v027](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v027-bs4-mtfl.md)): +1.8% tok/s/chip.
- ❌ [tokamax CE (fp32-cast recipe)](../hypotheses/qwen3-jax-tokamax-ce.md) — **refuted** ([v003](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v003-tkmce.md)): f32 weight cast offsets the streaming win; +6.3% step.
- ⛔ [tokamax CE mosaic_tpu bf16 inputs](../hypotheses/qwen3-jax-tokamax-ce-bf16-inputs.md) — **invalid** ([v006](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v006-tkmce-bf16.md)): kernel hardcodes f32 scratch Ref.
- ❌ [SparseCore collective offload](../hypotheses/qwen3-jax-sparsecore-offload.md) — **refuted** ([v009](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v009-sc-offload.md)): −12.7% tok/s at this op-point.
- ❌ [Splash block config sweep](../hypotheses/qwen3-jax-splash-fused-bwd.md) — **refuted** ([v028](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v028-splash-blocks-bs4.md)): sym-1024 −2.5%, all-2048 +0.1%. Third flat result (v016, v022, v028). Direction closed.
- ✅ [Splash context checkpointing](../hypotheses/qwen3-jax-splash-context-checkpoint.md) — **supported, NEW FRONTIER** ([v044](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v044-context-checkpoint.md)/[v045](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v045-ctxval.md)): +7.3% step, +6.7% tok/s/chip, +2.6 pp MFU; HBM −5 GiB; MaxText gap −9.9%→−3.3%. Merging to trunk.

## Variant-specific open hypotheses

**seq-2048 phase (opened 2026-06-12, user-directed; cluster = charles-v6e,
project `<your-project-2>`, <your-zone>, 3× v6e-8 on-demand — 1 slice usable,
2 held by foreign zombie Kueue reservations).** seq-2048 best on record:
31.4% MFU ([v007](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v007-bs2-vmem.md)) —
predates every seq-8192 frontier lever. Dispatch uses a dedicated kubeconfig
(`~/.kube/config-charles-v6e`; shared kubeconfig is clobbered by other users)
and literal `--project=<your-project-2> --zone=<your-zone>`.

**Phase extension CLOSED (2026-06-13)**: best-vs-best at 2k =
**statistical tie** ([analysis](../analyses/2026-06-13-qwen3-2k-stack-vs-maxtext.md)):
MaxText 7,915 (bs12 optimum; bs20 −16.6%, bs24 OOM on their bf16-logits
wall) vs our certified 7,871 (−0.56%). [Scan unroll=2](../hypotheses/qwen3-jax-2k-scan-unroll2.md)
refuted (v063: ARS flat — XLA schedules the body atomically; S(5) bug
confirmed offload-specific). S-effort space exhausted; remaining levers all
L / numerics-gated / quota-blocked (see the analysis ledger).

**Phase result (2026-06-13): batch arc complete at 40.5% MFU certified** — 11
experiments (v052–v062): continuity ✅ (v052) → flags/io shape-gated at bs2
(v053–v056) → **batch rescale on the remat substrate = the phase win**
(v057/v058, certified v060: **bs5, 7,871 tok/s/chip, 40.5% MFU, +28.3%**) →
bs6 OOM measured (v059) → chunked-CE unlock invalid on numerics (v061/v062,
⚠️ escalated). Resolved: [continuity](../hypotheses/qwen3-jax-2k-cluster-continuity.md)
supported; [frontier port](../hypotheses/qwen3-jax-2k-frontier-port.md)
refuted-at-bs2/subsumed-at-bs5; [batch rescale](../hypotheses/qwen3-jax-2k-batch-rescale.md)
supported+certified; [chunked-CE unlock](../hypotheses/qwen3-jax-2k-chunked-ce-batch-unlock.md)
retired.

## Retired hypotheses

(none yet)

## Knobs translation matrix

| Technique | 8B/v6e-8 | Notes |
|-----------|----------|-------|
| Splash attention (bq=2048, bkv=1024, fused bwd) | ✅ supported (v002) | llama3-lane block config transferred as-is to v6e; required q pre-scale + output transpose fixes vs that lane's dispatch |
| tokamax CE, fp32 boundary cast | ❌ refuted (v003) | llama3 recipe does NOT transfer: f32 `[4096,151936]` replicated cast (2.49 GB) erases the streaming win at fsdp=8/bs1 |
| tokamax CE mosaic_tpu, bf16 inputs | ⛔ invalid (v006) | kernel-support gap: bwd hardcodes f32 scratch Ref; re-check on tokamax upgrade |
| `xla_tpu_scoped_vmem_limit_kib=81920` | ✅ required for bs≥2 + splash (v005→v007) | default 32 MB < 33.84 MB dkv tile; v6e physical VMEM 96 MB |
| Batch scaling bs1→2 (with splash + vmem flag) | ✅ supported (v007): +40.5% tok/s/chip, +8 pp MFU | bs3 blocked: 28.13 GiB used, ~3 GiB headroom vs ~6 GiB/bs slope; needs remat |
| SparseCore collective offload (AR/RS/AG) | ❌ refuted (v009): −12.7% tok/s | llama3's +3.4% does NOT transfer to seq2048/bs2; its win was at seq8192/bs4-5 + scheduler bundle |
| HOST_OFFLOAD scheduler bundle | ❌ refuted (v010): flat at seq2048/bs2 (+40% compile) | shape-gated like SC offload; re-test at seq-8192 |
| chunked_xla CE | numerics rule (v008b): x MUST be f32 (lse follows x.dtype); w stays bf16 | memory case is seq-8192-only (f32 dw accumulator 2.49 GB) |
| save_attn remat (checkpoint_name q/k/v/attn_out) | ✅ seq-8192 enabler (v015); ❌ +27% at seq2048 | per-shape knob: ON at 8192, OFF below; dots-saveable policy NaNs (v012) |
| Partial remat (JAX_REMAT_SKIP_EVERY=5) | ✅ supported (v017): +6.3% tok/s at seq8192 | recompute tax linear in layers; 0.43 GiB HBM per exempt layer |
| Splash fwd bkv=2048 at seq8192 | ❌ refuted (v016): flat | bkv=1024 final on this stack (llama3's +0.7% doesn't reproduce w/ QK-norm GQA) |
| Scan-over-layers (--use_scan) | ✅ supported (v020): compile 3.4×, −8 GiB live-set, step parity | substrate tool; SKIP_EVERY n/a under scan |
| bs2 @ seq8192 (on scan) | ❌ refuted (v021): 5,701/chip < 5,898 bar | fits (no OOM) but scan's uniform-remat cost > batch amortization |
| tokamax splash + base2/fuse_recip | ❌ refuted (v022): neutral | llama3 splash transfers 0-for-3 on this stack; numerics-equivalent, kept non-default |
| Fused QK-norm+RoPE Pallas | 🅿 parked (HLO pre-filter 2026-06-12) | not co-fused by XLA (boundary open) but ceiling ≤1-2% at L effort; index principle: don't build norm kernels |
| seq 8192 full stack (splash+CE+remat+vmem) | ✅ supported (v015): 31.8% MFU, 27.62 GiB | collectives collapse to ~4.5% at this shape |
| Host-offload remat (offload_attn → pinned_host) | ✅ part of the v026 frontier config (standalone refuted: v024 unrolled liveness OOM; v025 scan-only 5,381 < bar with 416 ms DMA stalls) | Requires BOTH scan (liveness) AND the completed flag stack (DMA hiding: 416 ms → 0.7 ms). Only known bs3+-fitting substrate at seq8192. Merged to trunk 174efd7 |
| MaxText 25-flag stack (windowed-einsum disable + collective fusion + RS layout + DP overlap) | ✅ supported (v026): +10.3% at bs3/seq8192, FRONTIER; numerics unchanged | Windowed-einsum disable = −341 ms (ring was ACTIVE in our compiles at bs3); RS layout −90 ms data-formatting; also shrank HBM peak 27.34→19.66 GiB. SC-offload flags excluded (compile-fatal). Lane-standard for seq8192; untested at seq2048 |
| Batch scaling bs3→bs4 (on scan+offload+25-flag stack) | ✅ supported (v027): +1.8% tok/s/chip; actual HBM 23.72 GiB (75.9% cap) | RS-fusion bucket near fixed-cost (1.01x scale); conv+splash buckets linear with tokens; async-done trending super-linear (1.36x). Diminishing returns band confirmed; bs5 probe is low-confidence |
| Splash block config sweep (sym-1024 vs all-2048 vs hybrid) | ❌ refuted (v028) — third flat result (v016, v022, v028) | bq=2048 > bq=1024 by 142 ms/step (+2.6%); bkv=1024 stays optimal; splash cost is arithmetic not scheduling. Hybrid v027 config (bq=2048, bkv=1024, dkv=2048) confirmed optimal. Direction closed. |
| **Splash context checkpointing (SPLASH_RESIDUAL_CKPT_NAME=context + save_qkv_ctx)** | **✅ supported (v044/v045): +7.3% step, +6.7% tok/s/chip, −5 GiB HBM** | HLO confirmed: splash bwd uses `splash_mha_dkv_no_residuals` (saved out+lse from S(5) host); no `splash_mha_fwd_no_residuals` re-run; async-done +25 ms (DMA not fully hidden — refinement target). Frontier 6,265→6,715/chip (50-step certified); MaxText gap −9.9%→−3.3%. Merging to trunk. |
| (in,out) weight layout (JAX_WEIGHT_LAYOUT=io) at seq2048 | ❌ refuted ([v056](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v056-2kio.md)): −0.18%, within noise | **Per-shape knob**: +2.8% at seq8192 (v050/v051) does NOT transfer to 2k — [2,2048,·] activations don't pay the transpose toll. ON at 8192, OFF below. |
| Batch rescale at seq2048 on scan+save_qkv_ctx_hbm+context (bs2→bs4→bs5) | ✅ **supported, CERTIFIED ([v057](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v057-2kbs4.md)/[v058](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v058-2kbs5.md)/[v060](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v060-2kbs5val.md)): bs5 = 7,871/chip / 40.5% MFU, +28.3% over the v052 baseline** | The fixed ~190 ms collective floor amortizes (27.3% → ~15%); MXU 32.9% → 51.9%; HBM slope ~3.5–4.1 GiB/bs; bs6 OOM measured ([v059](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v059-2kbs6.md): wall = f32 CE logit pair). The remat tax that froze the 2k ladder pre-context-checkpointing is gone. |
| chunked_xla CE at seq2048 (as batch unlock) | ⛔ **invalid/retired ([v061](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v061-2kce8.md)/[v062](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v062-2kce5par.md))** | Memory unlock REAL (bs8 fits, +1.5% gross) but same-data parity FAILED: **systematic +0.006 loss offset** (padded-vocab phantom mass, ≈0.99× CE-grad scaling — [observation](../observations/chunked-ce-padded-vocab-loss-offset.md)) + 4.1% CE cost at bs5. ⚠️ same kernel is in the certified 8k frontier — human review requested. |
| 21-flag stack at seq2048 (on scan+remat carrier) | ❌ net-refuted at bs2 ([v055](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v055-2kstack.md)): −2.1%, but flags THEMSELVES fired (ring eliminated, −181.7 ms) | Carrier-coupled: scan+save_qkv_ctx_hbm adds +219 ms (exposed grad-ARS +143, MLP/norm recompute +57) at bs2. Peak HBM 15.61 GiB (49.9%) — reopens the batch ladder; batch amortization of the fixed 186 ms collective floor is the active hypothesis (v057). Unrolled+flags needs ≥120-min compile budget (scheduler flags × 36-layer graph). |

## Iteration ladder

1. Port Qwen3 8B to native JAX (Flax NNX), matching the torchax baseline number.
2. Apply the MaxText-style XLA flag stack + collective offload.
3. Cross-lane comparison vs the torchax frontier.

## See also

- [Qwen3 8B — torchax](qwen3-cc5-torchax.md) — primary lane (live frontier).
- Program: [`../experiments/qwen3_cc5_autoresearch_optimization/program.md`](../experiments/qwen3_cc5_autoresearch_optimization/program.md)
- Lane README: [`../experiments/qwen3_cc5_autoresearch_optimization/jax/README.md`](../experiments/qwen3_cc5_autoresearch_optimization/jax/README.md)

## Sources

- `wiki/experiments/qwen3_cc5_autoresearch_optimization/jax/` (scaffold).
