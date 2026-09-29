# Llama 3 8B — native-JAX stack experiments

Experiment writeups for the **native-JAX (Flax NNX)** path through the Llama 3 8B autoresearch program. Sibling of [`../../torchax/experiments/`](../../torchax/experiments/README.md) (the validated PyTorch-on-JAX path) and [`../../maxtext/experiments/`](../../maxtext/experiments/README.md) (the third-party reference baseline).

The native-JAX stack ended up beating both: at `bs=4 seq=8192 fsdp=8` on v6e-8 it runs at **~7,700 tok/s/chip / 43.3 % MFU** (mean of 3 reruns; peak 7,768/43.6 %) — **+8.9 % per chip vs MaxText reference 7,069/chip 44.6 %** and **+17.4 % vs the torchax frontier 6,559/chip 36.8 %** at the same hardware/model.

## Current best (trunk)

| Metric | Value | Source |
|--------|------:|--------|
| Throughput | **~7,700 tok/s/chip** (mean of 3 reruns; peak 7,768) | [exp 27/28b](2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md) |
| Reported MFU (v6e bf16 peak 918 TFLOPs/s/chip) | **~43.3 %** (peak 43.6 %) | exp 27/28b |
| Step time | 4,217 ms (bs=4 seq=8192) | exp 27/28b |
| vs MaxText reference (7,069/chip 44.6 %) | **+8.9 % per-chip** (peak +9.9 %) | — |
| Loss validation | bit-equivalent to minimal-flags baseline; max \|Δ\| = 0.0003 / 100 steps | [exp 65/66/67](2026-04-27-jax-exp65-67-loss-validation-100steps.md) |
| Stack | scan + AMP master + tokamax CE chunked_xla + tokamax-splash w/ base2/fuse_recip/mlc=30 + full MaxText XLA flag stack + SC offload of AR/RS/AG + nothing_saveable + bs=4 seq=8192 | — |

## Featured writeups

These three are the long-form experiment pages with full profile breakdowns and ablation tables:

- 🏆 **[2026-04-26 — exp 27/28b: SparseCore RS+AG offload frontier](2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md)** — the headline frontier writeup. Adds SparseCore offload of reduce-scatter and all-gather (we already had all-reduce) to relay all 3 FSDP collectives off the TC. Includes the consolidated **Wave 4–5 ablation table for exp 29-60** and the noise-band correction (mean ~7,700 vs single-run peak 7,768).
- 📜 **[2026-04-26 — exp 13/15/18 chronicle: MaxText XLA flag stack + bkv=2048 match](2026-04-26-jax-exp13-maxtext-xla-stack-bs5-accepted.md)** — earlier-session frontier discovery. The `HOST_OFFLOAD_FLAGS` scheduler bundle from `maxtext/benchmarks/xla_flags_library.py` was the breakthrough lever; +13.6 % from JAX baseline, then +0.7 % from matching MaxText's `bkv=2048`.
- 🧪 **[2026-04-27 — exp 65/66/67: 100-step loss-curve validation](2026-04-27-jax-exp65-67-loss-validation-100steps.md)** — three independent 100-step runs on synthetic data prove the optimization stack is bit-equivalent (max \|Δ\| = 0.0003) to the minimal-flags baseline at literally identical RNG seed. **+19.9 % per-chip throughput at zero loss-curve drift.**

## All experiments — chronological

### 2026-04-26 — JAX port + frontier discovery (exp 1-17)

Direct JAX port of the torchax frontier; characterising the noise band; discovering MaxText's XLA flag stack as the breakthrough lever.

| # | Experiment | Verdict | Headline |
|---|------------|---------|----------|
| 1e | [direct port baseline](2026-04-26-jax-exp1e-baseline-port-direct-accepted.md) | ✅ supported | 6,529/chip 36.6 % MFU — parity with torchax frontier |
| 3 | [profile capture](2026-04-26-jax-exp3-profile-bs3-baseline-accepted.md) | ✅ supported | identifies async-AR + matmul as primary cost |
| 4 | [vmem default localtmp](2026-04-26-jax-exp4-vmem-default-localtmp-accepted.md) | ✅ supported | operational fix: kubectl-cp profile pattern |
| 5 | [bs=2 density](2026-04-26-jax-exp5-bs2-density-accepted.md) | ✅ supported | 6,310/chip 35.4 % MFU |
| 6 | [bs=4 density (pre-XLA)](2026-04-26-jax-exp6-bs4-density-accepted.md) | ✅ supported | 6,420/chip 36.0 % MFU |
| 7 | [no-scan variant](2026-04-26-jax-exp7-bs3-noscan-rejected.md) | ❌ refuted | OOM — scan-over-layers mandatory at this shape |
| 8 | [dots_with_no_batch_dims_saveable](2026-04-26-jax-exp8-bs3-remat-dotsnobatch-rejected.md) | ❌ refuted | OOM — too aggressive |
| 9 | [bs=4 pre-MaxText-stack](2026-04-26-jax-exp9-bs4-frontier-pre-maxtext-xla-potential.md) | ⚠ inconclusive | compile error; superseded by exp 12 |
| 10 | [noop default stack noise calibration](2026-04-26-jax-exp10-noop-default-stack-accepted.md) | ✅ supported | bounds noise band before exp 12 |
| 11 | [everything_saveable](2026-04-26-jax-exp11-scan-remat-everything-saveable-rejected.md) | ❌ refuted | OOM at compile (saves all activations) |
| 12 | [🟢 MaxText XLA stack bs=3](2026-04-26-jax-exp12-maxtext-xla-stack-bs3-accepted.md) | ✅ supported | **+11.1 %** — breakthrough lever |
| 12b | [MaxText XLA stack bs=4](2026-04-26-jax-exp12b-maxtext-xla-stack-bs4-accepted.md) | ✅ supported | 7,402/chip 41.5 % MFU |
| 13 | [MaxText XLA stack bs=5 (chronicle)](2026-04-26-jax-exp13-maxtext-xla-stack-bs5-accepted.md) | ✅ supported | 7,415/chip 41.6 % MFU; long-form chronicle through exp 18 |
| 14 | [bs=6 density](2026-04-26-jax-exp14-bs6-density-rejected.md) | ❌ refuted | regresses at bs=6 |
| 15 | [exp 13 profile capture](2026-04-26-jax-exp15-exp13-profile-bs5-accepted.md) | ✅ supported | MXU 64.1 % / conv-fusion 57.2 % |
| 16 | [bs=7 attempt](2026-04-26-jax-exp16-bs7-attempt-rejected.md) | ❌ refuted | OOM by 2.26 GiB |
| 17 | [bs=8 attempt](2026-04-26-jax-exp17-bs8-attempt-rejected.md) | ❌ refuted | OOM by 172 MiB |

### 2026-04-27 — bkv match + perf-knob characterisation (exp 18-26)

| # | Experiment | Verdict | Headline |
|---|------------|---------|----------|
| 18 | [bkv=2048 match MaxText (prior frontier)](2026-04-27-jax-exp18-bkv2048-match-maxtext-bs5-accepted.md) | ✅ supported | **7,471/chip 41.9 % MFU** — prior frontier |
| 18b | [bkv=2048 at bs=3 (MaxText shape)](2026-04-27-jax-exp18b-bkv2048-bs3-accepted.md) | ✅ supported | +4.3 % vs MaxText 7,069 at their exact shape |
| 19 | [bkv=2048 at bs=4](2026-04-27-jax-exp19-bkv2048-bs4-accepted.md) | ✅ supported | density check |
| 19b | [bkv=2048 at bs=6](2026-04-27-jax-exp19b-bkv2048-bs6-rejected.md) | ❌ refuted | OOM (bkv=2048 raises memory floor) |
| 20 | [jax-experimental splash (perf-knob ablation)](2026-04-27-jax-exp20-jax-experimental-splash-accepted.md) | ✅ supported | 7,142/chip ≈ MaxText's 7,138 — quantifies tokamax-splash perf knobs as +4.4 % delta |
| 23b | [bq=1024 bkv=1024 symmetric](2026-04-27-jax-exp23b-bq1k-bkv1k-bs5-rejected.md) | ❌ refuted | -1.0 % (asymmetric wins) |
| 26 | [exp 18 frontier profile](2026-04-27-jax-exp26-frontier-profile-bs5-maxtext-accepted.md) | ✅ supported | finds async-RS at 5.0 % on TC — motivates exp 27 |

### 2026-04-27 — 🏆 Frontier advance (exp 27/28b)

| # | Experiment | Verdict | Headline |
|---|------------|---------|----------|
| 27 / 28 / 28b | [🏆 SparseCore RS+AG offload frontier writeup](2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md) | ✅ supported | **+4.0 % per-chip** over exp 18 → **7,768 / 43.6 % MFU** at bs=4 (peak run; mean 7,700/43.3 %). Add SparseCore offload of `reduce_scatter` + `all_gather` (we already had `all_reduce`) — relays all 3 FSDP collectives off the TC. |

### 2026-04-27 — Post-frontier ablation (exp 29-60, all refuted or within noise)

22 experiments in five waves; full table in [the frontier writeup](2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md#wave-45-ablation-summary-exp-3960). Each individually filed below.

| # | Experiment | Verdict | Headline |
|---|------------|---------|----------|
| 29 | [VMEM=131072](2026-04-27-jax-exp29-vmem-131072-bs4-rejected.md) | ❌ -2.9 % | more VMEM hurts |
| 30 | [bkv=2048 + full SC](2026-04-27-jax-exp30-bkv-2048-bs4-fullsc-rejected.md) | ❌ -0.2 % within noise | bkv lift doesn't compound |
| 31 | [bs=3 + full SC (MaxText shape)](2026-04-27-jax-exp31-bs3-fullsc-maxtext-shape-accepted.md) | ✅ supported | **+6.9 % vs MaxText** at their exact shape |
| 32 | [SPLASH_BQ=4096](2026-04-27-jax-exp32-splash-bq-4096-bs4-rejected.md) | ❌ -21.2 % | VMEM spill |
| 35 | [save_qkv_proj at bs=4](2026-04-27-jax-exp35-save-qkv-proj-bs4-rejected.md) | ❌ OOM | named-remat infrastructure shipped |
| 36 | [qkv_proj_offloaded bs=4](2026-04-27-jax-exp36-qkv-proj-offloaded-bs4-rejected.md) | ❌ -1.6 % | host PCIe > recompute savings |
| 37 | [qkv_proj_offloaded bs=6](2026-04-27-jax-exp37-qkv-proj-offloaded-bs6-rejected.md) | ❌ runtime OOM | host offload doesn't shrink workspace |
| 38 | [qkv_proj_offloaded bs=5](2026-04-27-jax-exp38-qkv-proj-offloaded-bs5-rejected.md) | ❌ -1.7 % | density-independent |
| 39 | [save_out_proj bs=4](2026-04-27-jax-exp39-save-out-proj-bs4-rejected.md) | ❌ OOM by 507 MiB | tight at bs=4 |
| 40 | [save_out_proj bs=3](2026-04-27-jax-exp40-save-out-proj-bs3-potential.md) | ⚠ +1.2 % at bs=3 | doesn't beat bs=4 frontier |
| 41 | [scan unroll=2](2026-04-27-jax-exp41-scan-unroll-2-bs4-rejected.md) | ❌ -2.8 % | bigger fusion window hurts |
| 42 | [no-bundle-aware-cost-model](2026-04-27-jax-exp42-no-bundle-aware-cost-model-rejected.md) | ❌ -2.6 % | doesn't apply to our shape |
| 43 | [enhanced-launch-barrier](2026-04-27-jax-exp43-enhanced-launch-barrier-rejected.md) | ❌ -1.1 % | no benefit |
| 44 | [async-collective-permute](2026-04-27-jax-exp44-async-collective-permute-potential.md) | ⚠ -1.0 % within noise | neutral |
| 45 | [no-megacore-fusion-allow-ags](2026-04-27-jax-exp45-no-megacore-fusion-allow-ags-potential.md) | ⚠ -0.8 % within noise | neutral |
| 46 | [combo acpermute+nomegafuseag](2026-04-27-jax-exp46-combo-acpermute-nomegafuseag-rejected.md) | ❌ -2.0 % | doesn't compound |
| 47 | [VMEM=65536](2026-04-27-jax-exp47-vmem-65536-bs4-rejected.md) | ❌ -4.2 % | hard refute |
| 48 | [VMEM=81920 (MOE level)](2026-04-27-jax-exp48-vmem-81920-mxt-moe-level-rejected.md) | ❌ -1.6 % | dense ≠ MoE optimum |
| 49 | [splash bkv=512](2026-04-27-jax-exp49-splash-bkv-512-bs4-rejected.md) | ❌ -2.2 % | smaller blocks worse |
| 50 | [validation re-run 1](2026-04-27-jax-exp50-validate-exp28b-rerun-accepted.md) | ✅ noise calibration | bounds run-to-run noise to ±0.7 % |
| 51 | [enable collective matmul](2026-04-27-jax-exp51-enable-collective-matmul-rejected.md) | ❌ -14.7 % | HARD REFUTE |
| 52 | [splash bkv_dkv=1024](2026-04-27-jax-exp52-splash-bkv-dkv-1024-rejected.md) | ❌ -3.3 % | bwd needs bigger blocks |
| 53 | [tokamax CE = mosaic_tpu](2026-04-27-jax-exp53-tokamax-ce-mosaic-tpu-rejected.md) | ❌ -4.4 % | chunked_xla confirmed best |
| 54 | [precast bf16 weights bs=4](2026-04-27-jax-exp54-precast-bf16-weights-bs4-rejected.md) | ❌ -1.1 % | XLA already fuses cast |
| 55 | [precast bf16 weights bs=5](2026-04-27-jax-exp55-precast-bf16-weights-bs5-rejected.md) | ❌ -0.7 % | density-independent |
| 56 | [validation re-run 2](2026-04-27-jax-exp56-validate-exp28b-rerun-2-accepted.md) | ✅ noise calibration | confirms ±0.7 % noise |
| 57 | [no overlap-compute-collective](2026-04-27-jax-exp57-no-overlap-compute-collective-rejected.md) | ❌ -1.1 % | confirm default `=true` |
| 58 | [no aggressive-opt-barrier](2026-04-27-jax-exp58-no-aggressive-opt-barrier-potential.md) | ⚠ -0.1 % within noise | neutral |
| 59 | [latency-hiding rerun=0](2026-04-27-jax-exp59-lat-hiding-rerun-0-potential.md) | ⚠ +0.1 % within noise | could disable to save compile time |
| 60 | [loop-invariant chain disabled](2026-04-27-jax-exp60-loop-inv-chain-disabled-potential.md) | ⚠ -0.7 % within noise | neutral |

### 2026-04-27 — 🧪 Loss-curve validation (exp 61-69)

| # | Experiment | Verdict | Headline |
|---|------------|---------|----------|
| 61 | [val 100 steps real-data (exhausted at step 9)](2026-04-27-jax-exp61-val100opt-realdata-exhausted-potential.md) | ⚠ inconclusive | wikitext-2 packs into ~9 batches at bs=4 seq=8192 |
| 62 | [val 100 steps base (real-data exhausted)](2026-04-27-jax-exp62-val100base-realdata-exhausted-potential.md) | ⚠ inconclusive | loss matches exp 61 step-for-step |
| 64 | [val 100 steps jax-exp splash (real-data exhausted)](2026-04-27-jax-exp64-val100ref-realdata-exhausted-potential.md) | ⚠ inconclusive | loss matches exp 61/62 |
| 65/66/67 | [🧪 100-step loss validation (synthetic, conclusive)](2026-04-27-jax-exp65-67-loss-validation-100steps.md) | ✅ supported | **max \|Δ\|=0.0003 across 3 stack configs** — bf16 noise floor |
| 68 | [lr=3e-5 syn bs=4 (MaxText comparison)](2026-04-27-jax-exp68-lr3e5-syn-bs4-accepted.md) | ✅ supported | identifies MaxText-curve gap as data-pipeline difference, not numerics |
| 69 | [lr=3e-5 syn bs=3 (MaxText shape)](2026-04-27-jax-exp69-lr3e5-bs3-syn-accepted.md) | ✅ supported | same lesson at MaxText shape |

## See also

- [`../`](../) — JAX trainer source.
- [`../../torchax/experiments/`](../../torchax/experiments/) — torchax sibling experiments.
- [`../../maxtext/experiments/`](../../maxtext/experiments/) — MaxText reference baseline.
- [`../../program.md`](../../program.md) — shared experiment protocol.
- [`../../../../hypotheses/`](../../../../hypotheses/) — open and refuted hypothesis pages (1 open: int8/AQT; 2 refuted via HLO inspection: Pallas RMSNorm+matmul, Pallas SwiGLU+down_proj).
