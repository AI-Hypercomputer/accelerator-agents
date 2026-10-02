---
title: "Why MaxText hits 45% MFU on Qwen3-8B and our jax lane hits 30% — code + profile teardown"
type: analysis
tags: [qwen3-cc, jax, maxtext, mfu-gap, profile, analysis]
model: qwen3-cc
created: 2026-06-02
updated: 2026-06-02
---

# Why MaxText reaches 45% MFU and our jax lane 30% (Qwen3-8B, v6e-8)

> [!warning] Re-framing (2026-06-16): the "45% MaxText" figure is **non-causal** and inflates the gap
> The MaxText `45.3%` here came from `tpu-recipes-v0.1.4`, which **predates MaxText's causal-mask `/2` fix** (commit `6288c233`, 2026-04-11) and counts attention non-causally. On the **causal** basis every lane uses, MaxText Qwen3-8B @8k is **39.8%**, not 45.3% — and the later jax frontier (cx 43.2% / cc5 39.9%) **meets-or-beats** it (consistent with throughput: cx 7,543 / cc5 6,959 vs MaxText 6,942). So most of the "MFU gap" analysed below is a **FLOP-accounting artifact**, not a real utilization deficit; the genuine residual is MaxText's offload-enabled bs3 throughput. See the [qwen3-cc-maxtext caveat](../models/qwen3-cc-maxtext.md) and the [llama3-8b finding](../experiments/llama3_8B_autoresearch_optimization/README.md#maxtext-mfu-is-reported-on-a-stale-non-causal-basis).

Teardown of the **measured** MaxText Qwen3-8B reference (seq8192 bs3 = **45.3% MFU / 6,942 tok/s/chip**)
against our hand-tuned native-JAX lane (seq8192 v009 = **30.4% / 5,305**; seq2048 v018 = **35.8% / 6,964**),
combining an xprof profile teardown of the MaxText run with a source read of the MaxText training path
(`tpu-recipes-v0.1.4`, `9f1820b47`). Answers: *what optimizations get MaxText to 45%, and which are
portable to our lane?*

## The gap is seq-specific, and that's the key clue

| seq | our jax | MaxText | jax tok/s/chip vs MaxText |
|-----|---------|---------|----------------------------|
| 2048 | v018 bs4: 35.8% / 6,964 | bs4: 38.0% / 7,116 | **97.9% — parity** |
| 8192 | v009 bs1: 30.4% / 5,305 | bs3: 45.3% / 6,942 | **76.4% — +31% gap** |

**At seq2048 (where both run bs4) we are at parity.** The gap is *entirely* in the seq8192 regime, where
MaxText runs **bs3** and we are stuck at **bs1**. So the gap is not "MaxText's matmuls are fundamentally
better" — at matched batch/shape our per-matmul efficiency equals theirs. The gap is **the seq8192
batch+collective regime**: MaxText fits bs3 and keeps the MXU full; we can't fit bs>1 and our collectives
don't overlap.

## Profile teardown — where MaxText's seq8192 step goes (45.3% MFU)

| Bucket | MaxText seq8192 bs3 | our jax v018 seq2048 bs4 | note |
|--------|---------------------|---------------------------|------|
| **MXU utilization** | **61.2%** | 48.3% | hardware counter — gap is REAL, not FLOP-normalization |
| matmul (conv fusion) | 48.3% | 51.4% | similar share; MaxText extracts more MXU/cycle (fewer, larger tiled dots — `fusion.684` 1,733 TF/call ×35/step) |
| splash attention | 33.2% | 12.3% | seq8192 is 4× seq2048 → O(seq²); proportionally correct, same kernel/blocks (2048) |
| loop-fusion (norms/acts) | 7.2% | 11.0% | HBM-bound RMSNorm/SiLU |
| **FSDP reduce-scatter** | **4.0% (async-overlapped)** | **12.9% (SYNCHRONOUS)** | the single biggest addressable gap |
| all-gather (fwd FSDP) | 1.9% (overlapped) | — | async |
| data-formatting (layout) | 1.8% | 4.5% | |
| host-offload copies | **<0.1% (free)** | — | 13.6k async-copy events fully pipelined behind compute |
| TC idle | 0.3% | (low) | both compute-bound |
| HBM peak | 25.83 / 31.25 GB = **82.7%** | 20.45 / 31.25 = 65.4% | MaxText fits bs3 with headroom via host-offload |

## The two real drivers (profile-confirmed)

1. **Collective overlap — RS 4.0% async vs our 12.9% synchronous (~+9 pp).** MaxText fuses the backward
   gradient reduce-scatter into the compute window (`async-collective-start/done` pairs in the xplane);
   ours runs serially ([observation: fsdp-rs-not-overlapped](../observations/qwen3-jax-fsdp-rs-not-overlapped.md)).
   This is the single largest absolute contributor.
2. **MXU occupancy — 61.2% vs 48.3% (+12.9 pp).** MaxText keeps the MXU pipeline fuller via fewer/larger
   tiled matmuls (the layer-cadence `fusion.684` signature) — a consequence of scan-compiled layers +
   rich logical-axis layout, not fused QKV/MLP (those are OFF for qwen3).

## Code teardown — what MaxText does that we don't

After removing levers we already share (**splash attention** ✓ v008, **the 7-flag scheduler bundle** ✓ v018,
**remat** ✓, **bf16-compute/f32-weights** ✓, **`sa_block=2048`** ✓, **`fused_qkv`/`fused_mlp` = OFF** ✓ —
confirmed not the gap), the genuine differences are:

| # | MaxText lever | Code | We have? | Portable? / impact |
|---|---------------|------|----------|--------------------|
| 1 | **scan-over-layers** (`scan_layers=True`, `nn.scan` over the 36 layers) | `decoders.py:425` | ❌ we Python-unroll | **The structural enabler.** Produces fewer/larger dots (→ MXU occupancy) AND the schedule on which collective-overlap flags work. Medium-L port. |
| 2 | **named-offload remat** — `save_and_offload_only_these_names(on_device=[decoder_layer_input], offload=[q/k/v/out_proj], dst=pinned_host)` | `decoders.py:284` | partial — our `--offload_remat` only does no-batch-dim dots | **Fits bs3@seq8192** (free: host-copy <0.1%). Exactly the piece v026 was 2.34G short of. S-M port. |
| 3 | **full collective-overlap flag set** (`async_collective_fusion` + `data_parallel_all_reduce_opt` + `overlap_compute_collective_tc` + SparseCore-offload-AR) | recipe `LIBTPU_INIT_ARGS` | ❌ — and our v023 attempt at async-fusion REGRESSED −3.5% | The +9pp RS-overlap lever — but see "revises" below: needs scan to be productive. |
| 4 | rich `logical_axis_rules` (systematic axis→mesh metadata) | `base.yml:293` | partial (explicit PartitionSpec) | feeds SPMD layout; contributes to MXU occupancy. Low-M. |
| 5 | CE via custom-vjp f32 logits (`cross_entropy_with_logits`) | `max_utils.py:557` | we stream via tokamax CE | different tradeoff; not the gap (loss-head is small at the profiled bucket). |

## Revises two earlier conclusions

- **"async-collective-fusion is refuted" ([v023](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v023-async-collective-fusion.md), −3.5%)** — judged on our *unrolled* graph, where stacking async-fusion on the already-overlap-tuned scheduler over-subscribed. MaxText runs the same family of flags productively *because the scanned graph gives the scheduler the structure to overlap the per-layer RS*. The flag isn't wrong; **it needs scan first.**
- **"scan is non-dominant / jax seq8192 at structural ceiling"** ([retrospective-2](2026-06-02-qwen3_cc-jax-retrospective-2.md), [v026 conclusion](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v026-offload-ce-s8k-bs3.md)) — the profile shows scan is the *enabler* of both real drivers (MXU occupancy + collective overlap). The seq8192 gap is **not** an unclosable structural-efficiency wall; it's the **bs1-vs-bs3 + sync-vs-async-collective regime**, both reachable via scan + named-offload. (The seq2048 parity result stands — there batch already matches, so there's nothing to gain.)

## Recommended path to close the seq8192 gap (ranked)

The levers are **interdependent**, not additive — they must land together:

1. **named-offload remat** (S-M) — port `save_and_offload_only_these_names` (decoder_layer_input on-device, q/k/v/out_proj → pinned_host) → fits **bs3@seq8192** (v026 was 2.34G short with the partial policy). *Necessary precondition.*
2. **scan-over-layers** (M-L) — `nn.scan`/`nnx` scan over the 36 layers → fewer/larger dots (MXU occupancy) + the schedule the overlap flags need. *The structural enabler.*
3. **full collective-overlap flag set on the scanned graph** (S) — re-test async-collective-fusion + data-parallel-overlap (+ SparseCore-offload-AR) *after* scan lands → target RS 12.9%→~4%.

Expected if all three land: bs3 fits, RS overlaps, MXU fills → tok/s/chip 5,305 → toward ~6,900 (the +31% / ~+15pp MFU). Effort is **L** (scan is a real NNX refactor + equivalence re-verification). Caveat: the JAX-native `nn.scan` over an NNX module + named-offload composition is non-trivial; this is multi-day work, not a flag flip.

## Anti-recommendation

Do **not** pursue named-offload *alone* to fit bs3 (the v026-followup) expecting an MFU win — without scan+collective-overlap, bs3 would still run synchronous RS and likely not amortize (v016 bs2 < bs1 evidence). The three levers are a package.

## Sources

- MaxText profile (45.3%): `gs://<your-bucket>/autoresearch/qwen3_cc/maxtext/2026-06-02-qwen3-8b-8192-ref/...` (profile-analyzer teardown, 2026-06-02).
- MaxText source: `/mnt/disks/persist/maxtext-tpu-recipes-v0.1.4/MaxText/` (`9f1820b47`) — `decoders.py:284,425`, `attentions.py`, `linears.py`, `base.yml`, `max_utils.py:557`.
- Our jax profiles: [v018](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v018-xla-flag-stack.md) (seq2048), [v009](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v009-splash-s8k-vmem.md) (seq8192).
- MaxText reference experiments: [seq8192](../experiments/qwen3_cc_autoresearch_optimization/maxtext/experiments/2026-06-02-maxtext-qwen3-8b-v6e8-ref-seq8192.md), [seq2048](../experiments/qwen3_cc_autoresearch_optimization/maxtext/experiments/2026-06-02-maxtext-qwen3-8b-v6e8-ref-seq2048.md).

## See also

- [Qwen3 8B — jax](../models/qwen3-cc-jax.md) · [Qwen3 8B — maxtext](../models/qwen3-cc-maxtext.md)
- [host-offload hypothesis](../hypotheses/qwen3-jax-host-offload-bs3-seq8192.md) (now superseded by the scan+offload+overlap package)
- [observation: fsdp-rs-not-overlapped](../observations/qwen3-jax-fsdp-rs-not-overlapped.md)
