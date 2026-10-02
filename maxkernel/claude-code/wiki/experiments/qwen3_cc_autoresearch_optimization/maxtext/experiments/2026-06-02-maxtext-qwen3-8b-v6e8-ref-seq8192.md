---
title: "MaxText Qwen3-8B v6e-8 reference baseline @ seq8192"
type: experiment
hypothesis: qwen3-maxtext-reference-baseline
model: qwen3-cc-maxtext
variant: "8B/v6e-8"
commit: 9f1820b47
status: supported
verdict: supported
tags: [qwen3-cc, maxtext, v6e-8, reference, baseline, seq8192]
created: 2026-06-02
updated: 2026-06-02
---

# MaxText Qwen3-8B v6e-8 reference baseline @ seq8192

> [!note] The **45.3% MFU is non-causal** — `tpu-recipes-v0.1.4` predates MaxText's causal-mask `/2` fix (commit `6288c233`, 2026-04-11). Causal-adjusted = **39.8% MFU** (same 6,942 tok/s/chip), which is the basis the jax/torchax lanes use. See the [qwen3-cc-maxtext model-page caveat](../../../../models/qwen3-cc-maxtext.md). Numbers below are left as-recorded (immutable).

## Hypothesis under test

**Hypothesis**: Upstream MaxText running `qwen3-8b.yml` on v6e-8 at the proven dense-8B recipe tuning
(seq8192, bs3, remat-custom + decoder/proj offloads, splash `sa_block_*=2048` + fused bwd, HOST_OFFLOAD +
SparseCore-offload + DISABLE_COLLECTIVE_MATMUL XLA stack) establishes the **reference-ceiling MFU** for
Qwen3-8B — the apples-to-apples number our hand-tuned jax (30.4% @ seq8192) and torchax (19.2%) lanes are
measured against. Expectation (by analogy to MaxText Llama-3.1-8B = 44.6% on this cluster): ~40–45% MFU.

**Mechanism**: `benchmark_runner xpk --model_name=qwen3_8b_8192_ref --base_docker_image=maxtext_base_image`
from the `tpu-recipes-v0.1.4` worktree (config mirrors `llama3_1-8b-8192-no-collective-matmul`; only
`model_type` differs → `qwen3-8b`, which adds QK-norm via the model YAML).

**Predicted signal**: steady-state TFLOP/s/device → MFU in the ~40–45% range; clean 20-step run; monotone
loss on synthetic data; xplane profile captured (steps 11–15).

**Falsification criterion**: this is a reference-establishment baseline — verdict `supported` if it runs
clean and yields a stable MFU number (the number itself is the deliverable, not a beat-the-frontier test).
`invalid` if it crashes (config/tokenizer/shape error) before steady state.

## Setup

- **Cluster**: <your-cluster> (v6e-8, 2 hosts × 4 chips, fsdp=8 via `ici_fsdp_parallelism=-1`).
- **MaxText**: worktree `/mnt/disks/persist/maxtext-tpu-recipes-v0.1.4` (`tpu-recipes-v0.1.4` / `9f1820b47`); venv `/mnt/disks/persist/venv-maxtext-v0.1.4`; base image `maxtext_base_image:latest` (rebuilt 2026-06-02 from `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.6.1-rev1`).
- **Submission**:
  ```bash
  python3 -m benchmarks.benchmark_runner xpk --project=<your-project> --zone=<your-zone> \
    --device_type=v6e-8 --num_slices=1 --cluster_name=<your-cluster> \
    --base_output_directory=gs://<your-bucket>/autoresearch/qwen3_cc/maxtext/2026-06-02-qwen3-8b-8192-ref \
    --model_name=qwen3_8b_8192_ref --base_docker_image=maxtext_base_image --xpk_path=/home/alekseyv_google_com/xpk
  ```
- **Workload**: `ale-qwen3-8b-8-1-060213-9sh`; pod `...-slice-job-0-0-mlbfw`. Entrypoint `python3 -m MaxText.train MaxText/configs/base.yml model_name=qwen3-8b ...`.
- **Tuning params** (config `qwen3_8b_8192_ref`, mirrors `llama3_1-8b-8192-no-collective-matmul`): `per_device_batch_size=3`, `max_target_length=8192`, `attention=flash`, `remat_policy=custom`, `decoder_layer_input=offload`, `out/query/key/value_proj=offload`, `sa_block_*=2048`, `sa_use_fused_bwd_kernel=True`, `dataset_type=synthetic`, `enable_checkpointing=False`, `profiler=xplane skip_first_n_steps_for_profiler=10 profiler_steps=5`, `steps=20`. QK-norm via `qwen3-8b.yml`. Global batch = 3×8×8192 = 196,608 tokens/step (= `total_weights`).
- **XLA flags**: MaxText `DENSE_VMEM_LIMIT + LAYOUT_FOR_ALL_REDUCE_SCATTER + DATA_PARALLEL_OVERLAP + CF_FOR_ALL_GATHER + ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_REDUCE + HOST_OFFLOAD_FLAGS + DISABLE_COLLECTIVE_MATMUL` (40+ flags). Notably includes `xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000` — accepted by MaxText's jax0.6.1 image (it crashed our qwen3-8b-jax:v006 build in [v019](../../jax/experiments/2026-06-02-v019-xla-flags-s8k.md); build-specific).

## Baseline comparison

This is the first MaxText run for Qwen3-8B (no published target exists). Cross-stack, **same architecture, same v6e-8, seq8192** — tok/s/chip is the FLOP-counter-independent metric:

| Stack | shape | MFU | tok/s/chip | note |
|-------|-------|-----|-----------|------|
| **MaxText (this run)** | bs3 seq8192 | **45.3%** | **6,942** | recipe: offload-enabled bs3 + full XLA stack |
| jax (our hand-tuned) | bs1 seq8192 ([v009](../../jax/experiments/2026-06-02-v009-splash-s8k-vmem.md)) | 30.4% | 5,305 | remat+splash, bs1 (couldn't fit bs>1 @ seq8192) |
| jax (our hand-tuned) | bs4 seq2048 ([v018](../../jax/experiments/2026-06-02-v018-xla-flag-stack.md)) | 35.8% | 6,964 | different seq — not directly comparable |
| torchax | seq2048 | 19.2% | 3,724 | loop not yet run |
| llama3-8b MaxText (cross-model yardstick) | bs3 seq8192 | 44.6% | 7,070 | same recipe, sibling arch |

**Headline gap**: at seq8192, MaxText gets **+31% tok/s/chip** over our jax lane (6,942 vs 5,305). The qwen3-8b MaxText number (45.3%) is ~on par with the llama3-8b MaxText reference (44.6%) — expected, near-identical architecture.

## Results

Steady-state (steps 15–19, post-profile, clean — `seconds` 3.533–3.541, Δ<8 ms):

| Metric | Value |
|--------|-------|
| TFLOP/s/device (median) | **415.8** (range 415.7–416.7) |
| **Per-chip MFU** | **45.3%** (415.8 / 918 v6e bf16 peak) |
| Tokens/s/device (median) | **6,942** (range 6,939–6,956) |
| Step time (median) | **3.540 s** |
| Aggregate TPS (8 chips) | ~55.5k tok/s |
| loss | 2.96 → 2.29 over steps 15–19, monotone (synthetic) |
| exit | clean (jobset `Completed`, both pods Succeeded); wall ~3m41s |

## Profile

- **xprof URL**: http://localhost:8791/?run=2026-06-02-qwen3-8b-8192-ref (point xprof at `gs://<your-bucket>/autoresearch/qwen3_cc/maxtext/2026-06-02-qwen3-8b-8192-ref/ale-qwen3-8b-8-1-060213-9sh/tensorboard`).
- **Run name**: `ale-qwen3-8b-8-1-060213-9sh`.
- **On-disk pointer**: [`raw/profiles/2026-06-02-maxtext-qwen3-8b-v6e8-ref-seq8192/`](../../../../../raw/profiles/2026-06-02-maxtext-qwen3-8b-v6e8-ref-seq8192/)
- **GCS**: `.../2026-06-02-qwen3-8b-8192-ref/ale-qwen3-8b-8-1-060213-9sh/tensorboard/plugins/profile/2026_06_02_13_55_04/` (`gke-tpu-0d0013c1-pzbz.xplane.pb` + `.trace.json.gz`, rank-0).
- **Steps captured**: 10–14 (`skip_first_n_steps_for_profiler=10`, `profiler_steps=5`).
- No HLO dump (recipe doesn't set `--xla_dump_to`).

### Deep bucket attribution (profile-analyzer, 2026-06-02)

**MXU utilization 61.2%** (vs our jax v018 48.3% — +12.9 pp, hardware counter) · TC idle 0.3% · HBM peak **25.83/31.25 GB = 82.7%** (bs3 fits w/ headroom via host-offload) · HBM BW util 23% (compute-bound).

| Bucket | % step | note |
|--------|--------|------|
| matmul (conv fusion) | 48.3% | AI 1,757 ≫ ridge 560; fewer/larger tiled dots (`fusion.684` 1,733 TF/call ×35/step) |
| splash attention | 33.2% | seq8192 O(seq²); bq=bkv=2048, fused_bwd; fwd 9.41ms/dkv 13.87ms per call |
| loop-fusion norms/acts | 7.2% | RMSNorm/SiLU, HBM-bound |
| **FSDP reduce-scatter** | **4.0% (async-overlapped)** | vs our 12.9% SYNCHRONOUS — `async-collective-start/done` pairs |
| all-gather (fwd) | 1.9% (overlapped) | |
| data-formatting | 1.8% | |
| host-offload copies | **<0.1% (free)** | 13.6k async-copy events fully pipelined behind compute |

Full cross-stack teardown: [2026-06-02-maxtext-vs-jax-qwen3-8b-mfu-gap.md](../../../../analyses/2026-06-02-maxtext-vs-jax-qwen3-8b-mfu-gap.md). The +14.9 pp MFU gap = collective overlap (RS 12.9%→4.0%, ~+9pp) + MXU occupancy (48.3%→61.2%), both downstream of **scan-over-layers** + **named-offload** (fits bs3).

## Verdict

**Supported** — reference baseline established. MaxText Qwen3-8B on v6e-8 at seq8192 runs clean at **45.3% MFU / 6,942 tok/s/chip**, on par with the llama3-8b MaxText reference (44.6%), confirming the recipe transfers to qwen3-8b (QK-norm included). This is the **true achievable ceiling** for this architecture/hardware and **reframes the jax-lane "ceiling"**: our jax seq8192 frontier (30.4% / 5,305 tok/s/chip, bs1) is **~15 pp / −31% tok/s/chip below** what MaxText achieves — the jax lane is NOT at the achievable ceiling, it's at the ceiling *of the levers we tried*. The decisive difference MaxText exploits and we didn't: **offload-enabled batch scaling at seq8192** — `decoder_layer_input=offload` + the four `*_proj=offload` (host-DRAM offload of layer inputs + QKV/out projections) let MaxText fit **bs3** at seq8192, where our jax lane was stuck at bs1 (remat alone insufficient; CE enabled bs2 but didn't help throughput). That offload recipe is a concrete, untried jax lever.

## Next hypotheses

- [jax host-offload of decoder-layer-input + projections @ seq8192](../../../../hypotheses/qwen3-jax-host-offload-bs3-seq8192.md) — port MaxText's `decoder_layer_input=offload` + proj-offload recipe to the jax trainer to fit bs3@seq8192; target closing the 5,305→~6,942 tok/s/chip gap. **Reopens the jax seq8192 frontier** (the MaxText reference proves ~45% is reachable).
- MaxText qwen3-8b @ seq2048 (`qwen3_8b_2048_ref`) — direct shape-match to the jax v018 35.8% / 6,964 tok/s/chip frontier; quantifies the seq2048 cross-stack gap. **Dispatch next.**
- MaxText ablations (mirror llama3): toggle SparseCore-offload, fused-bwd, collective-matmul to attribute the recipe's contributions for qwen3-8b.
