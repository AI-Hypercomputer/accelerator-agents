---
title: "Qwen3-8B jax v015 — save_attn remat (bs2 check + seq-8192 attempt)"
type: experiment
hypothesis: qwen3-jax-remat-save-attn
model: qwen3-cc5-jax
variant: "8B/v6e-8"
commit: fork v6e8-qwen3-8b-jax-20260612-v015-sattn
status: supported
verdict: supported
tags: [qwen3-cc5, jax, v6e-8, remat, seq8192, splash, chunked-xla]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3-8B jax v015 — save_attn remat (bs2 check + seq-8192 attempt)

Tests [qwen3-jax-remat-save-attn](../../../../hypotheses/qwen3-jax-remat-save-attn.md):
the splash-aware remat derivative of invalid [v012](2026-06-12-v012-remat.md).

## Hypothesis under test

**Hypothesis**: save_attn remat is numerically stable (no NaN — splash never
re-entered), costs ≤ 10% at bs2, and frees enough FFN-intermediate memory
that **seq-8192 bs1 (+ chunked CE f32-x) fits** (v013 needed −13.6 GB; FFN
intermediates are ~60% of its ~25 GB activation mass).

**Mechanism**: `checkpoint_name("attn_q/k/v/out")` +
`save_only_these_names` policy per layer; FFN gate/up/silu + norms
rematerialized in backward. CPU smoke: loss bit-identical remat-vs-none.

**Predicted signal**: phase 1 (bs2): smooth losses ≈ v007's series (CE adds
≤0.007 offset), step ≤ ~735 ms; phase 2 (seq8192 bs1 + CE): fits (~30 GB
projected) and produces the lane's first seq-8192 measurement.

**Falsification criterion**: NaN at any step → invalid (remat+splash deeper
issue); bs2 overhead > 10% → refuted on cost; seq-8192 OOM → refuted for the
seq-8192 goal (gap quantified for the next derivative).

## Setup

- Hardware: v6e-8, fsdp=8, tp=1, 1 slice of `<your-cluster>`.
- Image: `qwen3-8b-jax:v6e8-qwen3-8b-jax-20260612-v015-sattn` (FROM v008b
  image: dev tokamax; + CE-f32x trainer + save_attn remat modeling).
- Workload: `alekseyv-qwen3-cc5-jax-v015-sattn`.
- Phases (both splash + vmem flag, seed 0):
  ```bash
  # phase 1: stability + overhead at the frontier shape
  python -u train.py --batch_size=2 --seqlen=2048 --train_steps=12 \
      --use_splash=True --use_tokamax_ce=True --tokamax_ce_impl=chunked_xla \
      --remat_policy=save_attn
  # phase 2: the seq-8192 attempt (profiled)
  python -u train.py --batch_size=1 --seqlen=8192 --train_steps=20 \
      --use_splash=True --use_tokamax_ce=True --tokamax_ce_impl=chunked_xla \
      --remat_policy=save_attn \
      --profile_dir=<gcs_root> --profile_start_step=12 --profile_steps=3
  ```

## Results

Both phases exit 0; **no NaN anywhere** (save_attn fixed v012's instability);
all three init lines confirmed in both phases.

**Phase 1 (bs2/seq2048 + CE + save_attn)**: stable losses 12.09–12.11;
845.7 ms median (**+27% vs v007's no-remat 667 ms** — same overhead as v012's
policy, so the cost is inherent FFN recompute, not the splash re-entry; the
NaN was policy-specific). Remat stays OFF at seq 2048.

**Phase 2 (bs1/seq8192 + CE + save_attn) — FIRST SUCCESSFUL seq-8192 RUN**:

| Metric | Value |
|--------|-------|
| Steady step time | **1475.7 ms** |
| Throughput | **44,361 tok/s (5,545/chip)** |
| MFU | **31.8%** (above the 31.4% seq-2048 frontier, at the program-target shape) |
| Program HBM | **27.62 GiB / 31.25** (~3.6 GiB headroom; v013 without remat needed 44.88) |
| Loss | smooth 12.10 → 12.07 over 20 steps |

## Profile

- **xprof URL**: `http://localhost:8791/?run=2026-06-12-qwen3-jax-v015-sattn`
  (run `…/2026_06_12_03_33_33`); profiled steps 12–14 (phase 2).
- **GCS run dir**: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v015-sattn/plugins/profile/2026_06_12_03_33_33/`
- **Op profile** (exclude-idle, master-analyzed): `convolution fusion`
  **46.8%**, `custom-call` **24.0%** (= splash fwd 13.0% + splash dkv bwd
  10.7%; CE negligible), `loop fusion` 14.9% (distributed remat-recompute
  tax, no op > 0.7%), `data formatting` 5.2%, **collectives ≈ 4.5%** (vs ~30%
  at seq2048/bs2 — fully amortized at the target shape). FLOP util **33.4%**.
- **Predicted signal: CONFIRMED** — fits with headroom; stable; the
  collective story resolves itself at seq 8192.

## HLO Dump

- **GCS**: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v015-sattn/hlo/`
- Phase-2 module `module_6132.jit_train_step`: `Total bytes used: 27.62 GiB`.

## Verdict

**supported** — the headline hypothesis (stable splash-aware remat that
unlocks seq-8192) is confirmed: no NaN, fits with 3.6 GiB headroom, and the
target-shape MFU (31.8%) exceeds the seq-2048 frontier. The bs2/seq2048
sub-criterion (≤10% overhead) failed (+27%) → remat is a per-shape knob:
OFF below seq 8192, ON at 8192 (knobs matrix updated). **Merged to trunk
(`a2f1914`)**: CE-f32x trainer + save_attn remat modeling; trunk Dockerfile
rebased on the dev-tokamax layer; `latest` rebuilt + pushed.

## Observations

- Remat overhead (~27% at bs2/seq2048) is policy-independent (v012's
  dots-saveable showed the same 847 ms) — it's the FFN recompute cost; only
  the NaN was policy-specific.
- At seq 8192 the collective share collapses to ~4.5% — the SC-offload and
  scheduler-bundle retests queued for "the llama3 op-point" are now
  LOW-priority here (target bucket nearly empty; sub-5% rule).

## Next hypotheses

- [qwen3-jax-splash-bkv-8192](../../../../hypotheses/qwen3-jax-splash-bkv-8192.md) — splash block re-tune at seq 8192 (`SPLASH_BKV=2048`): targets the 23.7% splash share; llama3 matched MaxText with bkv=2048 at this seq (env-only, v016).
- [qwen3-jax-seq-8192](../../../../hypotheses/qwen3-jax-seq-8192.md) — landed; follow-ups: remat-scope tuning (use the 3.6 GiB headroom to save back part of the recompute), CE v_block at 8192.

## Sources

- Profile + HLO (GCS): `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v015-sattn/`
- Prior: [v012 (invalid remat)](2026-06-12-v012-remat.md), [v013 (seq8k OOM)](2026-06-12-v013-seq8k.md), [v014 (bs3 OOM)](2026-06-12-v014-bs3-ce.md)
