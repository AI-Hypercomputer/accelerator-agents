---
title: "Qwen3-8B jax v038 — 50-step validation + profile of the bs2/save_attn frontier candidate"
type: experiment
hypothesis: qwen3-jax-bs2-saveattn-frontier
model: qwen3-cc5-jax
variant: "8B/v6e-8"
commit: trunk 174efd7 (image digest bc8d7950; JAX_SCAN_UNROLL defaults to 1 — no code delta in play)
status: complete
verdict: supported
tags: [qwen3-cc5, jax, v6e-8, validation, save-attn, seq8192, frontier]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3-8B jax v038 — bs2/save_attn frontier validation

Certifies [v037](2026-06-12-v037-unroll-saveattn-bs2.md)'s control-arm
discovery (6,240 tok/s/chip / 35.8% MFU) per the v019 precedent: 50 steps,
long-horizon NaN/drift check, profile + HLO (the configuration was never
profiled).

## Hypothesis under test

[qwen3-jax-bs2-saveattn-frontier](../../../../hypotheses/qwen3-jax-bs2-saveattn-frontier.md):
sustained > 6,200 tok/s/chip over 50 steps. Profile question: what does the
bucket structure look like with NO offload at bs2 (async-done? splash
share? where does the remaining MaxText gap sit at this op-point)?

## Setup

- Hardware: v6e-8, fsdp=8, tp=1, 1 slice of `<your-cluster>`.
- Image: digest-pinned `qwen3-8b-jax@sha256:bc8d7950…`.
- Workload: `alekseyv-qwen3-cc5-jax-v038-bs2val`.
- Single arm: bs2 seq8192, scan + save_attn + splash + chunked CE f32-x +
  21-flag stack; **50 steps**, profiled 30–32, HLO dump.

## Results

50 steps clean, exit 0, no NaN, no drift. **Loss series at steps 0–2 is an
EXACT reproduction of v037 arm A** (12.1029 / 12.0996 / 12.0944) — config
+ seed deterministic. Steady state (steps 2–49 excl. profiler step 30):

| Metric | v037-A (20 steps) | **v038 (50 steps, canonical)** | v027 old frontier | MaxText |
|--------|-------------------|--------------------------------|--------------------|---------|
| Step (median) | 2,625.7 ms | **2,633.6 ms** (spread 0.30%, σ=2.2 ms) | 5,425.0 ms (262,144 tok) | 3,535 ms (bs3) |
| tok/s/chip | 6,240 | **6,221** | 6,040 | 6,953 |
| MFU | 35.8% | **35.7%** | 34.6% | ≈39.6% |
| vs PASS bar 6,200 | — | **PASS (+0.3%)** | — | — |
| vs old frontier 6,040 | — | **+3.0%** | — | — |
| vs MaxText | — | **−10.5%** | −13.1% | — |

The −0.30% vs v037-A is run-to-run noise. Loss 12.1029 → 12.0292 over 50
steps, monotone-ish, long-horizon stable.

## Profile

- **xprof URL**: http://localhost:8791/?run=2026-06-12-qwen3-jax-v038-bs2val/2026_06_12_12_36_28
- **Run name**: `2026-06-12-qwen3-jax-v038-bs2val/2026_06_12_12_36_28`
- **GCS**: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v038-bs2val/plugins/profile/2026_06_12_12_36_28/`
- **Steps captured**: 30–32; xprof traces, 2 hosts × 4 chips.

Bucket attribution (per chip per step; the new frontier's structure):

| Bucket | % | ms/step | vs v027 (bs4+offload) |
|---|---|---|---|
| conv fusion (matmul) | 48.6% | 1,280 | 46.0% |
| splash custom-call | 28.1% | 740 | 26.7% |
| **loop fusion (save_attn remat recompute)** | **9.5%** | **250** | 7.7% |
| RS fusion (grad reduce-scatter) | 7.5% | 198 | 2.7% |
| data formatting | 2.1% | 55 | 2.0% |
| all-gather (active, overlapped) | 0.9% | 24 | — |
| **async-done (exposed AG wait)** | **0.03%** | **0.9** | **11.5% / 624 ms** |
| idle | 0.3% | 8 | — |

**Headline structural finding: the exposed FSDP AG wait COLLAPSED from
624 ms (bs4+offload) to 0.9 ms — fully overlapped.** The "74% of the
MaxText gap" stall was an op-point artifact of bs4+offload, not a lane
property. At this frontier the remaining −10.5% to MaxText decomposes into
the save_attn remat recompute (250 ms, 9.5%) and MXU efficiency (**55.5%**
vs MaxText 61.9%).

- **HBM peak 28.62 / 31.25 GiB (91.6%)** — 2.63 GiB free. bs3+save_attn is
  a knife-edge (naive heap scaling: OOM by 0.27 GiB; fixed-infra model:
  fits by 0.48 GiB) → probed as [v039](2026-06-12-v039-bs3-saveattn.md).

## HLO Dump

None for this run — the compile cache was fully warm from v037 arm A (zero
recompilation), so XLA wrote no dumps. The configuration is pure trunk
`174efd7` code; its structural signatures (scan while, save_attn
checkpoint names, splash custom-calls, chunked-CE loops) were
HLO-verified in v015/v020/v027-era dumps of the same code paths.

## Verdict

**supported — FRONTIER SHIFT.** Canonical lane frontier is now
**2,633.6 ms / 49,772 TPS / 6,221 tok/s/chip / 35.7% MFU @ bs2/seq8192**
(50-step validated, spread 0.30%, exact loss reproducibility). Recipe:
scan + save_attn + splash + chunked CE f32-x + the 21-flag stack — zero
code delta vs trunk; launch-only. The bs4+offload config (v027, 6,040)
retires to alternate status. Session climb: 20.5% → **35.7% MFU**.

## Next hypotheses

- [qwen3-jax-bs3-saveattn](../../../../hypotheses/qwen3-jax-bs3-saveattn.md) — the knife-edge batch rung on the new substrate (v039, launch-only; OOM is decisive data either way).
- [qwen3-jax-int8-aqt](../../../../hypotheses/qwen3-jax-int8-aqt.md) — retrospective-2 #3; now doubly motivated: int8 weights halve the conv-fusion HBM traffic feeding the 48.6% bucket AND ~halve the 2.63 GiB-constrained weight memory (L; loss-parity validation required).

## Sources

- Profile + HLO (GCS): `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v038-bs2val/`
- Prior: [v037 (discovery)](2026-06-12-v037-unroll-saveattn-bs2.md), [v019 (validation precedent)](2026-06-12-v019-val50.md)
