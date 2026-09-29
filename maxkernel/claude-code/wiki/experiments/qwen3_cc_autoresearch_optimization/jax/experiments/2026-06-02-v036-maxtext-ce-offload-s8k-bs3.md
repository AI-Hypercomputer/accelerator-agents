---
title: "Qwen3-8B jax v036 — maxtext-CE + named-offload remat @ seq8192 bs3"
type: experiment
hypothesis: qwen3-jax-seq8192-kernel-gap
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: jax-maxtext-ce-2026-06-02
status: refuted
verdict: refuted
tags: [qwen3-cc, jax, cross-entropy, maxtext, custom-vjp, offload, scan, overlap, splash, remat, seq8192, bs3, v6e-8]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-8B jax v036 — maxtext-CE + named-offload remat @ seq8192 bs3

## Hypothesis under test

**Hypothesis**: the v035 profile attributed ~85% of the residual MaxText gap (+454 ms) to `nothing_saveable`
recompute (loop-fusion/norms 17.4% vs MaxText 7.2%). Adding `--offload_remat`
(`save_and_offload_only_these_names`, offloads the `checkpoint_name`-tagged Q/K/V/O-proj + mlpwi
activations to pinned host instead of recomputing) — now that maxtext-CE removed the tokamax f32[H,V]
weight-gather that confounded [v030](2026-06-02-v030-scan-offload-ce-s8k-bs3.md) — cuts the backward
recompute and lifts the seq8192 frontier above v035 (6,030) toward MaxText (6,942).

**Mechanism**: image `qwen3-8b-jax:v033-maxtext-ce`. `--use_remat --use_splash --use_scan --use_maxtext_ce
--offload_remat --batch_size=3 --seqlen=8192` + 7 scheduler + 4 overlap flags. Only change vs v035 is
`--offload_remat`.

**Predicted signal**: fits (offload frees HBM from the 90.7% nothing_saveable peak); loss ~12.05 matching
v035; tok/s/chip > v035 6,030 toward 6,942.

**Falsification criterion**: ≤ v035 6,030 beyond noise — the host round-trip isn't repaid even without the
tokamax gather (the recompute cost is in the RMSNorms, which the proj/mlpwi-tagged offload doesn't cover →
would point to needing a norm-inclusive offload/save policy instead).

## Setup

- **Hardware**: v6e-8 (fsdp=8), image `qwen3-8b-jax:v033-maxtext-ce`. Config = v035 + `--offload_remat`:
  `--use_remat --use_splash --use_scan --use_maxtext_ce --offload_remat --batch_size=3 --seqlen=8192` + 7
  scheduler + 4 overlap flags. Confirmed active incl. `[remat] ... save_and_offload(proj+mlpwi → pinned_host) ON`
  (NOT nothing_saveable) and `[ce] MaxText/T5X custom_vjp ... ON`; `per_chip_batch=3 global_batch=24`.
  (First dispatch refused — workload name >40 chars; re-ran as `alekseyv-qwen3-cc-jax-v036-off-s8k-bs3`.)

## Results — REGRESSES (offload not repaid)

| run | config | tok/s/chip | MFU | steady step (ms) |
|-----|--------|-----------|-----|------------------|
| **v035** | maxtext-CE bs3, **no offload** | **6,030** | **34.6%** | 4,075 |
| v036 | maxtext-CE bs3, **+offload** | 4,908 | 28.2% | 5,007 |
| MaxText | bs3 reference | 6,942 | 45.3% | 3,546 |

v036 = **−18.6% vs v035** (step +22%, 4,075 → 5,007 ms). Loss 12.0994 → 12.0511 monotone, no NaN, exit 0,
no OOM. Compile 53.7 s.

## Profile

- **Run name**: `2026-06-02-qwen3-jax-v036-maxtext-ce-offload-s8k-bs3` · xprof [`http://localhost:8791/?run=2026-06-02-qwen3-jax-v036-maxtext-ce-offload-s8k-bs3`](http://localhost:8791/?run=2026-06-02-qwen3-jax-v036-maxtext-ce-offload-s8k-bs3) · GCS `.../plugins/profile/2026_06_02_18_26_06/` (steps 12–14). On-disk: [`raw/profiles/2026-06-02-qwen3-jax-v036-maxtext-ce-offload-s8k-bs3/`](../../../../../raw/profiles/2026-06-02-qwen3-jax-v036-maxtext-ce-offload-s8k-bs3/). (No deep profile-analyzer pass — the headline regression is decisive; the prior v035 profile already attributed the buckets.)

## HLO Dump

- **GCS**: `.../2026-06-02-qwen3-jax-v036-maxtext-ce-offload-s8k-bs3/hlo/`.

## Verdict

**Refuted** (≤ v035, hits the falsification criterion). Named-offload makes bs3 **18.6% slower**, not faster,
even with maxtext-CE removing the tokamax weight-gather that confounded v030. **Two reasons, both decisive
for the lane:**
1. **The offload misses the actual recompute cost.** The v035 profile attributed the +454 ms to RMSNorm
   recompute (loop-fusion 17.4%). But `save_and_offload_only_these_names` offloads the `checkpoint_name`-tagged
   **proj/mlpwi outputs**, NOT the norm outputs — so the dominant norm recompute is untouched.
2. **Our pinned_host offload is not pipelined.** The host round-trip for the proj/mlpwi activations lands on
   the critical path (+22% step time), the opposite of MaxText's <0.1% overlapped host-copy. This is the same
   non-pipelined-offload limitation seen in v030 — now isolated cleanly (no tokamax confound).

So the remat-policy gap to MaxText is **not closable with our current offload mechanism**. **seq8192 frontier
stays v035 (34.6% / 6,030, no offload).** Closing the remaining gap would need either a *pipelined* host offload
(XLA/kernel-level — out of scope) or a norm-inclusive save policy that fits in HBM (already at 90.7%), plus the
secondary MXU/logical-axis tile-alignment lever.

## Next hypotheses

- **MXU occupancy via logical-axis-rules** (the v035 profile's secondary lever, +156 ms / ~+3-4%): align per-matmul
  GeMM tiles to MaxText's larger shapes via sharding-layout changes. Medium-high effort (sharding rewrite),
  uncertain payoff; deferred — captured in [qwen3-jax-seq8192-kernel-gap](../../../../hypotheses/qwen3-jax-seq8192-kernel-gap.md).
- [MaxText CE @ seq2048](../../../../hypotheses/qwen3-jax-maxtext-ce-seq2048.md) — cheap confirmation that maxtext-CE is the universal default CE (v037, dispatched).
- Pipelined host-offload — **out of scope** (XLA/kernel-authoring to overlap the pinned_host copy; our stock `offload_dot`/`save_and_offload` path lands on the critical path).
