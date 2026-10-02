---
title: "Qwen3-8B jax v031 — collective-pipeliner / ICI-AG-pipelining flags (corrected spellings) @ bs4/seq8192"
type: experiment
hypothesis: qwen3-jax-scan-ag-overlap
model: qwen3-cc5-jax
variant: "8B/v6e-8"
commit: trunk 174efd7 (image latest = v024-offload; launch-only)
status: complete
verdict: refuted
tags: [qwen3-cc5, jax, v6e-8, collectives, all-gather, pipelining, seq8192]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3-8B jax v031 — AG pipelining flags (corrected) @ bs4

Retry of [v030](2026-06-12-v030-ag-overlap.md) arm A with the flag
spellings that actually exist in this libtpu (strings audit): the
collective-pipeliner family targeting the 624 ms exposed AG wait at bs4.

## Hypothesis under test

[qwen3-jax-scan-ag-overlap](../../../../hypotheses/qwen3-jax-scan-ag-overlap.md).

**Arms (each bs4/seq8192 on the exact v027 config + 25-flag stack; only
the added flags differ):**
- **A — collective pipeliner + ICI AG pipelining**:
  `--xla_tpu_enable_collective_pipeliner=true
  --xla_tpu_enable_ici_ag_pipelining=true
  --xla_tpu_collective_fusion_pipeliner_all_gather=true` (profiled)
- **B — A + pipelined loop unrolling**:
  arm A flags + `--xla_tpu_enable_pipelined_loop_unrolling=true`

**Predicted signal**: async-done bucket < 624 ms; either arm > 6,161
tok/s/chip (frontier + 2%).

**Falsification criterion**: both arms ≤ 6,161 (or flags unknown again) →
the flag-level path is exhausted; remaining = manual weight prefetch (L)
or accept the residual gap as scan+GSPMD-structural.

## Setup

- Hardware: v6e-8, fsdp=8, tp=1, 1 slice of `<your-cluster>`.
- Image: `qwen3-8b-jax:v6e8-qwen3-8b-jax-20260612-v024-offload` (= latest).
- Workload: `alekseyv-qwen3-cc5-jax-v031-agpl`.
- 20 steps/arm; arm A profiled 12–14 + HLO.

## Results

- **Flag status**: `xla_tpu_collective_fusion_pipeliner_all_gather` is
  **RETIRED** in this libtpu (`ERROR: Accessing retired flag`, non-fatal,
  silently dropped — both arms). The other three flags were accepted.
- **Arm A (collective_pipeliner + ici_ag_pipelining)** — ran 20 steps
  clean, loss parity with v027 ±0.0001:

| Metric | v027 baseline | **Arm A** | delta |
|--------|----------------|-----------|-------|
| Step (median, excl 0–1 + 12–14) | 5,425.0 ms | **5,526.1 ms** | **+101 ms** |
| tok/s/chip | 6,040 | **5,930** | **−1.8%** |
| vs PASS bar 6,161 | — | FAIL | — |

  The pipeliner flags **regress** the frontier config. Compile also slower
  (~456 s to step 2 vs ~96 s).
- **Arm B (+ pipelined_loop_unrolling)** — **spot-preempted** ~57 s into
  compile (SIGTERM 143; Kueue reaped the jobset). The flag was ACCEPTED (no
  rejection) but is unmeasured → retried solo (without arm A's regressing
  flags) as [v032](2026-06-12-v032-loop-unroll.md).

## Profile

- **xprof URL**: http://localhost:8791/?run=2026-06-12-qwen3-jax-v031-agpl/2026_06_12_08_25_10
- **Run name**: `2026-06-12-qwen3-jax-v031-agpl/2026_06_12_08_25_10`
- **GCS**: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v031-agpl/plugins/profile/2026_06_12_08_25_10/`
- **Steps captured**: 12–14 (arm A).

Bucket attribution (per chip per step) vs v027:

| Bucket | v031-A ms | v027 ms | delta |
|---|---|---|---|
| conv fusion | 2,533 (45.7%) | 2,496 | +37 |
| custom-call (splash) | 1,451 (26.2%) | 1,448 | +3 (flat) |
| async-done | **607 (11.0%)** | 624 | **−17** |
| loop fusion | 415 | 418 | −3 |
| RS fusion | 144 | 146 | −2 |
| **all-gather (NEW explicit sync bucket)** | **138 (2.5%)** | **0** | **+138** |
| data formatting | 122 | 108 | +14 |

The +101 ms regression decomposes as: **+138 ms of brand-new synchronous
all-gather stall** (a bucket that does not exist in v027) − 17 ms async-done
relief + 37 ms conv-fusion scheduler disruption. MXU util 53.9% (flat).
copy-done (offload DMA) unchanged — the pipeliner did not touch the offload
pipeline.

## HLO Dump

- Arm A: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v031-agpl/hlo_a/`
  — 4 `jit_train_step` compile instances; module_0691 inspected (10,850
  lines). **Structural audit**: 32 synchronous `all-gather` ops (vs 17 in
  v027), **zero** `all-gather-start/done` async splits, zero loop rotation,
  zero windowed-einsum. The AGs carry `async_collective_name` hints but the
  scheduler compiled them synchronous regardless. The pipeliner restructured
  (more AGs, different modules) without producing the hoist-under-compute
  pattern.
- Arm B: `.../hlo_b/` (pre-step-0 compile artifacts only).

## Verdict

**refuted** (arm A) — a **fire-and-hurt** outcome: the flags demonstrably
changed compilation (32 vs 17 AGs, distinct schedule) but the intended
mechanism (AG-start hoisted under the previous layer's compute) never
materialized; instead 15 extra *synchronous* all-gathers appeared as a new
138 ms stall bucket. The collective-pipeliner flag route makes the frontier
config slower (−1.8%). Arm B's flag (`pipelined_loop_unrolling`) remains
the single untested knob → v032 is the FINAL flag-level probe before the
path is declared exhausted per the hypothesis's falsification criterion.

## Next hypotheses

- [qwen3-jax-scan-ag-overlap](../../../../hypotheses/qwen3-jax-scan-ag-overlap.md) — final flag probe v032 (`pipelined_loop_unrolling` solo); on refutation the hypothesis closes at flag level and [qwen3-jax-manual-weight-prefetch](../../../../hypotheses/qwen3-jax-manual-weight-prefetch.md) (L) becomes the only remaining route to the 624 ms.

## Sources

- Profile + HLO (GCS): `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v031-agpl/`
- Prior: [v030 (flag archaeology)](2026-06-12-v030-ag-overlap.md), [v027 (baseline)](2026-06-12-v027-bs4-mtfl.md), [gap analysis](../../../../analyses/2026-06-12-maxtext-vs-jax-bucket-diff.md)
