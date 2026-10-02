---
title: "Qwen3-8B jax v025 — host-offload remat + scan + bs3 @ seq 8192"
type: experiment
hypothesis: qwen3-jax-host-offload-remat
model: qwen3-cc5-jax
variant: "8B/v6e-8"
commit: fork v6e8-qwen3-8b-jax-20260612-v024-offload (same image as v024; launch-only diff)
status: complete
verdict: refuted
tags: [qwen3-cc5, jax, v6e-8, host-offload, remat, scan, seq8192, batch-seq]
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3-8B jax v025 — host-offload remat + scan + bs3 @ seq 8192

Retry of [v024](2026-06-12-v024-offload-bs3.md) with the structural fix it
identified: `--use_scan=True`. Scan bounds the backward's rematted-temp
liveness to ~one layer per iteration — the property that lets MaxText's
scanned decoder fit bs3 with offload
([observation](../../../../observations/host-offload-unrolled-backward-liveness.md)).

## Hypothesis under test

**Hypothesis** ([qwen3-jax-host-offload-remat](../../../../hypotheses/qwen3-jax-host-offload-remat.md)):
`offload_attn` remat **under scan** lets bs3 @ seq 8192 compile (v024's
59.41 G unrolled-backward liveness collapses to per-iteration working set)
and beat 5,898 tok/s/chip.

**Diff vs v024**: `--use_scan=True` added to both phases. Nothing else —
same image, same LIBTPU stack (vmem 81920 + HOST_OFFLOAD scheduler flags).

**Predicted signal**: ph1 (bs1) loss = seed-0 series (scan was bit-parity in
v020; offload was parity-exact in v024-ph1); ph2 (bs3) compiles — expected
live-set well under 31.25 G (scan alone saved 8.0 GiB at bs1 in v020; the
offloaded ~9 GiB attention mass is in host space) — and tok/s/chip > 5,898;
stretch ≥ 6,500 (MaxText 6,953).

**Falsification criterion**: bs3 OOM even under scan, or fits but
tok/s/chip ≤ 5,898 + 1% (host-transfer stalls / scan uniform-remat tax
~5.6% from v021 not recovered by bs3 amortization) → hypothesis refuted on
this substrate; residual MaxText gap documented as stack-structural.

## Setup

- Hardware: v6e-8, fsdp=8, tp=1, 1 slice of `<your-cluster>`.
- Image: `qwen3-8b-jax:v6e8-qwen3-8b-jax-20260612-v024-offload` (reused).
- Workload: `alekseyv-qwen3-cc5-jax-v025-off3s`.
- Phases (both seq8192, splash + chunked CE f32-x + offload_attn remat +
  scan + vmem flag + HOST_OFFLOAD scheduler flags; no SKIP_EVERY):
  ph1 = bs1 sanity (12 steps); ph2 = bs3 (20 steps, profiled 12–14, HLO dump).

## Results

**Phase 1 (bs1 + scan + offload_attn) — numerics PASS.** 12 steps, loss
series within ±0.0001 of the seed-0 references (12.0979 / 12.0985 /
12.0943), no NaN. Steady 1,597 ms — the offload transfer overhead is +8.6%
over the no-offload scan reference (~1,471 ms, v020-equivalent).

**Phase 2 (bs3 + scan + offload_attn) — FITS, runs clean, but slow.**
20 steps, exit 0, no OOM (v024's 59.41 G unrolled-liveness wall is gone —
scan bounds it, as predicted). Step-time spread < 8 ms across the window.

| Metric | baseline v019 (bs1, unrolled, save_attn+skip5) | this run (bs3, scan, offload_attn) | MaxText mt-v001 (bs3) | delta vs bar |
|--------|------------------------------------------------|-------------------------------------|------------------------|--------------|
| Step time | 1,389 ms (65,536 tok) | 4,566.6 ms (196,608 tok) | 3,535 ms (196,608 tok) | — |
| tok/s/chip | **5,898** | **5,381** | 6,953 | **−8.8%** |
| MFU (our accounting) | 33.6% | 30.9% | ≈39.6% | −2.7 pp |
| Loss | reference | clean descent 12.10→12.05, parity at bs1 | — | — |

bs1→bs3 amortization on this substrate is only +4.9% tok/s/chip (5,130 →
5,381), nowhere near enough to cover the scan uniform-remat tax (~5.6%,
v021) plus the offload transfer overhead (+8.6% at bs1). The PASS bar
(> 5,898) is missed; MaxText remains 29% faster at the identical shape and
mechanism — the residual gap is in *our* execution of the shape, not in
fitting it.

## Profile

- **xprof URL**: http://localhost:8791/?run=2026-06-12-qwen3-jax-v025-off3s/2026_06_12_06_58_27
- **Run name**: `2026-06-12-qwen3-jax-v025-off3s/2026_06_12_06_58_27`
- **On-disk/GCS**: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v025-off3s/plugins/profile/2026_06_12_06_58_27/`
- **Steps captured**: 12–14 (profiling overhead ~+42 ms on step 12)
- **Contents**: xprof trace (8 chips, 2 hosts) + HLO dump (separate dir below)

Bucket attribution (steady, steps 12–14), with v015 (bs1, save_attn,
unrolled, same seq) as reference:

| Bucket | v025 (bs3, scan, offload) | ~ms/step | v015 ref | Delta |
|---|---|---|---|---|
| conv fusion (matmul) | 44.3% | 2,023 | ~47% | −2.7 pp (bs3 amortization) |
| custom-call (splash + async coll.) | 23.8% | 1,087 | ~24% | ≈0 |
| loop fusion (scan remat recompute + CE) | 14.5% | 662 | ~14% | +0.5 pp |
| **async-done (DMA blocking)** | **9.1%** | **416** | ~1–2% | **+7 pp — host-offload stalls** |
| data formatting | 3.8% | 174 | ~9.3% | −5.5 pp (bs3 amortizes) |
| collectives (AG/AR/RS/permute) | ~2.7% | ~125 | ~4.5% | −1.8 pp |

- **HBM peak 27.34 / 31.25 GB (87.5%)**: stack 21.53 GB (scan body temps),
  heap 5.81 GB, **3.91 GB free** → bs4 does NOT fit (slope ~4 GB/batch),
  and reverting to save_attn at bs3 (18.12 GB back into HBM) is impossible.
- **Host-transfer accounting**: 18.12 GB/chip/step round-trips to S(5)
  (Q 7.25 + K 1.81 + V 1.81 + attn_out 7.25). copy-starts are free
  (prefetch works); 35 copy-done events/step avg 31.4 ms; ~62% of DMA
  latency hidden, **416 ms/step residual TensorCore stall** (≈ 320 ms
  attributable to offload, rest FSDP all-gather-done).
- **Windowed-einsum collective matmul is active** (collective-permute ring
  visible) — MaxText's recipe explicitly disables it
  (`--xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000`); the
  restructuring also hides inside the conv-fusion bucket.

## HLO Dump

- **GCS**: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v025-off3s/hlo/`
  (main module `module_0766.jit_train_step`, 4,680 instructions, scheduled).
- **Mechanism fully confirmed**: stacked scan-carry saves in host space —
  `bf16[36,3,32,8192,128]{…S(5)}` (Q), `bf16[36,3,8,8192,128]{…S(5)}` (K,V),
  `bf16[36,3,8192,32,128]{…S(5)}` (attn_out); `dynamic-update-slice` into
  S(5) in the forward scan body; 16 async `dynamic-slice-start` reads in the
  backward feeding `splash_mha_fwd_residuals`/`splash_mha_dkv` via
  copy-done. v024's unrolled 59.41 G liveness collapses to a bounded scan
  working set — fits at 27.34 G.
- Splash fwd ×4 instances (fwd + remat-fwd per backward variant) + dkv ×2;
  374 `rematted_computation` occurrences (scan uniform-remat recomputing
  FFN/RMSNorm in every backward iteration).
- chunked_xla CE while-loops present (b_block 1024 / v_block 2048); no
  mosaic CE; FSDP Async collectives present.
- `ConcatBitcast` bridges S(5) fetches into splash's VMEM operand layout —
  this is most of the 3.8% data-formatting bucket.

**Bottleneck verdict (profile-analyzer)**: largest recoverable cost is the
416 ms/step DMA blocking (offload-specific ≈ 320 ms), followed by the
662 ms scan uniform-remat recompute. Combined with the measured weak batch
amortization (+4.9% bs1→bs3), the offload route costs more than bs3 returns
on this stack.

## Verdict

**refuted** — pre-registered falsification criterion met: bs3 fits but
5,381 tok/s/chip ≤ 5,898 + 1%. v019 (bs1, unrolled, save_attn + skip5)
remains the frontier.

What this run *established* despite the verdict:
1. **Scan + offload is the only substrate where bs3 @ seq 8192 fits**
   (27.34/31.25 G) — the memory mechanism is validated end-to-end with
   exact loss parity.
2. **The MaxText gap is NOT primarily batch.** bs1→bs3 amortization on our
   stack is +4.9%, while MaxText leads our v019 by +17.9% at bs3. At the
   identical bs3 shape we are +1,032 ms vs MaxText (4,567 vs 3,535) with
   competitive matmul time — the excess decomposes into splash-bwd
   (1,087 ms bucket), scan recompute (662 ms), DMA stalls (416 ms), and a
   flag-stack delta (13 recipe flags unadopted, incl. windowed-einsum
   disable).

## Next hypotheses

- [qwen3-jax-maxtext-flag-completion](../../../../hypotheses/qwen3-jax-maxtext-flag-completion.md) — adopt the 13 residual MaxText recipe flags (windowed-einsum disable, async-collective-fusion family, RS layout, DP overlap, vmem 98304) on the v025 config; launch-only A/B (v026, dispatched).
- [qwen3-jax-splash-fused-bwd](../../../../hypotheses/qwen3-jax-splash-fused-bwd.md) — splash backward kernel config at the bs3 shape (fused bwd / block sweep); targets the 23.8% splash bucket; llama3 lane measured +30% kernel-time from `block_q=block_kv=1024, fused_bwd=True` autotune.
- [qwen3-jax-offload-decoder-input](../../../../hypotheses/qwen3-jax-offload-decoder-input.md) — (existing) now quantified: would free ~7.25 GB (the scan carry) → bs4 reachable; deprioritized by the measured weak batch amortization.

## Sources

- Profile + HLO (GCS): `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v025-off3s/`
- Prior: [v024 (unrolled OOM + mechanism proof)](2026-06-12-v024-offload-bs3.md), [v020 (scan, −8 GiB live-set)](2026-06-12-v020-scan.md), [mt-v001 (the reference)](../../maxtext/experiments/2026-06-12-mt-v001-base.md)
