---
title: "Qwen3-8B jax v028 — scan-over-layers + collective-overlap flags @ seq8192 bs1"
type: experiment
hypothesis: qwen3-jax-scan-layers
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: jax-scan-2026-06-02
status: supported
verdict: supported
tags: [qwen3-cc, jax, scan, collective-overlap, splash, remat, seq8192, v6e-8]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-8B jax v028 — scan + collective-overlap @ seq8192 bs1

## Hypothesis under test

**Hypothesis**: `nnx`/`lax.scan` over the 36 decoder layers (stacked-param storage; 1 compiled body)
gives the XLA scheduler the structure to **overlap the FSDP reduce-scatter** (the v018 profile's #1
non-matmul cost: 12.9% synchronous) — productive *only* on the scanned graph (the same async-collective-
fusion flags REGRESSED on the unrolled graph, [v023](2026-06-02-v023-async-collective-fusion.md)). Tested at
the bs1 seq8192 shape (fits, = v009) to isolate the scan+overlap benefit before scaling to bs3.

**Mechanism**: image `qwen3-8b-jax:v028-scan` (scan refactor — stacked layers via vmap-init + lax.scan;
CPU equivalence PASS, scan↔unrolled fwd/grad parity to 1e-6). Launch `--use_remat --use_splash --use_scan
--batch_size=1 --seqlen=8192` + the v018 7 scheduler flags + the 4 async-collective-fusion overlap flags.

**Predicted signal**: MFU > v009 30.4% / v019 30.7% (scan enables RS overlap → recover part of the 12.9%);
loss ~12.07 stable; exit 0. This is the scan-isolation step; the MaxText 45.3% target needs bs3 (next).

**Falsification criterion**: MFU ≤ 30.7% beyond noise (scan + overlap flags don't help at bs1 either —
would mean the seq8192 win genuinely needs bs3, not just scan).

## Setup

- **Hardware**: v6e-8 (fsdp=8), image `qwen3-8b-jax:v028-scan` (stacked-param scan). `--use_remat --use_splash --use_scan --batch_size=1 --seqlen=8192` + 7 scheduler flags + 4 async-collective-fusion overlap flags. All confirmed active; all 4 overlap flags accepted (no rejections).

## Results — NEW seq8192 frontier

| Metric | v009 (bs1, no scan) | v019 (+sched flags, no scan) | **v028 (scan + overlap)** | Δ vs v009 |
|--------|----------------------|------------------------------|----------------------------|-----------|
| MFU | 30.4% | 30.7% | **32.3%** | **+1.9 pp** |
| tok/s/chip | 5,305 | ~5,360 | **5,632** | **+6.2%** |
| steady step (ms) | ~1,540 | ~1,526 | **1,452** | −5.7% |
| step-0 compile (s) | long | long | **35.3** | scan = 1 body |
| loss | 12.10→12.07 | — | 12.10→12.07 stable | parity |

## Profile

- **xprof URL**: http://localhost:8791/?run=2026-06-02-qwen3-jax-v028-scan-overlap-s8k-bs1
- **Run name**: `2026-06-02-qwen3-jax-v028-scan-overlap-s8k-bs1`
- **On-disk pointer**: [`raw/profiles/2026-06-02-qwen3-jax-v028-scan-overlap-s8k-bs1/`](../../../../../raw/profiles/2026-06-02-qwen3-jax-v028-scan-overlap-s8k-bs1/)
- **GCS**: `.../2026-06-02-qwen3-jax-v028-scan-overlap-s8k-bs1/plugins/profile/2026_06_02_16_15_34/` (2 hosts; steps 12–14).
- Contents: xprof trace + HLO (~730 modules).

## HLO Dump

- **GCS**: `.../2026-06-02-qwen3-jax-v028-scan-overlap-s8k-bs1/hlo/` (~730 files; the scanned train-step is now a `while`-bodied module, not 36 unrolled layers).

## Verdict

**Supported — NEW seq8192 frontier, 32.3% MFU / 5,632 tok/s/chip** (+1.9 pp / +6.2% over v009 bs1). Confirms the [MFU-gap analysis](../../../../analyses/2026-06-02-maxtext-vs-jax-qwen3-8b-mfu-gap.md): **scan makes the collective-overlap flags productive** — the same 4 async-collective-fusion flags that *regressed* −3.5% on the unrolled graph ([v023](2026-06-02-v023-async-collective-fusion.md)) now help, because the scanned (`while`-bodied) graph gives the scheduler the structure to overlap the per-layer reduce-scatter. Bonus: compile time collapsed to 35.3 s (1 layer body vs 36). Loss parity, exit 0, scan numerically validated on CPU (equivalence PASS + scan↔unrolled fwd 9.5e-7 / grad 4.3e-8). **The scan + named-offload + overlap stack is the foundation; this is bs1 (5,632) — the MaxText 45.3% / 6,942 target needs bs3** (next: fix the CE f32 lm_head-weight wall to fit bs3, then run the full stack at bs3).

## Next hypotheses

- **Full stack @ bs3** (the MaxText-equivalent config) — scan + named-offload + CE (bf16 lm_head weight to clear the v027 4.64G wall) + bs3 + 7 scheduler + 4 overlap flags. Now that overlap works (this run) and scan amortizes, bs3 should finally amortize (unlike v016 bs2<bs1, which had sync collectives). **Dispatched next as v029.** Target 6,942 tok/s/chip.
- bs2 fallback if bs3 won't fit even with the CE-weight fix.
