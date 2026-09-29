---
title: "Exp 23 — AMP at bs=2 seq=4096 (ACCEPTED 33.1% MFU — best AMP-friendly density)"
type: experiment
tags: [llama3, torchax, amp, fp32-master, accepted]
hypothesis: llama3-torchax-amp-density
model: llama3-8b-torchax
created: 2026-04-25
updated: 2026-04-25
commit: "v6e8-llama3-8b-torchax-20260425-exp23-amp-bs2-seq4k (image hf-v7)"
branched_from: v6e8-llama3-8b-torchax-20260425-exp20-amp-fp32-master
verdict: supported
---

Same per-chip B·L = 8,192 as [exp 20](2026-04-25-exp20-amp-fp32-master-seq8k-accepted.md)
(`bs=1 seq=8192`) reached via a different shape: `bs=2 seq=4096`. **+1.5 pp
MFU vs exp 20** — bigger batch dimension parallelizes better at
identical compute density; less attention `O(L²)` overhead at the shorter
seq. The current best AMP-stack shape that fits.

## Setup

- Image `hf-v7` (AMP path), branched from exp 20.
- Stack: bf16 weights + bf16 compute + fp32 mu/nu master + per-layer remat
  + autotuned splash + VMEM=98304.
- Env-only diff vs exp 20: `--batch_size=2 --seqlen=4096`.

## Results

| Metric | Exp 20 (bs=1 seq=8192) | **Exp 23 (bs=2 seq=4096)** | Δ |
|---|---|---|---|
| Per-chip B·L | 8,192 | 8,192 | identical |
| Step time | 1,450 ms | **1,301 ms** | **−10 %** |
| Throughput (aggregate) | 45,184 TPS | **50,358 TPS** | **+11 %** |
| Per-chip TPS | 5,648 | **6,295** | **+11 %** |
| MFU | 31.6 % | **33.1 %** | **+1.5 pp** |

Same per-chip token count, ~10 % faster step time. Source: at seq=8192 the
attention kernel takes 7.944 ms × 32 layers = 254 ms / step (~17 % of total);
at seq=4096 attention drops to 2.402 ms × 32 = 77 ms (~6 % of total) — the
recovered budget translates directly to MFU.

## Verdict + reasoning

**Supported (accepted).** Exp 23 = the **best AMP-friendly stack** at
B·L=8,192 per chip. Exp 20 retains the program-target seq=8192 but at
−1.5 pp MFU due to the longer-seq attention overhead. Either is a valid
production target depending on which trade matters more (longer context
vs higher throughput).

## See also

- [Exp 20 (program-target seq, AMP)](2026-04-25-exp20-amp-fp32-master-seq8k-accepted.md) — same B·L, different shape.
- [Exp 17 (no AMP, seq=8192)](2026-04-25-exp17-layer-remat-bs1-seq8k-accepted.md) — bf16 mu/nu version.
- [Exp 22 (chunked CE), exp 26 (host-offload)](#) — both attempted to fit `bs=2 seq=8192 + AMP` (the next-density target); both rejected (see those pages).

## Sources

- Workload (XPK): `llama3-8b-exp23-amp-bs2-seq4k`.
- Image: `<your-registry>/test/llama3-8b-torchax-container:hf-v7`.
- Branch: `v6e8-llama3-8b-torchax-20260425-exp23-amp-bs2-seq4k`.
