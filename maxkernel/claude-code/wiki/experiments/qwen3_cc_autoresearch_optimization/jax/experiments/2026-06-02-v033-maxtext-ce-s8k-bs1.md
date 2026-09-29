---
title: "Qwen3-8B jax v033 — MaxText custom_vjp CE @ seq8192 bs1 (parity anchor)"
type: experiment
hypothesis: qwen3-jax-maxtext-ce
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: jax-maxtext-ce-2026-06-02
status: supported
verdict: supported
tags: [qwen3-cc, jax, cross-entropy, maxtext, custom-vjp, scan, overlap, splash, remat, seq8192, bs1, v6e-8]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-8B jax v033 — MaxText custom_vjp CE @ seq8192 bs1

## Hypothesis under test

**Hypothesis**: swapping our `_ce` (autodiff through `jax.nn.log_softmax`) for MaxText's verbatim T5X
`@jax.custom_vjp cross_entropy_with_logits` (one-hot targets, z_loss=0) is **numerically identical**
(CPU check: |Δloss|=0, |Δgrad|=0) and **does not regress** bs1 seq8192 vs the v028 frontier
(32.3% / 5,632 tok/s/chip). This is the parity anchor before the bs2 wall-test (v034).

**Mechanism**: image `qwen3-8b-jax:v033-maxtext-ce` (FROM v030-scan-full, swaps `train.py` only —
adds the module-level custom_vjp + `--use_maxtext_ce` flag; diff vs v030 is ONLY the CE additions,
verified). Config = v028 frontier + `--use_maxtext_ce=True` (NO tokamax CE, NO offload):
`--use_remat --use_splash --use_scan --use_maxtext_ce --batch_size=1 --seqlen=8192` + 7 scheduler + 4
async-collective-fusion overlap flags.

**Predicted signal**: ≈ v028 5,632 (within noise); loss ~12.07 stable matching v028; exit 0.

**Falsification criterion**: < 5,632 beyond noise ⇒ the custom_vjp graph compiles worse at bs1
(unexpected given identical math) ⇒ reconsider before the bs2 test.

## Setup

- **Hardware**: v6e-8 (fsdp=8), image `qwen3-8b-jax:v033-maxtext-ce` (FROM v030-scan-full, `train.py` swap only — adds the module-level T5X custom_vjp + `--use_maxtext_ce`; diff vs v030 verified to be exactly the CE additions). Config = v028 frontier + `--use_maxtext_ce=True` (no tokamax CE, no offload): `--use_remat --use_splash --use_scan --use_maxtext_ce --batch_size=1 --seqlen=8192` + 7 scheduler + 4 async-collective-fusion overlap flags. All four levers confirmed active in log: `[scan] ... ON`, `[attn] splash kernel ON`, `[remat] ... nothing_saveable ... ON`, `[ce] MaxText/T5X custom_vjp cross_entropy_with_logits (z_loss=0) ON`.
- **Launch note**: the first dispatch (`v033`, workload `...v033-mtce-s8k`) crashed at libtpu flag validation in 5s — two scheduler flags had a spurious `_tpu_` infix. Corrected to `--xla_latency_hiding_scheduler_rerun=2` / `--xla_max_concurrent_host_send_recv=100` (the `xla_` form this libtpu build accepts) and re-ran as workload `...v033b-mtce-s8k`. No code/image change.

## Results — PARITY (no regression)

| run | CE path | tok/s/chip | MFU | steady step (ms) |
|-----|---------|-----------|-----|------------------|
| v028 | `_ce` (autodiff log_softmax) | 5,632 | 32.3% | 1,452 |
| **v033** | **MaxText T5X custom_vjp** | **5,656** | **32.4%** | **1,446.8** |

Δ = **+0.4% / +24 tok/s/chip / +0.1 pp MFU / −5 ms** — within run-to-run noise → **parity**. Loss
12.0979 → 12.0720 monotone over 20 steps, no NaN/Inf, tracking v028's trajectory (identical math, as the
CPU check predicted: |Δloss|=0, |Δgrad|=0). Step 0 compile 36.2 s. Exit 0.

## Profile

- **Run name**: `2026-06-02-qwen3-jax-v033-maxtext-ce-s8k-bs1` · xprof [`http://localhost:8791/?run=2026-06-02-qwen3-jax-v033-maxtext-ce-s8k-bs1`](http://localhost:8791/?run=2026-06-02-qwen3-jax-v033-maxtext-ce-s8k-bs1) · GCS `.../plugins/profile/2026_06_02_17_40_21/` (steps 12–14). On-disk: [`raw/profiles/2026-06-02-qwen3-jax-v033-maxtext-ce-s8k-bs1/`](../../../../../raw/profiles/2026-06-02-qwen3-jax-v033-maxtext-ce-s8k-bs1/). Profile-analyzer dispatched to attribute the CE-backward bucket (custom_vjp vs autodiff log_softmax) for the bs2 (v034) memory-transient question.

## HLO Dump

- **GCS**: `.../2026-06-02-qwen3-jax-v033-maxtext-ce-s8k-bs1/hlo/`.

## Verdict

**Supported** (the parity/no-regression claim it tested). The MaxText custom_vjp CE is **parity-clean** at
bs1 seq8192 — +0.4% (within noise), loss-trajectory-identical, semantics preserved. The CE swap introduces
no regression, so it is a sound substrate for the bs2 wall-test (v034). It is **not a frontier-mover at
bs1** (the math is identical to `_ce`; the difference is purely the backward graph, which doesn't matter
when the CE backward isn't on the bs1 critical path) — **seq8192 frontier stays v028 (32.3% / 5,632), now
tied by v033**. The real question — whether the custom_vjp's lighter backward transient lets **bs2**
amortize — is v034.

## Next hypotheses

- [MaxText CE @ bs2](2026-06-02-v034-maxtext-ce-s8k-bs2.md) — **dispatched next.** Same config, `--batch_size=2`. The decisive seq8192-batch-wall test: does the custom_vjp's single-pass backward beat v031 plain-CE bs2 (5,553) and v028 bs1 (5,632)? > 5,632 = wall cracked; ≤ 5,553 = CE-independent wall (documented hard wall stands).
