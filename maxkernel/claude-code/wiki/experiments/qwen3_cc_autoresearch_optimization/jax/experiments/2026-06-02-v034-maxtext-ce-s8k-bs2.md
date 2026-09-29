---
title: "Qwen3-8B jax v034 — MaxText custom_vjp CE @ seq8192 bs2 (the batch-wall test)"
type: experiment
hypothesis: qwen3-jax-maxtext-ce
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: jax-maxtext-ce-2026-06-02
status: supported
verdict: supported
tags: [qwen3-cc, jax, cross-entropy, maxtext, custom-vjp, scan, overlap, splash, remat, seq8192, bs2, v6e-8]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-8B jax v034 — MaxText custom_vjp CE @ seq8192 bs2

## Hypothesis under test

**Hypothesis**: the documented seq8192 batch wall (plain-CE bs1 5,632 > bs2 5,553 (v031) > bs3 4,595
(v030)) is partly the CE backward's transient peak. MaxText's `custom_vjp` CE computes the logit
gradient in a single fused pass (`softmax − onehot`) rather than autodiff through `log_softmax`,
lowering that transient. If the bs>1 regression was CE-backward-transient-bound, **bs2 with maxtext-CE
beats v028 bs1 (5,632)** — the wall cracks. If it's collective/attention-bound, bs2 ≤ v031's 5,553 —
the CE swap doesn't move the wall.

**Mechanism**: image `qwen3-8b-jax:v033-maxtext-ce`. Config = v031 shape (bs2, no offload) but with
maxtext-CE instead of tokamax-CE: `--use_remat --use_splash --use_scan --use_maxtext_ce --batch_size=2
--seqlen=8192` + 7 scheduler + 4 overlap flags.

**Predicted signal**: bs2 tok/s/chip vs v031 plain-CE bs2 (5,553) and v028 bs1 (5,632).

**Falsification criterion**: ≤ v031 5,553 beyond noise ⇒ maxtext-CE doesn't change the batch wall
(wall is not CE-backward-bound) ⇒ the documented hard wall stands; only a pipelined offload kernel
remains. > 5,632 ⇒ wall cracked, pursue bs3.

## Setup

- **Hardware**: v6e-8 (fsdp=8), image `qwen3-8b-jax:v033-maxtext-ce`. Config = v033 + `--batch_size=2`:
  `--use_remat --use_splash --use_scan --use_maxtext_ce --batch_size=2 --seqlen=8192` + 7 scheduler + 4
  async-collective-fusion overlap flags. NO offload, NO tokamax CE. All four levers confirmed active
  (`[scan]/[attn]/[remat]/[ce]` ON); `[train] starting ... per_chip_batch=2 global_batch=16`.

## Results — WALL CRACKED (new seq8192 frontier)

| run | config | tok/s/chip | MFU | steady step (ms) |
|-----|--------|-----------|-----|------------------|
| v028 | scan+overlap, **bs1**, `_ce` | 5,632 | 32.3% | 1,452 |
| v031 | scan+overlap+CE, **bs2**, tokamax-CE (no offload) | 5,553 | 31.9% | 2,950 |
| **v034** | scan+overlap, **bs2**, **maxtext-CE** | **5,992** | **34.4%** | 2,732 |
| MaxText | bs3 reference | 6,942 | 45.3% | 3,546 |

v034 = **+6.4% vs v028 bs1 (5,632)** and **+7.9% vs v031 plain/tokamax-CE bs2 (5,553)**. Loss 12.1000 →
12.0575 monotone over 20 steps, no NaN/Inf, exit 0. Step 0 compile 39.2 s.

**Batch now amortizes at seq8192** — bs2 (5,992) beats bs1 (5,632), reversing the prior monotone
anti-amortization (v028 bs1 > v031 bs2 > v030 bs3). The only change vs v031 is the CE path:
MaxText's `custom_vjp` (single-pass fused `softmax − onehot` backward) instead of the tokamax streamed
kernel / autodiff log_softmax. So the prior seq8192 batch wall was **CE-backward-bound**, not
collective/attention-bound as the [closing analysis](../../../../analyses/2026-06-02-qwen3-cc-jax-maxtext-closing.md)
concluded — that conclusion is now corrected (see its contradiction block).

## Profile

- **Run name**: `2026-06-02-qwen3-jax-v034-maxtext-ce-s8k-bs2` · xprof [`http://localhost:8791/?run=2026-06-02-qwen3-jax-v034-maxtext-ce-s8k-bs2`](http://localhost:8791/?run=2026-06-02-qwen3-jax-v034-maxtext-ce-s8k-bs2) · GCS `.../plugins/profile/2026_06_02_17_53_14/` (steps 12–14). On-disk: [`raw/profiles/2026-06-02-qwen3-jax-v034-maxtext-ce-s8k-bs2/`](../../../../../raw/profiles/2026-06-02-qwen3-jax-v034-maxtext-ce-s8k-bs2/). Profile-analyzer to attribute the CE-backward bucket (custom_vjp vs the v033/v031 baselines) — pending.

## HLO Dump

- **GCS**: `.../2026-06-02-qwen3-jax-v034-maxtext-ce-s8k-bs2/hlo/`.

## Verdict

**Supported — NEW seq8192 frontier (34.4% / 5,992 tok/s/chip, bs2).** The MaxText custom_vjp CE is the
lever that lets batch amortize at seq8192: same config as v031 except the CE path, and bs2 jumps from
5,553 → 5,992 (now > bs1). The "documented hard wall" (v030/v031) was an artifact of the
tokamax/autodiff CE backward's transient, not a structural batch limit. Gap to MaxText (6,942) narrows to
**86.3%** (was 81%). Semantics preserved (loss-trajectory matches, monotone, no NaN; CPU check proved the
CE is bit-identical math). Pursuing bs3 next (v035) — the climb toward MaxText is reopened.

## Next hypotheses

- [MaxText CE @ bs3](2026-06-02-v035-maxtext-ce-s8k-bs3.md) — **dispatched.** Same config, `--batch_size=3`,
  NO offload. If bs2 amortizes, bs3 may amortize further (and maxtext-CE drops the ~4.6G tokamax f32[H,V]
  weight-gather that forced v030's offload, so bs3 may now fit without offload). Target > v034 5,992 toward
  MaxText 6,942.
- maxtext-CE at **seq2048** — re-test the v018 seq2048 frontier (35.8%) with maxtext-CE; the CE-backward
  share is smaller at seq2048 so likely neutral, but cheap to confirm now that the CE is a proven win.

