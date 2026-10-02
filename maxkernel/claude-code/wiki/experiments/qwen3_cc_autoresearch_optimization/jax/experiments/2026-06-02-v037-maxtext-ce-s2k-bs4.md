---
title: "Qwen3-8B jax v037 — MaxText custom_vjp CE @ seq2048 bs4 (universality check)"
type: experiment
hypothesis: qwen3-jax-maxtext-ce-seq2048
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: jax-maxtext-ce-2026-06-02
status: refuted
verdict: refuted
tags: [qwen3-cc, jax, cross-entropy, maxtext, custom-vjp, splash, remat, seq2048, bs4, v6e-8]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-8B jax v037 — MaxText custom_vjp CE @ seq2048 bs4

## Hypothesis under test

**Hypothesis**: maxtext-CE (the seq8192 wall-cracker, v034) is **neutral-or-better** at the seq2048 frontier
(v018, 35.8% / 6,964), making it the lane's universal default CE.

**Mechanism**: image `qwen3-8b-jax:v033-maxtext-ce`. The exact v018 seq2048 shape (`--use_remat --use_splash
--batch_size=4 --seqlen=2048`, NO scan — scan regresses at seq2048, v032) + `--use_maxtext_ce`. 7 scheduler +
4 overlap flags.

**Falsification criterion**: < v018 6,964 beyond noise ⇒ maxtext-CE regresses at seq2048 ⇒ it is NOT the
universal default; keep plain `_ce` at seq2048.

## Setup

- v6e-8 (fsdp=8), image `qwen3-8b-jax:v033-maxtext-ce`; `--use_remat --use_splash --use_maxtext_ce
  --batch_size=4 --seqlen=2048` + 7 scheduler + 4 overlap. Confirmed: `[attn] splash ON`, `[remat]
  nothing_saveable ON`, `[ce] MaxText/T5X custom_vjp ... ON`; `per_chip_batch=4 global_batch=32`. (No scan.)

## Results — REGRESSES at seq2048

| run | config | tok/s/chip | MFU | steady step (ms) |
|-----|--------|-----------|-----|------------------|
| **v018** | seq2048 bs4, plain `_ce` | **6,964** | **35.8%** | ~1,176 |
| v037 | seq2048 bs4, **maxtext-CE** | 6,486 | 33.4% | 1,263 |

v037 = **−6.9% vs v018** (step +7.4%). Loss 12.0986 → 12.0723 monotone, no NaN, exit 0.

## Profile

- **Run name**: `2026-06-02-qwen3-jax-v037-maxtext-ce-s2k-bs4` · xprof [`http://localhost:8791/?run=2026-06-02-qwen3-jax-v037-maxtext-ce-s2k-bs4`](http://localhost:8791/?run=2026-06-02-qwen3-jax-v037-maxtext-ce-s2k-bs4) · GCS `.../plugins/profile/2026_06_02_18_34_36/` (steps 12–14). On-disk: [`raw/profiles/2026-06-02-qwen3-jax-v037-maxtext-ce-s2k-bs4/`](../../../../../raw/profiles/2026-06-02-qwen3-jax-v037-maxtext-ce-s2k-bs4/). (No deep profile pass — the headline regression is decisive.)

## HLO Dump

- **GCS**: `.../2026-06-02-qwen3-jax-v037-maxtext-ce-s2k-bs4/hlo/`.

## Verdict

**Refuted** (< v018 beyond noise). maxtext-CE **regresses −6.9% at seq2048** — it is **NOT** the universal
default. The custom_vjp materializes a full one-hot `[BL,V]` f32 (here `[8192,151936]`≈5G) and saves
`exp_shifted`/`log_softmax` residuals; at seq2048 the logits are 4× smaller and the step is short (~1.2 s), so
that overhead is not repaid — whereas at seq8192 the same custom_vjp *lowers* the backward transient enough to
amortize batch (v034). **maxtext-CE is seq8192-specific** (like scan, v032). Conclusion for the lane:
`--use_maxtext_ce` at seq8192, plain `_ce` at seq2048. **seq2048 frontier stays v018 (35.8% / 6,964).**

## Next hypotheses

- None — maxtext-CE characterized at both sequence lengths (seq8192 win / seq2048 regress). Closes the
  maxtext-CE arc; see the [closing analysis](../../../../analyses/2026-06-02-qwen3-cc-jax-maxtext-ce-climb.md).
