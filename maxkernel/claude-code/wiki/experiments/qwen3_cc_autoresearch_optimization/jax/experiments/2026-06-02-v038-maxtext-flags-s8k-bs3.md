---
title: "Qwen3-8B jax v038 — MaxText XLA flag superset @ seq8192 bs3 (flags-only probe)"
type: experiment
hypothesis: qwen3-jax-xla-flag-stack
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: jax-maxtext-ce-2026-06-02
status: supported
verdict: supported
tags: [qwen3-cc, jax, libtpu-xla-flags, maxtext, scan, splash, remat, seq8192, bs3, v6e-8]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-8B jax v038 — MaxText XLA flag superset @ seq8192 bs3

## Hypothesis under test

**Hypothesis**: adding the ~9 XLA flags MaxText uses that we lack — LAYOUT_FOR_ALL_REDUCE_SCATTER (3),
DATA_PARALLEL_OVERLAP (2), CF_FOR_ALL_GATHER extras (2), HOST_OFFLOAD extras incl the `*_in_chain` flags (4)
— to the v035 frontier (no offload, no code change) is neutral-or-better and, crucially, **validates that
these flags are accepted by our libtpu build** before they're combined with the offload recipe (v039).

**Mechanism**: image `qwen3-8b-jax:v033-maxtext-ce`, v035 config (`--use_remat --use_splash --use_scan
--use_maxtext_ce --batch_size=3 --seqlen=8192`) + existing 12-flag bundle + the 11 new MaxText flags.

**Predicted signal**: compiles clean (all flags valid); tok/s/chip ≥ v035 6,030 (the RS-layout / DP-overlap
flags may help reduce-scatter; the offload-pipelining flags are likely no-ops without offload).

**Falsification criterion**: a flag is rejected (5 s crash → report the name, prune for v039); or < 6,030
beyond noise (a flag regresses → ablate).

## Setup

- Image `qwen3-8b-jax:v033-maxtext-ce`, v035 config (`--use_remat --use_splash --use_scan --use_maxtext_ce
  --batch_size=3 --seqlen=8192`) + the 12-flag base bundle + 11 new MaxText flags (LAYOUT_RS×3,
  DATA_PARALLEL_OVERLAP×2, CF extras×2, HOST_OFFLOAD extras×4 incl `*_in_chain`). NO offload, NO code change.
  All `[scan]/[attn]/[remat:nothing_saveable]/[ce]` ON; `per_chip_batch=3 global_batch=24`.

## Results — PARITY; all flags valid

| run | flags | tok/s/chip | MFU | steady step (ms) |
|-----|-------|-----------|-----|------------------|
| v035 | base 12 | 6,030 | 34.6% | 4,075 |
| **v038** | base 12 + **11 MaxText** | **6,029** | 34.6% | 4,075 |

Δ = **−0.02% (parity)**. **All 25 LIBTPU_INIT_ARGS flags accepted — zero rejections** (the key result: the
full MaxText flag superset is valid on our libtpu build, no pruning needed for v039). Loss 12.0994 → 12.0508
monotone, no NaN, exit 0. Compile 32.5 s.

## Profile

- **Run name**: `2026-06-02-qwen3-jax-v038-maxtext-flags-s8k-bs3` · xprof [`http://localhost:8791/?run=2026-06-02-qwen3-jax-v038-maxtext-flags-s8k-bs3`](http://localhost:8791/?run=2026-06-02-qwen3-jax-v038-maxtext-flags-s8k-bs3) · GCS `.../plugins/profile/2026_06_02_23_51_11/`. On-disk: [`raw/profiles/2026-06-02-qwen3-jax-v038-maxtext-flags-s8k-bs3/`](../../../../../raw/profiles/2026-06-02-qwen3-jax-v038-maxtext-flags-s8k-bs3/).

## HLO Dump

- **GCS**: dump dir present but EMPTY — the `--xla_dump_to=gs://...` path was silently skipped (XLA GCS-scheme registration). Not material for this flags-only parity run. (Future HLO dumps need a local `--xla_dump_to=/tmp/...` + copy.)

## Verdict

**Supported** (the claim it tested: the flag superset is valid + non-regressing). All 11 new MaxText XLA flags
compile on our libtpu build, and they're **neutral without offload** — expected, since the offload-pipelining
flags (`*_in_chain`, `aggressive_opt_barrier_removal`, `lhs_prioritize_async_depth_over_stall`) only act when
there are host transfers to overlap, and the RS-layout / data-parallel-overlap flags don't move the
recompute-bound frontier. **Their value is as the substrate for v039** (the offload recipe), which is where
the `*_in_chain` flags are designed to pipeline the host copy across scan. Frontier unchanged (v035 6,030).

## Next hypotheses

- [MaxText exact offload recipe](2026-06-02-v039-maxtext-offload-recipe-s8k-bs3.md) — **dispatched.** The validated
  full flag set + the corrected offload (decoder_layer_input + qkvo, recompute mlpwi). The decisive test of
  whether MaxText's recipe pipelines on our stack.
