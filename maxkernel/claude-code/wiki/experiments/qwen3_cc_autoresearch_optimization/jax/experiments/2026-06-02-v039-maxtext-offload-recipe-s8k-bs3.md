---
title: "Qwen3-8B jax v039 — MaxText exact offload recipe @ seq8192 bs3"
type: experiment
hypothesis: qwen3-jax-maxtext-offload-recipe
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: jax-offload-recipe-2026-06-02
status: refuted
verdict: refuted
tags: [qwen3-cc, jax, remat, offload, host-offload, maxtext, scan, splash, seq8192, bs3, v6e-8]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-8B jax v039 — MaxText exact offload recipe @ seq8192 bs3

## Hypothesis under test

**Hypothesis**: v036's offload regressed because it offloaded the WRONG tensors (the giant mlpwi, missing
decoder_layer_input). Replicating MaxText's exact recipe — offload `decoder_layer_input + q/k/v/o proj`,
**recompute** mlpwi — plus the MaxText offload-pipelining XLA flags (validated by v038), lifts the seq8192
frontier above v035 (6,030) toward MaxText (6,942).

**Mechanism**: image `qwen3-8b-jax:v039-offload-recipe` (FROM v033-maxtext-ce, `modeling_qwen3.py` swap:
decoder_layer_input `checkpoint_name` tag + offload-names = {decoder_layer_input, query_proj, key_proj,
value_proj, out_proj}, mlpwi dropped). `--use_remat --use_splash --use_scan --use_maxtext_ce --offload_remat
--batch_size=3 --seqlen=8192` + the validated MaxText flag superset (from v038). Equivalence PASS
(checkpoint_name = identity).

**Predicted signal**: fits; loss ~12.05 matching v035; tok/s/chip > v035 6,030 toward 6,942 (offload of the
attention path + recompute of the cheap-recompute MLP, with the host copy overlapped across scan via the
`*_in_chain` flags).

**Falsification criterion**: ≤ v035 6,030 — even MaxText's exact recipe doesn't pipeline on our libtpu/scan
build → the host-offload path genuinely differs at the runtime level (then the gap really is a build/kernel
wall). OOM → the offload-set change shifted the HBM balance unexpectedly.

## Setup

- Image `qwen3-8b-jax:v039-offload-recipe` (FROM v033-maxtext-ce, `modeling_qwen3.py` swap: offload-names =
  {decoder_layer_input, query_proj, key_proj, value_proj, out_proj}, mlpwi dropped → recomputed; verified at
  build time). `--use_remat --use_splash --use_scan --use_maxtext_ce --offload_remat --batch_size=3
  --seqlen=8192` + the full validated 23-flag MaxText superset (base 12 + 11 from v038). offload confirmed
  active (`[remat] save_and_offload ... ON`). **Note**: the `[remat]` log label still reads "proj+mlpwi" — a
  stale hardcoded string in `train.py` (not updated with the model-code change); the *actual* offloaded set is
  the corrected {decoder_layer_input + qkvo}, verified in the image. equivalence PASS.

## Results — REGRESSES vs no-offload (but beats the wrong-offload v036)

| run | offload set | tok/s/chip | MFU | steady step (ms) |
|-----|-------------|-----------|-----|------------------|
| **v035** | none (recompute all) | **6,030** | **34.6%** | 4,075 |
| **v039** | **MaxText exact** (decoder_input + qkvo; recompute mlpwi) | 5,630 | 32.3% | 4,365 |
| v036 | wrong (qkvo + **mlpwi**) | 4,908 | 28.2% | 5,007 |
| MaxText | (offload, pipelined) | 6,942 | 45.3% | 3,546 |

v039 = **−6.6% vs v035** (no-offload), **+14.7% vs v036** (wrong set). Loss 12.0994 → 12.0509 monotone, no
NaN, exit 0, no OOM. Compile 40.2 s.

## Profile

- **Run name**: `2026-06-02-qwen3-jax-v039-maxtext-offload-recipe-s8k-bs3` · xprof [`http://localhost:8791/?run=2026-06-02-qwen3-jax-v039-maxtext-offload-recipe-s8k-bs3`](http://localhost:8791/?run=2026-06-02-qwen3-jax-v039-maxtext-offload-recipe-s8k-bs3) · GCS `.../plugins/profile/2026_06_02_23_57_48/` (steps 12–14). On-disk: [`raw/profiles/2026-06-02-qwen3-jax-v039-maxtext-offload-recipe-s8k-bs3/`](../../../../../raw/profiles/2026-06-02-qwen3-jax-v039-maxtext-offload-recipe-s8k-bs3/).

## HLO Dump

- Not dumped (the `--xla_dump_to=gs://` path silently no-ops; v038 confirmed). Not material — the headline regression is decisive.

## Verdict

**Refuted** (≤ v035 no-offload). This is now a **rigorous** result: the corrected offload set is +14.7% over the
mis-tested v036, confirming v036's regression was partly the wrong tensors — but **MaxText's exact recipe
(correct tensor set + the full flag superset incl the `*_in_chain` host-offload-in-loop flags) still regresses
−6.6% vs no-offload.** The host round-trip costs ~400 tok/s/chip net even when the offloaded tensors and the
pipelining flags exactly match MaxText. So **host-offload is genuinely not viable on our JAX/scan/libtpu stack**
— the <0.1% pipelined host-copy MaxText achieves does not materialize for us, even with its config. This is a
**runtime/build-level difference**, not a config error (now established by testing the exact recipe, closing the
ambiguity the v036 verdict left). **seq8192 frontier stays v035 (34.6% / 6,030).** The MaxText offload advantage
is unreachable by config on this stack.

## Next hypotheses

- [Splash bkv 1024→2048](2026-06-02-v040-splash-bkv2048-s8k-bs3.md) — MaxText uses `sa_block_kv=2048` (we use 1024); cheap untried lever at the maxtext-CE frontier. **Dispatched next.**
- `DISABLE_COLLECTIVE_MATMUL` alternate flag — the `jf_spmd_threshold` form crashed v019; if another flag disables windowed-einsum it could lift MXU (53.6%→61%). Catalogued.
- Why our scan+offload doesn't pipeline (diagnostic): offload on the UNROLLED graph (use_scan=False) — if it pipelines there but not with scan, scan is the blocker. Catalogued (expensive: 36× compile).
