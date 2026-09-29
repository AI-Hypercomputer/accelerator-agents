---
title: "Qwen3 jax — host-offload of decoder-input + projections to fit bs3 @ seq8192"
type: hypothesis
model: qwen3-cc-jax
variants: ["8B/v6e-8"]
status: inconclusive
expected_gain: "+20-30% tok/s/chip @ seq8192 (close the gap to MaxText 6,942)"
confidence: medium-high
effort: M
origin: 2026-06-02-maxtext-qwen3-8b-v6e8-ref-seq8192
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3 jax — host-offload to fit bs3 @ seq8192

The MaxText reference run proved the **achievable seq8192 ceiling is 45.3% MFU / 6,942 tok/s/chip** —
**+31% tok/s/chip** over our jax frontier (v009, 30.4% / 5,305, bs1). The decisive lever MaxText uses and
the jax lane never tried: **host-DRAM offload of the decoder-layer input + the QKV/out projections**
(`decoder_layer_input=offload` + `out/query/key/value_proj=offload`, with `remat_policy=custom`), which
frees enough HBM to fit **bs3 at seq8192**. Our jax lane was stuck at bs1 there (remat alone insufficient;
tokamax-CE enabled bs2 but didn't raise throughput — [v016](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v016-s8k-ce-bs2.md)).

## Statement

Adding MaxText-style host-offload (offload decoder-layer input + the four attention projections to host
DRAM during the forward, recompute/reload on backward) to the jax trainer fits **bs3 (global 24) at
seq8192** and raises tok/s/chip materially above v009's 5,305 — target ≥ +20% (toward the MaxText 6,942
ceiling), with loss parity and no semantic change.

## Rationale

- **Reference proof**: MaxText hits 45.3% @ seq8192 bs3 on the identical architecture/hardware via exactly
  this offload recipe ([reference baseline](../experiments/qwen3_cc_autoresearch_optimization/maxtext/experiments/2026-06-02-maxtext-qwen3-8b-v6e8-ref-seq8192.md)). This is not speculative — it's a measured, transferable lever.
- **Category match**: this is a memory/scheduling (host-offload) lever — the *compiler/runtime* category that
  transfers to qwen3-jax (XLA scheduler flags gave +3.4 pp / v018), unlike kernel-swap levers that regress.
- The jax v018 profile already shows host-offload machinery active (`host_transfer_overlap_limit=24`,
  `copy-start/done` to S(4) host DRAM) — the infra exists; this hypothesis *directs* it at the
  layer-input + projection tensors to unlock the bigger batch, rather than just remat tensors.

## Proposed experiment

Code change (via `/edit-model-code`): add activation/projection host-offload to the jax trainer's decoder
layer (offload the layer input + Q/K/V/out projection activations to host, reload on backward), gated by a
flag; combine with the existing remat. Re-run `test_equivalence.py` for parity, build a tagged image, run
at seq8192 bs2 then bs3. Measure tok/s/chip vs v009 (5,305) and the MaxText ceiling (6,942).

## Risks

- JAX host-offload (`jax.experimental` device_put to host / `Format` memory_kind="pinned_host") is fiddlier
  than MaxText's integrated path; the port may not capture the full benefit. Effort M, possibly L.
- If bs3 still OOMs or the offload overhead isn't repaid, falls back to documenting the residual gap.

## Dependencies

Trainer/model edit + image build + equivalence re-verification. Supersedes the earlier "jax at ceiling"
read for the seq8192 frontier.

## Outcome — INCONCLUSIVE (bs3 not reached; throughput thesis undermined) — 2026-06-02

Explored via v025 (offload-only, bs2) and v026 (offload + tokamax-CE, bs3):
- **offload_remat mechanism VALIDATED**: `offload_dot_with_no_batch_dims(pinned_host)` activates and frees no-batch-dim activation HBM; CPU equivalence test passes; default-off `--offload_remat` flag committed to TRAINER_DIR.
- **bs3 NOT fitted**: v025 (offload, bs2) OOM'd +7.55G (LM-head logit wall); v026 (offload+CE, bs3) OOM'd by only **+2.34G** — CE removed the logit wall, offload freed no-batch dots; the last blocker is the batch-dim layer/projection activations (`bf16[B,8192,4096]`), which need MaxText's *named* offload (`decoder_layer_input`+`*_proj=offload`) — not yet ported (invasive).
- **Throughput thesis undermined**: even if bs3 fit, [v016](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v016-s8k-ce-bs2.md) shows bs2 (29.5%) < bs1 (30.4%) at seq8192 — our stack does NOT amortize batch there (CE-streaming + splash-backward costs). MaxText's bs3=45.3% reflects its per-token efficiency, not batch-fitting. So the ~31% gap is **structural** (MaxText's integrated pipeline), not closable by porting offload to enable a bigger batch.

**Verdict**: seq8192 frontier stays [v009](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v009-splash-s8k-vmem.md) (30.4%). Remaining test if revisited: named-offload to actually fit + measure bs3 (low payoff-prior given the amortization evidence).
