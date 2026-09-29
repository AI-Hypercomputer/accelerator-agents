---
title: "Qwen3 jax — MLP-only activation checkpointing (skip splash layers)"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: retired
expected_gain: "HBM headroom for bs6+ (frees ~8 GiB/step vs current offload_attn); net tok/s/chip depends on whether freed HBM enables a higher-batch op-point that offsets remat tax"
confidence: medium
effort: M
origin: 2026-06-12-v028-splash-blocks-bs4
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — MLP-only activation checkpointing (skip splash layers)

> **Retired 2026-06-12**: its purpose was the bs6 unlock, and
> [v029](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v029-bs5-probe.md)
> closed the batch axis at bs4 (bs5 fits but −0.5%) — more batch has no
> throughput payoff on this substrate. Reopen only if a future lever
> changes the per-step amortization curve.

*Hypothesis*: Applying `jax.checkpoint` only to the MLP sub-layer of each
decoder block — explicitly skipping the splash attention sub-layer — recovers
~8 GiB of activation HBM per step (MLP intermediates: gate, up, down, intermediate
activations for each of 32 layers) without triggering the v012 NaN that occurred
when splash outputs were caught by the dots-saveable policy. If MLP-only remat
provides sufficient headroom, bs6 @ seq8192 becomes feasible, potentially
delivering +3–5% tok/s/chip above bs4 (assuming the RS-fusion bucket continues
its near-fixed-cost behavior).

*Mechanism*: The v012 NaN was traced to the `dots_with_no_batch_dims_saveable`
policy recomputing splash attention outputs (which have non-trivial numerical
sensitivity). MLP-only checkpointing avoids this by using a `checkpoint_name`-
gated policy that only checkpoints ops inside the `mlp_block` name scope:
`jax.checkpoint(mlp_fn, policy=jax.checkpoint_policies.checkpoint_name('mlp_*'))`.
The scan loop in trunk already wraps each layer — the checkpoint boundary aligns
with the per-layer scan body.

*Rationale*: At bs4 (current frontier), HBM = 23.72 GiB (75.9% cap). bs5
projects 27.78 GiB (89%). bs6 projects ~31.84 GiB — would OOM. MLP intermediates
at bs4/seq8192/32-layers are approximately `[B, S, 4H]` × 3 (gate, up, down) +
intermediate = ~8 GiB/step (4 × 8192 × 22016 × 3 × 2 bytes × 32 layers =
~7.5 GiB). Recompute tax: one extra MLP forward per backward layer = +~3% step
(estimate from v017's partial-remat data: 0.43 GiB/exempt-layer = ~1.5 remat tax
per-layer). Net at bs5: may break even; at bs6: ~+6% raw batch amortization −3%
remat tax = +3% net.

*Falsification criterion*: (a) bs5 already clears without remat (see
[qwen3-jax-bs5-probe](qwen3-jax-bs5-probe.md) — if bs5 fits at 27.78 GiB, this
hypothesis is only needed for bs6), OR (b) the remat policy NaNs again (policy
scope too broad), OR (c) the tok/s/chip gain from the unlocked batch is < the
remat overhead (net < 0).

*Dependencies*: Should run after [qwen3-jax-bs5-probe](qwen3-jax-bs5-probe.md)
to establish whether bs5 needs it. Requires a code change to the scan body in
the trainer (M effort: add `--remat_policy=mlp_only` branch).

## See also

- [v012 (dots-saveable policy NaN + +27% overhead)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v012-remat.md)
- [v017 (partial remat, ~0.43 GiB/exempt layer)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v017-partial-remat.md)
- [qwen3-jax-bs5-probe](qwen3-jax-bs5-probe.md)
- [v028 (splash configs closed — motivates this direction)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v028-splash-blocks-bs4.md)
