---
title: "Qwen3 jax — tokamax-splash perf knobs"
type: hypothesis
model: qwen3-cc-jax
variants: ["8B/v6e-8"]
status: refuted
expected_gain: "1-4% MFU"
confidence: medium
effort: S
origin: 2026-06-02-v019-xla-flags-s8k
hlo_prefilter: passed
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3 jax — tokamax-splash perf knobs

The lane currently runs **upstream JAX** splash (`jax.experimental.pallas.ops.tpu.splash_attention`).
The trainer's `splash_attn.py` already supports the **tokamax-shipped** splash kernel + its perf
knobs (`use_base2_exp`, `fuse_reciprocal`, `max_logit_const`) behind `USE_TOKAMAX_SPLASH=1` (default
OFF). On the llama3-jax sibling lane this swap + knobs was a **+4.4%** win. Flag-only — the
`qwen3-8b-jax:v006-splash` image already has tokamax importable (confirmed v007); no rebuild.

## Statement

Enabling `USE_TOKAMAX_SPLASH=1` (with its default knobs `use_base2_exp`/`fuse_reciprocal` on) on the
v018 seq2048 frontier raises MFU ≥ +1 pp over 35.8%, with loss parity (the knobs are numerically
equivalent reformulations: base-2 exp, fused reciprocal of the softmax denominator).

## Rationale

- **Sibling-model evidence**: llama3-jax exp 72a — replacing upstream splash with tokamax-splash +
  `use_base2_exp=True, fuse_reciprocal=True` gave +4.4% over the `attention="flash"` path; both knobs
  contribute independently (~+0.7–0.8% each) and compound. The retrospective's transferable lever #4.
- `base2_exp` maps `exp(x)` → `exp2(x·log2e)` (cheaper on the MXU/VPU); `fuse_reciprocal` folds the
  softmax-denominator reciprocal into the output matmul. Pure reformulations — no semantic change.
- Splash is ~11% of the v018 step → a kernel-level speedup lands directly on the frontier.

## Proposed experiment

v020: v018 frontier launch + env `USE_TOKAMAX_SPLASH=1` (knobs default on). Compare MFU/tok-s-chip vs
35.8%; verify loss parity (≤ noise) and exit 0. If it wins, it joins the frontier config.

## Risks

- tokamax-splash may pick different default block sizes → re-tune via `SPLASH_*` env if it OOMs the
  scoped VMEM (keep `--xla_tpu_scoped_vmem_limit_kib=98304`).
- Gain may be smaller than llama3's (different head config); falsified if MFU ≤ 35.8% beyond noise.

## Dependencies

None — env-var flag-only on the existing frontier image (`splash_attn.py` already wired).
