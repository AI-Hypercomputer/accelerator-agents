---
title: "Qwen3 jax — scan-over-layers"
type: hypothesis
model: qwen3-cc-jax
variants: ["8B/v6e-8"]
status: open
expected_gain: "compile-HBM ↓; MFU neutral-to-small; may enable denser shapes"
confidence: low
effort: M
origin: 2026-06-02-v021-splash-bq1024
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3 jax — scan-over-layers

The model currently runs a **Python for-loop over 36 transformer layers** (unrolled into the HLO).
Replacing it with `nnx.scan` (a single scanned layer body over a stacked-param leading axis) is the
structural lever the llama3-jax sibling used. It cuts compile time and compile-time HBM, and can let
XLA schedule a tighter steady-state loop.

## Statement

Converting the layer stack to `nnx.scan` preserves loss (semantics-exact) and either improves MFU on
the v018 frontier or enables a denser shape (e.g. bs>4 at seq2048, or bs>1 at seq8192) that the
unrolled graph couldn't compile — net frontier gain ≥ +1 pp or a new feasible variant.

## Rationale

- **Category match**: qwen3-jax has shown that *compiler/structural* levers transfer from llama3-jax
  (XLA scheduler flags +3.4 pp / v018) while *kernel-swap* levers do not (tokamax-splash −8.5% / v020,
  SparseCore-offload refuted). scan is structural → in the transferable category.
- Unrolled 36-layer graph is large; scan shrinks the compiled program and reduces live activation
  ranges, which the v018 memory-pressure-aware scheduler may exploit.
- llama3-jax used scan as part of its 43.3% stack.

## Proposed experiment

Code change via `/edit-model-code`: wrap the decoder layer stack in `nnx.scan` (stack per-layer params
on a new leading axis; share the scanned body). Re-run the CPU equivalence test (`test_equivalence.py`)
to confirm loss/grad parity BEFORE TPU dispatch. Build a tagged image FROM the splash image, smoke-test
`--help`, dispatch on the v018 frontier shape. Measure MFU + compile time + loss parity; also probe
whether it unlocks a denser shape.

## Risks

- nnx.scan requires homogeneous layers + stacked params — a real refactor; correctness risk (must pass
  the equivalence test, else `invalid`).
- MFU upside is uncertain (scan is often compile-time-only); the win may be "enables a shape" not "+MFU".

## Dependencies

Trainer/model edit + image build + equivalence re-verification.

## Re-prioritized by the MaxText teardown (2026-06-02)

The [MaxText vs jax MFU-gap analysis](../analyses/2026-06-02-maxtext-vs-jax-qwen3-8b-mfu-gap.md) upgrades scan
from "low-prior, non-dominant" to **the structural enabler** of the seq8192 gap-closing package. The MaxText
profile (45.3% MFU) shows its advantage is (a) MXU occupancy 61.2% vs our 48.3% — fewer/larger tiled dots
from scan-compiled layers — and (b) reduce-scatter overlapped to 4.0% vs our synchronous 12.9%. Scan is the
graph structure on which BOTH land: it produces the larger dots AND the schedule the async-collective-fusion
flags need (our v023 async-fusion regressed precisely because it ran on the *unrolled* graph).

**Scan is necessary but not sufficient alone** — it is lever #2 of a 3-part package (named-offload to fit
bs3 + scan + collective-overlap flags), which must land together. Effort L (real NNX `nn.scan` refactor +
equivalence re-verification + named-offload composition). See the analysis page for the ranked plan.

## Scoping finding (2026-06-02) — scan needs stacked-param STORAGE, not a forward shim

Validated that `nnx.split` each layer → `jnp.stack` states → `jax.lax.scan` one body is numerically exact
(toy: max|unrolled−scan| = 0.0). BUT this "stack at forward" shortcut is **infeasible for perf**: stacking
the 36 layers' weights into a `[36,…]` array at runtime materializes a **full ~16 GB second copy** of all
layer params (alongside the optimizer's originals) → OOM. So scan **cannot** be a low-blast-radius forward
shim; it requires **storing** params stacked (leading layer axis), i.e. the full refactor:
- `modeling_qwen3.py`: build ONE decoder layer with `[36,…]` params (nnx.vmap-init) + `nnx.scan` forward.
- `weight_loader.py`: HF `model.layers.{i}.*` → `stacked_layer.*[i]` (rework navigation).
- `sharding.py`: prepend a (replicated) layer axis to every per-layer rule.
- `test_equivalence.py`: rework the per-layer grad-by-HF-name comparison.

Multi-hour, high-risk, 4-file, equivalence-gated. Payoff at the bs1 shape that fits is **modest/uncertain**
(scan's clear benefit is enabling collective-overlap, ~+a few pp on the RS bucket; the big MXU/bs3 gains are
confounded with batch + blocked by the CE-weight wall, [v027](../experiments/qwen3_cc_autoresearch_optimization/jax/experiments/2026-06-02-v027-named-offload-ce-s8k-bs3.md)).
