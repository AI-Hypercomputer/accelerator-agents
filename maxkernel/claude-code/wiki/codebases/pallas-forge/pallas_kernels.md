---
title: "pallas-forge — Pallas kernels"
type: concept
tags: [pallas, kernels, tpu, pallas-forge]
created: 2026-07-04
updated: 2026-07-04
---

pallas-forge is primarily an **autotune harness** (its grounded concepts are the tune-runner /
tune-config / tune-report trio), but its `pallas_forge/kernels/` package ships **4 self-contained,
novel-to-this-repo example Pallas kernels** — all `mosaic_tpu`, all pedagogical "worked example"
quality (they exist to be tuned and to teach the TPU tiling/alignment rules, not vendored from
jax/tokamax). One matmul, one memory-bound fused norm, and two compute-bound fused GLU activations.
Every kernel is `@jax.jit`-wrapped with block sizes as static (traced-once-per-config) tuning knobs.

## Kernels

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `tiled_matmul` | mosaic_tpu | Blocked `x @ w` on a 3D grid `(grid_m, grid_n, k_tiles)` — K iterated by the grid (no in-kernel `dynamic_slice`); fp32 accumulator kept in VMEM across K tiles via `o_ref[...] +=`, zeroed by `pl.when(program_id(2)==0)`; pads to block-aligned, casts to input dtype in one post-pass. Knobs: `block_m` (×8), `block_k`/`block_n` (×128), `num_stages`. | [matmul.py](../../../raw/code/pallas-forge/pallas_forge/kernels/matmul.py#L32) · [catalog](catalog/pallas_forge/kernels/matmul.md) | stable (example) | novel |
| `fused_rmsnorm_residual` | mosaic_tpu | Memory-bound VPU kernel fusing `new_residual = x + residual` then `rmsnorm(new_residual) * weight` in a single HBM pass; 1D grid over 8-token tiles (`TOKENS_PER_TILE=8` satisfies the sublane-alignment rule); returns `(output, new_residual)`. Demonstrates VPU-vs-MXU tradeoff and when fusion beats XLA auto-fusion. | [rmsnorm.py](../../../raw/code/pallas-forge/pallas_forge/kernels/rmsnorm.py#L34) · [catalog](catalog/pallas_forge/kernels/rmsnorm.md) | stable (example) | novel |
| `fused_swiglu` | mosaic_tpu | Compute-bound fused gated activation `silu(x @ w_gate) * (x @ w_up)` on a 2D grid `(grid_m, grid_n)`; both projections computed in-kernel with `preferred_element_type=fp32`, saving two HBM round-trips for the gate/up intermediates. Knobs: `block_m`, `block_n`, `num_stages`. | [swiglu.py](../../../raw/code/pallas-forge/pallas_forge/kernels/swiglu.py#L124) · [catalog](catalog/pallas_forge/kernels/swiglu.md) | stable (example) | novel |
| `fused_geglu` | mosaic_tpu | GELU variant of the above — `gelu(x @ w_gate) * (x @ w_up)`. Shares the single `_swiglu_kernel` body via `use_gelu=True`; same grid/knobs as `fused_swiglu`. | [swiglu.py](../../../raw/code/pallas-forge/pallas_forge/kernels/swiglu.py#L157) · [catalog](catalog/pallas_forge/kernels/swiglu.md) | stable (example) | novel |

**Imported (not defined here):** none — pallas-forge's kernels are all locally authored. Its value-add
over the kernels themselves is the *tuning* layer (`pallas_forge.tune`) that sweeps the block-size and
`num_stages` knobs and produces roofline reports.

## Notes
- **Vendoring lineage:** none. These are original teaching kernels (the matmul docstring calls itself
  "the *hello world* of Pallas"). They are the tuning targets for the harness, not production kernels.
- **Autotune surface:** every kernel exposes its block sizes (`block_m` ×8, `block_k`/`block_n` ×128 on
  TPU) and `num_stages` (DMA pipeline depth) as `static_argnames`, so each config traces + lowers
  exactly once. `num_stages` is forwarded through `pallas_call_compat` (`_compat.py`) and silently
  dropped on JAX versions that don't expose the knob — a portability shim, not a kernel feature.
- **Alignment rules encoded:** `block_k`/`block_n` divisible by 128 (last-dim lane), `block_m` divisible
  by 8 (sublane); the norm kernel hard-codes `TOKENS_PER_TILE=8`. Wrappers pad up and slice back so
  callers can pass non-aligned shapes.
- **Numerical care:** matmul keeps the K-axis accumulator in fp32 in VMEM and casts once at the end
  (avoids per-tile bf16 rounding); GLU kernels use `preferred_element_type=jnp.float32` on both `jnp.dot`s.
- **Reconciliation with the cross-repo directory (2026-04-23):** pallas-forge is listed there only under
  the `pedagogical` bucket (~10 sources) and as an already-ingested "roofline+autotune" codebase, with a
  noted **forward-only (no `custom_vjp`) blocker** — the kernels have no backward pass, so they can't drop
  into a training step as-is. Source has not materially diverged since that survey.

## See also
- [pallas-forge — overview](overview.md) — the repo's grounded wiki front door (autotune-harness framing).
- [pallas-forge — perf surfaces](../pallas-forge.md) — hand-curated codebase page.
- [tune-runner](concepts/pallas_forge-tune-runner.md) · [tune-config](concepts/pallas_forge-tune-config.md) · [tune-report](concepts/pallas_forge-tune-report.md) — the autotune layer that sweeps these kernels' knobs.
- [Pallas kernel directory (catalog)](../../analyses/2026-04-23-pallas-kernel-directory.md) — cross-repo catalog; pallas-forge appears in its `pedagogical` bucket, not a dedicated lab subpage.
