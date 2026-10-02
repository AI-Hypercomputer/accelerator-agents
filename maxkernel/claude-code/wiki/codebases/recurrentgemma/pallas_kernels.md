---
title: "recurrentgemma — Pallas kernels"
type: concept
tags: [pallas, kernels, tpu, recurrentgemma]
created: 2026-07-04
updated: 2026-07-04
---

RecurrentGemma defines **exactly one Pallas kernel, `mosaic_tpu`**: `linear_rnn_pallas_kernel` in
[`recurrentgemma/jax/pallas.py`](../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py) — the
canonical, first-principles **linear-recurrent-unit (LRU) scan** `h_t = a_t · h_{t-1} + x_t` that backs the
Griffin RG-LRU block. It is **novel-to-this-repo** (not vendored from jax/tokamax; it is instead the *ancestor*
of Apple AxLearn's Mamba Pallas scan) and is distinguished by two features rare in public Pallas kernels:
support for **complex accumulators** (`complex_lib.Complex`) and a **multi-shard correction** path for
sequence-parallel training. The sibling `scan.py` provides `jax.lax.scan` and `associative_scan` fallbacks —
those are **not** Pallas.

## Kernels

### SSM / scan

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `linear_rnn_pallas_kernel` (inner `linear_rnn_loop_body`) | mosaic_tpu | LRU scan `h_t = a_t·h_{t-1} + x_t`. **One kernel body computes both forward and backward** via a `backprop` flag that flips the index arithmetic (the VJP of a linear recurrence is itself a reversed linear recurrence with `a` shifted one step). 3-D grid (batch × hidden-dim × sequence, sequence innermost); carry refs (`h_carry_ref`, `a_prod_carry_ref`) persist across sequence-grid steps — zeroed only at `program_id(2)==0` — turning Pallas grid iteration into the outer scan loop, with an inner `jax.lax.fori_loop`. Optionally accumulates `prod(a)` for multi-shard correction. | [pallas.py#L520](../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L520) (loop body [#L477](../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L477); `pl.pallas_call` in `linear_rnn_pallas_call` [#L674](../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py#L674)) | stable | novel (ancestor of AxLearn Mamba scan) |

**Not Pallas (fallback scans in `scan.py`):** `lru_linear_scan` (`jax.lax.scan`) and `lru_associative_scan`
(`jax.lax.associative_scan`) are pure-XLA alternatives selected by `ScanType`; `LINEAR_PALLAS` resolves to the
kernel above via `lru_pallas_scan`. Listed here only to disambiguate — no `pl.pallas_call`.

## Notes

**Full mechanism** is documented in the grounded concept page
[recurrentgemma-jax-pallas](concepts/recurrentgemma-jax-pallas.md) (custom-VJP forward/backward split,
`compute_pallas_kernel_spec` grid/tile derivation, `multi_shard_correction` algebra, padding invariants). Key
points for reuse:

- **Complex accumulators.** Because Mosaic cannot lower native `jnp` complex, complex inputs are converted to
  the repo's `complex_lib.Complex` wrapper before the kernel; `get_acc_dtype` fixes the fp32 accumulation dtype.
  This is unusual — most public Pallas kernels are real-only.
- **Multi-shard (sequence-parallel) correction.** `linear_rnn_shard_corrected_pallas_call` runs every sequence
  shard's kernel independently with a *local* zero initial state (also tracking `a_prod`), then
  `multi_shard_correction` all-gathers each shard's boundary `(h_last, a_prod_last)` and folds back the missing
  `prod(a_per_shard) · h_from_previous_shard` term — a Blelloch-style prefix correction instead of a sequential
  cross-device dependency chain.
- **Padding invariant.** `pad_array_to_divisible` pads `x` with `0.0` but `a` with `1.0` (a multiplicative
  gate — padding `a` with `0` would truncate the decay and corrupt the correction math).

**Autotune / config surface** (via `single_shard_rnn_scan` / `pallas_lru`): `pallas_max_seq_block_size`
(default 256), `pallas_min_seq_block_size` (16), `pallas_pad_seq_to_min_block_size`,
`pallas_pad_last_dim_to_128`. `compute_tile_size` picks the largest power-of-two block ≤ max dividing the seq
length; hidden dim must be divisible by `singleton_tile_size=128` (hard `ValueError` otherwise).

**Hardware matrix.** `mosaic_tpu` only (with a GPU special-case in `compute_tile_size` that runs the whole
sequence in one block). No Mosaic-GPU or Triton kernel. Callers: `scan.py` → `layers.py` `RecurrentBlock` →
`griffin.py`.

**Directory catalog** row (subpage 04) matches source; commit pin `2efa84d` (2026-02-06). The kernel is the
**direct ancestor of AxLearn's `mamba_kernels.py`** SSM scan and remains the only public LRU Pallas scan built
from first principles for Griffin-family models.

## See also
- [recurrentgemma — overview](overview.md) — the repo's grounded wiki front door.
- [recurrentgemma-jax-pallas](concepts/recurrentgemma-jax-pallas.md) — the full grounded concept page for this kernel (mechanism, custom-VJP split, multi-shard correction).
- [pallas (catalog)](catalog/recurrentgemma/jax/pallas.md) — per-symbol catalog page (`linear_rnn_pallas_call`, `_lru`, `PallasKernelSpec`, `multi_shard_correction`).
- [Pallas kernel directory (catalog)](../../analyses/2026-04-23-pallas-kernel-directory.md) — the cross-repo Pallas catalog; this repo appears in [subpage 04 — Research labs](../../analyses/pallas-kernel-directory/04-research-labs.md).

## Sources
- [raw/code/recurrentgemma/recurrentgemma/jax/pallas.py](../../../raw/code/recurrentgemma/recurrentgemma/jax/pallas.py) — commit `2efa84d`
- [raw/code/recurrentgemma/recurrentgemma/jax/scan.py](../../../raw/code/recurrentgemma/recurrentgemma/jax/scan.py) — fallback (non-Pallas) scans
