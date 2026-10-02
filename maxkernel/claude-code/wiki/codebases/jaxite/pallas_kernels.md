---
title: "jaxite — Pallas kernels"
type: concept
tags: [pallas, kernels, tpu, jaxite, fhe]
created: 2026-07-04
updated: 2026-07-04
---

`jaxite` is the survey's **only non-ML Pallas author**: its four `pl.pallas_call` kernels (all
`mosaic_tpu`, all research-grade) implement **CGGI-style Fully Homomorphic Encryption (FHE)
polynomial multiplication** — negacyclic vector-matrix polymul mod `X^N + 1` over 32-bit integers,
built as `int32 → byte-split → bf16 matmul → f32 accumulate → shift-reassemble`. They live entirely
in `jaxite/jaxite_cggi/` (`polymul_kernel.py` + `matrix_utils.py`); the CKKS `ntt.py` and the
`jaxite_ec` elliptic-curve MSM code are pure JAX with no Pallas. All kernels are novel-to-this-repo.
Source paths diverged from the 2026-04-23 catalog (see Notes).

## Kernels

### CGGI polynomial multiplication (FHE)

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `toeplitz_kernelized` (`_toeplitz`) | mosaic_tpu | Builds an `(n,n)` negacyclic Toeplitz + sign matrix from a length-`n` vector using `pltpu.roll`. The 128-strided rolls lower to **register aliasing (zero hardware instructions)** — a comment notes the roll "produces no hardware instructions". TPU v5+, `n % 128 == 0`, input+output must fit VMEM. | [matrix_utils.py#L686](../../../raw/code/jaxite/jaxite/jaxite_cggi/matrix_utils.py#L686) · [catalog](catalog/jaxite/jaxite_cggi/matrix_utils.md) | research | novel |
| `_decomposed_vector_matrix_polymul` (`vec_mat_polymul_kernel`) | mosaic_tpu | **Headline kernel.** Fuses Toeplitz construction (in-kernel `pltpu.roll` + sign flip) with the CGGI-reduced integer matmul, computing decomposed vector×matrix polymul in one `pallas_call`. Pads the RLWE dim `m` to 8 (TPU sublane granularity; `pltpu.roll` requires multiple-of-8 sublanes). `vmem_limit_bytes` bumped to 32 MiB. | [polymul_kernel.py#L199](../../../raw/code/jaxite/jaxite/jaxite_cggi/polymul_kernel.py#L199) · [catalog](catalog/jaxite/jaxite_cggi/polymul_kernel.md) | research | novel |
| `negacyclic_vector_matrix_polymul_bat` (`_toeplitz_chunk`) | mosaic_tpu | Batched variant. `PrefetchScalarGridSpec` over `grid=(18,)` with `dimension_semantics=("parallel",)`; each grid step builds one batch's Toeplitz matrix, then `bat_matmul` does the reduced einsum `"cmnpq,cnkq->cmkp"`. Falls back to a vmap reference when `n % 128 != 0` or TPU < v5. | [polymul_kernel.py#L315](../../../raw/code/jaxite/jaxite/jaxite_cggi/polymul_kernel.py#L315) · [catalog](catalog/jaxite/jaxite_cggi/polymul_kernel.md) | research | novel |
| `i32_matmul` (`i32_matmul_unreduced`) | mosaic_tpu | Test-isolation helper: wraps the bare `_i32_matmul_unreduced` byte-split matmul core in a `pallas_call` so the matmul can be validated independently of the Toeplitz path. | [polymul_kernel.py#L347](../../../raw/code/jaxite/jaxite/jaxite_cggi/polymul_kernel.py#L347) · [catalog](catalog/jaxite/jaxite_cggi/polymul_kernel.md) | research | novel |

## Notes

**The integer-matmul trick.** TPU MXUs have no native `int32×int32` matmul, so the compute cores split
each 32-bit operand into four bytes, cast to `bfloat16`, matmul with `preferred_element_type=float32`
(f32 accumulate for precision), cast back to `int32`, then left-shift and sum the four partial products
to reconstruct the `uint32` result:
- `_i32_matmul_unreduced` (polymul_kernel.py#L32) — the general form: four byte-planes on each side,
  loops `rhs_shift in range(0, 32, 8)`, accumulating `left_shift(raw_out, out_shift_base + rhs_shift)`.
- `_i32_matmul_unreduced_cggi` (polymul_kernel.py#L85) — **the CGGI 4× optimization**. Source comment:
  *"rhs is the Toeplitz matrix of the decomposed vector. The values are small (e.g. < 64) because the
  decomposition base < 8. We can cast to bfloat16 directly without splitting into bytes, reducing the
  number of matmuls by 4x."* Only the LHS is byte-split; the small-valued RHS goes straight to bf16.
- `bat_matmul` (polymul_kernel.py#L63) — the batched core: `u8(c,m,n,4,4) @ u32(c,n,k)` via
  `jnp.einsum("cmnpq,cnkq->cmkp")` with per-byte shift-factors `[0,8,16,24]`.

**`pltpu.roll` register-aliasing is the perf hook.** Toeplitz construction would normally be an `O(n²)`
data-movement blow-up; here the 128-aligned rolls are compiled to register-index remapping and emit no
hardware instructions, so the Toeplitz matrix is materialized essentially for free inside VMEM.

**Hardware / shape matrix.** All kernels are gated on **TPU v5+** and `n % 128 == 0`; non-conforming
shapes fall back to `fallback_vector_matrix_polymul` (a `jax.vmap` of `poly_dot_product`) or the
sign-matrix path, both of which the source notes "do not lower well on TPU". These are `mosaic_tpu`
only — no GPU variant.

**Not Pallas (for the record).** The CKKS number-theoretic transform (`jaxite/jaxite_ckks/ntt.py`) is
pure JAX (Barrett reduction + BAT utils, no `pallas_call`), and `jaxite_ec` (elliptic-curve MSM /
Pippenger) contains no Pallas kernels. The FHE polymul kernels above are jaxite's entire Pallas
surface.

**Source divergence vs the 2026-04-23 catalog.** Subpage 05 §5.4 located these kernels at
`jaxite/jaxite_lib/polymul_kernel.py`. In the current checkout they live under
`jaxite/jaxite_cggi/polymul_kernel.py`, and `toeplitz_kernelized` has moved out to
`jaxite/jaxite_cggi/matrix_utils.py`. The catalog's single-row summary also predates the
`_i32_matmul_unreduced_cggi` 4× optimization and the batched `negacyclic_vector_matrix_polymul_bat`
path being split out — both are covered as distinct rows here. Kernel semantics (byte-split bf16 matmul
+ shift reassembly, CGGI 4× reduction) are unchanged.

## See also
- [jaxite — overview](overview.md) — the repo's grounded wiki front door.
- [polymul_kernel — catalog](catalog/jaxite/jaxite_cggi/polymul_kernel.md) · [matrix_utils — catalog](catalog/jaxite/jaxite_cggi/matrix_utils.md) — per-symbol grounded pages.
- [jaxite_cggi decomposition](concepts/jaxite-jaxite_cggi-decomposition.md) · [jaxite_cggi RLWE](concepts/jaxite-jaxite_cggi-rlwe.md) — the FHE primitives these kernels serve.
- [Pallas kernel directory (catalog)](../../analyses/2026-04-23-pallas-kernel-directory.md) — the cross-repo Pallas catalog; this repo appears in [subpage 05 — frameworks & quant §5.4](../../analyses/pallas-kernel-directory/05-frameworks-quant.md).
