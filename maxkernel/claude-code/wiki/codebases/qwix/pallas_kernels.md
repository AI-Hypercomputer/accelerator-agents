---
title: "qwix — Pallas kernels"
type: concept
tags: [pallas, kernels, tpu, qwix]
created: 2026-07-04
updated: 2026-07-04
---

qwix ships **two novel-to-this-repo Mosaic-TPU Pallas kernels**, both in `qwix/contrib/kernels/` and both quantized GEMMs: a `QArray × QArray → f32/bf16` matmul with per-tile scales, and an LHS-fused variant that quantizes the activation on-chip and reuses it in VMEM. Both are `mosaic_tpu` only, tuned for TPUv5, and marked actively-developed / contrib (not stable API). A third file, `qwix/_src/core/pallas.py`, is a `QArray`-aware `pallas_call` **lifting layer**, not a compute kernel. The source has **diverged materially** from the 2026-04-23 directory catalog (see Notes).

## Kernels

| Kernel | Backend | What it does | Source | Stability | Novel/Vendored |
|--------|---------|--------------|--------|-----------|----------------|
| `quantized_matmul` / `quantized_matmul_kernel` | mosaic_tpu | `QArray × QArray → dtype` GEMM over a `(cdiv(m,bm), cdiv(n,bn), cdiv(k,bk))` grid; int matmul into `int32`, then per-tile dequant `(xy * sx) * sy` accumulated in a VMEM f32 scratch; **sub-block M/N/K loops** (default 128³) overlap MXU + VPU with the `emit_pipeline` DMA; k-axis is the `ARBITRARY` reduction dim (zero-init at `program_id(2)==0`, write-out at last k-step) | [quantized_matmul.py#L138](../../../raw/code/qwix/qwix/contrib/kernels/quantized_matmul.py#L138) | experimental (contrib; "actively developed", TPUv5-focused) | novel |
| `lhs_fused_quantized_matmul` / `lhs_fused_qmm_kernel` | mosaic_tpu | Fuses **on-chip absmax int8 quantization of the LHS** into the matmul: `quantize_a_tile` computes `s = max(|x|)/127.49`, rounds to int8; quantized LHS tiles are cached in a per-block `xq_list`/`sx_list` and **reused across all N-subtiles** (quantized once at `nloop==0`); RHS is pre-quantized with `sy` scales; same int32-accumulate → `(xy*sx)*sy` dequant → f32 VMEM scratch pattern | [lhs_fused_quantized_matmul.py#L57](../../../raw/code/qwix/qwix/contrib/kernels/lhs_fused_quantized_matmul.py#L57) | experimental (contrib; absmax-int8 only, no zero-point, no stochastic rounding yet) | novel |

### Imported / not a kernel (lifting layer)

- `pallas_call` + `update_block_specs_for_qarray` + `transform_block_specs_for_tpu` — [`qwix/_src/core/pallas.py`](../../../raw/code/qwix/qwix/_src/core/pallas.py). A lifted `jax.pallas_call` that accepts `QArray` (value + scale + zero-point) as pytree leaves, derives scale-tile `BlockSpec`s, and applies a memory-saving minor/second-minor transpose for trailing-dim-1 scale tensors (avoids 128-lane padding waste). **Not a compute kernel** — the QArray-plumbing analogue of AQT's `pallas_call.py`, which it supersedes. `qwix/_src/qconfig.py` also intercepts `jax.experimental.pallas.pallas_call` to disable qwix op-interception during kernel tracing.

## Notes

**Vendoring / lineage.** Both kernels are authored in qwix (Apache-2.0, `Copyright 2026 Google LLC`), not vendored from jax/tokamax. The `pallas_call` lifting layer re-does AQT's `aqt/jax/v2/pallas/pallas_call.py` pattern (qwix is the explicit AQT successor); AQT itself ships **no compute Pallas kernel**, so qwix is where the real quantized-GEMM kernels now live in this lineage.

**Divergence from the 2026-04-23 catalog (subpage 05.2).** The source has moved on since the directory was written:
- The catalog's `QuantizedMatmulConfig(bm=128, bk=128, bn=128)` with `INTERPRET: bool = True` hard-coded is **gone**. Current `QuantizedMatmulConfig` defaults are **`bm=256, bk=512, bn=1024`** ("determined by TPUv5 sweep"), and the kernel no longer carries an `INTERPRET` flag — it now dispatches through `pl.kernel` + `pltpu.emit_pipeline` + `pltpu.create_tensorcore_mesh` (tensorcore-mesh / megacore pipeline API) rather than a plain `pl.pallas_call`.
- The gating helper moved from `can_use_qmm_in_dot_general` on the old `qwix/contrib/kernels/quantized_matmul.py` path to a pair `can_use_qmm` / `can_use_qmm_in_dot_general` here (still requires no zero-point, contracting dims `(1,)/(0,)`, block sizes dividing matrix dims, scale-tile ≥128 per axis, f32 scales only).
- A **second kernel, `lhs_fused_quantized_matmul.py`, is new** and absent from the catalog.

**Autotune surface.** No autotuner — block sizes (`bm/bk/bn`) and sub-block caps (`max_sublock_size_{m,n,k}`, default 128) are caller-chosen dataclass fields, with a "TPUv5 sweep" comment as the only guidance. `quantized_matmul` requires `bm/bk/bn` to divide `m/k/n` and per-axis scale tiles to divide the grid (else raises). `reduction_dtype = jnp.int32` is hard-coded; accumulation scratch is f32 VMEM `(bm, bn)`.

**Gotchas for reuse.** Both are contrib, not stable API ("Integration into the main qwix library" is listed future work). `quantized_matmul` needs pre-expanded 2-D scales (`(sm,sk)` / `(sk,sn)`; 1-D scales must be reshaped to `(sm,1)`/`(1,sk)` by the caller) and rejects non-f32 scales. `lhs_fused_quantized_matmul` supports **absmax int8 only** (no zero-point, no stochastic rounding), quantizes the LHS with `max_val = iinfo(int8).max + 0.49` rounding bias, and takes `sm` (LHS scale rows) as a static arg. Both target TPUv5 and are untested on other generations per their docstrings.

## See also

- [qwix — overview](overview.md) — the repo's grounded wiki front door.
- [Pallas kernel directory (catalog)](../../analyses/2026-04-23-pallas-kernel-directory.md) — the cross-repo Pallas catalog; qwix appears in [subpage 05 §5.2](../../analyses/pallas-kernel-directory/05-frameworks-quant.md).

## Sources

- [`raw/code/qwix/qwix/contrib/kernels/quantized_matmul.py`](../../../raw/code/qwix/qwix/contrib/kernels/quantized_matmul.py) (commit `e621789`)
- [`raw/code/qwix/qwix/contrib/kernels/lhs_fused_quantized_matmul.py`](../../../raw/code/qwix/qwix/contrib/kernels/lhs_fused_quantized_matmul.py) (commit `e621789`)
- [`raw/code/qwix/qwix/_src/core/pallas.py`](../../../raw/code/qwix/qwix/_src/core/pallas.py) — QArray lifting layer (not a kernel)
