---
title: "qwix (google/qwix)"
type: codebase
tags: [qwix, quantization, pallas, mosaic-tpu, qarray, aqt-successor, qat, ptq]
repo: google/qwix
commit: b966dc4
created: 2026-04-23
updated: 2026-04-23
---

> **Grounded front door → [qwix overview](qwix/overview.md).** This hand page is the thin, on-demand companion: cross-ecosystem perf verdicts, loop-experiment backlinks, and Pallas-kernel-directory cross-links that the code-regenerated overview intentionally omits. For architecture, mechanisms, and per-symbol detail, start at the overview.

Google's quantization framework — **successor to [AQT](aqt.md)**. Provides a `QArray` tensor type and `pallas_call` wrappers that make Pallas kernels quantization-aware. Used by [maxtext](maxtext.md) Megablox for int4/int8 MoE. TPU Pallas surface concentrated in `qwix/contrib/kernels/quantized_matmul.py`. Per-kernel detail in [§5 of the Pallas kernel directory](../analyses/pallas-kernel-directory/05-frameworks-quant.md).

## Architecture

- `qwix/_src/core/pallas.py` — `QArray`-aware `pallas_call` wrapper (pytree lifting + scale-tile pass-through).
- `qwix/contrib/kernels/quantized_matmul.py` — per-tile scale W8A8 matmul for QArray; **`INTERPRET = True`** hard-coded (runs in Pallas interpret mode — portable but slower).
- Rest of `qwix/` — QAT/PTQ training machinery (out of scope).

## Key abstractions

| Component | File | Backend | Notes |
|---|---|---|---|
| `QArray` pallas_call wrapper | `qwix/_src/core/pallas.py` | lifting | Makes Pallas kernels see real tensors with associated scale tiles |
| `quantized_matmul` | `qwix/contrib/kernels/quantized_matmul.py` | `mosaic_tpu` (interpret) | Per-tile scale W8A8; prototype-quality |

## Performance-relevant surfaces

### 1. `QArray`-aware `pallas_call` lifting pattern
Pytree lifting that makes scale tiles transparent to kernel code. The right layer to introduce new quantization optimization hypotheses — use `qwix.pallas.pallas_call` (not upstream `pl.pallas_call`) when operating on quantized tensors.

### 2. AQT successor status
qwix replaces AQT. AQT's four Pallas files are deprecated. For new quantization hypotheses, target qwix.

### 3. `INTERPRET = True` in contrib kernel
The `contrib/` quantized matmul runs in Pallas interpret mode. For perf, use the maxtext Megablox path which bridges qwix's `QArray` into a real Mosaic-TPU kernel.

### 4. Integration with MaxText
`maxtext/kernels/megablox/backend.py` imports `qwix.pallas` for int4/int8 GMM. Precedent for any new Pallas quant hypothesis.

## Connections

- [Pallas kernel directory §5](../analyses/pallas-kernel-directory/05-frameworks-quant.md)
- [aqt](aqt.md) — deprecated predecessor.
- [maxtext](maxtext.md) — main consumer (Megablox int4/int8).
- [tpu-inference](tpu-inference.md) — alternative quantized matmul path (blockwise, not per-tile).
- [int8-quantization](../concepts/int8-quantization.md)

## Sources

- `raw/code/qwix/qwix/_src/core/pallas.py`
- `raw/code/qwix/qwix/contrib/kernels/quantized_matmul.py`
- Upstream: <https://github.com/google/qwix>
