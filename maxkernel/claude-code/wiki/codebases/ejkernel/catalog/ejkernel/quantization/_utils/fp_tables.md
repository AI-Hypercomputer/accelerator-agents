---
title: 'Module: ejkernel/quantization/_utils/fp_tables.py'
type: catalog
provenance: extracted
module: ejkernel/quantization/_utils/fp_tables.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.quantization._utils.fp_tables`/_
symbols:
  _get_e2m1_table: get_e2m1_table().
  _get_e4m3_table: get_e4m3_table().
  _get_e4m3_q_threshold_map: get_e4m3_q_threshold_map().
  _get_nf4_table: get_nf4_table().
  _get_e2m1_threshold_map: get_e2m1_threshold_map().
  _get_e4m3_table_q: get_e4m3_table_q().
  _get_nf4_threshold_map: get_nf4_threshold_map().
  _get_e2m1_max: get_e2m1_max().
  _get_e4m3_max: get_e4m3_max().
  _decode_e4m3_codes: decode_e4m3_codes().
  _decode_e2m1_codes: decode_e2m1_codes().
  _build_threshold_map: build_threshold_map().
  _make_fp_table: make_fp_table().
  _build_nf4_table: build_nf4_table().
---
# Module: [`ejkernel/quantization/_utils/fp_tables.py`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/fp_tables.py)

## Functions
- `_build_nf4_table()` — [`L176`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/fp_tables.py#L176) — Build the NF4 (NormalFloat 4-bit) codebook tensor.
- `_build_threshold_map(codebook: jnp.ndarray)` — [`L148`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/fp_tables.py#L148) — Build sorted-index and midpoint-boundary tensors for threshold quantization.
- `_decode_e2m1_codes(codes: jnp.ndarray, *, dtype: jnp.dtype)` — [`L33`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/fp_tables.py#L33) — Decode E2M1 (FP4) codes to float values without a lookup table.
- `_decode_e4m3_codes(codes: jnp.ndarray, *, dtype: jnp.dtype)` — [`L62`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/fp_tables.py#L62) — Decode E4M3 (FP8) codes to float values without a lookup table.
- `_get_e2m1_max()` — [`L248`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/fp_tables.py#L248) — Get the maximum representable absolute value in E2M1 format.
- `_get_e2m1_table()` — [`L209`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/fp_tables.py#L209) — Get the E2M1 (FP4) lookup table.
- `_get_e2m1_threshold_map()` — [`L290`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/fp_tables.py#L290) — Get cached threshold map (sorted_idx, boundaries) for E2M1 codebook.
- `_get_e4m3_max()` — [`L258`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/fp_tables.py#L258) — Get the maximum representable absolute value in E4M3 format.
- `_get_e4m3_q_threshold_map()` — [`L295`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/fp_tables.py#L295) — Get cached threshold map (sorted_idx, boundaries) for E4M3-quant codebook.
- `_get_e4m3_table()` — [`L222`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/fp_tables.py#L222) — Get the E4M3 (FP8) lookup table.
- `_get_e4m3_table_q()` — [`L234`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/fp_tables.py#L234) — Get E4M3 table with NaN replaced by infinity for quantization.
- `_get_nf4_table()` — [`L270`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/fp_tables.py#L270) — Get the NF4 (NormalFloat 4-bit) codebook.
- `_get_nf4_threshold_map()` — [`L300`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/fp_tables.py#L300) — Get cached threshold map (sorted_idx, boundaries) for NF4 codebook.
- `_make_fp_table(exp_bits: int, mant_bits: int, *, nan_all_ones: bool)` — [`L101`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/fp_tables.py#L101) — Generate a lookup table for a minifloat format.

