---
title: 'Module: ejkernel/quantization/_utils/bitpack.py'
type: catalog
provenance: extracted
module: ejkernel/quantization/_utils/bitpack.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.quantization._utils.bitpack`/_
symbols:
  _unpack_bits: unpack_bits().
  _pack_bits: pack_bits().
  _pack_bits_fast_grouped: pack_bits_fast_grouped().
  _unpack_bits_fast_grouped: unpack_bits_fast_grouped().
  _pack_bits_u1_fast: pack_bits_u1_fast().
  _pack_bits_u2_fast: pack_bits_u2_fast().
  _pack_bits_u4_fast: pack_bits_u4_fast().
  _pack_bits_u8_fast: pack_bits_u8_fast().
  _unpack_bits_u1_fast: unpack_bits_u1_fast().
  _unpack_bits_u2_fast: unpack_bits_u2_fast().
  _unpack_bits_u4_fast: unpack_bits_u4_fast().
  _unpack_bits_u8_fast: unpack_bits_u8_fast().
  _pack_bits_generic: pack_bits_generic().
  _unpack_bits_generic: unpack_bits_generic().
---
# Module: [`ejkernel/quantization/_utils/bitpack.py`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/bitpack.py)

## Functions
- `_pack_bits(values: jax.Array, bits: int, *, prefer_fast_u4_u8: bool = True, strict_shape_alignment: bool = False)` — [`L185`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/bitpack.py#L185) — Pack quantized codes into uint32 words (LSB-first).
- `_pack_bits_fast_grouped(values: jax.Array, *, bits: int, values_per_word: int, mask: int, strict_shape_alignment: bool)` — [`L79`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/bitpack.py#L79) — Fast grouped packer for bit-widths that tile evenly into uint32.
- `_pack_bits_generic(values: jax.Array, bits: int)` — [`L32`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/bitpack.py#L32) — Generic bit-packer supporting arbitrary bit-widths.
- `_pack_bits_u1_fast(values: jax.Array, *, strict_shape_alignment: bool)` — [`L121`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/bitpack.py#L121) — Fast path for packing uint1-like codes into uint32.
- `_pack_bits_u2_fast(values: jax.Array, *, strict_shape_alignment: bool)` — [`L132`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/bitpack.py#L132) — Fast path for packing uint2-like codes into uint32.
- `_pack_bits_u4_fast(values: jax.Array, *, strict_shape_alignment: bool)` — [`L143`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/bitpack.py#L143) — Fast path for packing uint4-like codes into uint32.
- `_pack_bits_u8_fast(values: jax.Array, *, strict_shape_alignment: bool)` — [`L154`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/bitpack.py#L154) — Fast path for packing uint8-like codes into uint32.
- `_unpack_bits(packed: jax.Array, n: int, bits: int, *, prefer_fast_u4_u8: bool = True)` — [`L220`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/bitpack.py#L220) — Unpack quantized codes from uint32 words (LSB-first).
- `_unpack_bits_fast_grouped(packed: jax.Array, n: int, *, bits: int, values_per_word: int, mask: int)` — [`L105`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/bitpack.py#L105) — Fast grouped unpacker for bit-widths that tile evenly into uint32.
- `_unpack_bits_generic(packed: jax.Array, n: int, bits: int)` — [`L56`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/bitpack.py#L56) — Generic bit-unpacker supporting arbitrary bit-widths.
- `_unpack_bits_u1_fast(packed: jax.Array, n: int)` — [`L165`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/bitpack.py#L165) — Fast path for unpacking uint1-like codes from uint32.
- `_unpack_bits_u2_fast(packed: jax.Array, n: int)` — [`L170`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/bitpack.py#L170) — Fast path for unpacking uint2-like codes from uint32.
- `_unpack_bits_u4_fast(packed: jax.Array, n: int)` — [`L175`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/bitpack.py#L175) — Fast path for unpacking uint4-like codes from uint32.
- `_unpack_bits_u8_fast(packed: jax.Array, n: int)` — [`L180`](../../../../../../../raw/code/ejkernel/ejkernel/quantization/_utils/bitpack.py#L180) — Fast path for unpacking uint8-like codes from uint32.

