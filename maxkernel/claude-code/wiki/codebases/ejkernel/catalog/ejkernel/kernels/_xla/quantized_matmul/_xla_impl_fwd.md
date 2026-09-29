---
title: 'Module: ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._xla.quantized_matmul._xla_impl_fwd`/
symbols:
  quantized_matmul: quantized_matmul().
  _blocked_quantized_matmul.decode_tile: _blocked_quantized_matmul().decode_tile().
  _blocked_quantized_matmul: _blocked_quantized_matmul().
  _operate: _operate().
  _packed_words_for_values: _packed_words_for_values().
  _resolve_qparams: _resolve_qparams().
  _pad_2d: _pad_2d().
  _decode_tile_nvfp4: _decode_tile_nvfp4().
  _blocked_quantized_matmul.load_q_tile: _blocked_quantized_matmul().load_q_tile().
  _blocked_quantized_matmul.n_loop: _blocked_quantized_matmul().n_loop().
  _blocked_quantized_matmul.k_loop: _blocked_quantized_matmul().k_loop().
  _pad_2d_optional: _pad_2d_optional().
  _ceil_div: _ceil_div().
  _decode_tile_nf4: _decode_tile_nf4().
  _decode_tile_mxfp4: _decode_tile_mxfp4().
  _decode_tile_mxfp8: _decode_tile_mxfp8().
  _decode_tile_nvfp8: _decode_tile_nvfp8().
  _blocked_quantized_matmul.n_loop._decode_k: _blocked_quantized_matmul().n_loop()._decode_k().
  _blocked_quantized_matmul.n_loop.m_loop: _blocked_quantized_matmul().n_loop().m_loop().
  _blocked_quantized_matmul.n_loop.m_loop.k_body: _blocked_quantized_matmul().n_loop().m_loop().k_body().
  _lcm: _lcm().
  _bit_aligned_values: _bit_aligned_values().
  _dot_general: _dot_general().
  _decode_tile_affine: _decode_tile_affine().
  _blocked_quantized_matmul.load_group_tile: _blocked_quantized_matmul().load_group_tile().
---
# Module: [`ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py)

## Functions
- `_bit_aligned_values(bits: int)` — [`L89`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py#L89) — Return the value count that starts and ends on a 32-bit word boundary.
- `_blocked_quantized_matmul(x: jax.Array, w_q: jax.Array, scales: jax.Array, biases: jax.Array | None, *, transpose: bool, group_size: int, bits: int, mode: str, block_m: int, block_n: int, block_k: int, use_bf16: bool)` — [`L248`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py#L248) — Execute blocked quantized matmul with fused dequantization.
- `_ceil_div(a: int, b: int)` — [`L75`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py#L75) — Compute ceiling division of a by b.
- `_decode_k(k_idx: jax.Array)` — [`L473`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py#L473)
- `_decode_tile_affine(q: jax.Array, scale_tile: jax.Array, bias_tile: jax.Array | None)` — [`L122`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py#L122) — Decode affine quantized tile using additive-bias form.
- `_decode_tile_mxfp4(q: jax.Array, scale_tile: jax.Array)` — [`L163`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py#L163) — Decode MXFP4 (E2M1) quantized tile with E8M0 shared exponent.
- `_decode_tile_mxfp8(q: jax.Array, scale_tile: jax.Array)` — [`L180`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py#L180) — Decode MXFP8 (E4M3) quantized tile with E8M0 shared exponent.
- `_decode_tile_nf4(q: jax.Array, scale_tile: jax.Array)` — [`L148`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py#L148) — Decode NF4 quantized tile using codebook lookup.
- `_decode_tile_nvfp4(q: jax.Array, scale_tile: jax.Array)` — [`L197`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py#L197) — Decode NVFP4 (E2M1 with E4M3 scale) quantized tile.
- `_decode_tile_nvfp8(q: jax.Array, scale_tile: jax.Array)` — [`L214`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py#L214) — Decode NVFP8 (E4M3 with E4M3 scale) quantized tile.
- `_dot_general(a: jax.Array, b: jax.Array)` — [`L230`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py#L230) — Perform matrix multiplication with float32 output accumulation.
- `_lcm(a: int, b: int)` — [`L80`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py#L80) — Compute least common multiple of two positive integers.
- `_operate(x, w, scales, biases, transpose, group_size, bits, mode, block_m, block_n, block_k, use_bf16, allow_dense_fallback, gemv_mode, revsplit_k, revsplit_k_parts)` — [`L539`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py#L539) — Execute quantized matmul with automatic path selection.
- `_packed_words_for_values(values: int, bits: int)` — [`L94`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py#L94) — Return the number of uint32 words needed for a packed bitstream.
- `_pad_2d(x: jax.Array, pad0: int, pad1: int)` — [`L99`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py#L99) — Pad a 2D array with zeros on the right and bottom edges.
- `_pad_2d_optional(x: jax.Array | None, pad0: int, pad1: int)` — [`L115`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py#L115) — Pad a 2D array if not None, otherwise return None.
- `_resolve_qparams(mode: str, group_size: int | None, bits: int | None)` — [`L56`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py#L56) — Resolve quantization parameters and convert mode to backend representation.
- `decode_tile(off_k: int, off_n: int)` — [`L399`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py#L399) — Dequantize a single weight tile using the configured quantization mode.
- `k_body(idx: int, carry: jax.Array)` — [`L495`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py#L495) — Inner loop body for K-dimension accumulation.
- `k_loop(k_idx: int, acc: jax.Array, *, off_m: int, off_n: int)` — [`L433`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py#L433) — Accumulate one K-dimension tile into the output accumulator.
- `load_group_tile(off_k: int, off_n: int)` — [`L363`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py#L363) — Load per-group scales and affine additive biases for a weight tile.
- `load_q_tile(off_k: int, off_n: int)` — [`L337`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py#L337) — Load and unpack a tile of quantized weight codes.
- `m_loop(m_idx: int, out_local: jax.Array)` — [`L479`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py#L479) — Process one M-dimension block row for a fixed N column.
- `n_loop(n_idx: int, out_buf: jax.Array)` — [`L456`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py#L456) — Process one N-dimension block column across all M rows.
- `quantized_matmul(x: Float[Array, "m k"], w: Array, scales: Array, zeros: Array | None = None, transpose: bool = False, group_size: int | None = None, bits: int | None = None, mode: str = "affine", axis: QuantizationAxis | None = None, gemv_mode: GemvMode = "auto", revsplit_k: RevSplitKMode = "auto", revsplit_k_parts: int | None = None, *, block_m: int = 128, block_n: int = 128, block_k: int = 64, use_bf16: bool = True, allow_dense_fallback: bool = True, num_warps: int | None = None, num_stages: int | None = None, split_k: int | None = None)` — [`L631`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/quantized_matmul/_xla_impl_fwd.py#L631) — Compute quantized matrix multiplication using JAX/XLA.

