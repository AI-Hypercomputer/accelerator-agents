---
title: 'Module: ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.quantized_matmul._pallas_impl_core`/
symbols:
  _ceil_div: _ceil_div().
  get_predecoded_dense_weight: get_predecoded_dense_weight().
  is_packed_tpu_legal_forward: is_packed_tpu_legal_forward().
  _dequantize_tile: _dequantize_tile().
  get_qmm_tpu_vmem_limit_bytes: get_qmm_tpu_vmem_limit_bytes().
  _packed_words_for_values: _packed_words_for_values().
  is_packed_tpu_legal_input_grad: is_packed_tpu_legal_input_grad().
  pallas_dense_matmul: pallas_dense_matmul().
  _pad_2d: _pad_2d().
  _normalize_tpu_blocks: _normalize_tpu_blocks().
  _bit_aligned_values: _bit_aligned_values().
  _is_2d_blockspec_legal: _is_2d_blockspec_legal().
  _unpack_packed_bits: _unpack_packed_bits().
  estimate_qmm_tpu_vmem_limit_bytes: estimate_qmm_tpu_vmem_limit_bytes().
  _expand_groups: _expand_groups().
  _predecode_dense_weight: _predecode_dense_weight().
  _pad_2d_optional: _pad_2d_optional().
  get_qmm_tpu_path: get_qmm_tpu_path().
  choose_packed_n_subtile: choose_packed_n_subtile().
  get_predecode_cache_max_items: get_predecode_cache_max_items().
  get_predecode_max_bytes: get_predecode_max_bytes().
  _PREDECODE_CACHE._PREDECODE_CACHE: _PREDECODE_CACHE._PREDECODE_CACHE.
  _lcm: _lcm().
  _decode_e4m3: _decode_e4m3().
  get_predecode_cache_enabled: get_predecode_cache_enabled().
  _parse_int_env: _parse_int_env().
  _get_tpu_version: _get_tpu_version().
  _is_tracer: _is_tracer().
  _unpack_bits_4_8: _unpack_bits_4_8().
  _PREDECODE_CACHE_LOCK: _PREDECODE_CACHE_LOCK.
  _decode_e2m1: _decode_e2m1().
  _QMM_PATHS: _QMM_PATHS.
  _DEFAULT_PREDECODE_CACHE_MAX_ITEMS: _DEFAULT_PREDECODE_CACHE_MAX_ITEMS.
  _DEFAULT_PREDECODE_MAX_BYTES: _DEFAULT_PREDECODE_MAX_BYTES.
  _DEFAULT_TPU_VMEM_LIMIT_BYTES: _DEFAULT_TPU_VMEM_LIMIT_BYTES.
  _DEFAULT_TPU_V7_VMEM_LIMIT_BYTES: _DEFAULT_TPU_V7_VMEM_LIMIT_BYTES.
  _parse_bool_env: _parse_bool_env().
  _decode_nf4: _decode_nf4().
  _device_key: _device_key().
  _estimate_predecode_bytes: _estimate_predecode_bytes().
  pallas_dense_matmul._kernel: pallas_dense_matmul()._kernel().
  pallas_dense_matmul._kernel._zero_acc: pallas_dense_matmul()._kernel()._zero_acc().
  pallas_dense_matmul._kernel._store: pallas_dense_matmul()._kernel()._store().
  __all__: __all__.
---
# Module: [`ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py)

## Functions
- `_bit_aligned_values(bits: int)` — [`L94`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L94) — Return the value count that starts and ends on a 32-bit word boundary. — documented in [ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core.md)
- `_ceil_div(a: int, b: int)` — [`L72`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L72) — Compute ceiling division of a by b. — documented in [ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core.md)
- `_decode_e2m1(code: jax.Array)` — [`L449`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L449) — Decode E2M1 (MXFP4) codes to float32. — documented in [ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core.md)
- `_decode_e4m3(code: jax.Array)` — [`L469`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L469) — Decode E4M3 (FP8) codes to float32. — documented in [ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core.md)
- `_decode_nf4(code: jax.Array)` — [`L491`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L491) — Decode NF4 (NormalFloat4) codes to float32. — documented in [ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core.md)
- `_dequantize_tile(q: jax.Array, scales: jax.Array, biases: jax.Array | None, mode: str, group_size: int)` — [`L653`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L653) — Dequantize an unpacked code tile to float32 using per-group scales. — documented in [ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core.md)
- `_device_key(arr: jax.Array)` — [`L759`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L759) — Extract a hashable device identifier from a JAX array for caching.
- `_estimate_predecode_bytes(k: int, n: int)` — [`L793`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L793) — Estimate the memory footprint of a predecoded dense bfloat16 weight.
- `_expand_groups(values: jax.Array, group_size: int, width: int)` — [`L634`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L634) — Repeat per-group values to match the full output width.
- `_get_tpu_version()` — [`L394`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L394) — Return numeric TPU version, or -1 when unavailable.
- `_is_2d_blockspec_legal(block0: int, block1: int, dim0: int, dim1: int)` — [`L163`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L163) — Check whether a 2D BlockSpec satisfies Mosaic TPU lowering constraints.
- `_is_tracer(x: object)` — [`L781`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L781) — Check whether *x* is a JAX abstract tracer (i.e. not a concrete value).
- `_kernel(lhs_ref, rhs_ref, out_ref, acc_ref)` — [`L965`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L965)
- `_lcm(a: int, b: int)` — [`L85`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L85) — Compute least common multiple for positive integers.
- `_normalize_tpu_blocks(block_m: int, block_n: int, block_k: int)` — [`L136`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L136) — Round block dimensions up to TPU-friendly tile sizes.
- `_packed_words_for_values(values: int, bits: int)` — [`L99`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L99) — Return the number of uint32 words needed for a packed bitstream.
- `_pad_2d(x: jax.Array, pad0: int, pad1: int)` — [`L104`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L104) — Zero-pad a 2D array along both dimensions.
- `_pad_2d_optional(x: jax.Array | None, pad0: int, pad1: int)` — [`L120`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L120) — Zero-pad a 2D array if it is not None.
- `_parse_bool_env(name: str, default: bool)` — [`L317`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L317) — Parse a boolean from an environment variable.
- `_parse_int_env(name: str, default: int)` — [`L334`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L334) — Parse a non-negative integer from an environment variable.
- `_predecode_dense_weight(w_q: jax.Array, scales: jax.Array, biases: jax.Array | None, *, group_size: int, bits: int, mode: str)` — [`L720`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L720) — Materialize a dense bfloat16 weight matrix from quantized representation.
- `_store()` — [`L980`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L980)
- `_unpack_bits_4_8(words: jax.Array, bits: int)` — [`L629`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L629) — Backward-compatible alias for packed 1-bit through 8-bit unpacking.
- `_unpack_packed_bits(words: jax.Array, bits: int)` — [`L565`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L565) — Unpack 1-bit through 8-bit quantized values from packed 32-bit words.
- `_zero_acc()` — [`L969`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L969)
- `choose_packed_n_subtile(*, block_n: int, group_size: int, value_alignment: int, max_subtile: int = 256)` — [`L430`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L430) — Pick an N-subtile width for packed kernels to reduce unpack/dequant pressure.
- `estimate_qmm_tpu_vmem_limit_bytes(*, io_bytes: int, scratch_bytes: int, has_double_buffer: bool)` — [`L416`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L416) — Estimate per-kernel VMEM usage and cap it to device budget.
- `get_predecode_cache_enabled()` — [`L367`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L367) — Check whether the predecoded dense weight LRU cache is enabled.
- `get_predecode_cache_max_items()` — [`L376`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L376) — Return the maximum number of items in the predecode LRU cache.
- `get_predecode_max_bytes()` — [`L385`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L385) — Return the maximum byte size allowed for a single predecoded weight buffer.
- `get_predecoded_dense_weight(w_q: jax.Array, scales: jax.Array, biases: jax.Array | None, *, group_size: int, bits: int, mode: str)` — [`L806`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L806) — Get or compute a dense bfloat16 weight from quantized tensors. — documented in [ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core.md)
- `get_qmm_tpu_path()` — [`L354`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L354) — Read the preferred QMM TPU execution path from the environment.
- `get_qmm_tpu_vmem_limit_bytes()` — [`L406`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L406) — Return VMEM compiler budget for TPU QMM kernels.
- `is_packed_tpu_legal_forward(x: jax.Array, w_q: jax.Array, scales: jax.Array, *, group_size: int, bits: int, block_m: int, block_n: int, block_k: int)` — [`L182`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L182) — Check if the packed fused TPU Pallas path is legal for forward matmul. — documented in [ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core.md)
- `is_packed_tpu_legal_input_grad(dy: jax.Array, w_q: jax.Array, scales: jax.Array, *, group_size: int, bits: int, block_m: int, block_n: int, block_k: int)` — [`L249`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L249) — Check if the packed fused TPU Pallas path is legal for the input gradient.
- `pallas_dense_matmul(lhs: jax.Array, rhs: jax.Array, *, transpose_rhs: bool, block_m: int, block_n: int, block_k: int)` — [`L900`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L900) — Tiled dense matrix multiplication on TPU via Pallas. — documented in [ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core.md)

## Module values
- `_DEFAULT_PREDECODE_CACHE_MAX_ITEMS` — [`L63`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L63) — documented in [ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core.md)
- `_DEFAULT_PREDECODE_MAX_BYTES` — [`L64`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L64) — documented in [ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core.md)
- `_DEFAULT_TPU_V7_VMEM_LIMIT_BYTES` — [`L66`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L66)
- `_DEFAULT_TPU_VMEM_LIMIT_BYTES` — [`L65`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L65)
- `_PREDECODE_CACHE` — [`L68`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L68) — documented in [ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core.md)
- `_PREDECODE_CACHE_LOCK` — [`L69`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L69) — documented in [ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core](../../../../../../concepts/ejkernel-kernels-_pallas-tpu-quantized_matmul-_pallas_impl_core.md)
- `_QMM_PATHS` — [`L62`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L62)
- `__all__` — [`L1024`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/quantized_matmul/_pallas_impl_core.py#L1024)

