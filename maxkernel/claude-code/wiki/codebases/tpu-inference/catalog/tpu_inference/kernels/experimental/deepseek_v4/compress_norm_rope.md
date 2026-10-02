---
title: 'Module: tpu_inference/kernels/experimental/deepseek_v4/compress_norm_rope.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/experimental/deepseek_v4/compress_norm_rope.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.experimental.deepseek_v4.compress_norm_rope`/
symbols:
  compress_norm_rope_store: compress_norm_rope_store().
  compress_norm_rope_store_indexer: compress_norm_rope_store_indexer().
  unpack_state_cache: unpack_state_cache().
  pack_state_cache: pack_state_cache().
  shared_sparse_cache_shape: shared_sparse_cache_shape().
  shared_indexer_cache_shape: shared_indexer_cache_shape().
  _to_byte_lane: _to_byte_lane().
  _from_byte_lane: _from_byte_lane().
  PACKING: PACKING.
  compress_norm_rope: compress_norm_rope().
  _align_to: _align_to().
  unpack_sparse_kv_cache: unpack_sparse_kv_cache().
  unpack_indexer_kv_cache: unpack_indexer_kv_cache().
  quantize_fp8_ue8m0: quantize_fp8_ue8m0().
  _state_chunk_dims: _state_chunk_dims().
  gather_state_windows: gather_state_windows().
  _boundary_dest: _boundary_dest().
  sparse_packed_width: sparse_packed_width().
  indexer_packed_width: indexer_packed_width().
  interleaved_rope: interleaved_rope().
---
# Module: [`tpu_inference/kernels/experimental/deepseek_v4/compress_norm_rope.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compress_norm_rope.py)

## Functions
- `_align_to(value: int, multiple: int)` — [`L124`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compress_norm_rope.py#L124)
- `_boundary_dest(positions: jax.Array, slot_mapping: jax.Array, kv_slot_mapping: jax.Array, compress_ratio: int, num_slots: int)` — [`L333`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compress_norm_rope.py#L333)
- `_from_byte_lane(b: jax.Array, dtype)` — [`L58`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compress_norm_rope.py#L58) — Inverse of ``_to_byte_lane``: read trailing bytes back as ``dtype``.
- `_state_chunk_dims(cache_shape, state_block_size: int, state_dim: int)` — [`L155`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compress_norm_rope.py#L155) — Geometry of state token in cache.
- `_to_byte_lane(x: jax.Array)` — [`L46`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compress_norm_rope.py#L46) — Reinterpret each element of ``x``'s trailing dim as raw bytes.
- `compress_norm_rope(kv_window: jax.Array, score_window: jax.Array, valid_mask: jax.Array, rms_weight: jax.Array, cos_sin_cache: jax.Array, compressed_pos: jax.Array, rms_eps: float, rope_head_dim: int)` — [`L249`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compress_norm_rope.py#L249) — Window softmax-pool, RMSNorm, and interleaved RoPE.
- `compress_norm_rope_store(cache: jax.Array, state_cache: jax.Array, positions: jax.Array, slot_mapping: jax.Array, block_table: jax.Array, token_to_req_indices: jax.Array, kv_slot_mapping: jax.Array, rms_weight: jax.Array, cos_sin_cache: jax.Array, state_block_size: int, head_dim: int, rope_head_dim: int, compress_ratio: int, overlap: bool, rms_eps: float, quant_block: int)` — [`L345`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compress_norm_rope.py#L345) — Compress, norm, RoPE, and write boundary KV into the shared cache.
- `compress_norm_rope_store_indexer(cache: jax.Array, state_cache: jax.Array, positions: jax.Array, slot_mapping: jax.Array, block_table: jax.Array, token_to_req_indices: jax.Array, kv_slot_mapping: jax.Array, rms_weight: jax.Array, cos_sin_cache: jax.Array, state_block_size: int, head_dim: int, rope_head_dim: int, compress_ratio: int, overlap: bool, rms_eps: float, quant_block: int)` — [`L417`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compress_norm_rope.py#L417) — Indexer (head_dim=128) twin of ``compress_norm_rope_store``.
- `gather_state_windows(state_cache: jax.Array, positions: jax.Array, block_table: jax.Array, token_to_req_indices: jax.Array, block_size: int, head_dim: int, compress_ratio: int, overlap: bool)` — [`L276`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compress_norm_rope.py#L276) — Gather ``[kv_window, score_window, valid_mask]`` from the paged cache.
- `indexer_packed_width(head_dim: int, quant_block: int)` — [`L87`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compress_norm_rope.py#L87) — Bytes per token in the packed indexer (head_dim=128) KV cache.
- `interleaved_rope(x: jax.Array, cos_sin: jax.Array, rope_head_dim: int)` — [`L214`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compress_norm_rope.py#L214) — Interleaved (GPT-J) RoPE on the trailing ``rope_head_dim`` elements.
- `pack_state_cache(cache: jax.Array, state: jax.Array)` — [`L198`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compress_norm_rope.py#L198) — Write an f32 state view ``[num_pages, sb, state_dim]`` into ``cache``.
- `quantize_fp8_ue8m0(x: jax.Array, block_size: int)` — [`L67`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compress_norm_rope.py#L67) — Block fp8 quantization with UE8M0 (power-of-two) block scales.
- `shared_indexer_cache_shape(num_pages: int, page_size: int, head_dim: int, quant_block: int)` — [`L142`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compress_norm_rope.py#L142) — MLA-style shape of the shared state+KV ``uint8`` buffer (indexer path).
- `shared_sparse_cache_shape(num_pages: int, page_size: int, nope_dim: int, rope_head_dim: int, quant_block: int)` — [`L128`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compress_norm_rope.py#L128) — MLA-style shape of the shared state+KV ``uint8`` buffer (sparse path).
- `sparse_packed_width(nope_dim: int, rope_head_dim: int, quant_block: int)` — [`L80`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compress_norm_rope.py#L80) — Bytes per token in the packed sparse (head_dim=512) KV cache.
- `unpack_indexer_kv_cache(kv_cache: jax.Array, head_dim: int, quant_block: int)` — [`L110`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compress_norm_rope.py#L110) — Split the packed indexer KV cache into ``(fp8, scale)`` views.
- `unpack_sparse_kv_cache(kv_cache: jax.Array, nope_dim: int, rope_head_dim: int, quant_block: int)` — [`L93`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compress_norm_rope.py#L93) — Split the packed sparse KV cache into native-dtype component views.
- `unpack_state_cache(cache: jax.Array, state_block_size: int, state_dim: int)` — [`L182`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compress_norm_rope.py#L182) — Read the shared buffer's f32 state view ``[num_pages, sb, state_dim]``.

## Module values
- `PACKING` — [`L121`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/deepseek_v4/compress_norm_rope.py#L121)

