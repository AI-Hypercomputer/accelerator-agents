---
title: 'Module: python/sgl_jax/srt/kernels/simple_gla/simple_gla.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/simple_gla/simple_gla.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.simple_gla.simple_gla`/
symbols:
  chunk_fwd_h_kernel_varlen: chunk_fwd_h_kernel_varlen().
  chunk_simple_gla_fwd_varlen: chunk_simple_gla_fwd_varlen().
  simple_gla_fwd: simple_gla_fwd().
  _chunk_fwd_o_pl: _chunk_fwd_o_pl().
  chunk_fwd_o: chunk_fwd_o().
  exp: exp().
  assert_shape: assert_shape().
  _align_varlen_inputs: _align_varlen_inputs().
  assert_shape_or_none: assert_shape_or_none().
  fused_recurrent_simple_gla: fused_recurrent_simple_gla().
  get_interpret: get_interpret().
  _scan_segment: _scan_segment().
  _scan_varlen: _scan_varlen().
  _chunk_fwd_h_kernel_varlen: _chunk_fwd_h_kernel_varlen().
  _chunk_fwd_o_kernel: _chunk_fwd_o_kernel().
  _unalign_output: _unalign_output().
  _chunk_fwd_o_pl._reshape_bt: _chunk_fwd_o_pl()._reshape_bt().
  _align_varlen_inputs._gather_and_mask: _align_varlen_inputs()._gather_and_mask().
  _build_align_gather_idx: _build_align_gather_idx().
  SimpleGLAKernelMode: SimpleGLAKernelMode#
  _scan_segment.step: _scan_segment().step().
  _scan_varlen.step: _scan_varlen().step().
  _build_chunk_map: _build_chunk_map().
  chunk_fwd_h_kernel_varlen.k_index_map: chunk_fwd_h_kernel_varlen().k_index_map().
  chunk_fwd_h_kernel_varlen.gk_index_map: chunk_fwd_h_kernel_varlen().gk_index_map().
  chunk_fwd_h_kernel_varlen.v_index_map: chunk_fwd_h_kernel_varlen().v_index_map().
  chunk_fwd_h_kernel_varlen.h0_index_map: chunk_fwd_h_kernel_varlen().h0_index_map().
  chunk_fwd_h_kernel_varlen.ht_index_map: chunk_fwd_h_kernel_varlen().ht_index_map().
  chunk_fwd_h_kernel_varlen.h_index_map: chunk_fwd_h_kernel_varlen().h_index_map().
  _compute_t_aligned: _compute_t_aligned().
  SimpleGLAKernelMode.FUSED_CHUNK: SimpleGLAKernelMode#FUSED_CHUNK.
  _chunk_fwd_h_kernel_varlen._.reset_state: _chunk_fwd_h_kernel_varlen()._().reset_state().
  _chunk_fwd_h_kernel_varlen._.store_fn: _chunk_fwd_h_kernel_varlen()._().store_fn().
  _chunk_fwd_h_kernel_varlen._.write_final: _chunk_fwd_h_kernel_varlen()._().write_final().
  SimpleGLAKernelMode.CHUNK: SimpleGLAKernelMode#CHUNK.
---
# Module: [`python/sgl_jax/srt/kernels/simple_gla/simple_gla.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py)

## Classes
### `SimpleGLAKernelMode`  ·  implements/extends Enum
- def: [`python/sgl_jax/srt/kernels/simple_gla/simple_gla.py:935`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L935)
- doc: Simple GLA kernel implementation mode.
- signature: `class SimpleGLAKernelMode(enum.Enum):`
- members:
  - `CHUNK` — [`L938`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L938)
  - `FUSED_CHUNK` — [`L939`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L939)
- used by: [`simple_gla_fwd`](simple_gla.md#simple_gla_fwd)

## Functions
- `_align_varlen_inputs(q, k, v, cu_seqlens_dev, chunk_size, T_aligned)` — [`L819`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L819) — Pad each sequence to a multiple of chunk_size and rebuild cu_seqlens.
- `_build_align_gather_idx(cu_seqlens, aligned_cu, T_aligned)` — [`L797`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L797) — For each position in the aligned layout, return (orig_pos, is_valid).
- `_build_chunk_map(cu_seqlens, T_sum, BT)` — [`L311`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L311)
- `_chunk_fwd_h_kernel_varlen(k_ref, v_ref, h0_ref, gk_ref, g_gamma_ref, cu_seqlens_ref, chunk_to_seq, seq_real_lens_ref, h_ref, ht_ref, scratch_ref, *, BT, BS)` — [`L321`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L321)
- `_chunk_fwd_o_kernel(q_ref, k_ref, v_ref, h_ref, g_ref, g_gamma_ref, scale_ref, o_ref, *, BT: int)` — [`L564`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L564) — Pallas kernel for chunk_fwd_o.
- `_chunk_fwd_o_pl(q: jax.Array, k: jax.Array, v: jax.Array, h: jax.Array, *, g: jax.Array | None = None, g_gamma: jax.Array | None = None, scale: float, chunk_size: int = 64)` — [`L644`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L644) — Pallas launcher for chunk_fwd_o on the uniform-length path.
- `_compute_t_aligned(T_orig, N, chunk_size)` — [`L812`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L812) — Static upper bound for the per-seq-aligned packed length.
- `_gather_and_mask(x)` — [`L829`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L829)
- `_reshape_bt(x, D)` — [`L662`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L662)
- `_scan_segment(q: jax.Array, k: jax.Array, v: jax.Array, *, g: jax.Array | None, g_gamma: jax.Array | None, scale: float, initial_state: jax.Array | None, reverse: bool)` — [`L82`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L82) — Run recurrent Simple GLA over one dense segment.
- `_scan_varlen(q: jax.Array, k: jax.Array, v: jax.Array, *, g: jax.Array | None, g_gamma: jax.Array | None, scale: float, initial_state: jax.Array | None, reverse: bool, cu_seqlens: jax.Array)` — [`L134`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L134) — Run recurrent Simple GLA over packed varlen data with one JAX scan.
- `_unalign_output(o_aligned, cu_seqlens_orig, aligned_cu, T_orig)` — [`L840`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L840) — Scatter the aligned-layout output back to the original packed layout.
- `assert_shape(x: jax.Array | list[jax.Array] | tuple[jax.Array, ...], expected_shape: list[int] | tuple[int, ...], name: str | list[str] | tuple[str, ...] = "tensor")` — [`L51`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L51)
- `assert_shape_or_none(x: jax.Array | list[jax.Array | None] | tuple[jax.Array | None, ...] | None, expected_shape: list[int] | tuple[int, ...], name: str | list[str] | tuple[str, ...] = "tensor")` — [`L32`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L32)
- `chunk_fwd_h_kernel_varlen(k: jax.Array, v: jax.Array, g: jax.Array | None = None, g_gamma: jax.Array | None = None, gk: jax.Array | None = None, gv: jax.Array | None = None, h0: jax.Array | None = None, output_final_state: bool = False, cu_seqlens_dev: jax.Array | None = None, chunk_size: int = 128, split_size: int | None = None, states_in_fp32: bool = False, seq_real_lens: jax.Array | None = None)` — [`L422`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L422)
- `chunk_fwd_o(q: jax.Array, k: jax.Array, v: jax.Array, h: jax.Array, *, g: jax.Array | None = None, g_gamma: jax.Array | None = None, scale: float | None = None, cu_seqlens_cpu: jax.Array | None = None, cu_seqlens_dev: jax.Array | None = None, chunk_size: int = 64)` — [`L745`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L745) — Chunk forward output computation.
- `chunk_simple_gla_fwd_varlen(q: jax.Array, k: jax.Array, v: jax.Array, *, g: jax.Array | None = None, g_gamma: jax.Array | None = None, scale: float | None = None, h0: jax.Array | None = None, use_ht: bool = False, cu_seqlens_cpu: jax.Array | None = None, cu_seqlens_dev: jax.Array | None = None, chunk_size: int = 64)` — [`L856`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L856)
- `exp(x)` — [`L67`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L67)
- `fused_recurrent_simple_gla(q: jax.Array, k: jax.Array, v: jax.Array, g: jax.Array | None = None, g_gamma: jax.Array | None = None, scale: float | None = None, initial_state: jax.Array | None = None, output_final_state: bool = False, reverse: bool = False, cu_seqlens: np.ndarray | jax.Array | None = None)` — [`L209`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L209) — Simple GLA fused recurrent forward for decode-friendly execution.
- `get_interpret()` — [`L71`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L71)
- `gk_index_map(head_index, k_index, _, t_index)` — [`L473`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L473)
- `h0_index_map(head_index, k_index, v_index, t_index)` — [`L479`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L479)
- `h_index_map(head_index, k_index, v_index, t_index)` — [`L485`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L485)
- `ht_index_map(head_index, k_index, v_index, t_index)` — [`L482`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L482)
- `k_index_map(head_index, k_index, _, t_index)` — [`L470`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L470)
- `reset_state()` — [`L358`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L358)
- `simple_gla_fwd(q: jax.Array, k: jax.Array, v: jax.Array, *, g: jax.Array | None = None, g_gamma: jax.Array | None = None, scale: float | None = None, h0: jax.Array | None = None, use_ht: bool = False, cu_seqlens_cpu: jax.Array | None = None, cu_seqlens_dev: jax.Array | None = None, chunk_size: int = 64, mode: SimpleGLAKernelMode = SimpleGLAKernelMode.FUSED_CHUNK)` — [`L942`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L942)
- `step(h, xs)` — [`L111`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L111)
- `step(carry, xs)` — [`L174`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L174)
- `store_fn()` — [`L366`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L366)
- `v_index_map(head_index, _, v_index, t_index)` — [`L476`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L476)
- `write_final()` — [`L408`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/simple_gla/simple_gla.py#L408)

