---
title: 'Module: python/sgl_jax/srt/kernels/kda/kda.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/kda/kda.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.kda.kda`/
symbols:
  chunk_kda_fwd: chunk_kda_fwd().
  chunk_gated_delta_rule_fwd_h: chunk_gated_delta_rule_fwd_h().
  kda_fwd_intra: kda_fwd_intra().
  chunk_local_cumsum_vector: chunk_local_cumsum_vector().
  assert_shape: assert_shape().
  kda_fwd_intra._make_spec: kda_fwd_intra()._make_spec().
  chunk_kda_fwd_o_gk: chunk_kda_fwd_o_gk().
  kda_gate_chunk_cumsum: kda_gate_chunk_cumsum().
  pallas_kda_gate_cumsum: pallas_kda_gate_cumsum().
  _align_seqs: _align_seqs().
  exp2: exp2().
  kda_fwd_intra.gather: kda_fwd_intra().gather().
  prepare_chunk_indices: prepare_chunk_indices().
  chunk_kda_fwd_o_gk.gather: chunk_kda_fwd_o_gk().gather().
  exp: exp().
  _kda_fwd_intra_kernel: _kda_fwd_intra_kernel().
  kda_fwd_intra._scatter: kda_fwd_intra()._scatter().
  _chunk_gated_delta_rule_fwd_kernel: _chunk_gated_delta_rule_fwd_kernel().
  chunk_gated_delta_rule_fwd_h._t_index_map: chunk_gated_delta_rule_fwd_h()._t_index_map().
  _chunk_kda_fwd_o_gk_pl_kernel: _chunk_kda_fwd_o_gk_pl_kernel().
  kda_fwd_intra._to_bhnd: kda_fwd_intra()._to_bhnd().
  align_up: align_up().
  cdiv: cdiv().
  assert_shape_or_none: assert_shape_or_none().
  get_interpret: get_interpret().
  _chunk_cumsum_kernel: _chunk_cumsum_kernel().
  _prepare_chunk_offsets: _prepare_chunk_offsets().
  _unalign_output: _unalign_output().
  _RCP_LN2: _RCP_LN2.
  _cdiv_jax: _cdiv_jax().
  _pad_to_multiple_jax: _pad_to_multiple_jax().
  pad_to_multiple: pad_to_multiple().
  chunk_local_cumsum_vector._index_map: chunk_local_cumsum_vector()._index_map().
  prepare_lens: prepare_lens().
  _VMEM_HW_LIMIT_BYTES: _VMEM_HW_LIMIT_BYTES.
  _chunk_cumsum_kernel.body: _chunk_cumsum_kernel().body().
  _solve_unit_lower_triangular: _solve_unit_lower_triangular().
  kda_fwd_intra.gather.extract: kda_fwd_intra().gather().extract().
  chunk_gated_delta_rule_fwd_h._h_index_map: chunk_gated_delta_rule_fwd_h()._h_index_map().
  chunk_kda_fwd_o_gk.gather.extract: chunk_kda_fwd_o_gk().gather().extract().
  _align_seqs._build_gather: _align_seqs()._build_gather().
  _align_seqs.repack_4d: _align_seqs().repack_4d().
  _align_seqs.repack_3d: _align_seqs().repack_3d().
  _unalign_output._build_gather: _unalign_output()._build_gather().
---
# Module: [`python/sgl_jax/srt/kernels/kda/kda.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py)

## Functions
- `_align_seqs(tensors_4d, tensors_3d, cu_seqlens, align)` — [`L1044`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L1044)
- `_build_gather(i, gather_idx)` — [`L1053`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L1053)
- `_build_gather(i, gather_idx)` — [`L1083`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L1083)
- `_cdiv_jax(x: jax.Array, y: int)` — [`L39`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L39)
- `_chunk_cumsum_kernel(cu_seqlens_ref, chunk_indices_ref, s_ref, o_ref, *, BT: int, NT: int, REVERSE: bool, HAS_SCALE: bool, scale: float)` — [`L139`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L139)
- `_chunk_gated_delta_rule_fwd_kernel(seqlens_ref, chunk_offsets_ref, k_ref, v_ref, w_ref, g_ref, gk_ref, h0_ref, h_ref, v_new_ref, ht_ref, scratch_ref, *, NT, USE_G, USE_GK, USE_INITIAL_STATE, STORE_FINAL_STATE, SAVE_NEW_VALUE, USE_EXP2)` — [`L563`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L563)
- `_chunk_kda_fwd_o_gk_pl_kernel(q_ref, v_ref, g_ref, h_ref, A_ref, o_ref, *, BT, scale, USE_EXP2)` — [`L834`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L834)
- `_h_index_map(n, h, nt, seqlens_ref, chunk_offsets_ref)` — [`L756`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L756)
- `_index_map(i_s, i_bb, *_)` — [`L249`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L249)
- `_kda_fwd_intra_kernel(q_ref, k_ref, g_ref, beta_ref, v_ref, u_out_ref, w_out_ref, qg_out_ref, kg_out_ref, Aqk_out_ref, Akk_inv_out_ref, *, chunk_size, head_dim, value_dim, scale, disable_recompute, safe_gate)` — [`L328`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L328)
- `_make_spec(last_dim)` — [`L493`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L493)
- `_pad_to_multiple_jax(x: jax.Array, multiple: int | list, axis: int | list, val)` — [`L53`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L53)
- `_prepare_chunk_offsets(seqlens, chunk_size)` — [`L555`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L555)
- `_scatter(chunks_r, D)` — [`L536`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L536)
- `_solve_unit_lower_triangular(A, b)` — [`L278`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L278)
- `_t_index_map(n, h, nt, seqlens_ref, chunk_offsets_ref)` — [`L751`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L751)
- `_to_bhnd(x)` — [`L480`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L480)
- `_unalign_output(o, orig_cu_seqlens, aligned_cu_seqlens, T_out)` — [`L1079`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L1079)
- `align_up(x: int, align: int)` — [`L43`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L43)
- `assert_shape(x, expected_shape, name="tensor")` — [`L107`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L107)
- `assert_shape_or_none(x, expected_shape, name="tensor")` — [`L92`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L92)
- `body(i_t, _)` — [`L153`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L153)
- `cdiv(x: int, y: int)` — [`L34`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L34)
- `chunk_gated_delta_rule_fwd_h(k, w, u, g=None, gk=None, initial_state=None, output_final_state=False, chunk_size=64, save_new_value=True, use_exp2=True, cu_seqlens=None, chunk_indices=None)` — [`L648`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L648)
- `chunk_kda_fwd(q: jax.Array, k: jax.Array, v: jax.Array, g: jax.Array, beta: jax.Array, scale: float, initial_state: jax.Array, output_final_state: bool, cu_seqlens: jax.Array, use_qk_l2norm_in_kernel: bool = False, chunk_indices: jax.Array | None = None, chunk_size: int = 64, safe_gate: bool = True, lower_bound: float | None = None, use_gate_in_kernel: bool = False, A_log: jax.Array | None = None, dt_bias: jax.Array | None = None, disable_recompute: bool = False, return_intermediate_states: bool = False, cp_context: None = None, transpose_state_layout: bool = False)` — [`L1118`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L1118) — KDA chunked forward pass for variable-length sequences (varlen).
- `chunk_kda_fwd_o_gk(q, v, g, A, h, scale, *, cu_seqlens, chunk_indices=None, chunk_size=64, use_exp2=False)` — [`L881`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L881)
- `chunk_local_cumsum_vector(g: jax.Array, chunk_size: int, reverse: bool = False, scale: float | None = None, cu_seqlens: jax.Array | None = None, head_first: bool = False, output_dtype: jnp.dtype | None = jnp.float32, chunk_indices: jax.Array | None = None)` — [`L184`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L184)
- `exp(x)` — [`L119`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L119)
- `exp2(x)` — [`L123`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L123)
- `extract(start)` — [`L467`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L467)
- `extract(start)` — [`L925`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L925)
- `gather(x_pad, D)` — [`L466`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L466)
- `gather(x_pad, D)` — [`L924`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L924)
- `get_interpret()` — [`L127`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L127)
- `kda_fwd_intra(q, k, v, gk, beta, scale, cu_seqlens, chunk_size=64, chunk_indices=None, safe_gate=True, disable_recompute=False)` — [`L416`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L416)
- `kda_gate_chunk_cumsum(g, A_log, chunk_size, scale=None, dt_bias=None, cu_seqlens=None, output_dtype=jnp.float32, chunk_indices=None, lower_bound=None)` — [`L978`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L978)
- `pad_to_multiple(x, multiple: int, axis: int, val)` — [`L48`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L48)
- `pallas_kda_gate_cumsum(g, chunk_size, reverse=False, scale=_RCP_LN2, cu_seqlens=None, head_first=False, output_dtype=jnp.float32, chunk_indices=None)` — [`L1014`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L1014)
- `prepare_chunk_indices(cu_seqlens: jax.Array, chunk_size: int, max_T: int | None = None)` — [`L74`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L74)
- `prepare_lens(cu_seqlens: jax.Array)` — [`L70`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L70)
- `repack_3d(t)` — [`L1068`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L1068)
- `repack_4d(t)` — [`L1065`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L1065)

## Module values
- `_RCP_LN2` — [`L975`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L975)
- `_VMEM_HW_LIMIT_BYTES` — [`L136`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/kda/kda.py#L136)

