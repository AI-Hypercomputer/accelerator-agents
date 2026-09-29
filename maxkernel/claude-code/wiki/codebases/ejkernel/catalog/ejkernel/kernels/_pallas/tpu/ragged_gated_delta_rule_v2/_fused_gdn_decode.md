---
title: 'Module: ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_fused_gdn_decode.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_fused_gdn_decode.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.ragged_gated_delta_rule_v2._fused_gdn_decode`/
symbols:
  fused_gdn_decode: fused_gdn_decode().
  _decode_kernel_main: _decode_kernel_main().
  _decode_kernel_main.token_map: _decode_kernel_main().token_map().
  _validate_decode_inputs: _validate_decode_inputs().
  _default_decode_bt: _default_decode_bt().
  _decode_kernel_main._inner_kernel: _decode_kernel_main()._inner_kernel().
  _decode_kernel_main._first_load: _decode_kernel_main()._first_load().
  _decode_kernel_main._inner_kernel._prefetch: _decode_kernel_main()._inner_kernel()._prefetch().
  _decode_kernel_main._inner_kernel._process_token: _decode_kernel_main()._inner_kernel()._process_token().
  _decode_kernel_main._inner_kernel._wait_prev_store: _decode_kernel_main()._inner_kernel()._wait_prev_store().
  _decode_kernel_main._inner_kernel._start_store: _decode_kernel_main()._inner_kernel()._start_store().
  _decode_kernel_main._drain_other: _decode_kernel_main()._drain_other().
---
# Module: [`ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_fused_gdn_decode.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_fused_gdn_decode.py)

## Functions
- `_decode_kernel_main(q_hbm, k_hbm, v_hbm, g_hbm, b_hbm, state_indices_ref, distribution_ref, _state_init_ref, o_hbm, state_hbm, h_bufs, h_load_sems, h_store_sems, *, h: int, k_dim: int, v_dim: int, bt: int)` — [`L141`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_fused_gdn_decode.py#L141) — Pallas program that runs the fused GDN recurrent decode update.
- `_default_decode_bt(*, t: int, h: int, k_dim: int, v_dim: int, dtype, state_dtype)` — [`L100`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_fused_gdn_decode.py#L100) — Choose a token block size that fits comfortably in TPU VMEM.
- `_drain_other()` — [`L415`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_fused_gdn_decode.py#L415) — Drain the second-to-last block's outstanding state store after the pipeline finishes.
- `_first_load(i_t=i_t)` — [`L220`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_fused_gdn_decode.py#L220) — Kick off the asynchronous HBM-to-VMEM load of the first pipeline block's recurrent state.
- `_inner_kernel(q_ref, k_ref, v_ref, g_ref, b_ref, o_ref, h_bufs_s, state_indices_s, h_load_sems_s, h_store_sems_s)` — [`L238`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_fused_gdn_decode.py#L238) — Run one ``emit_pipeline`` step over a block of decode tokens with software-pipelined state DMAs.
- `_prefetch(i_t=i_t)` — [`L281`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_fused_gdn_decode.py#L281) — Start the asynchronous load of one token's state slab for the next pipeline block.
- `_process_token(i_t=i_t)` — [`L307`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_fused_gdn_decode.py#L307) — Apply the gated delta-rule recurrent update for a single decode token in VMEM.
- `_start_store(i_t=i_t)` — [`L370`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_fused_gdn_decode.py#L370) — Start the asynchronous store of one token's updated state slab back to the HBM state pool.
- `_validate_decode_inputs(q: jax.Array, k: jax.Array, v: jax.Array, g: jax.Array, state: jax.Array, state_indices: jax.Array, b: jax.Array)` — [`L34`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_fused_gdn_decode.py#L34) — Validate fused decode inputs and return shape metadata.
- `_wait_prev_store()` — [`L354`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_fused_gdn_decode.py#L354) — Wait for the state store issued two blocks earlier to complete before reusing this double buffer.
- `fused_gdn_decode(q: jax.Array, k: jax.Array, v: jax.Array, g: jax.Array, b: jax.Array, state: jax.Array, state_indices: jax.Array, distribution: jax.Array)` — [`L430`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_fused_gdn_decode.py#L430) — Run the fused TPU decode kernel for Gated Delta Rule recurrent attention.
- `token_map(i)` — [`L196`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_fused_gdn_decode.py#L196) — Map a pipeline block index to its dynamic token slice into HBM activation refs.

