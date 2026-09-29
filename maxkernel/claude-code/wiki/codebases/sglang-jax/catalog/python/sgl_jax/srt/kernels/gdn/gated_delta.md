---
title: 'Module: python/sgl_jax/srt/kernels/gdn/gated_delta.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/gdn/gated_delta.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.gdn.gated_delta`/
symbols:
  decode_gated_delta_rule_ref: decode_gated_delta_rule_ref().
  ragged_gated_delta_rule_ref: ragged_gated_delta_rule_ref().
  ragged_gated_delta_rule_ref.scan_fn: ragged_gated_delta_rule_ref().scan_fn().
  jax_causal_conv1d_prefill: jax_causal_conv1d_prefill().
  jax_causal_conv1d_update: jax_causal_conv1d_update().
  _l2norm: _l2norm().
  _scatter_idx0_safe: _scatter_idx0_safe().
  _gated_delta_step: _gated_delta_step().
---
# Module: [`python/sgl_jax/srt/kernels/gdn/gated_delta.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gdn/gated_delta.py)

## Functions
- `_gated_delta_step(state: jax.Array, q_t: jax.Array, k_t: jax.Array, v_t: jax.Array, g_t: jax.Array, beta_t: jax.Array)` — [`L52`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gdn/gated_delta.py#L52) — Single gated delta step.
- `_l2norm(x: jax.Array, eps: float = 0.000001)` — [`L47`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gdn/gated_delta.py#L47)
- `_scatter_idx0_safe(buf: jax.Array, state_indices: jax.Array, val: jax.Array)` — [`L81`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gdn/gated_delta.py#L81) — ``buf.at[state_indices].set(val)`` with an ``idx == 0`` no-op guard.
- `decode_gated_delta_rule_ref(mixed_qkv: jax.Array, b: jax.Array, a: jax.Array, recurrent_state: jax.Array, A_log: jax.Array, dt_bias: jax.Array, state_indices: jax.Array, *, n_kq: int, n_v: int, d_k: int, d_v: int, has_initial_state: jax.Array | None = None)` — [`L479`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gdn/gated_delta.py#L479) — Decode-only gated delta-rule (parallel single-step across the batch).
- `jax_causal_conv1d_prefill(x: jax.Array, weight: jax.Array, bias: jax.Array | None = None, cu_seqlens: jax.Array | None = None, conv_state: jax.Array | None = None, state_indices: jax.Array | None = None, has_initial_state: jax.Array | None = None, activation: str | None = None)` — [`L95`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gdn/gated_delta.py#L95) — Depthwise causal conv1d over a ragged-batched packed sequence.
- `jax_causal_conv1d_update(x: jax.Array, conv_state: jax.Array, state_indices: jax.Array, weight: jax.Array, bias: jax.Array | None = None, activation: str | None = None, has_initial_state: jax.Array | None = None)` — [`L255`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gdn/gated_delta.py#L255) — Single-token causal conv1d update.
- `ragged_gated_delta_rule_ref(mixed_qkv: jax.Array, b: jax.Array, a: jax.Array, recurrent_state: jax.Array, A_log: jax.Array, dt_bias: jax.Array, cu_seqlens: jax.Array, state_indices: jax.Array, has_initial_state: jax.Array, *, n_kq: int, n_v: int, d_k: int, d_v: int)` — [`L327`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gdn/gated_delta.py#L327) — Ragged gated delta-rule forward (extend / chunked-prefill).
- `scan_fn(state_buf, xs)` — [`L438`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gdn/gated_delta.py#L438)

