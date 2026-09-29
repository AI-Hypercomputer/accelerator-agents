---
title: 'Module: ejkernel/kernels/_xla/ring_attention/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_xla/ring_attention/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._xla.ring_attention._interface`/
symbols:
  ring_attention: ring_attention().
  _ring_attention: _ring_attention().
  ring_attention._divisor_or_self: ring_attention()._divisor_or_self().
---
# Module: [`ejkernel/kernels/_xla/ring_attention/_interface.py`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ring_attention/_interface.py)

## Functions
- `_divisor_or_self(length: int, candidate: int)` — [`L341`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ring_attention/_interface.py#L341) — Find the largest divisor of length that is at most candidate.
- `_ring_attention(query: chex.Array, key: chex.Array, value: chex.Array, bias: chex.Array | None = None, q_segment_ids: chex.Array | None = None, kv_segment_ids: chex.Array | None = None, q_position_ids: chex.Array | None = None, kv_position_ids: chex.Array | None = None, softmax_aux: chex.Array | None = None, axis_name: str | None = None, float32_logits: bool = True, softmax_scale: float | None = None, query_chunk_size: int = 512, key_chunk_size: int = 512, causal_block_size: int | None = None, deterministic: bool = True, dropout_rng: PRNGKeyArray | None = None, pdrop: float = 0, dtype: DTypeLike = jnp.float32, policy=jax.checkpoint_policies.nothing_saveable, precision: lax.PrecisionLike = jax.lax.Precision.DEFAULT, prevent_cse: bool = True, sliding_window: int | tuple[int, int] | None = None, logits_soft_cap: float | None = None, attention_sink_size: int = 0, causal: bool = False)` — [`L104`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ring_attention/_interface.py#L104) — Compute ring attention with blockwise transformers (internal, with custom VJP).
- `ring_attention(query: Float[Array, "batch seq_len_q num_heads head_dim"], key: Float[Array, "batch seq_len_k num_kv_heads head_dim"], value: Float[Array, "batch seq_len_k num_kv_heads head_dim"], q_segment_ids: Int[Array, "batch seq_len_q"] | None = None, kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None, q_position_ids: Int[Array, "batch seq_len_q"] | None = None, kv_position_ids: Int[Array, "batch seq_len_k"] | None = None, softmax_aux: Float[Array, num_sinks] | None = None, bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None, mask_builder: Callable[[int, int, int, int, int], Mask] | None = None, sliding_window: int | tuple[int, int] | None = None, chunk_size: int | None = None, causal: bool = False, logits_soft_cap: float | None = None, softmax_scale: float | None = None, axis_name: str | None = None, fwd_params: FwdParams | None = None, bwd_params: BwdParams | None = None, fused_backward: bool = False)` — [`L237`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/ring_attention/_interface.py#L237) — Compute ring attention for distributed long-sequence processing.

