---
title: 'Module: ejkernel/kernels/_xla/attention/_xla_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_xla/attention/_xla_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._xla.attention._xla_impl_fwd`/
symbols:
  attention: attention().
  _normalize_softmax_aux: _normalize_softmax_aux().
---
# Module: [`ejkernel/kernels/_xla/attention/_xla_impl_fwd.py`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/attention/_xla_impl_fwd.py)

## Functions
- `_normalize_softmax_aux(softmax_aux: Float[Array, ...] | None, *, num_q_heads: int, num_kv_heads: int, dtype: jnp.dtype)` — [`L82`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/attention/_xla_impl_fwd.py#L82) — Normalize softmax_aux into per-(kv_head, rep) sink logits for GQA/MQA.
- `attention(query: Float[Array, "batch seq_len num_q_heads head_dim"], key: Float[Array, "batch kv_len num_kv_heads head_dim"], value: Float[Array, "batch kv_len num_kv_heads vhead_dim"], attention_mask: Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | None = None, bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None, init_bias: Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None, deterministic: bool = True, dropout_rng: PRNGKeyArray | None = None, softmax_aux: Float[Array, num_sinks] | None = None, softmax_scale: float | None = None, logits_soft_cap: float | None = None, dtype: DTypeLike | None = jnp.bfloat16, softmax_dtype: DTypeLike | None = None, dropout_prob: float = 0, causal: bool = False, sliding_window: int | tuple[int, int] | None = None)` — [`L146`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/attention/_xla_impl_fwd.py#L146) — Compute multi-head attention using standard JAX operations.

