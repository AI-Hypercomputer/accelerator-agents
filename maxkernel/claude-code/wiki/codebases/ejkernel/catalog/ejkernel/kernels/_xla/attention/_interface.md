---
title: 'Module: ejkernel/kernels/_xla/attention/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_xla/attention/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._xla.attention._interface`/
symbols:
  attention: attention().
  __all__: __all__.
---
# Module: [`ejkernel/kernels/_xla/attention/_interface.py`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/attention/_interface.py)

## Functions
- `attention(query: Float[Array, "batch seq_len num_q_heads head_dim"], key: Float[Array, "batch kv_len num_kv_heads head_dim"], value: Float[Array, "batch kv_len num_kv_heads vhead_dim"], attention_mask: Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | None = None, bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None, init_bias: Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None, deterministic: bool = True, dropout_rng: PRNGKeyArray | None = None, softmax_aux: Float[Array, num_sinks] | None = None, softmax_scale: float | None = None, logits_soft_cap: float | None = None, dtype: DTypeLike | None = jnp.bfloat16, softmax_dtype: DTypeLike | None = None, dropout_prob: float = 0, causal: bool = False, sliding_window: int | tuple[int, int] | None = None, fwd_params: FwdParams | None = None, bwd_params: BwdParams | None = None, *, weights_block_q: int = 64, weights_block_k: int = 64)` — [`L35`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/attention/_interface.py#L35) — Compute multi-head attention using standard JAX operations.

## Module values
- `__all__` — [`L149`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/attention/_interface.py#L149)

