---
title: 'Module: ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_chunked.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_chunked.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._xla.fused_cross_entropy._xla_impl_chunked`/
symbols:
  blockwise_cross_entropy: blockwise_cross_entropy().
  chunked_vocab_cross_entropy: chunked_vocab_cross_entropy().
  chunked_token_cross_entropy: chunked_token_cross_entropy().
  _logsumexp_chunked: _logsumexp_chunked().
  _apply_sparse_label_smoothing: _apply_sparse_label_smoothing().
  chunked_token_cross_entropy._chunk: chunked_token_cross_entropy()._chunk().
  blockwise_cross_entropy.process_block: blockwise_cross_entropy().process_block().
  blockwise_cross_entropy.full_body: blockwise_cross_entropy().full_body().
  chunked_token_cross_entropy.body: chunked_token_cross_entropy().body().
  _logsumexp_chunked.max_body: _logsumexp_chunked().max_body().
  _logsumexp_chunked.sum_body: _logsumexp_chunked().sum_body().
  _label_smoothing_params: _label_smoothing_params().
  __all__: __all__.
---
# Module: [`ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_chunked.py`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_chunked.py)

## Functions
- `_apply_sparse_label_smoothing(log_z: jax.Array, target_logit: jax.Array, sum_logits: jax.Array, *, vocab_size: int, label_smoothing: float, dtype: jnp.dtype)` — [`L103`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_chunked.py#L103) — Label-smoothed sparse cross-entropy from streamed sufficient statistics.
- `_chunk(chunk_logits, chunk_targets, chunk_weights)` — [`L333`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_chunked.py#L333)
- `_label_smoothing_params(vocab_size: int, label_smoothing: float, dtype: jnp.dtype)` — [`L82`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_chunked.py#L82) — Return ``(confidence, low_confidence, normalizing_constant)`` for smoothing.
- `_logsumexp_chunked(x: jnp.ndarray, chunk_size: int)` — [`L44`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_chunked.py#L44) — ``logsumexp`` over the last axis computed in ``chunk_size`` vocab slices.
- `blockwise_cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray, weights: jnp.ndarray | None = None, *, ignore_index: int = -100, label_smoothing: float = 0, z_loss: float = 0, reduction: str = "sum", block_size: int = 8192, compute_dtype: jnp.dtype = jnp.float32, checkpoint: bool = True)` — [`L183`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_chunked.py#L183) — Single-pass online-softmax blockwise sparse cross-entropy.
- `body(i, carry)` — [`L357`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_chunked.py#L357)
- `chunked_token_cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray, weights: jnp.ndarray | None = None, *, ignore_index: int = -100, label_smoothing: float = 0, z_loss: float = 0, reduction: str = "sum", token_chunk_size: int = 8192, compute_dtype: jnp.dtype = jnp.float32)` — [`L301`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_chunked.py#L301) — Sparse cross-entropy with token/row-dimension chunking.
- `chunked_vocab_cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray, weights: jnp.ndarray | None = None, *, ignore_index: int = -100, label_smoothing: float = 0, z_loss: float = 0, reduction: str = "mean", chunk_size: int = 8192, compute_dtype: jnp.dtype = jnp.float32)` — [`L127`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_chunked.py#L127) — Sparse cross-entropy with vocabulary-dimension chunking.
- `full_body(i, carry)` — [`L262`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_chunked.py#L262)
- `max_body(i, m)` — [`L55`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_chunked.py#L55)
- `process_block(start, size, m, log_z, o, sum_logits, best_logit, best_id)` — [`L238`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_chunked.py#L238)
- `sum_body(i, s)` — [`L67`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_chunked.py#L67)

## Module values
- `__all__` — [`L389`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_chunked.py#L389)

