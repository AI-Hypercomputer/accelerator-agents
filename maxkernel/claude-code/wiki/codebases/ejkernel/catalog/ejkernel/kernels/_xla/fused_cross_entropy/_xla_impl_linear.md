---
title: 'Module: ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_linear.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_linear.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._xla.fused_cross_entropy._xla_impl_linear`/
symbols:
  fused_linear_cross_entropy: fused_linear_cross_entropy().
  fused_linear_cross_entropy._chunk_loss: fused_linear_cross_entropy()._chunk_loss().
  fused_linear_cross_entropy._chunk_step: fused_linear_cross_entropy()._chunk_step().
  fused_linear_cross_entropy._accumulate: fused_linear_cross_entropy()._accumulate().
  PerChunkCEFn: PerChunkCEFn.
  _default_token_chunk_size: _default_token_chunk_size().
  fused_linear_cross_entropy._project: fused_linear_cross_entropy()._project().
  fused_linear_cross_entropy._chunk_step._skip: fused_linear_cross_entropy()._chunk_step()._skip().
  __all__: __all__.
---
# Module: [`ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_linear.py`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_linear.py)

## Functions
- `_accumulate(i, carry)` — [`L261`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_linear.py#L261)
- `_chunk_loss(chunk_hidden, chunk_targets, chunk_weights)` — [`L153`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_linear.py#L153)
- `_chunk_step(ch, ct, cw, i)` — [`L245`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_linear.py#L245)
- `_default_token_chunk_size(seq_len: int, vocab_size: int | None, dtype_bytes: int)` — [`L54`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_linear.py#L54) — Pick a token chunk that keeps the transient ``[chunk, V]`` logits ~<=1 GiB.
- `_project(chunk_hidden)` — [`L141`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_linear.py#L141)
- `_skip()` — [`L248`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_linear.py#L248)
- `fused_linear_cross_entropy(hidden: jnp.ndarray, targets: jnp.ndarray, weights: jnp.ndarray | None = None, *, lm_head_weight: jnp.ndarray | None = None, lm_head_bias: jnp.ndarray | None = None, lm_head_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None, logit_softcap: float | None = None, ignore_index: int = -100, label_smoothing: float = 0, z_loss: float = 0, reduction: str = "mean", token_chunk_size: int = 0, compute_dtype: jnp.dtype | None = None, checkpoint: bool = True, per_chunk_ce_fn: PerChunkCEFn | None = None, vocab_parallel_axis: str | None = None, sparse_skip: bool = False, sparse_reduce_axes: tuple[str, ...] = ())` — [`L65`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_linear.py#L65) — Token-chunked fused linear cross-entropy.

## Module values
- `PerChunkCEFn` — [`L48`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_linear.py#L48)
- `__all__` — [`L286`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/fused_cross_entropy/_xla_impl_linear.py#L286)

