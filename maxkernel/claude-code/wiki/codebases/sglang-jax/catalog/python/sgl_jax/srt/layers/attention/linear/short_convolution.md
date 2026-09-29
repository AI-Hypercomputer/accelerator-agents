---
title: 'Module: python/sgl_jax/srt/layers/attention/linear/short_convolution.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/layers/attention/linear/short_convolution.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.layers.attention.linear.short_convolution`/
symbols:
  short_convolution: short_convolution().
  _decode_conv: _decode_conv().
  _extend_conv: _extend_conv().
  _resolve_activation: _resolve_activation().
  _ACTIVATION_FNS._ACTIVATION_FNS: _ACTIVATION_FNS._ACTIVATION_FNS.
  _apply_activation: _apply_activation().
  _normalize_weight: _normalize_weight().
  __all__: __all__.
---
# Module: [`python/sgl_jax/srt/layers/attention/linear/short_convolution.py`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/short_convolution.py)

## Functions
- `_apply_activation(y: jax.Array, activation_fn: Callable[[jax.Array], jax.Array] | None)` — [`L103`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/short_convolution.py#L103)
- `_decode_conv(x: jax.Array, conv_kernel: jax.Array, cache: jax.Array, bias: jax.Array | None, activation_fn: Callable[[jax.Array], jax.Array] | None)` — [`L113`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/short_convolution.py#L113)
- `_extend_conv(x: jax.Array, conv_kernel: jax.Array, cache: jax.Array, cu_seqlens: jax.Array, bias: jax.Array | None, activation_fn: Callable[[jax.Array], jax.Array] | None)` — [`L131`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/short_convolution.py#L131)
- `_normalize_weight(weight: jax.Array)` — [`L95`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/short_convolution.py#L95) — Reduce common conv-weight layouts to ``[D, K]``.
- `_resolve_activation(activation: str | Callable[[jax.Array], jax.Array] | None)` — [`L39`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/short_convolution.py#L39) — Resolve an activation spec to a callable (or None for identity).
- `short_convolution(x: jax.Array, weight: jax.Array, cache: jax.Array, cu_seqlens: jax.Array | None, forward_mode: ForwardMode, bias: jax.Array | None = None, activation: str | Callable[[jax.Array], jax.Array] | None = "silu")` — [`L56`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/short_convolution.py#L56) — Depthwise causal conv1d with per-sequence cache.

## Module values
- `_ACTIVATION_FNS` — [`L28`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/short_convolution.py#L28)
- `__all__` — [`L191`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/linear/short_convolution.py#L191)

