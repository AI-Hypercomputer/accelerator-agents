---
title: 'Module: qwix/_src/core/stochastic_rounding.py'
type: catalog
provenance: extracted
module: qwix/_src/core/stochastic_rounding.py
status: fresh
symbol_base: scip-python python qwix 0.0.0 `qwix._src.core.stochastic_rounding`/
symbols:
  NoiseFn.__call__: NoiseFn#__call__().
  get_noise_fn: get_noise_fn().
  NoiseFn.method: NoiseFn#method.
  uniform_noise: uniform_noise().
  NoiseFn: NoiseFn#
  NoiseFn.key: NoiseFn#key.
  NoiseFn.channelwise_noise_axes: NoiseFn#channelwise_noise_axes.
  _get_noise_shape: _get_noise_shape().
  low_bit_uniform_noise: low_bit_uniform_noise().
---
# Module: [`qwix/_src/core/stochastic_rounding.py`](../../../../../../../raw/code/qwix/qwix/_src/core/stochastic_rounding.py)

## Classes
### `NoiseFn`
- def: [`qwix/_src/core/stochastic_rounding.py:85`](../../../../../../../raw/code/qwix/qwix/_src/core/stochastic_rounding.py#L85)
- doc: Noise function for stochastic rounding that is JAX PyTree compatible.
- signature: `class NoiseFn:`
- members:
  - `channelwise_noise_axes` — [`L90`](../../../../../../../raw/code/qwix/qwix/_src/core/stochastic_rounding.py#L90)
  - `key` — [`L88`](../../../../../../../raw/code/qwix/qwix/_src/core/stochastic_rounding.py#L88)
  - `method` — [`L89`](../../../../../../../raw/code/qwix/qwix/_src/core/stochastic_rounding.py#L89)
- protocol/private: `__call__`[`L92`](../../../../../../../raw/code/qwix/qwix/_src/core/stochastic_rounding.py#L92)
- uses (calls/refs, reference-scoped): [`uniform_noise`](stochastic_rounding.md#uniform_noise), [`_get_noise_shape`](stochastic_rounding.md#_get_noise_shape), [`low_bit_uniform_noise`](stochastic_rounding.md#low_bit_uniform_noise)
- used by: [`get_noise_fn`](stochastic_rounding.md#get_noise_fn), [`dlhs_stochastic_rounding_noise_fn`](dot_general_qt.md#DotGeneralQtConfig.dlhs_stochastic_rounding_noise_fn), [`drhs_stochastic_rounding_noise_fn`](dot_general_qt.md#DotGeneralQtConfig.drhs_stochastic_rounding_noise_fn)

## Functions
- `_get_noise_shape(channelwise_noise_axes: Sequence[int], shape: tuple[int, ...])` — [`L75`](../../../../../../../raw/code/qwix/qwix/_src/core/stochastic_rounding.py#L75)
- `get_noise_fn(method: str, key: jax.Array, channelwise_noise_axes: Sequence[int] = (0,))` — [`L105`](../../../../../../../raw/code/qwix/qwix/_src/core/stochastic_rounding.py#L105) — Returns a noise function for stochastic rounding. — documented in [qwix-_src-providers-qt](../../../../concepts/qwix-_src-providers-qt.md)
- `low_bit_uniform_noise(key: jax.Array, shape: tuple[int, ...])` — [`L22`](../../../../../../../raw/code/qwix/qwix/_src/core/stochastic_rounding.py#L22) — Random float32 numbers in nearly (-0.5, 0.5) of shape `shape`.
- `uniform_noise(key: jax.Array, shape: tuple[int, ...])` — [`L67`](../../../../../../../raw/code/qwix/qwix/_src/core/stochastic_rounding.py#L67) — Uniform noise.

