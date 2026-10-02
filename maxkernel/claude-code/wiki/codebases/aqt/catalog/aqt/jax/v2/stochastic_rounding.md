---
title: 'Module: aqt/jax/v2/stochastic_rounding.py'
type: catalog
provenance: extracted
module: aqt/jax/v2/stochastic_rounding.py
status: fresh
symbol_base: scip-python python aqt 0.0.0 `aqt.jax.v2.stochastic_rounding`/
symbols:
  NoiseFn: NoiseFn.
  _degenerate_noise_shape: _degenerate_noise_shape().
  JaxUniform: JaxUniform#
  JaxUniform.__call__: JaxUniform#__call__().
  RandomCenteredUniform: RandomCenteredUniform#
  RandomCenteredUniform.__call__: RandomCenteredUniform#__call__().
---
# Module: [`aqt/jax/v2/stochastic_rounding.py`](../../../../../../../raw/code/aqt/aqt/jax/v2/stochastic_rounding.py)

## Classes
### `JaxUniform`
- def: [`aqt/jax/v2/stochastic_rounding.py:36`](../../../../../../../raw/code/aqt/aqt/jax/v2/stochastic_rounding.py#L36)
- doc: Jax uniform noise.
- signature: `class JaxUniform:`
- protocol/private: `__call__`[`L39`](../../../../../../../raw/code/aqt/aqt/jax/v2/stochastic_rounding.py#L39)

### `RandomCenteredUniform`
- def: [`aqt/jax/v2/stochastic_rounding.py:50`](../../../../../../../raw/code/aqt/aqt/jax/v2/stochastic_rounding.py#L50)
- doc: Customized efficient implementation for random centered uniform noise.
- signature: `class RandomCenteredUniform:`
- members:
  - `__call__(self, shape: tuple[int, ...], key: jax.Array, noise_sharing_axes: Sequence[int]=())` — [`L53`](../../../../../../../raw/code/aqt/aqt/jax/v2/stochastic_rounding.py#L53) — Generates uniform number in [-0.5, 0.5].

## Functions
- `_degenerate_noise_shape(shape: tuple[int, ...], noise_sharing_axes: Sequence[int]=())` — [`L25`](../../../../../../../raw/code/aqt/aqt/jax/v2/stochastic_rounding.py#L25) — Degenerate the given shape to 1 for the broadcasting axes.

## Module values
- `NoiseFn` — [`L22`](../../../../../../../raw/code/aqt/aqt/jax/v2/stochastic_rounding.py#L22)

