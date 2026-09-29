---
title: 'Module: src/maxtext/layers/normalizations.py'
type: catalog
provenance: extracted
module: src/maxtext/layers/normalizations.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.layers.normalizations`/
symbols:
  RMSNorm.__call__: RMSNorm#__call__().
  RMSNorm: RMSNorm#
  Qwen3NextRMSNormGated.rms_norm: Qwen3NextRMSNormGated#rms_norm.
  Qwen3NextRMSNormGated.__call__: Qwen3NextRMSNormGated#__call__().
  RMSNorm.with_scale: RMSNorm#with_scale.
  RMSNorm.scale: RMSNorm#scale.
  GlobalRMSNorm: GlobalRMSNorm#
  GlobalRMSNorm.__call__: GlobalRMSNorm#__call__().
  Qwen3NextRMSNorm: Qwen3NextRMSNorm().
  rms_norm: rms_norm().
  Qwen3NextRMSNormLinen: Qwen3NextRMSNormLinen.
  RMSNorm.dtype: RMSNorm#dtype.
  RMSNorm.epsilon: RMSNorm#epsilon.
  RMSNorm.shard_mode: RMSNorm#shard_mode.
  RMSNorm.parameter_memory_host_offload: RMSNorm#parameter_memory_host_offload.
  RMSNorm.scale_offset: RMSNorm#scale_offset.
  Qwen3NextRMSNormGated.epsilon: Qwen3NextRMSNormGated#epsilon.
  Qwen3NextRMSNormGated.dtype: Qwen3NextRMSNormGated#dtype.
  RMSNorm.__init__: RMSNorm#__init__().
  RMSNorm.num_features: RMSNorm#num_features.
  RMSNorm.weight_dtype: RMSNorm#weight_dtype.
  RMSNorm.kernel_axes: RMSNorm#kernel_axes.
  RMSNorm.scale_init: RMSNorm#scale_init.
  Qwen3NextRMSNormGated: Qwen3NextRMSNormGated#
  Qwen3NextRMSNormGated.__init__: Qwen3NextRMSNormGated#__init__().
  Qwen3NextRMSNormGated.num_features: Qwen3NextRMSNormGated#num_features.
  Qwen3NextRMSNormGated.weight_dtype: Qwen3NextRMSNormGated#weight_dtype.
  l2norm: l2norm().
---
# Module: [`src/maxtext/layers/normalizations.py`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py)

## Classes
### `GlobalRMSNorm`  ·  implements/extends RMSNorm
- def: [`src/maxtext/layers/normalizations.py:95`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L95)
- doc: Applies RMSNorm over the last two dimensions (Heads * HeadDim).
- signature: `class GlobalRMSNorm(RMSNorm):`
- protocol/private: `__call__`[`L101`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L101)
- uses (calls/refs, reference-scoped): [`__call__`](normalizations.md#RMSNorm.__call__), [`RMSNorm`](normalizations.md#RMSNorm)
- used by: [`RMSNorm`](normalizations.md#RMSNorm)

### `Qwen3NextRMSNormGated`  ·  implements/extends Module
- def: [`src/maxtext/layers/normalizations.py:152`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L152)
- doc: This applies RMS Normalization and then a gated activation function (SiLU).
- signature: `class Qwen3NextRMSNormGated(nnx.Module):`
- members:
  - `__call__(self, hidden_states: Array, gate: Array)` — [`L183`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L183) — Applies RMSNorm and then a SiLU gate.
  - `dtype` — [`L170`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L170)
  - `epsilon` — [`L169`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L169)
  - `num_features` — [`L168`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L168)
  - `rms_norm` — [`L172`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L172)
  - `weight_dtype` — [`L171`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L171)
- protocol/private: `__init__`[`L167`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L167)
- uses (calls/refs, reference-scoped): [`RMSNorm`](normalizations.md#RMSNorm)

### `RMSNorm`  ·  implements/extends Module
- def: [`src/maxtext/layers/normalizations.py:33`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L33)
- doc: RMS normalization.
- signature: `class RMSNorm(nnx.Module):`
- members:
  - `__call__(self, x: jnp.ndarray, out_sharding: NamedSharding | None = None)` — [`L69`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L69) — Applies layer normalization on the input.
  - `dtype` — [`L53`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L53)
  - `epsilon` — [`L52`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L52)
  - `kernel_axes` — [`L56`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L56)
  - `num_features` — [`L51`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L51)
  - `parameter_memory_host_offload` — [`L58`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L58)
  - `scale` — [`L62`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L62)
  - `scale_init` — [`L57`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L57)
  - `scale_offset` — [`L59`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L59)
  - `shard_mode` — [`L55`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L55)
  - `weight_dtype` — [`L54`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L54)
  - `with_scale` — [`L60`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L60)
- protocol/private: `__init__`[`L36`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L36)
- uses (calls/refs, reference-scoped): [`GlobalRMSNorm`](normalizations.md#GlobalRMSNorm)
- used by: [`rms_norm`](normalizations.md#Qwen3NextRMSNormGated.rms_norm), [`GlobalRMSNorm`](normalizations.md#GlobalRMSNorm), [`Qwen3NextRMSNorm`](normalizations.md#Qwen3NextRMSNorm), [`Qwen3NextRMSNormLinen`](normalizations.md#Qwen3NextRMSNormLinen), [`__call__`](normalizations.md#GlobalRMSNorm.__call__), [`rms_norm`](normalizations.md#rms_norm)

## Functions
- `Qwen3NextRMSNorm(num_features: int, epsilon: float = 0.000001, dtype: DType = None, weight_dtype: DType = None, shard_mode=None, kernel_axes=None, parameter_memory_host_offload=None, *, rngs: nnx.Rngs)` — [`L117`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L117) — Used for input and post attention layernorms
- `l2norm(x: Array, dim: int = -1, eps: float = 0.000001)` — [`L233`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L233) — L2 normalization function. Normalizes a vector to have a length of 1.
- `rms_norm(num_features: int, epsilon: float = 0.000001, dtype: Any = jnp.float32, weight_dtype: Any = jnp.float32, shard_mode: ShardMode = ShardMode.AUTO, kernel_axes: tuple[None | str, ...] = (), scale_init: Initializer = nn.initializers.ones, name: None | str = None, parameter_memory_host_offload: bool = False, with_scale: bool = True)` — [`L203`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L203) — Creates a RMSNorm module.

## Module values
- `Qwen3NextRMSNormLinen` — [`L249`](../../../../../../../raw/code/maxtext/src/maxtext/layers/normalizations.py#L249)

