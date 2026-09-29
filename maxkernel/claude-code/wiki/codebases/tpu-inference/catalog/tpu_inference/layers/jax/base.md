---
title: 'Module: tpu_inference/layers/jax/base.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/jax/base.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.jax.base`/
symbols:
  create_param: create_param().
  _init_fn: _init_fn.
  Config.maybe_apply_overrides: Config#maybe_apply_overrides().
  logger: logger.
  sharded_initializer: sharded_initializer.
  Config.__post_init__: Config#__post_init__().
  scale_initializer: scale_initializer.
  Config._overrides_str: Config#_overrides_str().
  Config._get_overrides_dict: Config#_get_overrides_dict().
  Initializer: Initializer.
  Config: Config#
  Config.from_cfg: Config#from_cfg().
---
# Module: [`tpu_inference/layers/jax/base.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/base.py)

## Classes
### `Config`
- def: [`tpu_inference/layers/jax/base.py:38`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/base.py#L38)
- doc: Base configuration class with a robust factory method.
- signature: `class Config:`
- members:
  - `_get_overrides_dict(self)` — [`L98`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/base.py#L98) — Return the overrides from all of the possible vllm sections.
  - `from_cfg(cls, cfg: dict[str, Any] | None = None, **kwargs)` — [`L47`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/base.py#L47) — Creates a config instance from a dictionary and/or keyword arguments.
  - `maybe_apply_overrides(self)` — [`L86`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/base.py#L86) — Update the args with additional_configs, hf_overrides, and override_generation_config settings.
- protocol/private: `__post_init__`[`L145`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/base.py#L145), `_overrides_str`[`L94`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/base.py#L94)
- uses (calls/refs, reference-scoped): [`logger`](base.md#logger)

## Functions
- `create_param(rngs: nnx.Rngs, shape: tuple[int, ...], sharding: Sharding = (), dtype: Any = jnp.float32, random_init=False)` — [`L149`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/base.py#L149) — documented in [tpu_inference-layers-jax-attention](../../../../concepts/tpu_inference-layers-jax-attention.md)

## Module values
- `Initializer` — [`L28`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/base.py#L28)
- `_init_fn` — [`L34`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/base.py#L34)
- `logger` — [`L29`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/base.py#L29)
- `scale_initializer` — [`L32`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/base.py#L32)
- `sharded_initializer` — [`L33`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/base.py#L33)

