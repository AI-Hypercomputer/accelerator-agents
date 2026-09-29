---
title: 'Module: jax/_src/pallas/mosaic/random.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic/random.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic.random`/
symbols:
  to_pallas_key: to_pallas_key().
  tpu_key_impl: tpu_key_impl.
  tpu_internal_stateful_impl: tpu_internal_stateful_impl.
  sample_block: sample_block().
  _make_stateful_sampler: _make_stateful_sampler().
  is_pallas_impl: is_pallas_impl().
  _seed_func: _seed_func().
  _random_bits: _random_bits().
  _fold_in: _fold_in().
  _pl_stateful_random_bits: _pl_stateful_random_bits().
  Shape: Shape.
  _make_stateful_sampler.new_sampler: _make_stateful_sampler().new_sampler().
  _split: _split().
  _pl_stateful_seed_func: _pl_stateful_seed_func().
  _pl_stateful_split: _pl_stateful_split().
  stateful_bits: stateful_bits.
  stateful_uniform: stateful_uniform.
  stateful_bernoulli: stateful_bernoulli.
  stateful_normal: stateful_normal.
  KeylessSampleFnType: KeylessSampleFnType.
  unwrap_pallas_seed: unwrap_pallas_seed.
  wrap_pallas_seed: wrap_pallas_seed.
  _pl_stateful_fold_in: _pl_stateful_fold_in().
  set_seed: set_seed.
  SampleFnType: SampleFnType.
---
# Module: [`jax/_src/pallas/mosaic/random.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/random.py)

## Functions
- `_fold_in(key: jax_prng.PRNGKeyArray, data: typing.Array)` — [`L75`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/random.py#L75)
- `_make_stateful_sampler(sampler: SampleFnType)` — [`L137`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/random.py#L137) — Converts a jax.random sampling function to a stateful version.
- `_pl_stateful_fold_in(key: typing.Array, data: typing.Array)` — [`L117`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/random.py#L117)
- `_pl_stateful_random_bits(key: typing.Array, bit_width: int, shape: Shape)` — [`L112`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/random.py#L112)
- `_pl_stateful_seed_func(seed: jnp.int32)` — [`L107`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/random.py#L107)
- `_pl_stateful_split(key: typing.Array, shape: Shape)` — [`L121`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/random.py#L121)
- `_random_bits(key: typing.Array, bit_width: int, shape: Shape)` — [`L69`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/random.py#L69)
- `_seed_func(seed: jnp.int32)` — [`L65`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/random.py#L65)
- `_split(key: typing.Array, shape: Shape)` — [`L82`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/random.py#L82)
- `is_pallas_impl(impl: jax_prng.PRNGImpl)` — [`L60`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/random.py#L60) — Returns True if the PRNGImpl is a Pallas-specific implementation.
- `new_sampler(*args, **kwargs)` — [`L147`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/random.py#L147)
- `sample_block(sampler_fn: SampleFnType, global_key: jax.Array, block_size: Shape, tile_size: Shape, total_size: Shape, block_index: tuple[typing.ArrayLike, ...] | None = None, **kwargs)` — [`L166`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/random.py#L166) — Samples a block of random values with invariance guarantees.
- `to_pallas_key(key: jax.Array)` — [`L40`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/random.py#L40) — Helper function for converting non-Pallas PRNG keys into Pallas keys.

## Module values
- `KeylessSampleFnType` — [`L33`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/random.py#L33)
- `SampleFnType` — [`L32`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/random.py#L32)
- `Shape` — [`L31`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/random.py#L31)
- `set_seed` — [`L35`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/random.py#L35)
- `stateful_bernoulli` — [`L162`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/random.py#L162)
- `stateful_bits` — [`L160`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/random.py#L160)
- `stateful_normal` — [`L163`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/random.py#L163)
- `stateful_uniform` — [`L161`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/random.py#L161)
- `tpu_internal_stateful_impl` — [`L126`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/random.py#L126)
- `tpu_key_impl` — [`L88`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/random.py#L88)
- `unwrap_pallas_seed` — [`L36`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/random.py#L36)
- `wrap_pallas_seed` — [`L37`](../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/random.py#L37)

