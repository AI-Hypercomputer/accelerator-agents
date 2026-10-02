---
title: 'Module: jax/experimental/pallas/ops/tpu/random/threefry.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/tpu/random/threefry.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.tpu.random.threefry`/
symbols:
  threefry_2x32_count: threefry_2x32_count().
  threefry_2x32_count.kernel: threefry_2x32_count().kernel().
  plthreefry_random_bits: plthreefry_random_bits().
  BLOCK_SIZE: BLOCK_SIZE.
  plthreefry_prng_impl: plthreefry_prng_impl.
  Shape: Shape.
---
# Module: [`jax/experimental/pallas/ops/tpu/random/threefry.py`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/random/threefry.py)

## Functions
- `kernel(key_ref, out_ref)` — [`L62`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/random/threefry.py#L62)
- `plthreefry_random_bits(key, bit_width: int, shape: Shape)` — [`L90`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/random/threefry.py#L90)
- `threefry_2x32_count(key, shape: Shape, unpadded_shape: Shape, block_size: tuple[int, int])` — [`L30`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/random/threefry.py#L30) — Generates random bits using the Threefry hash function.

## Module values
- `BLOCK_SIZE` — [`L27`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/random/threefry.py#L27)
- `Shape` — [`L25`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/random/threefry.py#L25)
- `plthreefry_prng_impl` — [`L112`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/random/threefry.py#L112)

