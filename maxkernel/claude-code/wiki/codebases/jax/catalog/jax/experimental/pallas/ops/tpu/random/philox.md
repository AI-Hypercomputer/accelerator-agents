---
title: 'Module: jax/experimental/pallas/ops/tpu/random/philox.py'
type: catalog
provenance: extracted
module: jax/experimental/pallas/ops/tpu/random/philox.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax.experimental.pallas.ops.tpu.random.philox`/
symbols:
  philox_4x32_kernel: philox_4x32_kernel().
  philox_4x32_kernel.kernel: philox_4x32_kernel().kernel().
  philox_4x32_count: philox_4x32_count().
  philox_4x32: philox_4x32().
  plphilox_prng_impl: plphilox_prng_impl.
  philox_split: philox_split().
  BLOCK_SIZE: BLOCK_SIZE.
  philox_random_bits: philox_random_bits().
  Shape: Shape.
  mul32_hi_lo: mul32_hi_lo().
  philox_fold_in: philox_fold_in().
  K_HI_32: K_HI_32.
  K_LO_32: K_LO_32.
  MUL_A: MUL_A.
  MUL_B: MUL_B.
---
# Module: [`jax/experimental/pallas/ops/tpu/random/philox.py`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/random/philox.py)

## Functions
- `kernel(offset_ref, key_ref, out_ref)` — [`L116`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/random/philox.py#L116)
- `mul32_hi_lo(x: jax.Array, y: jax.Array)` — [`L38`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/random/philox.py#L38) — Multiplies 2 32-bit values and returns the hi+low bits of the result.
- `philox_4x32(hi0, lo0, hi1, lo1, k_hi, k_lo, rounds=10)` — [`L55`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/random/philox.py#L55) — Philox 4x32 keyed hash function.
- `philox_4x32_count(key, shape: Shape, offset: typing.ArrayLike = 0, fuse_output: bool = True)` — [`L155`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/random/philox.py#L155) — Convenience function to call philox_4x32_kernel with padded shapes.
- `philox_4x32_kernel(key, shape: Shape, unpadded_shape: Shape, block_size: tuple[int, int], offset: typing.ArrayLike = 0, fuse_output: bool = True)` — [`L77`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/random/philox.py#L77) — Generates random bits using the Philox keyed hash function.
- `philox_fold_in(key, data)` — [`L197`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/random/philox.py#L197)
- `philox_random_bits(key, bit_width: int, shape: Shape)` — [`L191`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/random/philox.py#L191)
- `philox_split(key, shape: Shape)` — [`L185`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/random/philox.py#L185) — Splits the key into two keys of the same shape.

## Module values
- `BLOCK_SIZE` — [`L28`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/random/philox.py#L28)
- `K_HI_32` — [`L32`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/random/philox.py#L32)
- `K_LO_32` — [`L33`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/random/philox.py#L33)
- `MUL_A` — [`L34`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/random/philox.py#L34)
- `MUL_B` — [`L35`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/random/philox.py#L35)
- `Shape` — [`L26`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/random/philox.py#L26)
- `plphilox_prng_impl` — [`L202`](../../../../../../../../../../raw/code/jax/jax/experimental/pallas/ops/tpu/random/philox.py#L202)

