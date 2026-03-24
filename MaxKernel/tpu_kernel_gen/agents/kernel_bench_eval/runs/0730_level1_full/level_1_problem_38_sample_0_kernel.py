# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))
block_batch = 8


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel to normalize rows by their L1 norm."""
  block_d = 1024  # Process dim in chunks to avoid memory issues

  # Pass 1: Compute the L1 norm for each row in the block.
  row_sums = jnp.zeros(block_batch, dtype=x_ref.dtype)

  def sum_body(j, acc):
    # Use pl.dslice for dynamic slicing. Using it for both dimensions,
    # even when one is static, can help avoid layout compilation issues.
    # The modern syntax `ref[idx]` is preferred over `pl.load`.
    idx = (pl.dslice(0, block_batch), pl.dslice(j * block_d, block_d))
    x_chunk = x_ref[idx]
    return acc + jnp.sum(jnp.abs(x_chunk), axis=1)

  row_sums = jax.lax.fori_loop(0, dim // block_d, sum_body, row_sums)
  row_sums = row_sums[:, None]  # Reshape for broadcasting

  # Pass 2: Normalize and store the result.
  def normalize_body(j, _):
    idx = (pl.dslice(0, block_batch), pl.dslice(j * block_d, block_d))
    x_chunk = x_ref[idx]
    normalized_chunk = x_chunk / row_sums
    # The modern syntax `ref[idx] = val` is preferred over `pl.store`.
    out_ref[idx] = normalized_chunk

  jax.lax.fori_loop(0, dim // block_d, normalize_body, None)


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size // block_batch,),
  in_specs=[pl.BlockSpec(block_shape=(block_batch, dim), index_map=lambda i: (i * block_batch, 0))],
  out_specs=pl.BlockSpec(block_shape=(block_batch, dim), index_map=lambda i: (i * block_batch, 0)),
)(x).block_until_ready()
