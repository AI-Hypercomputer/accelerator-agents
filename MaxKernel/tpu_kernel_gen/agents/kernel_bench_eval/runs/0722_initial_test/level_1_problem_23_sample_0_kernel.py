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

# Define a block size for the batch dimension that is compatible with TPU hardware.
# The dimension of a block that is not fully sized must be a multiple of 8.
batch_block_size = 8
# Define a block size for the reduction dimension. This must be small enough
# to fit in VMEM and should ideally be a divisor of the dimension size.
dim_block_size = 1024


def kernel(x_ref, out_ref):
  # Pass 1: Find the max of each row for numerical stability.
  # We iterate over the large `dim` dimension in smaller chunks.
  row_max = jnp.full((batch_block_size,), -jnp.inf)
  for j in range(dim // dim_block_size):
    offset = j * dim_block_size
    x_chunk = pl.load(x_ref, (pl.dslice(None), pl.dslice(offset, dim_block_size)))
    max_in_chunk = jnp.max(x_chunk, axis=1)
    row_max = jnp.maximum(row_max, max_in_chunk)

  # Pass 2: Compute the denominator.
  denominator = jnp.zeros((batch_block_size,))
  row_max_bcast = row_max[:, None]  # Prepare for broadcasting
  for j in range(dim // dim_block_size):
    offset = j * dim_block_size
    x_chunk = pl.load(x_ref, (pl.dslice(None), pl.dslice(offset, dim_block_size)))
    numerator_chunk = jnp.exp(x_chunk - row_max_bcast)
    denominator += jnp.sum(numerator_chunk, axis=1)

  denominator_bcast = denominator[:, None]  # Prepare for broadcasting

  # Pass 3: Compute the final result and store it back to HBM.
  for j in range(dim // dim_block_size):
    offset = j * dim_block_size
    x_chunk = pl.load(x_ref, (pl.dslice(None), pl.dslice(offset, dim_block_size)))
    numerator_chunk = jnp.exp(x_chunk - row_max_bcast)
    output_chunk = numerator_chunk / denominator_bcast
    pl.store(out_ref, (pl.dslice(None), pl.dslice(offset, dim_block_size)), output_chunk)


# Computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size // batch_block_size,),
  in_specs=[pl.BlockSpec((batch_block_size, dim), lambda i: (i * batch_block_size, 0))],
  out_specs=pl.BlockSpec((batch_block_size, dim), lambda i: (i * batch_block_size, 0)),
)(x).block_until_ready()
