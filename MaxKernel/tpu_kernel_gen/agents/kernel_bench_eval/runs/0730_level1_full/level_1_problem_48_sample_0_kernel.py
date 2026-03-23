# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.experimental import pallas as pl

# Initialization
dim = 1
batch_size = 16
dim1 = 256
dim2 = 256
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim1, dim2))
block_size_batch = 8
block_size_dim1 = 64
block_size_dim2 = 128


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel to compute the mean along axis=1."""
  # Get the grid indices for the batch and dim2 dimensions.
  i = pl.program_id(axis=0)
  j = pl.program_id(axis=1)

  # Compute the base offsets for this kernel instance.
  batch_offset = i * block_size_batch
  dim2_offset = j * block_size_dim2

  # Initialize an accumulator for the sum.
  acc = jnp.zeros(out_ref.shape, dtype=x.dtype)

  # Loop over the reduction dimension (dim1).
  def body(k, current_acc):
    # Load a slice of x.
    x_slice = pl.load(
      x_ref,
      (
        pl.dslice(batch_offset, block_size_batch),
        pl.dslice(k * block_size_dim1, block_size_dim1),
        pl.dslice(dim2_offset, block_size_dim2),
      ),
    )
    # Sum over the reduction axis of the slice and add to the accumulator.
    return current_acc + jnp.sum(x_slice, axis=1)

  # Run the loop.
  acc = lax.fori_loop(0, dim1 // block_size_dim1, body, acc)

  # Compute the mean and write to the output tile.
  out_ref[...] = acc / dim1


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, dim2), x.dtype),
  grid=(batch_size // block_size_batch, dim2 // block_size_dim2),
  in_specs=[pl.BlockSpec(x.shape, lambda i, j: (0, 0, 0))],
  out_specs=pl.BlockSpec(
    block_shape=(block_size_batch, block_size_dim2),
    index_map=lambda i, j: (i * block_size_batch, j * block_size_dim2),
  ),
)(x).block_until_ready()
