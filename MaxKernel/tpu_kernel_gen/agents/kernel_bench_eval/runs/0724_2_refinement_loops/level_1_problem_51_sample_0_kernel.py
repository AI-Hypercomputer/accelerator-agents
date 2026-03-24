# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim1 = 256
dim2 = 256
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim1, dim2))
b_batch = 8
b_dim2 = 128


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel to compute argmax along axis=1.

  This kernel receives a block of the input data `x_ref` and computes the
  argmax along its second dimension (axis=1). The result is written to the
  corresponding block in the output `out_ref`.

  Args:
    x_ref: A reference to a block of the input tensor of shape
      (b_batch, dim1, b_dim2).
    out_ref: A reference to a block of the output tensor of shape
      (b_batch, b_dim2) where the result is stored.
  """
  # Initialize max values and corresponding indices
  max_vals = jnp.full(out_ref.shape, -jnp.inf, dtype=x_ref.dtype)
  max_indices = jnp.zeros(out_ref.shape, dtype=jnp.int32)

  # Iterate over the reduction axis (dim1)
  for k in range(dim1):
    current_vals = x_ref[:, k, :]
    # Find where the current values are greater than the max values
    should_update = current_vals > max_vals
    # Update max values and indices
    max_vals = jnp.where(should_update, current_vals, max_vals)
    max_indices = jnp.where(should_update, k, max_indices)

  # Store the final indices in the output block
  out_ref[...] = max_indices


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, dim2), jnp.int32),
  grid=(batch_size // b_batch, dim2 // b_dim2),
  in_specs=[pl.BlockSpec(block_shape=(b_batch, dim1, b_dim2), index_map=lambda i, j: (i, 0, j))],
  out_specs=pl.BlockSpec(block_shape=(b_batch, b_dim2), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()
