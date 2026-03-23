# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim1 = 256
dim2 = 256
dim = 1
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim1, dim2))

# Define block sizes that are TPU-compatible
# For the output (rank 2), the last two dimensions of the block must be
# divisible by 8 and 128 respectively.
b_batch = 8
b_dim2 = 128

# The grid is determined by how many blocks fit into the full array.
grid = (batch_size // b_batch, dim2 // b_dim2)


def kernel(x_ref, out_ref):
  """Pallas kernel for argmin.

  Args:
    x_ref: Input array block.
    out_ref: Output array block to store the indices.
  """
  # The input `x_ref` is a block of the original `x` array.
  # We compute `argmin` along the second axis (dim=1) of this block.
  # The result is then written to the corresponding output block `out_ref`.
  # Initialize the minimum values with the first slice and indices with 0.
  min_vals = x_ref[:, 0, :]
  min_indices = jnp.zeros(out_ref.shape, dtype=jnp.int32)

  # Iterate over the reduction axis to find the argmin.
  def body(k, state):
    min_vals, min_indices = state
    current_vals = x_ref[:, k, :]
    # Update mins and indices where a smaller value is found.
    new_min_vals = jnp.minimum(min_vals, current_vals)
    new_min_indices = jnp.where(current_vals < min_vals, k, min_indices)
    return new_min_vals, new_min_indices

  # The loop starts from 1 since we initialized with the 0-th element.
  _, min_indices = lax.fori_loop(1, dim1, body, (min_vals, min_indices))

  out_ref[...] = min_indices


# Computation
result = pl.pallas_call(
  kernel,
  # The output shape is (batch_size, dim2) with an integer dtype for indices.
  out_shape=jax.ShapeDtypeStruct((batch_size, dim2), jnp.int32),
  grid=grid,
  # in_specs describes how to slice the input array `x`.
  # To compute an (8, 128) block of the output, the kernel needs an
  # (8, 256, 128) block of the input, since argmin is on axis=1.
  in_specs=[pl.BlockSpec(block_shape=(b_batch, dim1, b_dim2), index_map=lambda i, j: (i, 0, j))],
  # out_specs describes how the output is laid out. Each kernel instance (i, j)
  # writes to a unique (8, 128) block.
  out_specs=pl.BlockSpec(block_shape=(b_batch, b_dim2), index_map=lambda i, j: (i, j)),
)(x).block_until_ready()
