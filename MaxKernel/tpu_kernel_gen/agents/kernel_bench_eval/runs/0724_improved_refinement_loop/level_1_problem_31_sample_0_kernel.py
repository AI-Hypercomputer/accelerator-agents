# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
alpha = 1.0
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))
# Define the block size for the inner dimension, adhering to TPU constraints.
# The dimension size is 16384, which is divisible by 128.
block_dim = 128


# Computation
def kernel(x_ref, out_ref):
  """
  Pallas kernel for the ELU activation function.

  Args:
    x_ref: A reference to a block of the input tensor `x`.
    alpha: The scalar alpha value for the ELU computation.
    out_ref: A reference to a block of the output tensor where the result
      is stored.
  """
  # Load the input data from the reference into a local variable.
  x = x_ref[...]

  # Compute the ELU activation using jnp.where for conditional logic.
  # The formula for ELU is:
  # - x if x > 0
  # - alpha * (exp(x) - 1) if x <= 0
  result = jnp.where(x > 0, x, alpha * (jnp.exp(x) - 1))

  # Write the computed result back to the output reference, performing the
  # operation in-place from the perspective of the caller.
  out_ref[...] = result


# The computation is an element-wise ELU. We can parallelize by chunking
# the input tensor `x`. We create a 1D grid that tiles across the `dim`
# dimension. Each kernel instance processes a `(batch_size, block_dim)` slice.
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  # Create a 1D grid of size `dim / block_dim`.
  grid=(dim // block_dim,),
  in_specs=[
    # For `x`, map the grid index `i` to a block at `(0, i)`.
    # The full `batch_size` dimension is used in each block.
    pl.BlockSpec(block_shape=(batch_size, block_dim), index_map=lambda i: (0, i)),
  ],
  # The output `result` is chunked identically to the input `x`.
  out_specs=pl.BlockSpec(block_shape=(batch_size, block_dim), index_map=lambda i: (0, i)),
)(x).block_until_ready()
