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
# For TPU compatibility, the second-to-last dimension of a 2D block
# must be divisible by 8. We will process rows in blocks of 8.
block_size_batch = 8


# Computation
def kernel(x_ref, y_ref):
  """
  Pallas kernel to perform L2 normalization on each row of a matrix block.

  This kernel translates the following JAX computation:
  y = x / jnp.linalg.norm(x, ord=2, axis=1, keepdims=True)

  Args:
    x_ref: A reference to the input block of shape (block_size_batch, dim).
    y_ref: A reference to the output block of shape (block_size_batch, dim).
           The result of the normalization will be written here in-place.
  """
  # Load the input data from SRAM into registers for computation.
  x = x_ref[...]

  # Manually compute the L2 norm for each row in the block.
  # jnp.linalg.norm is not a pallas primitive.
  # The L2 norm is the square root of the sum of the squares of the elements.
  # - `axis=1`: The sum is computed along each row.
  # - `keepdims=True`: The result has shape (block_size_batch, 1), which is
  #   necessary for broadcasting during the division step.
  norm = jnp.sqrt(jnp.sum(x * x, axis=1, keepdims=True))

  # Divide each row by its corresponding norm and write the result to the
  # output buffer. This performs the normalization.
  y_ref[...] = x / norm


y = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size // block_size_batch,),
  in_specs=[pl.BlockSpec(block_shape=(block_size_batch, dim), index_map=lambda i: (i, 0))],
  out_specs=pl.BlockSpec(block_shape=(block_size_batch, dim), index_map=lambda i: (i, 0)),
)(x).block_until_ready()
