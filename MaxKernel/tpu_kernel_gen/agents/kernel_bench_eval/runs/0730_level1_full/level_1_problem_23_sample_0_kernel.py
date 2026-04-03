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


# Computation
def kernel(x_ref, out_ref):
  """
  Computes the softmax of a 1D array.
  This kernel is designed to be called in a grid where each program
  instance handles one row of the input matrix.
  """
  # x_ref and out_ref are the entire matrices.
  # We use pl.prange to explicitly parallelize over the batch dimension.
  for i in pl.prange(batch_size):
    # Get a single row.
    row = x_ref[i, :]

    # Compute softmax in a numerically stable way.
    # 1. Find the maximum value in the row.
    max_val = jnp.max(row)
    # 2. Subtract the max value to prevent overflow.
    x_shifted = row - max_val
    # 3. Exponentiate the shifted values.
    numerator = jnp.exp(x_shifted)
    # 4. Sum the exponentiated values to get the denominator.
    denominator = jnp.sum(numerator)
    # 5. Compute the final softmax result for the row.
    result_row = numerator / denominator

    # Write the result to the output buffer in HBM.
    out_ref[i, :] = result_row


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(1,),
  in_specs=[pl.BlockSpec(block_shape=(batch_size, dim), index_map=lambda i: (0, 0))],
  out_specs=pl.BlockSpec(block_shape=(batch_size, dim), index_map=lambda i: (0, 0)),
)(x).block_until_ready()
