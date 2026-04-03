# Imports
import jax
import jax.nn
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim))
log_softmax_dim = 1


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for log_softmax.

  Args:
    x_ref: A reference to a row of the input matrix.
    out_ref: A reference to a row of the output matrix.
  """
  # Load the whole row. The BlockSpec specifies that each kernel instance
  # gets a view of a full row.
  row = x_ref[0]

  # The log-sum-exp trick for numerical stability.
  # 1. Find the maximum value in the current row.
  max_val = jnp.max(row)

  # 2. Compute the log of the sum of the exponentiated stable values.
  # This is the log of the denominator of the softmax.
  numerator = jnp.exp(row - max_val)
  log_denominator = jnp.log(jnp.sum(numerator))

  # 4. Compute the log_softmax and write it to the output.
  output_row = (row - max_val) - log_denominator
  out_ref[0] = output_row


output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size,),
  in_specs=[pl.BlockSpec(block_shape=(1, dim), index_map=lambda i: (i, 0))],
  out_specs=pl.BlockSpec(block_shape=(1, dim), index_map=lambda i: (i, 0)),
)(x).block_until_ready()
