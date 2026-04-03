# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim_size = 16384
log_softmax_dim = 1
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim_size))


def kernel(x_ref, out_ref):
  """
  Pallas kernel for computing log_softmax along the second axis.

  This kernel implements the log-softmax operation using the log-sum-exp trick
  for numerical stability. The computation is applied to a block of the input
  tensor.

  Args:
    x_ref: A reference to the input block in SRAM.
    out_ref: A reference to the output block in SRAM where the result is stored.
  """
  # The log_softmax implementation is based on the log-sum-exp trick:
  # log_softmax(x) = x - log(sum(exp(x)))

  # 1. Find the maximum value for each row in the block.
  #    `keepdims=True` ensures the result's shape is (block_shape[0], 1),
  #    which allows it to be broadcast correctly for the subtraction.
  max_val = jnp.max(x_ref[...], axis=1, keepdims=True)

  # 2. Subtract the max value from each element in the row.
  #    This improves numerical stability when computing the exponential.
  x_minus_max = x_ref[...] - max_val

  # 3. Calculate the log of the sum of the exponentiated values.
  #    This is the "log-sum-exp" part of the trick.
  log_sum_exp = jnp.log(jnp.sum(jnp.exp(x_minus_max), axis=1, keepdims=True))

  # 4. Compute the final log_softmax result and store it in the output buffer.
  out_ref[...] = x_minus_max - log_sum_exp


# Computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size // 8,),
  in_specs=[pl.BlockSpec(block_shape=(8, dim_size), index_map=lambda i: (i, 0))],
  out_specs=pl.BlockSpec(block_shape=(8, dim_size), index_map=lambda i: (i, 0)),
)(x).block_until_ready()
