# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
dim_features = 16384
key = random.PRNGKey(0)
x = random.normal(key, (batch_size, dim_features))
batch_block_size = 8


# Computation
def kernel(x_ref, out_ref):
  """
  Pallas kernel for computing log_softmax row-wise.

  This kernel implements the log_softmax operation using the log-sum-exp trick
  for numerical stability. The computation is applied to each row of the
  input block independently.

  Args:
    x_ref: A reference to the input block of data.
    out_ref: A reference to the output block where the result is stored.
  """
  # Load the input block from SRAM into registers.
  x = x_ref[...]

  # The log_softmax operation is defined as:
  # log_softmax(x) = x - log(sum(exp(x)))
  # To ensure numerical stability, the log-sum-exp trick is used:
  # log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
  # Therefore, log_softmax(x) = (x - max(x)) - log(sum(exp(x - max(x)))).

  # 1. Find the maximum value in each row for numerical stability.
  #    The shape of max_val will be (batch_block_size, 1).
  max_val = jnp.max(x, axis=1, keepdims=True)

  # 2. Center the values by subtracting the max.
  #    This is broadcasted across the `dim_features` dimension.
  centered_x = x - max_val

  # 3. Compute the log of the sum of exponentiated centered values.
  #    This is the second term in the stable log_softmax formula.
  log_sum_exp = jnp.log(jnp.sum(jnp.exp(centered_x), axis=1, keepdims=True))

  # 4. Compute the final log_softmax value.
  log_softmax_output = centered_x - log_sum_exp

  # 5. Write the result to the output reference, performing the operation in-place.
  out_ref[...] = log_softmax_output


output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size // batch_block_size,),
  in_specs=[pl.BlockSpec(block_shape=(batch_block_size, dim_features), index_map=lambda i: (i, 0))],
  out_specs=pl.BlockSpec(block_shape=(batch_block_size, dim_features), index_map=lambda i: (i, 0)),
)(x).block_until_ready()
