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
# For TPU compatibility, the batch dimension of the block must be divisible by 8.
batch_block_size = 8
# For TPU compatibility, the dimension of the block must be divisible by 128.
block_dim = 128

# Reshape the input tensor to be 3D.
x_reshaped = x.reshape(batch_size, dim // block_dim, block_dim)


# Computation
def kernel(x_ref, out_ref):
  """
  Pallas kernel for log_softmax.
  The kernel receives a 3D block of the input tensor.
  """
  # Find the maximum value for each row for numerical stability by reducing
  # along each axis sequentially.
  max_in_tile = jnp.max(x_ref[...], axis=2, keepdims=True)
  row_max = jnp.max(max_in_tile, axis=1, keepdims=True)

  # Subtract the maximum value from the input tensor.
  x_shifted = x_ref[...] - row_max

  # Compute the log of the sum of the exponentials of the shifted tensor,
  # also reducing along each axis sequentially.
  sum_exp_in_tile = jnp.sum(jnp.exp(x_shifted), axis=2, keepdims=True)
  log_sum_exp = jnp.log(jnp.sum(sum_exp_in_tile, axis=1, keepdims=True))

  # Compute the log_softmax value.
  log_softmax_output = x_shifted - log_sum_exp

  # Write the result to the output reference.
  out_ref[...] = log_softmax_output


output_reshaped = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x_reshaped.shape, x_reshaped.dtype),
  grid=(batch_size // batch_block_size,),
  in_specs=[
    pl.BlockSpec(
      (batch_block_size, dim // block_dim, block_dim),
      lambda i: (i * batch_block_size, 0, 0),
    )
  ],
  out_specs=pl.BlockSpec(
    (batch_block_size, dim // block_dim, block_dim),
    lambda i: (i * batch_block_size, 0, 0),
  ),
)(x_reshaped)

# Reshape the output back to the original 2D shape.
output = output_reshaped.reshape(x.shape).block_until_ready()
