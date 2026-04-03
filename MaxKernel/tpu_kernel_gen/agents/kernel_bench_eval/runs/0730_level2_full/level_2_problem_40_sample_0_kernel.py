# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_features = 64
out_features = 128
key = random.PRNGKey(0)
key_x, key_weight, key_bias = random.split(key, 3)
x = random.normal(key_x, (batch_size, in_features))
weight = random.normal(key_weight, (out_features, in_features))
bias = random.normal(key_bias, (out_features,))
batch_block_size = 32


# Computation
def kernel(x_ref, weight_ref, bias_ref, y_ref):
  """Pallas kernel for a fused linear operation with scaling and addition.

  This kernel computes the following sequence of operations:
  1. y = jnp.dot(x, weight.T) + bias
  2. original_y = y
  3. y = y * scaling_factor
  4. y = y + original_y

  Args:
    x_ref: A reference to a block of the input tensor 'x'.
    weight_ref: A reference to the weight matrix.
    bias_ref: A reference to the bias vector.
    y_ref: A reference to a block of the output tensor 'y' for in-place update.
  """
  # The scaling factor is a constant in the original computation.
  scaling_factor = 0.5

  # Compute y = jnp.dot(x, weight.T) + bias
  # This intermediate result is also the 'original_y' from the source.
  original_y = jnp.dot(x_ref[...], weight_ref[...].T) + bias_ref[...]

  # Apply the scaling and add the original value.
  # This combines the last two steps: y = y * scaling_factor + original_y
  final_y = original_y * scaling_factor + original_y

  # Write the final result to the output buffer.
  y_ref[...] = final_y


y = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_features), x.dtype),
  grid=(batch_size // batch_block_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(batch_block_size, in_features), index_map=lambda i: (i, 0)),
    pl.BlockSpec(block_shape=(out_features, in_features), index_map=lambda i: (0, 0)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(batch_block_size, out_features), index_map=lambda i: (i, 0)),
)(x, weight, bias).block_until_ready()
