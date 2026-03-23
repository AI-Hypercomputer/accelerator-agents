# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
input_size = 512
hidden_size = 1024
scale_factor = 2.0
clamp_min = -10.0
clamp_max = 10.0
key = random.PRNGKey(0)
key_x, key_weight, key_bias = random.split(key, 3)
x = random.normal(key_x, (batch_size, input_size))
weight = random.normal(key_weight, (hidden_size, input_size))
bias = random.normal(key_bias, (hidden_size,))
weight_t = weight.T


# Computation
def kernel(x_ref, weight_t, bias, out_ref):
  """
  Pallas kernel for a sequence of neural network operations.

  Args:
    x_ref: A reference to a slice of the input tensor `x`.
    weight_t: The transposed weight matrix.
    bias: The bias vector.
    out_ref: A reference to the output tensor.
  """
  # Constants from the source computation
  scale_factor = 2.0
  clamp_min = -10.0
  clamp_max = 10.0

  # 1. x = jnp.matmul(x, weight.T) + bias
  # x_ref[...] has shape (8, input_size)
  # weight_t has shape (input_size, hidden_size)
  # The result of the matmul is (8, hidden_size)
  y = jnp.matmul(x_ref[...], weight_t)
  y = y + bias

  # 2. x = x * scale_factor
  y = y * scale_factor

  # 3. x = x + x
  y = y + y

  # 4. x = jnp.clip(x, clamp_min, clamp_max)
  y = jnp.clip(y, clamp_min, clamp_max)

  # 5. x = jax.nn.logsumexp(x, axis=1, keepdims=True)
  # The input `y` has shape (8, hidden_size).
  # The result has shape (8, 1).
  y = jax.nn.logsumexp(y, axis=1, keepdims=True)

  # 6. x = x * jax.nn.mish(x)
  y = y * jax.nn.mish(y)

  # Write the final result to the output.
  out_ref[...] = y


# Final output after all computations
final_output_shape = (batch_size, 1)

# Construct the pallas_call
x = pl.pallas_call(
  lambda x_ref, out_ref: kernel(x_ref, weight_t, bias, out_ref),
  out_shape=jax.ShapeDtypeStruct(final_output_shape, x.dtype),
  grid=(batch_size // 8,),
  in_specs=[
    pl.BlockSpec(block_shape=(8, input_size), index_map=lambda i: (i * 8, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(8, 1), index_map=lambda i: (i * 8, 0)),
)(x).block_until_ready()
