# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
input_size = 10
hidden_size = 20

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

x = random.normal(key_x, (batch_size, input_size))
linear_layer = nn.Dense(features=hidden_size)
params = linear_layer.init(key_params, x)["params"]
kernel_matrix = params["kernel"]
bias_vector = params["bias"]


# Computation
def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  """
  Pallas kernel for a simple neural network layer.

  This kernel performs the following operations:
  1. A linear transformation (matrix multiplication with a kernel and addition of a bias).
  2. A sigmoid activation function.
  3. A sum reduction across the feature dimension.

  Args:
    x_ref: A reference to a slice of the input data `x`. For each program instance,
      this will be a single row of `x` with shape (1, input_size).
    kernel_ref: A reference to the weight matrix of the dense layer. This is the
      full matrix of shape (input_size, hidden_size), shared across all programs.
    bias_ref: A reference to the bias vector of the dense layer. This is the full
      vector of shape (hidden_size,), shared across all programs.
    out_ref: A reference to a slice of the output buffer. For each program
      instance, this is a scalar-like slice of shape (1, 1) where the result
      of the computation for the corresponding input row is stored.
  """
  # Perform the linear transformation: y = x @ W + b
  y = jnp.dot(x_ref[...], kernel_ref[...])[:, :hidden_size]
  y = y + bias_ref[...]

  # Apply the sigmoid activation function
  y = jax.nn.sigmoid(y)

  # Sum the results along the hidden dimension and write to the output
  out_ref[...] = jnp.sum(y, axis=1, keepdims=True)


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, 1), x.dtype),
  grid=(batch_size // 8,),
  in_specs=[
    pl.BlockSpec(block_shape=(8, input_size), index_map=lambda i: (i * 8, 0)),
    pl.BlockSpec(block_shape=(input_size, hidden_size), index_map=lambda i: (0, 0)),
    pl.BlockSpec(block_shape=(hidden_size,), index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(8, 1), index_map=lambda i: (i * 8, 0)),
)(x, kernel_matrix, bias_vector).block_until_ready()
