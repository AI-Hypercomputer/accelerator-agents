# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
input_size = 10
hidden_size = 20

key = random.PRNGKey(0)
key, x_key, params_key = random.split(key, 3)

x = random.normal(x_key, (batch_size, input_size))
linear1 = nn.Dense(features=hidden_size)
params = linear1.init(params_key, x)


def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  """Pallas kernel for a dense layer followed by a sigmoid activation.

  This kernel computes `sigmoid(x @ W + b)` for a block of data.

  Args:
    x_ref: A reference to the input data block.
    kernel_ref: A reference to the weight matrix (kernel).
    bias_ref: A reference to the bias vector.
    out_ref: A reference to the output block to be written to.
  """
  # Perform the linear transformation (x @ W + b)
  # x_ref[...] has shape (8, input_size)
  # kernel_ref[...] has shape (input_size, hidden_size)
  # The result of the matmul has shape (8, hidden_size)
  y = x_ref[...] @ kernel_ref[...]

  # Add the bias vector.
  # bias_ref[...] has shape (hidden_size,) and is broadcast across the batch dim.
  y = y + bias_ref[...]

  # Apply the sigmoid activation function and write to the output.
  out_ref[...] = jax.nn.sigmoid(y)


# Computation
x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, hidden_size), x.dtype),
  grid=(batch_size // 8,),
  in_specs=[
    pl.BlockSpec(block_shape=(8, input_size), index_map=lambda i: (i, 0)),
    pl.BlockSpec(block_shape=(input_size, hidden_size), index_map=lambda i: (0, 0)),
    pl.BlockSpec(block_shape=(hidden_size,), index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(8, hidden_size), index_map=lambda i: (i, 0)),
)(x, params["params"]["kernel"], params["params"]["bias"]).block_until_ready()

x = jnp.sum(x, axis=1)
x = nn.logsumexp(x, axis=0)
