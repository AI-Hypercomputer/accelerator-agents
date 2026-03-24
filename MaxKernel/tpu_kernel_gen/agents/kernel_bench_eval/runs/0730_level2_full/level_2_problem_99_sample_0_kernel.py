# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_features = 100
out_features = 10

key = random.PRNGKey(0)
key_x, key_weight, key_bias = random.split(key, 3)

x = random.normal(key_x, (batch_size, in_features))
weight = random.normal(key_weight, (out_features, in_features))
bias = random.normal(key_bias, (out_features,))

batch_block_size = 8


# Computation
def kernel(x_ref, weight_ref, bias_ref, out_ref):
  """
  Pallas kernel for a sequence of neural network operations.

  This kernel performs the following steps:
  1. A linear transformation (matrix multiplication with transposed weights and bias addition).
  2. Applies the GELU activation function.
  3. Applies the Softmax activation function along the feature axis.

  Args:
    x_ref: A reference to the input data block.
    weight_ref: A reference to the weight matrix.
    bias_ref: A reference to the bias vector.
    out_ref: A reference to the output block to store the result.
  """
  # Perform the linear transformation: y = x @ w.T + b
  y = jnp.dot(x_ref[...], weight_ref[...].T) + bias_ref[...]
  # Apply the GELU activation function element-wise
  y = jax.nn.gelu(y)
  # Apply the Softmax function along the feature dimension (axis=1)
  out_ref[...] = jax.nn.softmax(y, axis=1)


x = pl.pallas_call(
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
