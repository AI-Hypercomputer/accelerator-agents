# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_features = 1024
out_features = 512
divisor = 2.0
key = random.PRNGKey(0)
key_x, key_weight, key_bias = random.split(key, 3)
x = random.normal(key_x, (batch_size, in_features))
weight = random.normal(key_weight, (out_features, in_features))
bias = random.normal(key_bias, (out_features,))

b_batch = 128
b_out = 128


def kernel(x_ref, weight_ref, bias_ref, out_ref, divisor):
  # Perform the dot product of the input block with the transposed weight block.
  y = jnp.dot(x_ref[...], weight_ref[...].T)

  # Add the bias vector. Broadcasting handles the addition across the batch dimension.
  # By explicitly adding a dimension, we give a hint to the compiler that helps resolve the layout mismatch.
  y = y + bias_ref[None, ...]

  # Apply the ReLU activation function.
  y = nn.relu(y)

  # Divide the result by the divisor.
  y = y / divisor

  # Store the final result in the output buffer.
  out_ref[...] = y


# Computation
result = pl.pallas_call(
  lambda x_ref, w_ref, b_ref, o_ref: kernel(x_ref, w_ref, b_ref, o_ref, divisor),
  out_shape=jax.ShapeDtypeStruct(x.shape[:-1] + (out_features,), x.dtype),
  grid=(batch_size // b_batch, out_features // b_out),
  in_specs=[
    pl.BlockSpec(block_shape=(b_batch, in_features), index_map=lambda i, j: (i, 0)),
    pl.BlockSpec(block_shape=(b_out, in_features), index_map=lambda i, j: (j, 0)),
    pl.BlockSpec(block_shape=(b_out,), index_map=lambda i, j: (j,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(b_batch, b_out), index_map=lambda i, j: (i, j)),
)(x, weight, bias).block_until_ready()
