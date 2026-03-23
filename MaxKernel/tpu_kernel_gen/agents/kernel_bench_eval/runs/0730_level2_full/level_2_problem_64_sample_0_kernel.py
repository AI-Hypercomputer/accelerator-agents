# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax.linen import gelu, leaky_relu
from jax.experimental import pallas as pl
from jax.scipy.special import logsumexp

# Initialization
batch_size = 128
in_features = 1024
out_features = 512
b_batch = 8

key = random.PRNGKey(0)
key_x, key_weight, key_bias = random.split(key, 3)

x = random.normal(key_x, (batch_size, in_features))
weight = random.normal(key_weight, (out_features, in_features))
bias = random.normal(key_bias, (out_features,))


# Computation
def kernel(x_ref, weight_ref, bias_ref, out_ref):
  # Load the data from references into registers.
  x = x_ref[...]
  weight = weight_ref[...]
  bias = bias_ref[...]

  # Perform the matrix multiplication and add the bias.
  # x has shape (b_batch, in_features).
  # weight.T has shape (in_features, out_features).
  # The result has shape (b_batch, out_features).
  x = jnp.matmul(x, jnp.transpose(weight)) + bias

  # Apply logsumexp along the feature dimension.
  # The result has shape (b_batch, 1).
  x = logsumexp(x, axis=1, keepdims=True)

  # Apply leaky_relu twice.
  x = leaky_relu(x, negative_slope=0.01)
  x = leaky_relu(x, negative_slope=0.01)

  # Apply gelu twice.
  x = gelu(x)
  x = gelu(x)

  # Write the final result to the output reference.
  out_ref[...] = x


x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, 1), x.dtype),
  grid=(batch_size // b_batch,),
  in_specs=[
    pl.BlockSpec(block_shape=(b_batch, in_features), index_map=lambda i: (i, 0)),
    pl.BlockSpec(block_shape=(out_features, in_features), index_map=lambda i: (0, 0)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(b_batch, 1), index_map=lambda i: (i, 0)),
)(x, weight, bias).block_until_ready()
