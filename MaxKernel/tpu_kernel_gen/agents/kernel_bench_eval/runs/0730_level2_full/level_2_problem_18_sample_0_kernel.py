# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl
from jax.scipy.special import logsumexp

# Initialization
batch_size = 128
in_features = 10
out_features = 5
key = random.PRNGKey(0)
key, x_key, weight_key, bias_key = random.split(key, 4)
x = random.normal(x_key, (batch_size, in_features))
weight = random.normal(weight_key, (out_features, in_features))
bias = random.normal(bias_key, (out_features,))
batch_block_size = 8


# Computation
def kernel(x_ref, weight_ref, bias_ref, out_ref):
  # Load data from memory into registers.
  x = x_ref[...]
  weight = weight_ref[...]
  bias = bias_ref[...]

  # Perform the linear transformation (dot product + bias)
  x = jnp.dot(x, weight.T) + bias

  # Perform the sequence of reduction operations.
  # Note: After the first sum, the dimension along axis=1 is 1.
  # Subsequent reductions along axis=1 are identity operations but are
  # included for a faithful translation of the original code.
  x = jnp.sum(x, axis=1, keepdims=True)
  x = jnp.max(x, axis=1, keepdims=True)
  x = jnp.mean(x, axis=1, keepdims=True)
  x = logsumexp(x, axis=1, keepdims=True)
  x = logsumexp(x, axis=1, keepdims=True)

  # Store the final result in the output buffer.
  out_ref[...] = x


x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, 1), x.dtype),
  grid=(batch_size // batch_block_size,),
  in_specs=[
    pl.BlockSpec(
      block_shape=(batch_block_size, in_features),
      index_map=lambda i: (i * batch_block_size, 0),
    ),
    pl.BlockSpec(block_shape=(out_features, in_features), index_map=lambda i: (0, 0)),
    pl.BlockSpec(block_shape=(out_features,), index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(batch_block_size, 1),
    index_map=lambda i: (i * batch_block_size, 0),
  ),
)(x, weight, bias).block_until_ready()
