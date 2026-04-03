# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
features = 64
dim1 = 256
dim2 = 256
key = random.PRNGKey(0)
key_x, key_init = random.split(key)

# Flax layers expect channel-last format (N, H, W, C)
x = random.normal(key_x, (batch_size, dim1, dim2, features))


# Computation
def kernel(x_ref, out_ref):
  # The input x_ref corresponds to a slice of shape (1, dim1, dim2, features)
  # for a specific batch item, as defined by the grid and BlockSpec.
  x = x_ref[...]

  # Calculate the mean and variance across the spatial dimensions (dim1, dim2) for each channel.
  # The axes of reduction are (1, 2) for the (1, dim1, dim2, features) slice.
  mean = jnp.mean(x, axis=(1, 2), keepdims=True)
  var = jnp.var(x, axis=(1, 2), keepdims=True)

  # Epsilon is a small float added to variance to avoid division by zero.
  # The default value in flax.linen.InstanceNorm is 1e-5.
  epsilon = 1e-5

  # Apply the instance normalization formula.
  # Since use_scale and use_bias are False, we don't use gamma or beta.
  normalized_x = (x - mean) / jnp.sqrt(var + epsilon)

  # Write the result back to the output buffer.
  out_ref[...] = normalized_x


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  grid=(batch_size,),
  in_specs=[pl.BlockSpec(block_shape=(1, dim1, dim2, features), index_map=lambda n: (n, 0, 0, 0))],
  out_specs=pl.BlockSpec(block_shape=(1, dim1, dim2, features), index_map=lambda n: (n, 0, 0, 0)),
)(x).block_until_ready()
