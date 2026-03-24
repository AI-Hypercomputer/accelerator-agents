# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
subtract_value = 0.5
pool_kernel_size = 2

key = random.PRNGKey(0)
key, x_key, params_key = random.split(key, 3)

x = random.normal(x_key, (batch_size, height, width, in_channels))
conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size))
params = conv.init(params_key, x)["params"]

# Perform convolution using standard JAX since it's not supported in Pallas
conv_out = conv.apply({"params": params}, x)


# Computation
def kernel(x_ref, out_ref):
  # Constants from the source code
  subtract_value = 0.5

  x = x_ref[...]

  # Subtract the constant value
  x = x - subtract_value

  # Apply the hard_swish activation function
  x = jax.nn.hard_swish(x)

  # Compute the maximum value over the pooling window (axes 1 and 2)
  x = jnp.max(x, axis=(1, 2))

  # Apply the mish activation function
  x = jax.nn.mish(x)

  # Write the final result to the output reference
  out_ref[...] = x.reshape(1, 1, 1, out_channels)


x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(
    (batch_size, height // pool_kernel_size, width // pool_kernel_size, out_channels), x.dtype
  ),
  grid=(batch_size, height // pool_kernel_size, width // pool_kernel_size),
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, pool_kernel_size, pool_kernel_size, out_channels),
      index_map=lambda b, i, j: (b, i * pool_kernel_size, j * pool_kernel_size, 0),
    )
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 1, 1, out_channels), index_map=lambda b, i, j: (b, i, j, 0)),
)(conv_out).block_until_ready()
