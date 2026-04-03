# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 8
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
stride = (2, 2, 2)
padding = 1

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# JAX uses channel-last convention (N, D, H, W, C)
x_shape = (batch_size, depth, height, width, in_channels)
x = random.normal(key_x, x_shape)

conv_transpose = nn.ConvTranspose(
  features=out_channels, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=True
)
params = conv_transpose.init(key_params, x)["params"]

# The final output shape after all computations.
final_shape = (batch_size, 5, 10, 10, 1)

# Perform unsupported operations outside of Pallas.
conv_out = conv_transpose.apply({"params": params}, x)

# First max pooling operation.
pooled_out = nn.max_pool(conv_out, window_shape=(2, 2, 2), strides=(2, 2, 2), padding="VALID")

# Second max pooling operation.
pooled_out = nn.max_pool(pooled_out, window_shape=(3, 3, 3), strides=(3, 3, 3), padding="VALID")


# Computation
def sum_kernel(x_ref, out_ref):
  """
  Pallas kernel for sum operation.
  """
  # Sum over the channel dimension and write to the output.
  out_ref[...] = jnp.sum(x_ref[...], axis=-1, keepdims=True)


x = pl.pallas_call(
  sum_kernel,
  out_shape=jax.ShapeDtypeStruct(final_shape, x.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(1, *pooled_out.shape[1:]), index_map=lambda i: (i, 0, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, *final_shape[1:]), index_map=lambda i: (i, 0, 0, 0, 0)),
)(pooled_out).block_until_ready()
