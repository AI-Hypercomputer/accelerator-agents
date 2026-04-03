# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 32
out_channels = 64
height, width = 16, 16
kernel_size = 4
stride = 2
padding = 1
maxpool_kernel_size = 2
maxpool_stride = 2
hardtanh_min = -1
hardtanh_max = 1

key = random.PRNGKey(0)
key_input, key_init = random.split(key)

x = random.normal(key_input, (batch_size, height, width, in_channels))
conv_transpose = nn.ConvTranspose(
  features=out_channels, kernel_size=(kernel_size, kernel_size), strides=(stride, stride), padding=padding
)
params = conv_transpose.init(key_init, x)["params"]

# Perform ConvTranspose and MaxPool outside of the Pallas kernel, as they are not supported.
y_conv = conv_transpose.apply({"params": params}, x)
y_maxpool = jax.lax.reduce_window(
  y_conv,
  -jnp.inf,
  jax.lax.max,
  window_dimensions=(1, maxpool_kernel_size, maxpool_kernel_size, 1),
  window_strides=(1, maxpool_stride, maxpool_stride, 1),
  padding="VALID",
)


# The Pallas kernel will handle the subsequent operations.
def kernel(y_maxpool_ref, out_ref):
  # Define constants for the operations
  hardtanh_min = -1
  hardtanh_max = 1

  # Load input from memory into registers
  y = y_maxpool_ref[...]

  # 1. Clip (emulates HardTanh)
  y = jnp.clip(y, a_min=hardtanh_min, a_max=hardtanh_max)

  # 2. Mean over spatial dimensions (H, W)
  y = jnp.mean(y, axis=(1, 2), keepdims=True)

  # 3. Tanh activation
  y = jnp.tanh(y)

  # Write the final result to the output buffer
  out_ref[...] = y


# Final output shape calculation
final_out_shape = (batch_size, 1, 1, out_channels)
y_maxpool_shape = y_maxpool.shape

# Computation
result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(final_out_shape, x.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, y_maxpool_shape[1], y_maxpool_shape[2], y_maxpool_shape[3]),
      index_map=lambda i: (i, 0, 0, 0),
    ),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 1, 1, out_channels), index_map=lambda i: (i, 0, 0, 0)),
)(y_maxpool).block_until_ready()
