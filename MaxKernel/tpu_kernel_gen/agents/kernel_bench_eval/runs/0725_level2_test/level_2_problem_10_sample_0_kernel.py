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
key_init, key_x = random.split(key)

# Note: JAX uses channels-last convention (N, H, W, C)
x = random.normal(key_x, (batch_size, height, width, in_channels))

conv_transpose_layer = nn.ConvTranspose(
  features=out_channels, kernel_size=(kernel_size, kernel_size), strides=(stride, stride), padding=padding
)
params = conv_transpose_layer.init(key_init, x)["params"]

# Perform the ConvTranspose and MaxPool operations, which are unsupported
# inside a Pallas TPU kernel, using standard JAX.
y_conv = conv_transpose_layer.apply({"params": params}, x)
y_pool = nn.max_pool(
  y_conv,
  window_shape=(maxpool_kernel_size, maxpool_kernel_size),
  strides=(maxpool_stride, maxpool_stride),
  padding="VALID",
)

# The Pallas kernel will take the result of the max pooling as its input.
# We need to know its shape for the BlockSpec.
y_pool_height, y_pool_width = y_pool.shape[1], y_pool.shape[2]


# Computation
def kernel(y_pool_ref, out_ref):
  """Pallas kernel implementing a sequence of neural network operations.

  This kernel performs the following steps for each item in the batch:
  1. Hard Tanh activation.
  2. Mean reduction over the spatial dimensions.
  3. Tanh activation.

  Args:
    y_pool_ref: A reference to a slice of the max pooling output,
      corresponding to a single batch item.
      Shape: (1, H_pool_out, W_pool_out, C_out).
    out_ref: A reference to the output buffer for a single batch item.
      Shape: (1, 1, 1, C_out).
  """
  # --- Define layer parameters based on the source computation ---
  hardtanh_min = -1
  hardtanh_max = 1

  # 1. Hard Tanh activation
  # This is a clipping operation.
  y = jnp.clip(y_pool_ref[...], hardtanh_min, hardtanh_max)

  # 2. Mean over spatial dimensions (H, W)
  # axis=(1, 2) corresponds to the H and W dimensions in NHWC format.
  y = jnp.mean(y, axis=(1, 2), keepdims=True)

  # 3. Tanh activation
  y = jnp.tanh(y)

  # 4. Write final result to the output reference
  out_ref[...] = y


result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, 1, 1, out_channels), x.dtype),
  grid=(batch_size,),
  in_specs=[
    # Spec for the output of the max pooling: process one batch item
    # per kernel instance.
    pl.BlockSpec(
      block_shape=(1, y_pool_height, y_pool_width, out_channels),
      index_map=lambda i: (i, 0, 0, 0),
    ),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 1, 1, out_channels), index_map=lambda i: (i, 0, 0, 0)),
)(y_pool).block_until_ready()
