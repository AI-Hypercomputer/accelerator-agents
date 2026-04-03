# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2

key = random.PRNGKey(0)
key_params, key_x = random.split(key)

conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size, kernel_size))
# JAX uses channel-last convention: (N, D, H, W, C)
x_shape = (batch_size, depth, height, width, in_channels)
x = random.normal(key_x, x_shape)
params = conv.init(key_params, x)["params"]

# Computation
# Define the dimension numbers for the 3D convolution.
# 'NDHWC' for input/output: (Batch, Depth, Height, Width, Channels)
# 'DHWIO' for kernel: (Depth, Height, Width, In-channels, Out-channels)
dimension_numbers = ("NDHWC", "DHWIO", "NDHWC")

# 1. Perform 3D convolution using JAX, as it's not supported in Pallas.
conv_out = jax.lax.conv_general_dilated(
  lhs=x,
  rhs=params["kernel"],
  window_strides=(1, 1, 1),
  padding="SAME",
  dimension_numbers=dimension_numbers,
)


def activation_kernel(conv_out_ref, bias_ref, out_ref):
  """
  Pallas kernel for a sequence of bias add and softmax.

  This kernel processes a single data instance from a batch. It performs the
  following operations in sequence:
  1. Adds a bias term.
  2. Applies a softmax function along the channel dimension.

  Args:
    conv_out_ref: A reference to the input tensor (result of a convolution)
      for a single batch item, with shape
      (1, depth, height, width, out_channels).
    bias_ref: A reference to the convolution bias, with shape (out_channels,).
    out_ref: A reference to the output tensor, which will be populated by this
      kernel. Its shape is (1, depth, height, width, out_channels).
  """
  # 2. Add the bias term. The bias is broadcast across the spatial dimensions.
  conv_out_with_bias = conv_out_ref[...] + bias_ref[...]

  # 3. Apply softmax activation function along the last (channel) axis.
  softmax_out = jax.nn.softmax(conv_out_with_bias, axis=-1)

  # 4. Write the final result to the output buffer in HBM.
  out_ref[...] = softmax_out


# The output shape of the activation kernel is the same as the convolution output.
out_struct = jax.ShapeDtypeStruct(conv_out.shape, conv_out.dtype)

# To avoid memory exhaustion, we tile the computation across the height
# dimension, reducing the data processed by each kernel instance.
height_block_size = height // 4

# We parallelize the computation over the batch dimension and height blocks.
softmax_out = pl.pallas_call(
  activation_kernel,
  out_shape=out_struct,
  grid=(batch_size, 4),
  # Input specifications:
  # 1. The convolution output tensor is chunked along batch and height.
  # 2. The convolution bias is broadcast to all instances.
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, depth, height_block_size, width, out_channels),
      index_map=lambda i, j: (i, 0, j, 0, 0),
    ),
    pl.BlockSpec(block_shape=params["bias"].shape, index_map=lambda i, j: (0,)),
  ],
  # Output specification:
  # The output tensor is chunked similarly along batch and height.
  out_specs=pl.BlockSpec(
    block_shape=(1, depth, height_block_size, width, out_channels),
    index_map=lambda i, j: (i, 0, j, 0, 0),
  ),
)(conv_out, params["bias"])

# 5. Perform two consecutive 3D max-pooling operations using JAX, as
#    `reduce_window` is not supported in Pallas.
pool_window_shape = (1, 2, 2, 2, 1)
pooled_out_1 = jax.lax.reduce_window(
  operand=softmax_out,
  init_value=-jnp.inf,
  computation=jnp.maximum,
  window_dimensions=pool_window_shape,
  window_strides=pool_window_shape,
  padding="VALID",
)

pooled_out_2 = jax.lax.reduce_window(
  operand=pooled_out_1,
  init_value=-jnp.inf,
  computation=jnp.maximum,
  window_dimensions=pool_window_shape,
  window_strides=pool_window_shape,
  padding="VALID",
)

x = pooled_out_2.block_until_ready()
