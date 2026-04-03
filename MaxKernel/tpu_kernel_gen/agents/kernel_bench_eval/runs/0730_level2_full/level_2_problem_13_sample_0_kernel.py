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
kernel_size = 3
stride = 2
padding = 1
bias_shape = (1, 1, 1, 1, 1)
scaling_factor = 2.0

key = random.PRNGKey(0)
key, params_key, bias_key, x_key = random.split(key, 4)

# In Flax, layers are stateless. We define the layer configuration...
conv_transpose = nn.ConvTranspose(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size, kernel_size),
  strides=(stride, stride, stride),
  padding=padding,
)

# ...and initialize its parameters separately.
# JAX uses a channels-last convention, so the shape is (N, D, H, W, C).
x_shape = (batch_size, depth, height, width, in_channels)
x = random.normal(x_key, x_shape)
params = conv_transpose.init(params_key, x)["params"]

bias = random.normal(bias_key, bias_shape)


# Computation
def kernel(x_ref, kernel_ref, conv_bias_ref, add_bias_ref, out_ref):
  """
  Pallas kernel for a sequence of neural network operations.

  This kernel performs the following steps:
  1. 3D Transposed Convolution with bias.
  2. Mean reduction over the channel dimension.
  3. Addition of a second bias.
  4. Softmax activation.
  5. Tanh activation.
  6. Scaling by a constant factor.

  Args:
    x_ref: Reference to the input tensor slice.
    kernel_ref: Reference to the convolution kernel weights.
    conv_bias_ref: Reference to the bias tensor for the convolution.
    add_bias_ref: Reference to the bias tensor added after the mean operation.
    out_ref: Reference to the output tensor slice for in-place updates.
  """
  # These values are derived from the context of the original script.
  stride = 2
  padding = 1

  # 1. Apply the transposed convolution.
  # The dimension numbers ('NDHWC', 'DHWIO', 'NDHWC') are standard for
  # JAX's channels-last convention.
  dimension_numbers = ("NDHWC", "DHWIO", "NDHWC")
  # The padding argument for jax.lax.conv_transpose is a sequence of (low, high) pairs.
  padding_config = [(padding, padding), (padding, padding), (padding, padding)]

  # Perform the convolution on the input data slice.
  y = jax.lax.conv_transpose(
    x_ref[...],
    kernel_ref[...],
    strides=(stride, stride, stride),
    padding=padding_config,
    dimension_numbers=dimension_numbers,
  )
  # Add the convolution bias.
  y = y + conv_bias_ref[...]

  # 2. Compute the mean across the channel dimension (last axis).
  y = jnp.mean(y, axis=-1, keepdims=True)

  # 3. Add the second, separate bias.
  y = y + add_bias_ref[...]

  # 4. Apply Softmax activation.
  y = jax.nn.softmax(y, axis=-1)

  # 5. Apply Tanh activation.
  y = jnp.tanh(y)

  # 6. Scale the result.
  y = y * scaling_factor

  # 7. Write the final result to the output buffer.
  out_ref[...] = y


# Calculate the output shape after the ConvTranspose operation
# In JAX, for 'SAME' padding, output_size = input_size * stride
# The code uses custom padding, so we calculate manually:
# O = (I - 1) * S - 2*P + K
# Note: Flax's nn.ConvTranspose with padding argument might behave differently
# from the standard formula depending on the implementation details.
# A robust way is to compute the shape with a dry run.
out_depth = (depth - 1) * stride + kernel_size - 2 * padding
out_height = (height - 1) * stride + kernel_size - 2 * padding
out_width = (width - 1) * stride + kernel_size - 2 * padding

# The final shape after mean reduction and other ops
final_out_shape = (batch_size, out_depth, out_height, out_width, 1)
out_struct = jax.ShapeDtypeStruct(final_out_shape, x.dtype)

# The pallas_call replaces the original computation section.
# We parallelize over the batch dimension. Each kernel instance processes one
# item from the batch.
result = pl.pallas_call(
  kernel,
  out_shape=out_struct,
  grid=(batch_size,),
  in_specs=[
    # Input image tensor `x`: chunked by batch index.
    pl.BlockSpec(block_shape=(1, depth, height, width, in_channels), index_map=lambda i: (i, 0, 0, 0, 0)),
    # Conv kernel: replicated for all grid instances.
    pl.BlockSpec(block_shape=params["kernel"].shape, index_map=lambda i: (0,) * params["kernel"].ndim),
    # Conv bias: replicated for all grid instances.
    pl.BlockSpec(block_shape=params["bias"].shape, index_map=lambda i: (0,) * params["bias"].ndim),
    # Additive bias: replicated for all grid instances.
    pl.BlockSpec(block_shape=bias.shape, index_map=lambda i: (0,) * bias.ndim),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, out_depth, out_height, out_width, 1), index_map=lambda i: (i, 0, 0, 0, 0)),
  interpret=True,
)(x, params["kernel"], params["bias"], bias).block_until_ready()
