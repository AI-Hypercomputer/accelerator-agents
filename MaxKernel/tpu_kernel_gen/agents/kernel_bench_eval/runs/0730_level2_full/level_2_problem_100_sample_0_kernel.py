# Imports
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 32
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
min_value = -1.0
divisor = 2.0

key = random.PRNGKey(0)
key, x_key, params_key = random.split(key, 3)

conv_transpose = nn.ConvTranspose(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size, kernel_size),
  strides=(stride, stride, stride),
  padding=(padding, padding, padding),
)
x = random.normal(x_key, (batch_size, depth, height, width, in_channels))
params = conv_transpose.init(params_key, x)["params"]


# Computation
def kernel(x_ref, kernel_ref, bias_ref, out_ref, min_value, divisor):
  """
  Pallas kernel for 3D transposed convolution, clipping, and division.

  This kernel performs the following operations:
  1. 3D transposed convolution on the input `x_ref` with `kernel_ref`.
  2. Adds the `bias_ref` to the result of the convolution.
  3. Clips the values to be at least `min_value`.
  4. Divides the result by `divisor`.
  5. Stores the final result in `out_ref`.

  Args:
    x_ref: Input tensor reference of shape (1, depth, height, width, in_channels).
    kernel_ref: Convolution kernel reference of shape (kernel_depth, kernel_height, kernel_width, in_channels, out_channels).
    bias_ref: Bias vector reference of shape (out_channels,).
    out_ref: Output tensor reference of shape (1, out_depth, out_height, out_width, out_channels).
    min_value: The minimum value for the clipping operation.
    divisor: The value to divide the tensor by.
  """
  # Define convolution parameters
  strides = (2, 2, 2)
  padding = ((1, 1), (1, 1), (1, 1))
  dimension_numbers = ("NDHWC", "DHWIO", "NDHWC")

  # Perform the 3D transposed convolution
  # The input x_ref has a batch dimension of 1, which we apply the convolution on.
  y = jax.lax.conv_transpose(
    x_ref[...], kernel_ref[...], strides, padding, dimension_numbers=dimension_numbers, transpose_kernel=True
  )

  # Add the bias. The bias has shape (out_channels,) and will be broadcasted
  # to the shape of y.
  y = y + bias_ref[...]

  # Apply element-wise clipping
  y = jnp.clip(y, a_min=min_value)

  # Apply element-wise division
  y = y / divisor

  # Store the final result in the output buffer
  out_ref[...] = y


# The original computation involves a 3D transposed convolution followed by
# element-wise clip and division.
# The output shape of the nn.ConvTranspose with the given parameters is
# (16, 32, 64, 64, 16).
# We will parallelize the computation across the batch dimension. Each kernel
# instance will process one item from the batch.

# Define the output shape and dtype.
# The output shape of the convolution is (batch_size, out_depth, out_height, out_width, out_channels)
# which is (16, 32, 64, 64, 16). The dtype remains the same as the input.
output_shape_struct = jax.ShapeDtypeStruct((batch_size, 32, 64, 64, out_channels), x.dtype)

# The grid will iterate over the batch dimension.
grid = (batch_size,)

# Define BlockSpecs for inputs and outputs.
# Each kernel instance (indexed by `i`) gets a slice of the input and output
# corresponding to the i-th element in the batch.
# The kernel and bias are shared across all instances.

# Input specs
in_specs = [
  # Input tensor `x`
  pl.BlockSpec(block_shape=(1, *x.shape[1:]), index_map=lambda i: (i, 0, 0, 0, 0)),
  # Convolution kernel weights
  pl.BlockSpec(block_shape=params["kernel"].shape, index_map=lambda i: (0,) * params["kernel"].ndim),
  # Convolution bias
  pl.BlockSpec(block_shape=params["bias"].shape, index_map=lambda i: (0,) * params["bias"].ndim),
]

# Output spec
out_specs = pl.BlockSpec(block_shape=(1, *output_shape_struct.shape[1:]), index_map=lambda i: (i, 0, 0, 0, 0))

# Create a new kernel with min_value and divisor partially applied.
# The new kernel's signature will only contain the array arguments.
kernel_with_constants = partial(kernel, min_value=min_value, divisor=divisor)

x = pl.pallas_call(
  kernel_with_constants,
  out_shape=output_shape_struct,
  grid=grid,
  in_specs=in_specs,
  out_specs=out_specs,
)(x, params["kernel"], params["bias"]).block_until_ready()
