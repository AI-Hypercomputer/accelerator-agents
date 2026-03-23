# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl
from jax.experimental.pallas.ops.tpu import convolution

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3

key = random.PRNGKey(0)
x_key, params_key = random.split(key)

# JAX convention is channels-last: (N, H, W, C)
x = random.normal(x_key, (batch_size, height, width, in_channels))
conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size))
params = conv.init(params_key, x)["params"]


# Computation
def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  """
  Pallas kernel for a sequence of convolution, scaling, and reduction.

  This kernel performs the following operations on a block of data:
  1. 2D Convolution: Applies a convolution with 'VALID' padding. The input
     `x_ref` is a 16x16 patch which includes a 1-pixel halo needed for the
     3x3 kernel to produce an 8x8 output. We slice the required 10x10
     region from this patch.
  2. Bias Addition: Adds a bias term to the convolution result.
  3. Scaling: Multiplies the result by a constant scale factor.
  4. Reduction: Computes the minimum value across the feature channels.

  Args:
    x_ref: A reference to the input data block, including a halo.
    kernel_ref: A reference to the convolution kernel weights.
    bias_ref: A reference to the convolution bias vector.
    out_ref: A reference to the output data block.
  """
  # Define a constant used in the computation
  scale_factor = 2.0

  # For a 3x3 kernel and an 8x8 output with 'VALID' padding, we need a
  # 10x10 input patch. We slice this from the larger 16x16 `x_ref` which
  # was loaded for TPU memory alignment purposes.
  x_patch = x_ref[:, :10, :10, :]

  # 1. Perform the 2D convolution.
  # The dimension numbers specify the data layout:
  # 'NHWC' for input/output (Batch, Height, Width, Channels)
  # 'HWIO' for the kernel (Height, Width, In Channels, Out Channels)
  conv_out = convolution.conv(
    lhs=x_patch,
    rhs=kernel_ref[...],
    window_strides=(1, 1),
    padding="VALID",
    dimension_numbers=("NHWC", "HWIO", "NHWC"),
  )

  # Add the bias term. It will be broadcast across the spatial dimensions.
  conv_with_bias = conv_out + bias_ref[...]

  # 2. Scale the result.
  scaled_result = conv_with_bias * scale_factor

  # 3. Reduce by taking the minimum along the channel axis.
  # `keepdims=True` ensures the output rank is preserved, matching `out_ref`.
  final_result = jnp.min(scaled_result, axis=-1, keepdims=True)

  # Write the final result to the output buffer.
  out_ref[...] = final_result


# The output shape for a 'VALID' convolution is smaller than the input.
out_height = height - kernel_size + 1
out_width = width - kernel_size + 1

# The grid needs to cover the entire output, using ceiling division.
grid_h = (out_height + 7) // 8
grid_w = (out_width + 7) // 8

x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((x.shape[0], out_height, out_width, 1), x.dtype),
  grid=(x.shape[0], grid_h, grid_w),
  in_specs=[
    # Input image 'x'. For each 8x8 output tile, we need a 10x10 input
    # tile. We load a larger, TPU-compatible 16x16 block.
    pl.BlockSpec(
      block_shape=(1, 16, 16, x.shape[3]),
      index_map=lambda i, j, k: (i, j * 8, k * 8, 0),
    ),
    # The convolution kernel weights are read-only and used by all instances.
    pl.BlockSpec(
      block_shape=params["kernel"].shape,
      index_map=lambda i, j, k: (0, 0, 0, 0),
    ),
    # The convolution bias is also read-only and used by all instances.
    pl.BlockSpec(block_shape=params["bias"].shape, index_map=lambda i, j, k: (0,)),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, 8, 8, 1),
    index_map=lambda i, j, k: (i, j * 8, k * 8, 0),
  ),
)(x, params["kernel"], params["bias"]).block_until_ready()
