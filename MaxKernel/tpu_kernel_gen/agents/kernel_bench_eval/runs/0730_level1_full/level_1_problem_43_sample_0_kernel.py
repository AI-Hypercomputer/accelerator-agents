# Imports
import math

import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
channels = 32
dim1 = 64
dim2 = 64
dim3 = 64
kernel_size = 3
stride = 2
padding = 1
dilation = 3

key = random.PRNGKey(0)
# JAX uses channels-last convention (N, D, H, W, C)
x = random.normal(key, (batch_size, dim1, dim2, dim3, channels))


# Computation
def kernel(x_ref, out_ref):
  """Pallas kernel for 3D max pooling.

  This kernel implements the equivalent of lax.reduce_window with lax.max
  for a 3D input tensor.

  Args:
    x_ref: Input tensor block.
    out_ref: Output tensor block to be populated.
  """
  # Hardcoded parameters from the source computation
  kernel_size = 3
  stride = 2
  dilation = 3
  channels = x_ref.shape[-1]

  # Get the shape of the output block
  _, block_d, block_h, block_w, _ = out_ref.shape

  # Iterate over each element in the output block.
  for i in range(block_d):
    for j in range(block_h):
      for k in range(block_w):
        # Initialize the maximum value for the current output element.
        max_val = jnp.full((channels,), -jnp.inf, dtype=x_ref.dtype)

        # Iterate over each element in the pooling window.
        for kd in range(kernel_size):
          for kh in range(kernel_size):
            for kw in range(kernel_size):
              # Calculate the coordinates in the input block (x_ref).
              d_in = i * stride + kd * dilation
              h_in = j * stride + kh * dilation
              w_in = k * stride + kw * dilation

              # Update the maximum value.
              max_val = jnp.maximum(max_val, x_ref[0, d_in, h_in, w_in, :])

        # Write the result to the output block.
        out_ref[0, i, j, k, :] = max_val


# Calculate output dimensions
out_dim = math.floor((dim1 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
output_shape = (batch_size, out_dim, out_dim, out_dim, channels)

# Define block sizes for spatial dimensions
block_d = 8
block_h = 8
block_w = 8

# Define the shape for the output argument to pallas_call
out_shape = jax.ShapeDtypeStruct(output_shape, jnp.float32)

# Calculate the size of the input block needed for one output block
dilated_kernel_size = dilation * (kernel_size - 1) + 1
in_block_d_unpadded = (block_d - 1) * stride + dilated_kernel_size
in_block_h_unpadded = (block_h - 1) * stride + dilated_kernel_size
in_block_w_unpadded = (block_w - 1) * stride + dilated_kernel_size

# Pad to the next multiple of 8 for TPU compatibility
in_block_d = (in_block_d_unpadded + 7) // 8 * 8
in_block_h = (in_block_h_unpadded + 7) // 8 * 8
in_block_w = (in_block_w_unpadded + 7) // 8 * 8


output = pl.pallas_call(
  kernel,
  out_shape=out_shape,
  grid=(
    batch_size,
    math.ceil(out_dim / block_d),
    math.ceil(out_dim / block_h),
    math.ceil(out_dim / block_w),
  ),
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, in_block_d, in_block_h, in_block_w, channels),
      index_map=lambda b, i, j, k: (
        b,
        i * block_d * stride - padding,
        j * block_h * stride - padding,
        k * block_w * stride - padding,
        0,
      ),
    )
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, block_d, block_h, block_w, channels),
    index_map=lambda b, i, j, k: (b, i * block_d, j * block_h, k * block_w, 0),
  ),
)(x).block_until_ready()
