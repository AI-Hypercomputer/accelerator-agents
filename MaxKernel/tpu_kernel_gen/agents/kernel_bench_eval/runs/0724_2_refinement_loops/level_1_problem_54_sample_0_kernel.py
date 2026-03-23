# Imports
import math

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
depth = 64
width = 64
height = 64
stride = 1
padding = 0
dilation = 1
groups = 1
bias = False

key = random.PRNGKey(0)
key_input, key_params = random.split(key)

# JAX/Flax uses channels-last convention: (N, D, H, W, C)
x = random.normal(key_input, (batch_size, depth, width, height, in_channels))

conv3d = nn.Conv(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size, kernel_size),
  strides=(stride, stride, stride),
  padding=padding,
  kernel_dilation=(dilation, dilation, dilation),
  feature_group_count=groups,
  use_bias=bias,
)
params = conv3d.init(key_params, x)["params"]


# Computation
def kernel(x_ref, w_ref, out_ref):
  """Pallas kernel for 3D convolution."""
  # Get the kernel dimensions.
  kd, kh, kw = w_ref.shape[0:3]

  # Use nested lax.fori_loop to iterate over the output block's spatial dimensions.
  # This prevents unrolling and handles boundary blocks where out_ref.shape is smaller
  # than the nominal block size.
  def d_loop_body(od, _):
    def h_loop_body(oh, _):
      def w_loop_body(ow, _):
        # Accumulator for the output pixel, initialized to zeros.
        acc = jnp.zeros(out_ref.shape[-1], dtype=x_ref.dtype)

        # The loops over the kernel dimensions (z, y, x) are small (3x3x3)
        # and can be unrolled by the compiler without issues.
        for z in range(kd):
          for y in range(kh):
            for x_kernel in range(kw):
              # Calculate the corresponding input coordinates.
              input_d = od + z
              input_h = oh + y
              input_w = ow + x_kernel

              # Load the input vector and kernel weights.
              input_val = x_ref[0, input_d, input_h, input_w, :]
              kernel_val = w_ref[z, y, x_kernel, :, :]

              # Compute the dot product and accumulate.
              acc += jnp.sum(input_val[:, None] * kernel_val, axis=0)

        # Store the final computed value for the output pixel.
        out_ref[0, od, oh, ow, :] = acc
        return None  # No loop-carried state

      lax.fori_loop(0, out_ref.shape[3], w_loop_body, None)
      return None  # No loop-carried state

    lax.fori_loop(0, out_ref.shape[2], h_loop_body, None)
    return None  # No loop-carried state

  lax.fori_loop(0, out_ref.shape[1], d_loop_body, None)


# Extract the kernel weights from the Flax parameters.
# The shape is (KD, KH, KW, C_in, C_out) -> (3, 3, 3, 3, 64)
w = params["kernel"]

# --- Pallas Call Configuration ---

# 1. Define Output Shape & Data Type
# For a 'VALID' convolution (padding=0, stride=1), output dim = input_dim - kernel_dim + 1
# O_D = 64 - 3 + 1 = 62
# O_H = 64 - 3 + 1 = 62
# O_W = 64 - 3 + 1 = 62
# Final shape: (N, D, H, W, C_out) -> (16, 62, 62, 62, 64)
output_shape = (
  x.shape[0],
  x.shape[1] - w.shape[0] + 1,
  x.shape[2] - w.shape[1] + 1,
  x.shape[3] - w.shape[2] + 1,
  w.shape[4],
)

# 2. Define Grid and Block Sizes
# We tile the computation over the spatial dimensions of the output.
# Let's choose a block size of 8 for each spatial dimension.
b_out_d, b_out_h, b_out_w = 8, 8, 8

# The grid will have one dimension for the batch and one for each tiled spatial dimension.
grid = (
  output_shape[0],  # Batch dimension
  math.ceil(output_shape[1] / b_out_d),  # Tiled depth
  math.ceil(output_shape[2] / b_out_h),  # Tiled height
  math.ceil(output_shape[3] / b_out_w),  # Tiled width
)  # -> (16, 8, 8, 8)

# 3. Define Input Block Shapes
# Calculate the required input patch size to produce one output block.
# I_block = (O_block - 1) * Stride + K_size
# Since stride=1, I_block = O_block - 1 + K_size
b_in_d = (b_out_d - 1) + w.shape[0]  # (8 - 1) + 3 = 10
b_in_h = (b_out_h - 1) + w.shape[1]  # (8 - 1) + 3 = 10
b_in_w = (b_out_w - 1) + w.shape[2]  # (8 - 1) + 3 = 10

# For TPU compatibility, the second-to-last dim of the input block must be divisible by 8.
# We pad the required width (10) to the next multiple of 8, which is 16.
b_in_w_padded = math.ceil(b_in_w / 8) * 8  # -> 16

# --- Pallas Call Invocation ---

output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=grid,
  in_specs=[
    # Spec for input tensor 'x'
    pl.BlockSpec(
      block_shape=(1, b_in_d, b_in_h, b_in_w_padded, x.shape[-1]),
      index_map=lambda n, d, h, w_idx: (n, d * b_out_d, h * b_out_h, w_idx * b_out_w, 0),
    ),
    # Spec for kernel weights 'w'. Each kernel instance gets the full weights.
    pl.BlockSpec(block_shape=w.shape, index_map=lambda *_: (0,) * w.ndim),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, b_out_d, b_out_h, b_out_w, w.shape[-1]),
    index_map=lambda n, d, h, w_idx: (n, d * b_out_d, h * b_out_h, w_idx * b_out_w, 0),
  ),
)(x, w).block_until_ready()
