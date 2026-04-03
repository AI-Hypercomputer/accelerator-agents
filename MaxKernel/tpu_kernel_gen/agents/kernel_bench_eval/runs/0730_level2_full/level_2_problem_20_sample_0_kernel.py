# Imports
import flax.linen as nn
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
stride = (2, 2, 2)
padding = "SAME"
bias_shape = (1, 1, 1, 1, out_channels)  # Shape for channels-last broadcasting

key = random.PRNGKey(0)
key, x_key, params_key, bias_key = random.split(key, 4)

# JAX uses channels-last data format (NDHWC)
x = random.normal(x_key, (batch_size, depth, height, width, in_channels))

# We use nn.ConvTranspose to initialize the kernel weights and perform the convolution.
conv_transpose_layer = nn.ConvTranspose(features=out_channels, kernel_size=kernel_size, strides=stride, padding=padding)
params = conv_transpose_layer.init(params_key, x)["params"]
bias = random.normal(bias_key, bias_shape)

# Perform the 3D transposed convolution using standard JAX.
# This operation is not supported inside a Pallas kernel on TPU.
conv_out = conv_transpose_layer.apply({"params": params}, x)


# Define a Pallas kernel for the subsequent element-wise operations.
def fused_elementwise_kernel(conv_out_ref, bias_ref, out_ref):
  # Load the convolution output, which will be used multiple times.
  original_conv_out = conv_out_ref[...]

  # Apply the sequence of element-wise operations.
  # 1. Add bias
  fused_x = original_conv_out + bias_ref[...].astype(original_conv_out.dtype)
  # 2. Add the original convolution output
  fused_x = fused_x + original_conv_out
  # 3. Multiply by the original convolution output
  fused_x = fused_x * original_conv_out
  # 4. Add the original convolution output again
  fused_x = fused_x + original_conv_out

  # Write the final result to the output buffer.
  out_ref[...] = fused_x


# The output shape of the Pallas call is the same as the convolution output.
output_shape = conv_out.shape

# Define the block shapes for tiling to fit into SRAM.
# We reduce the block size further to avoid memory exhaustion.
block_d = output_shape[1] // 4
block_h = output_shape[2] // 4
block_w = output_shape[3] // 4

grid = (
  batch_size,
  output_shape[1] // block_d,
  output_shape[2] // block_h,
  output_shape[3] // block_w,
)
tiled_block_shape = (1, block_d, block_h, block_w, out_channels)

# Call the Pallas kernel to perform the fused element-wise operations.
x = pl.pallas_call(
  fused_elementwise_kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, conv_out.dtype),
  grid=grid,
  in_specs=[
    pl.BlockSpec(
      block_shape=tiled_block_shape,
      index_map=lambda i, j, k, l: (i, j * block_d, k * block_h, l * block_w, 0),
    ),
    pl.BlockSpec(block_shape=bias.shape, index_map=lambda i, j, k, l: (0, 0, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(
    block_shape=tiled_block_shape,
    index_map=lambda i, j, k, l: (i, j * block_d, k * block_h, l * block_w, 0),
  ),
)(conv_out, bias).block_until_ready()
