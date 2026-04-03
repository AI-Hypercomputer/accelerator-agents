# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = (3, 3)
stride = (2, 2)
padding = "SAME"
# output_padding is not a direct parameter in Flax; 'SAME' padding is the idiomatic way to handle this upsampling case.
bias_shape = (1, 1, 1, out_channels)  # JAX uses NHWC format
scaling_factor = 2.0

key = random.PRNGKey(0)
key, x_key, params_key, bias_key = random.split(key, 4)

# JAX uses channels-last (NHWC) convention
x = random.normal(x_key, (batch_size, height, width, in_channels))

conv_transpose = nn.ConvTranspose(features=out_channels, kernel_size=kernel_size, strides=stride, padding=padding)
params = conv_transpose.init(params_key, x)["params"]
bias = random.normal(bias_key, bias_shape)

# The kernel from Flax has shape (KH, KW, IC, OC).
# For the 'HWOI' dimension numbering, we need (KH, KW, OC, IC).
# We perform the transpose outside the Pallas kernel, as transpose is not supported inside.
transposed_kernel_param = jnp.transpose(params["kernel"], (0, 1, 3, 2))


# Computation
def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  """
  Pallas kernel for a sequence of operations: ConvTranspose, bias add, clip, scale, clip, descale.
  """
  # Hard-coded constants from the original computation
  strides = (2, 2)
  padding = "SAME"
  scaling_factor = 2.0
  # Use 'HWOI' to match the pre-transposed kernel
  dn = ("NHWC", "HWOI", "NHWC")

  # Perform the transposed convolution.
  # The input x_ref has shape (1, H, W, IC) for each program instance.
  # The kernel_ref is pre-transposed to (KH, KW, OC, IC) to match 'HWOI'.
  # The output of conv_transpose will have shape (1, H*stride, W*stride, OC).
  conv_out = jax.lax.conv_transpose(x_ref[...], kernel_ref[...], strides=strides, padding=padding, dimension_numbers=dn)

  # Since the output BlockSpec covers only one row of the output feature map,
  # we need to slice the result of the convolution to match the output block.
  # The program_id(1) corresponds to the row index 'j' in the output grid.
  output_row_index = pl.program_id(1)
  # Dynamically slice the convolution output to get the specific row this kernel instance is responsible for.
  # The shape of the slice will be (1, 1, W*stride, OC), matching out_ref.
  output_slice = jax.lax.dynamic_slice_in_dim(conv_out, output_row_index, 1, axis=1)

  # Apply the sequence of post-processing operations.
  # Note: bias_ref has shape (1, 1, 1, OC) and will broadcast correctly.
  output_slice = output_slice + bias_ref[...]
  output_slice = jnp.clip(output_slice, a_min=0.0, a_max=1.0)
  output_slice = output_slice * scaling_factor
  output_slice = jnp.clip(output_slice, a_min=0.0, a_max=1.0)
  output_slice = output_slice / scaling_factor

  # Write the final result to the output buffer.
  out_ref[...] = output_slice


x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, height * stride[0], width * stride[1], out_channels), x.dtype),
  grid=(batch_size, height * stride[0]),
  in_specs=[
    pl.BlockSpec(block_shape=(1, height, width, in_channels), index_map=lambda i, j: (i, 0, 0, 0)),
    pl.BlockSpec(block_shape=transposed_kernel_param.shape, index_map=lambda i, j: (0, 0, 0, 0)),
    pl.BlockSpec(block_shape=bias.shape, index_map=lambda i, j: (0, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 1, width * stride[1], out_channels), index_map=lambda i, j: (i, j, 0, 0)),
)(x, transposed_kernel_param, bias).block_until_ready()
