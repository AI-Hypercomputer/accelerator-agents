# Imports
import math

import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
width = 128
height = 128
stride = 1
padding = "VALID"
output_padding = 0
groups = 1
bias = False
key = random.PRNGKey(0)
key_x, key_init = random.split(key)

# JAX expects channels-last data by default: (N, H, W, C)
x = random.normal(key_x, (batch_size, height, width, in_channels))

conv_transpose2d = nn.ConvTranspose(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size),
  strides=(stride, stride),
  padding=padding,
  use_bias=bias,
)
variables = conv_transpose2d.init(key_init, x)


def kernel(x_ref, kernel_ref, out_ref):
  """
  Pallas kernel for 2D transposed convolution.

  This kernel computes a tile of the transposed convolution output. It works
  from an output-centric perspective: for each pixel in the output tile, it
  iterates over the kernel's weights, finds the corresponding input pixel,
  and accumulates the product.

  Args:
    x_ref: A reference to the input tensor tile.
    kernel_ref: A reference to the convolution kernel.
    out_ref: A reference to the output tensor tile to be written to.
  """
  # Get the logical indices of this program instance in the grid.
  # These indices determine which output tile this instance is responsible for.
  i = pl.program_id(1)
  j = pl.program_id(2)

  # Initialize the output tile with zeros.
  out_ref[...] = jnp.zeros_like(out_ref)

  # Define the shape of the output tile this kernel instance will compute.
  # These dimensions come from the `block_shape` in `out_specs`.
  BLOCK_H, BLOCK_W = 8, 128

  # Calculate the base offsets for the output tile in the global output tensor.
  h_out_base = i * BLOCK_H
  w_out_base = j * BLOCK_W

  # The kernel is 3x3. We iterate over each position in the kernel.
  for kh in range(3):
    for kw in range(3):
      # For a given output coordinate (h_out, w_out) and kernel offset (kh, kw),
      # the corresponding input coordinate is (h_out - kh, w_out - kw).
      # We calculate the top-left corner of the input patch to read.
      h_in_start = h_out_base - kh
      w_in_start = w_out_base - kw

      # Load the input block using dynamic slices. `pl.load` with `evasion_val`
      # will automatically handle out-of-bounds accesses by padding with that value.
      input_block = pl.load(
        x_ref,
        (
          0,
          pl.dslice(h_in_start, BLOCK_H),
          pl.dslice(w_in_start, BLOCK_W),
          slice(None),  # Load all input channels
        ),
        evasion_val=0.0,
      )

      # Get the kernel slice for the current (kh, kw) position.
      # kernel_ref has shape (3, 3, 32, 64).
      # kernel_slice will have shape (32, 64).
      kernel_slice = kernel_ref[kh, kw, :, :]

      # Perform a batched matrix multiplication (dot product) between the
      # masked input block and the kernel slice.
      # (BLOCK_H, BLOCK_W, 32) @ (32, 64) -> (BLOCK_H, BLOCK_W, 64)
      contribution = jnp.dot(input_block, kernel_slice)

      # Accumulate the result into the output tile.
      # out_ref has shape (1, BLOCK_H, BLOCK_W, 64).
      out_ref[0, :, :, :] += contribution


# Computation
output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, 130, 130, out_channels), x.dtype),
  grid=(batch_size, math.ceil(130 / 8), math.ceil(130 / 128)),
  in_specs=[
    pl.BlockSpec(block_shape=(1, 128, 128, 32), index_map=lambda n, i, j: (n, 0, 0, 0)),
    pl.BlockSpec(block_shape=(3, 3, 32, 64), index_map=lambda n, i, j: (0, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 8, 128, 64), index_map=lambda n, i, j: (n, i * 8, j * 128, 0)),
)(x, variables["params"]["kernel"]).block_until_ready()
