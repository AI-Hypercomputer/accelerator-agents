# Imports
import math

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
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
key_x, key_params = random.split(key)

x = random.normal(key_x, (batch_size, depth, width, height, in_channels))
conv3d = nn.Conv(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size, kernel_size),
  strides=stride,
  padding=padding,
  kernel_dilation=dilation,
  feature_group_count=groups,
  use_bias=bias,
)
variables = conv3d.init(key_params, x)

# Define block sizes for tiling
BLOCK_D, BLOCK_W = 8, 8

# Calculate output dimensions
out_depth = depth - kernel_size + 1
out_width = width - kernel_size + 1
out_height = height - kernel_size + 1

# Calculate input block size based on receptive field
IN_BLOCK_D = BLOCK_D + kernel_size - 1
IN_BLOCK_W = BLOCK_W + kernel_size - 1

# Define the output shape and type
out_struct = jax.ShapeDtypeStruct((batch_size, out_depth, out_width, out_height, out_channels), x.dtype)


def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 3D convolution.

  Args:
    x_ref: Input tile.
    kernel_ref: Convolution kernel.
    out_ref: Output tile to write to.
  """
  # Load inputs from HBM into SRAM
  x = x_ref[...]
  kernel_val = kernel_ref[...]

  # Get kernel dimensions for slicing
  (kd, kw, kh, _, out_channels) = kernel_val.shape
  # Get output block dimensions from the output reference shape
  # The shape is (1, BLOCK_D, BLOCK_W, out_height, out_channels)
  _, block_d, block_w, out_h, _ = out_ref.shape

  # Iterate over the output block dimensions
  for d in range(block_d):
    for w in range(block_w):
      for h in range(out_h):
        # For each output pixel (d, w, h), initialize an accumulator.
        acc = jnp.zeros((out_channels,), dtype=x.dtype)
        # Iterate over the receptive field (kernel dimensions)
        for rd in range(kd):
          for rw in range(kw):
            for rh in range(kh):
              # Slice the input vector corresponding to the current
              # receptive field position.
              in_vec = x[0, d + rd, w + rw, h + rh, :]
              # Slice the corresponding kernel weights.
              k_mat = kernel_val[rd, rw, rh, :, :]
              # Perform a dot product and accumulate the result.
              acc += jnp.dot(in_vec, k_mat)

        # Write the final computed pixel to the output reference.
        # The indices (0, d, w, h) are relative to the out_ref block.
        out_ref[0, d, w, h, :] = acc


# Computation
result = pl.pallas_call(
  kernel,
  out_shape=out_struct,
  grid=(
    batch_size,
    math.ceil(out_depth / BLOCK_D),
    math.ceil(out_width / BLOCK_W),
  ),
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, IN_BLOCK_D, IN_BLOCK_W, height, x.shape[-1]),
      index_map=lambda i, j, k: (i, j * BLOCK_D, k * BLOCK_W, 0, 0),
    ),
    pl.BlockSpec(
      block_shape=variables["params"]["kernel"].shape,
      index_map=lambda i, j, k: tuple([0] * variables["params"]["kernel"].ndim),
    ),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, BLOCK_D, BLOCK_W, out_height, out_channels),
    index_map=lambda i, j, k: (i, j * BLOCK_D, k * BLOCK_W, 0, 0),
  ),
)(x, variables["params"]["kernel"]).block_until_ready()
