# Imports
import jax
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 128

key = random.PRNGKey(0)
key_init, key_x = random.split(key)

# JAX uses (N, H, W, C) format by default
x = random.normal(key_x, (batch_size, height, width, in_channels))

conv2d = nn.Conv(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size),
  strides=1,
  padding="VALID",  # padding=0 in torch is 'VALID' in jax
  kernel_dilation=1,
  feature_group_count=1,
  use_bias=False,
)
variables = conv2d.init(key_init, x)
w = variables["params"]["kernel"]  # The kernel weights


def kernel(x_ref, w_ref, out_ref):
  """
  Pallas kernel for 2D convolution.

  This kernel computes a tile of the 2D convolution output. It takes a slice
  of the input feature map `x_ref` and the full convolution weights `w_ref`
  to produce a corresponding slice of the output `out_ref`.

  Args:
    x_ref: A reference to a block of the input tensor. This block is a window
      of the input data required to compute the output tile.
    w_ref: A reference to the complete kernel weights tensor.
    out_ref: A reference to the output tensor block where the result of the
      convolution for this tile will be written.
  """
  x = x_ref[...]
  w = w_ref[...]

  # Iterate over the width of the output tile.
  for ow in range(out_ref.shape[2]):
    # Extract the input patch corresponding to the current output pixel.
    patch = x[:, :, ow : ow + kernel_size, :]

    # The convolution is a dot product between the patch and the kernel weights.
    # We contract over the spatial dimensions of the kernel (H, W) and the
    # input channels (I).
    out_pixel = jax.lax.dot_general(patch, w, dimension_numbers=(((1, 2, 3), (0, 1, 2)), ((), ())))

    # Write the result to the corresponding location in the output tile.
    out_ref[0, 0, ow, :] = out_pixel.reshape((out_channels,)).astype(out_ref.dtype)


# Calculate output shape
out_height = height - kernel_size + 1
out_width = width - kernel_size + 1
out_shape = jax.ShapeDtypeStruct((batch_size, out_height, out_width, out_channels), x.dtype)

# Define block sizes for tiling
# We tile the output width dimension
block_w = 8
# The required input width is larger due to the kernel halo
in_block_w_req = block_w + kernel_size - 1
# We pad the input block width to the next multiple of 8 to satisfy TPU constraints
in_block_w_padded = (in_block_w_req + 7) & ~7

# Define grid dimensions
grid_w = (out_width + block_w - 1) // block_w
grid = (batch_size, out_height, grid_w)


# Computation
# The pallas_call replaces the conv2d.apply(variables, x)
result = pl.pallas_call(
  kernel,
  out_shape=out_shape,
  grid=grid,
  in_specs=[
    # Spec for input image 'x'
    pl.BlockSpec(
      block_shape=(1, kernel_size, in_block_w_padded, in_channels),
      index_map=lambda b, h, j: (b, h, j * block_w, 0),
    ),
    # Spec for kernel weights 'w'
    pl.BlockSpec(block_shape=w.shape, index_map=lambda b, h, j: (0, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 1, block_w, out_channels), index_map=lambda b, h, j: (b, h, j * block_w, 0)),
)(x, w)
