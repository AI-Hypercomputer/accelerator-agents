# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5)
height = 256
width = 128
stride = (1, 1)
# In Flax, padding=(0,0) is equivalent to the string 'VALID'
padding = "VALID"
dilation = (1, 1)
groups = 1
bias = False

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# JAX/Flax expect channel-last data: (N, H, W, C)
x = random.normal(key_x, (batch_size, height, width, in_channels))

conv2d = nn.Conv(
  features=out_channels,
  kernel_size=kernel_size,
  strides=stride,
  padding=padding,
  kernel_dilation=dilation,
  feature_group_count=groups,
  use_bias=bias,
)
variables = conv2d.init(key_params, x)
kernel_weights = variables["params"]["kernel"]

# Calculate output shape
out_height = (height - (kernel_size[0] - 1) * dilation[0] - 1) // stride[0] + 1
out_width = (width - (kernel_size[1] - 1) * dilation[1] - 1) // stride[1] + 1
output_shape = (batch_size, out_height, out_width, out_channels)

# Define block sizes for tiling
# Each work item will compute a tile of the output
out_h_block = 32
out_w_block = 16


def kernel(x_ref, kernel_ref, out_ref):
  """
  Pallas kernel for 2D convolution.

  This kernel computes a tile of the output of a 2D convolution.
  It iterates over the kernel's spatial dimensions, and for each position,
  it computes the dot product between the corresponding input patch and
  the kernel weights, accumulating the result.

  Args:
    x_ref: A reference to the input tile.
    kernel_ref: A reference to the convolution kernel weights.
    out_ref: A reference to the output tile to be written to.
  """
  # Initialize an accumulator for the output tile with zeros.
  # The shape of the accumulator matches the output tile size.
  acc = jnp.zeros((1, out_h_block, out_w_block, out_channels), dtype=x.dtype)

  # The kernel weights are loaded entirely into SRAM.
  # We squeeze the batch dimension as it's always 1.
  kernel_val = kernel_ref[...]

  # Iterate over the spatial dimensions of the kernel (height and width).
  for kh in range(kernel_size[0]):
    for kw in range(kernel_size[1]):
      # Slice the input tile to get the patch corresponding to the current
      # kernel position. The slice size matches the output tile size.
      # The starting position of the slice is determined by the kernel indices.
      x_slice = jax.lax.dynamic_slice(x_ref[0], (kh, kw, 0), (out_h_block, out_w_block, in_channels))

      # Perform the dot product between the input slice and the corresponding
      # part of the kernel. This computes the convolution for one spatial
      # kernel position over all input and output channels for the entire tile.
      acc += jnp.dot(x_slice, kernel_val[kh, kw, :, :])

  # Write the final accumulated values to the output buffer.
  # The shape of acc must match the shape of out_ref for this assignment.
  out_ref[...] = acc


# Computation
output = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=(
    batch_size,
    pl.cdiv(out_height, out_h_block),
    pl.cdiv(out_width, out_w_block),
  ),
  in_specs=[
    pl.BlockSpec(
      block_shape=(
        1,
        (out_h_block - 1) * stride[0] + kernel_size[0],
        (out_w_block - 1) * stride[1] + kernel_size[1],
        in_channels,
      ),
      index_map=lambda b, h, w: (
        b,
        h * out_h_block * stride[0],
        w * out_w_block * stride[1],
        0,
      ),
    ),
    pl.BlockSpec(block_shape=kernel_weights.shape, index_map=lambda b, h, w: (0, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, out_h_block, out_w_block, out_channels),
    index_map=lambda b, h, w: (b, h * out_h_block, w * out_w_block, 0),
  ),
)(x, kernel_weights).block_until_ready()
