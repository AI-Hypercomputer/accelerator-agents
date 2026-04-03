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
kernel_size = 3
depth = 64
width = 64
height = 64
stride = 1
padding = "VALID"
dilation = 1
groups = 1
bias = False

key = random.PRNGKey(0)
key_input, key_params = random.split(key)

# JAX expects channel-last data: (N, D, H, W, C)
x = random.normal(key_input, (batch_size, depth, width, height, in_channels))

# Flax's nn.Conv is general and handles 3D convolutions.
conv3d = nn.Conv(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size, kernel_size),
  strides=(stride, stride, stride),
  padding=padding,
  kernel_dilation=(dilation, dilation, dilation),
  feature_group_count=groups,
  use_bias=bias,
)
variables = conv3d.init(key_params, x)
w = variables["params"]["kernel"]

# Calculate output dimensions
out_depth = (depth - kernel_size) // stride + 1
out_height = (height - kernel_size) // stride + 1
out_width = (width - kernel_size) // stride + 1
out_shape_struct = jax.ShapeDtypeStruct((batch_size, out_depth, out_height, out_width, out_channels), x.dtype)

# Define a block size for tiling the width dimension
block_w = 128


# Computation
def kernel(x_ref, w_ref, out_ref):
  """Pallas kernel for 3D convolution."""
  # Initialize an accumulator with zeros for the output tile.
  # The shape of the accumulator is simplified for the dot product.
  acc = jnp.zeros((block_w, out_channels), dtype=x_ref.dtype)

  # Iterate over the 3D kernel dimensions (depth, height, width).
  # This is the sliding window part of the convolution.
  for kd in range(kernel_size):
    for kh in range(kernel_size):
      for kw in range(kernel_size):
        # Slice the input tile (x_ref) to get the current receptive field.
        # The slice is of shape (block_w, in_channels).
        # The offsets (kd, kh, kw) are relative to the start of the input
        # tile (x_ref).
        x_slice = jax.lax.dynamic_slice_in_dim(
          x_ref.reshape(-1, in_channels),
          kd * (kernel_size * ((block_w - 1) * stride + kernel_size))
          + kh * ((block_w - 1) * stride + kernel_size)
          + kw,
          block_w,
          axis=0,
        )

        # Reshape the input slice and the corresponding kernel part for matrix
        # multiplication. The dot product computes the convolution for this
        # specific kernel position.
        # x_slice reshaped to: (block_w, in_channels)
        # w_ref slice reshaped to: (in_channels, out_channels)
        # Resulting dot product shape: (block_w, out_channels)
        acc += jnp.dot(x_slice, w_ref[kd, kh, kw, :, :])

  # Write the accumulated result to the output memory block.
  # The result is reshaped to match the 5D output block shape.
  out_ref[...] = acc.reshape(out_ref.shape)


# Pallas call to replace the computation
output = pl.pallas_call(
  kernel,
  out_shape=[out_shape_struct],
  grid=(
    batch_size,
    out_depth,
    out_height,
    (out_width + block_w - 1) // block_w,
  ),
  in_specs=[
    # x_ref spec
    pl.BlockSpec(
      block_shape=(
        1,
        kernel_size,
        kernel_size,
        (block_w - 1) * stride + kernel_size,
        in_channels,
      ),
      index_map=lambda n, od, oh, ow_i: (
        n,
        od * stride,
        oh * stride,
        ow_i * block_w * stride,
        0,
      ),
    ),
    # w_ref spec
    pl.BlockSpec(
      block_shape=w.shape,
      index_map=lambda *_: (0,) * w.ndim,
    ),
  ],
  out_specs=[
    pl.BlockSpec(
      block_shape=(1, 1, 1, block_w, out_channels),
      index_map=lambda n, od, oh, ow_i: (n, od, oh, ow_i * block_w, 0),
    )
  ],
  compiler_params=dict(mosaic=dict(dimension_semantics=("parallel", "parallel", "parallel", "parallel"))),
)(x, w).block_until_ready()
