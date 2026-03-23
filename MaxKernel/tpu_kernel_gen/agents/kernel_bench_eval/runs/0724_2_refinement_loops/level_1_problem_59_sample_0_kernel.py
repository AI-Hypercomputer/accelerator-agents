# Imports
import math

import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 256
depth = 10

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# JAX/Flax convention is channels-last: (N, H, W, D, C)
x = random.normal(key_x, (batch_size, height, width, depth, in_channels))

conv3d = nn.Conv(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size, 1),
  strides=(1, 1, 1),
  padding="VALID",  # Corresponds to padding=0 in PyTorch
  kernel_dilation=(1, 1, 1),
  feature_group_count=1,
  use_bias=False,
)
variables = conv3d.init(key_params, x)


# Computation
def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 3D convolution."""
  # kernel_ref shape: (KH, KW, 1, C_in, C_out)
  # x_ref shape: (1, H_in, W_in, D_in, C_in)
  # out_ref shape: (1, H_out, W_out, D_out, C_out)
  _, _, _, D_out, C_out = out_ref.shape

  # Load data from Refs into JAX arrays.
  x_tile = x_ref[...]
  kernel_weights = kernel_ref[...]

  # Iterate over the output tile dimensions
  for h in range(out_ref.shape[1]):
    for w in range(out_ref.shape[2]):
      # The kernel depth is 1, so the convolution is applied independently
      # at each depth slice of the input.
      for d in range(D_out):
        # Extract the input patch (receptive field) for the output at (h, w, d)
        # using a static slice, which is supported in Pallas.
        x_patch = jax.lax.slice(
          x_tile,
          start_indices=(0, h, w, d, 0),
          limit_indices=(1, h + kernel_size, w + kernel_size, d + 1, in_channels),
        )
        # x_patch shape: (1, KH, KW, 1, C_in)

        # Perform the convolution for a single output pixel using einsum.
        # This contracts over kernel height, width, depth, and input channels.
        # x_patch shape: (b=1, h=KH, w=KW, d=1, c=C_in)
        # kernel_weights shape: (h=KH, w=KW, d=1, c=C_in, o=C_out)
        # Result shape: (b=1, o=C_out)
        out_pixel = jnp.einsum("bhwdc,hwdco->bo", x_patch, kernel_weights)

        # Write the resulting pixel (all output channels) to the output tile.
        out_ref[0, h, w, d, :] = out_pixel.squeeze()


# Calculate output dimensions based on 'VALID' padding and stride of 1
H_out = height - kernel_size + 1
W_out = width - kernel_size + 1
D_out = depth - 1 + 1  # Kernel depth is 1

# Define block sizes for tiling the output's spatial dimensions
BLOCK_H = 16
BLOCK_W = 16

# Calculate grid size. The grid iterates over the batch and output tiles.
grid = (batch_size, math.ceil(H_out / BLOCK_H), math.ceil(W_out / BLOCK_W))

# Define the shape of the kernel weights tensor
kernel_weights_shape = (kernel_size, kernel_size, 1, in_channels, out_channels)

# Define the shape of the input patch required to compute one output block
in_block_h = BLOCK_H + kernel_size - 1
in_block_w = BLOCK_W + kernel_size - 1

# The pallas_call replaces the `conv3d.apply` call.
output = pl.pallas_call(
  kernel,  # The user-defined Pallas kernel function
  out_shape=jax.ShapeDtypeStruct((batch_size, H_out, W_out, D_out, out_channels), x.dtype),
  grid=grid,
  in_specs=[
    # Spec for the input tensor `x`
    pl.BlockSpec(
      block_shape=(1, in_block_h, in_block_w, depth, in_channels),
      index_map=lambda b, h_idx, w_idx: (b, h_idx * BLOCK_H, w_idx * BLOCK_W, 0, 0),
    ),
    # Spec for the kernel weights, which are not tiled
    pl.BlockSpec(block_shape=kernel_weights_shape, index_map=lambda *_: (0,) * len(kernel_weights_shape)),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, BLOCK_H, BLOCK_W, D_out, out_channels),
    index_map=lambda b, h_idx, w_idx: (b, h_idx * BLOCK_H, w_idx * BLOCK_W, 0, 0),
  ),
)(x, variables["params"]["kernel"]).block_until_ready()
