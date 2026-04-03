# Imports
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

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# Note: JAX uses channels-last (NHWC) by default
x = random.normal(key_x, (batch_size, height, width, in_channels))

conv2d = nn.Conv(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size),
  strides=1,
  padding="VALID",
  kernel_dilation=1,
  feature_group_count=1,
  use_bias=False,
)
params = conv2d.init(key_params, x)["params"]

# Define the tiling strategy and grid for the Pallas kernel.
# We tile over batch, height, and width to manage memory usage.
# We choose block sizes that satisfy TPU alignment constraints.
# For the output, the second-to-last dim (width) must be a multiple of 8.
# For the input, the second-to-last dim (width) of the block must also be a
# multiple of 8. The input block width is bW_out + kernel_size - 1.
# Let bW_out = 128. Then output block width is 128 (multiple of 8).
# Input block width is 128 + 3 - 1 = 130, not a multiple of 8.
# Let's use a smaller bW_out. If bW_out = 126, it's not a multiple of 8.
# Let's choose bW_out = 120. It's a multiple of 8.
# Input block width = 120 + 3 - 1 = 122. Not a multiple of 8.
# The key is that the *loaded slice* must conform, not the logical block.
# We can use a block size for width that is a multiple of 128 for simplicity
# and to ensure good performance.
bH_out = 32
bW_out = 128
out_height = height - kernel_size + 1
out_width = width - kernel_size + 1
grid_h = (out_height + bH_out - 1) // bH_out
grid_w = (out_width + bW_out - 1) // bW_out
grid = (batch_size, grid_h, grid_w)


# Computation
def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 2D convolution."""
  acc = jnp.zeros_like(out_ref[0])
  for ky in range(kernel_size):
    for kx in range(kernel_size):
      in_slice = jax.lax.slice(x_ref[0], (ky, kx, 0), (ky + bH_out, kx + bW_out, in_channels))
      kernel_w = kernel_ref[ky, kx, :, :]
      acc += jnp.einsum("hwi,io->hwo", in_slice, kernel_w, preferred_element_type=x_ref.dtype)
  out_ref[0] = acc


# The kernel parameter is passed as a single, non-tiled block to each instance.
kernel_spec = pl.BlockSpec(
  block_shape=params["kernel"].shape, index_map=lambda b, h, w: tuple([0] * params["kernel"].ndim)
)

result = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_height, out_width, out_channels), x.dtype),
  grid=grid,
  in_specs=[
    pl.BlockSpec(
      block_shape=(1, bH_out + kernel_size - 1, bW_out + kernel_size - 1, in_channels),
      index_map=lambda b, h, w: (b, h * bH_out, w * bW_out, 0),
    ),
    kernel_spec,
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, bH_out, bW_out, out_channels), index_map=lambda b, h, w: (b, h * bH_out, w * bW_out, 0)
  ),
)(x, params["kernel"]).block_until_ready()
