# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
height_in = 128
width_in = 256
stride = 1
padding = "VALID"
# output_padding is not a direct parameter in flax.linen.ConvTranspose
# output_padding = 0
groups = 1
bias = False

key = random.PRNGKey(0)
key_x, key_init = random.split(key)

# JAX expects channels-last format (NHWC) by default
x = random.normal(key_x, (batch_size, height_in, width_in, in_channels))

conv_transpose2d = nn.ConvTranspose(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size),
  strides=(stride, stride),
  padding=padding,
  # Correct parameter for grouped convolutions in Flax is 'feature_group_count'
  # but it is not supported in ConvTranspose. The equivalent is to reshape
  # the input and output channels and use a standard convolution.
  # For groups=1, this parameter is not needed.
  use_bias=bias,
)
variables = conv_transpose2d.init(key_init, x)


# Computation
def kernel(x_ref, kernel_ref, out_ref):
  """Pallas kernel for 2D transposed convolution."""
  for h in pl.prange(height_in):
    for w in pl.prange(width_in):
      # x_val: (32,)
      x_val = x_ref[0, h, w, :]
      # kernel_ref: (3, 3, 32, 64)
      # update: (3, 3, 64)
      update = jnp.einsum("c,hwCo->hwo", x_val, kernel_ref, preferred_element_type=jnp.float32)
      # Atomically add the update to the output slice
      out_ref.at[0, h : h + kernel_size, w : w + kernel_size, :].add(update)


# The kernel weights are stored in the 'params' dictionary within the variables
# Flax ConvTranspose stores weights as (H, W, C_in, C_out)
kernel_weights = variables["params"]["kernel"]

# The pallas_call replaces the original nn.ConvTranspose().apply() call.
# It computes the transposed convolution by tiling the input space.
# Each kernel instance computes the contribution of a 128x256 input tile
# to the output.
output = pl.pallas_call(
  kernel,
  # The output shape is calculated based on input dimensions, kernel size, and stride.
  # (128-1)*1+3 = 130 for height, (256-1)*1+3 = 258 for width.
  out_shape=jax.ShapeDtypeStruct(
    (batch_size, height_in + kernel_size - 1, width_in + kernel_size - 1, out_channels), x.dtype
  ),
  # Grid is tiled over batch, input height, and input width.
  grid_spec=pl.GridSpec(
    grid=(batch_size,),
    block_mapping=[
      pl.BlockMapping(block_shape=(1, height_in, width_in, in_channels), index_map=lambda i: (i, 0, 0, 0)),
      pl.BlockMapping(
        block_shape=(kernel_size, kernel_size, in_channels, out_channels), index_map=lambda i: (0, 0, 0, 0)
      ),
      pl.BlockMapping(
        block_shape=(1, height_in + kernel_size - 1, width_in + kernel_size - 1, out_channels),
        index_map=lambda i: (i, 0, 0, 0),
      ),
    ],
  ),
  in_specs=[
    pl.BlockSpec(memory_space=pl.TpuMemorySpace.ANY),
    pl.BlockSpec(memory_space=pl.TpuMemorySpace.ANY),
  ],
  out_specs=pl.BlockSpec(memory_space=pl.TpuMemorySpace.VMEM),
)(x, kernel_weights).block_until_ready()
