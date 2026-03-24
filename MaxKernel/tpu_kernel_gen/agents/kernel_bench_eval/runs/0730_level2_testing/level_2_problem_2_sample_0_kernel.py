# Imports
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (1, 1, 1, out_channels)
scaling_factor = 2.0

key = random.PRNGKey(0)
key, conv_key, bias_key, x_key = random.split(key, 4)

# Note: JAX/Flax convention is channel-last (N, H, W, C)
x_shape = (batch_size, height, width, in_channels)
x = random.normal(x_key, x_shape)
bias = random.normal(bias_key, bias_shape)

# The combination of stride, padding, and output_padding in the original
# PyTorch code results in doubling the spatial dimensions (32x32 -> 64x64).
# In Flax, padding='SAME' with a stride of 2 achieves the same effect.
conv_transpose = nn.ConvTranspose(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size),
  strides=(stride, stride),
  padding="SAME",
  use_bias=False,
)

params = conv_transpose.init(conv_key, x)["params"]

# Computation
# The conv_transpose primitive is not implemented in Pallas on TPU.
# We must run it outside the Pallas kernel.
conv_out = conv_transpose.apply({"params": params}, x)


def kernel(conv_out_ref, bias_ref, out_ref):
  """
  Pallas kernel for a sequence of element-wise operations:
  bias add, clip, scale, clip, divide.
  """
  # Load data
  conv_block = conv_out_ref[...]
  # Bias is broadcast.
  bias_block = bias_ref[...]

  # Fused operations
  y = conv_block + bias_block
  y = jnp.clip(y, a_min=0.0, a_max=1.0)
  y = y * 2.0
  y = jnp.clip(y, a_min=0.0, a_max=1.0)
  out_ref[...] = y / 2.0


# Define block sizes for the fused kernel.
# The grid will iterate over the output of the convolution.
output_shape = conv_out.shape
out_block_h = 16
out_block_w = 16

# We use Pallas to fuse the element-wise operations that follow the
# convolution.
x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
  grid=(
    batch_size,
    output_shape[1] // out_block_h,
    output_shape[2] // out_block_w,
  ),
  in_specs=[
    # Input is the result of the convolution.
    pl.BlockSpec(
      block_shape=(1, out_block_h, out_block_w, out_channels),
      index_map=lambda b, h, w: (b, h * out_block_h, w * out_block_w, 0),
    ),
    # Bias
    pl.BlockSpec(block_shape=bias.shape, index_map=lambda *_: tuple([0] * bias.ndim)),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, out_block_h, out_block_w, out_channels),
    index_map=lambda b, h, w: (b, h * out_block_h, w * out_block_w, 0),
  ),
)(conv_out, bias).block_until_ready()
