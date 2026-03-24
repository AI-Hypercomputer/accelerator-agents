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
kernel_size = 3
stride = 2
padding = 1
output_padding = 1

key = random.PRNGKey(0)
key_x, key_params, key_bias = random.split(key, 3)

# JAX uses NHWC channel format by default.
x = random.normal(key_x, (batch_size, height, width, in_channels))

# The PyTorch ConvTranspose2d with kernel_size=3, stride=2, padding=1,
# and output_padding=1 doubles the input spatial dimensions (32x32 -> 64x64).
# In Flax, padding='SAME' with strides=(2, 2) achieves the same output shape.
conv_transpose = nn.ConvTranspose(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size),
  strides=(stride, stride),
  padding="SAME",
  use_bias=False,
)
params = conv_transpose.init(key_params, x)["params"]

# The original PyTorch bias has shape (C, H, W) -> (16, 1, 1).
# The equivalent shape in JAX's NHWC format is (H, W, C) -> (1, 1, 16).
# This shape is necessary to replicate the specific broadcasting behavior
# of the original script.
bias = random.normal(key_bias, (1, 1, out_channels))

# Perform the transposed convolution outside of the Pallas kernel, as it is
# not a supported primitive in Pallas on TPU.
conv_out = conv_transpose.apply({"params": params}, x)


# Computation
def kernel(conv_out_ref, bias_ref, out_ref):
  """
  Pallas kernel for a sequence of operations including reductions,
  activation, and bias addition. The ConvTranspose is done outside the kernel.
  """
  # The input conv_out_ref has shape (1, 64, 64, 16).
  x = conv_out_ref[...]

  # Reduce along the channel dimension (axis=3).
  # The input shape is (1, 64, 64, 16).
  # The output shape is (1, 64, 64, 1).
  x = jnp.min(x, axis=3, keepdims=True)

  # Reduce along the height dimension (axis=1).
  # The input shape is (1, 64, 64, 1).
  # The output shape is (1, 1, 64, 1).
  x = jnp.sum(x, axis=1, keepdims=True)

  # Apply the GELU activation function element-wise.
  # The shape remains (1, 1, 64, 1).
  x = jax.nn.gelu(x)

  # Add the bias. This involves broadcasting.
  # x shape: (1, 1, 64, 1)
  # bias_ref shape: (1, 1, 16)
  # The result is broadcast to shape (1, 1, 64, 16).
  x = x + bias_ref[...]

  # Write the final result to the output buffer.
  out_ref[...] = x


x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((128, 1, 64, 16), x.dtype),
  grid=(128,),
  in_specs=[
    pl.BlockSpec(block_shape=(1, 64, 64, 16), index_map=lambda i: (i, 0, 0, 0)),
    pl.BlockSpec(block_shape=(1, 1, 16), index_map=lambda i: (0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, 1, 64, 16), index_map=lambda i: (i, 0, 0, 0)),
)(conv_out, bias).block_until_ready()
