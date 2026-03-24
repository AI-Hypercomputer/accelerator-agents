# Imports
import flax.linen as nn
import jax
import jax.random as random
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
multiplier_shape = (1, 1, 1, out_channels)
key = random.PRNGKey(0)
key_x, key_conv, key_multiplier = random.split(key, 3)
x = random.normal(key_x, (batch_size, height, width, in_channels))
conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size))
conv_params = conv.init(key_conv, x)["params"]
multiplier = random.normal(key_multiplier, multiplier_shape)

# The dimension numbers for a standard Flax Conv on JAX.
dn = ("NHWC", "HWIO", "NHWC")
# Perform the convolution.
conv_out = jax.lax.conv_general_dilated(
  x,
  conv_params["kernel"],
  window_strides=(1, 1),
  padding="SAME",
  dimension_numbers=dn,
)


# Computation
def kernel(conv_out_ref, bias_ref, multiplier_ref, y_ref):
  """
  Pallas kernel that applies a sequence of layers: Conv, multiplication,
  leaky_relu, and gelu.
  """
  # Add the bias.
  x = conv_out_ref[...] + bias_ref[...]
  # Apply element-wise multiplication.
  x = x * multiplier_ref[...]
  # Apply activation functions.
  x = nn.leaky_relu(x)
  y = nn.gelu(x)
  # Store the result.
  y_ref[...] = y


y = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, height, width, out_channels), x.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(1, height, width, out_channels), index_map=lambda i: (i, 0, 0, 0)),
    pl.BlockSpec(block_shape=conv_params["bias"].shape, index_map=lambda i: (0,)),
    pl.BlockSpec(block_shape=multiplier.shape, index_map=lambda i: (0, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, height, width, out_channels), index_map=lambda i: (i, 0, 0, 0)),
)(conv_out, conv_params["bias"], multiplier).block_until_ready()
