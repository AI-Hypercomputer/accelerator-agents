# Imports
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import pallas as pl
from jax.experimental.pallas.ops.tpu import convolve

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
input_shape = (batch_size, height, width, in_channels)
bias_shape = (1, 1, 1, out_channels)

key = random.PRNGKey(0)
key, conv_key, bias_key, x_key = random.split(key, 4)

# Use nn.Conv to initialize weights for a standard convolution
flax_conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size))
params = flax_conv.init(conv_key, jnp.ones(input_shape))["params"]
bias = random.normal(bias_key, bias_shape)
x = random.normal(x_key, input_shape)


# Computation
def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  """Pallas kernel for a convolution, ReLU, and bias add sequence."""
  # Perform the 2D convolution using the Pallas-specific convolution primitive.
  # 'NHWC' specifies the layout for input/output: (Batch, Height, Width, Channels).
  # 'HWIO' specifies the layout for the kernel: (Height, Width, Input channels, Output channels).
  # The strides are (1, 1) and padding is 'SAME' to match the default nn.Conv behavior.
  conv_out = convolve(
    x_ref[...], kernel_ref[...], window_strides=(1, 1), padding="SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
  )

  # Apply the ReLU activation function element-wise.
  relu_out = nn.relu(conv_out)

  # Add the bias term. Broadcasting ensures it's added to all spatial locations.
  final_out = relu_out + bias_ref[...]

  # Write the final result to the output reference.
  out_ref[...] = final_out


# The pallas_call invocation remains the same.
# It processes one batch item at a time.
y = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, height, width, out_channels), x.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(1, height, width, in_channels), index_map=lambda n: (n, 0, 0, 0)),
    pl.BlockSpec(block_shape=params["kernel"].shape, index_map=lambda _: (0, 0, 0, 0)),
    pl.BlockSpec(block_shape=bias.shape, index_map=lambda _: (0, 0, 0, 0)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, height, width, out_channels), index_map=lambda n: (n, 0, 0, 0)),
)(x, params["kernel"], bias).block_until_ready()
