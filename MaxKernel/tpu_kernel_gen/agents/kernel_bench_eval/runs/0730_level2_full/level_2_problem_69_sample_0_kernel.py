# Imports
import jax
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl
from jax.experimental.pallas.ops.tpu import conv_general_dilated as tpu_conv

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# JAX/Flax uses channels-last convention (N, H, W, C)
x = random.normal(key_x, (batch_size, height, width, in_channels))

# In Flax, layers are stateless. We initialize the layer and its parameters separately.
conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size))
params = conv.init(key_params, x)["params"]


# Computation
def kernel(x_ref, kernel_ref, bias_ref, out_ref):
  """
  Pallas kernel for a sequence of Convolution, Hard Swish, and ReLU.

  Args:
    x_ref: Input tensor reference for a single batch item.
    kernel_ref: Convolution kernel weights reference.
    bias_ref: Convolution bias reference.
    out_ref: Output tensor reference.
  """
  # The input x_ref already has a batch dimension of 1.
  x = x_ref[...]

  # Define the dimension numbers for the convolution.
  # JAX/Flax convention is NHWC for input/output and HWIO for the kernel.
  dn = ("NHWC", "HWIO", "NHWC")

  # Perform the 2D convolution using the Pallas TPU-specific operator.
  y = tpu_conv(
    x,
    kernel_ref[...],
    window_strides=(1, 1),
    padding="VALID",
    dimension_numbers=dn,
  )

  # Add the bias.
  y = y + bias_ref[...].astype(y.dtype)

  # Apply the non-linear activation functions in sequence.
  y = nn.hard_swish(y)
  y = nn.relu(y)

  # Write the final result to the output reference.
  out_ref[...] = y


out_height = height - kernel_size + 1
out_width = width - kernel_size + 1

x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, out_height, out_width, out_channels), x.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(1, height, width, in_channels), index_map=lambda i: (i, 0, 0, 0)),
    pl.BlockSpec(block_shape=params["kernel"].shape, index_map=lambda i: (0, 0, 0, 0)),
    pl.BlockSpec(block_shape=params["bias"].shape, index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(
    block_shape=(1, out_height, out_width, out_channels),
    index_map=lambda i: (i, 0, 0, 0),
  ),
)(x, params["kernel"], params["bias"]).block_until_ready()
