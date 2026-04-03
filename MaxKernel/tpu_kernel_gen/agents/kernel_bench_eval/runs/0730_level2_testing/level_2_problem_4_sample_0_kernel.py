# Imports
import jax
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3

key = random.PRNGKey(0)
key_x, key_params = random.split(key)

# JAX uses channels-last convention by default
x = random.normal(key_x, (batch_size, height, width, in_channels))
conv = nn.Conv(features=out_channels, kernel_size=(kernel_size, kernel_size))
params = conv.init(key_params, x)["params"]


# Computation
def kernel(conv_out_ref, bias_ref, out_ref):
  """Pallas kernel for a bias add followed by two Mish activations.

  Args:
    conv_out_ref: Convolution output data reference.
    bias_ref: Convolution bias reference.
    out_ref: Output data reference.
  """
  # Add the bias to the convolution result. The bias will be broadcasted.
  y = conv_out_ref[...] + bias_ref[...]

  # Apply the first Mish activation function.
  y = jax.nn.mish(y)

  # Apply the second Mish activation function.
  y = jax.nn.mish(y)

  # Write the final result to the output buffer.
  out_ref[...] = y


# Perform the convolution operation using standard JAX.
# This operation is not supported inside a Pallas kernel on TPU.
conv_out = jax.lax.conv_general_dilated(
  lhs=x, rhs=params["kernel"], window_strides=(1, 1), padding="SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
)

# Use pallas_call to fuse the bias add and Mish activations.
x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct(conv_out.shape, conv_out.dtype),
  grid=(batch_size,),
  in_specs=[
    pl.BlockSpec(block_shape=(1, height, width, out_channels), index_map=lambda i: (i, 0, 0, 0)),
    pl.BlockSpec(block_shape=(out_channels,), index_map=lambda i: (0,)),
  ],
  out_specs=pl.BlockSpec(block_shape=(1, height, width, out_channels), index_map=lambda i: (i, 0, 0, 0)),
)(conv_out, params["bias"]).block_until_ready()
