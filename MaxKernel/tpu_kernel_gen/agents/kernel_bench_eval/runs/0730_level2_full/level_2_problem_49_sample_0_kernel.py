# Imports
import jax
import jax.lax
import jax.nn
import jax.random as random
from flax import linen as nn
from jax.experimental import pallas as pl

# Initialization
batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias = True

key = random.PRNGKey(0)
key, x_key, params_key = random.split(key, 3)

# In JAX/Flax, the convention is channels-last: (N, D, H, W, C)
x = random.normal(x_key, (batch_size, D, H, W, in_channels))

# The 'padding' argument in Flax's ConvTranspose is different from PyTorch's.
# To match PyTorch's behavior with stride=2, padding=1, output_padding=1,
# we need to calculate the padding manually.
# The formula for PyTorch output size: O = (I-1)*S - 2*P + K + OP
# For a dimension of size 16: O = (16-1)*2 - 2*1 + 3 + 1 = 32.
# The 'padding' in Flax ConvTranspose should be 'SAME' to get the standard
# transposed convolution output size, which is I*S = 16*2 = 32.
conv_transpose = nn.ConvTranspose(
  features=out_channels,
  kernel_size=(kernel_size, kernel_size, kernel_size),
  strides=(stride, stride, stride),
  padding="SAME",
  use_bias=bias,
)
params = conv_transpose.init(params_key, x)["params"]


# Computation
def kernel(conv_out_ref, bias_ref, out_ref):
  """
  Pallas kernel for a sequence of bias add, softmax, and sigmoid.
  This kernel is applied after the 3D transposed convolution.
  """
  # Load the slice of the convolution output for the current program.
  y_slice = conv_out_ref[...]

  # Add the bias term. The bias vector is broadcast across the spatial dimensions.
  y_biased = y_slice + bias_ref[...]

  # Apply the softmax function along the channel dimension (last axis).
  y_softmax = jax.nn.softmax(y_biased, axis=-1)

  # Apply the sigmoid activation function.
  y_sigmoid = jax.nn.sigmoid(y_softmax)

  # Write the final result to the output buffer.
  out_ref[...] = y_sigmoid


# First, perform the 3D transposed convolution using standard JAX/Flax.
# This operation is not supported inside a Pallas kernel on TPU.
conv_out = jax.lax.conv_transpose(
  x,
  params["kernel"],
  strides=(stride, stride, stride),
  padding="SAME",
  dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
)

# Then, use pallas_call to fuse the subsequent element-wise operations.
x = pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDtypeStruct((batch_size, D * stride, H * stride, W * stride, out_channels), x.dtype),
  grid=(batch_size, D * stride),
  in_specs=[
    pl.BlockSpec(
      (1, 1, H * stride, W * stride, out_channels),
      lambda i, j: (i, j, 0, 0, 0),
    ),
    pl.BlockSpec(params["bias"].shape, lambda i, j: (0,)),
  ],
  out_specs=pl.BlockSpec(
    (1, 1, H * stride, W * stride, out_channels),
    lambda i, j: (i, j, 0, 0, 0),
  ),
)(conv_out, params["bias"]).block_until_ready()
